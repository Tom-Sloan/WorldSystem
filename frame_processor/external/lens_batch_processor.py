"""
Batch processor for Google Lens API calls.

This module provides efficient batch processing of object identification
requests with deduplication, rate limiting, and error handling.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set, Tuple
import numpy as np
from dataclasses import dataclass, field
import time
from collections import deque
import cv2
import hashlib

from core.utils import get_logger
from core.config import Config
from external.lens_identifier import LensIdentifier, VisualSimilarityCache

logger = get_logger(__name__)


@dataclass
class BatchItem:
    """Single item in a batch."""
    item_id: str
    object_id: str
    image: np.ndarray
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher priority items processed first
    retries: int = 0
    last_error: Optional[str] = None


@dataclass 
class BatchResult:
    """Result from batch processing."""
    batch_id: str
    items_processed: int
    items_succeeded: int
    items_failed: int
    items_cached: int
    processing_time_ms: float
    results: Dict[str, Any]  # item_id -> result


class LensBatchProcessor:
    """
    Batch processor for Google Lens API calls.
    
    Features:
    - Automatic batching with size and time limits
    - Visual deduplication within batches
    - Priority queue for important items
    - Retry logic for failed items
    - Comprehensive monitoring
    """
    
    def __init__(self, config: Config, lens_identifier: LensIdentifier, publisher=None):
        self.config = config
        self.lens_identifier = lens_identifier
        self.publisher = publisher
        self.result_callback = None
        
        # Batch configuration
        self.batch_size = config.lens_batch_size
        self.batch_wait_ms = config.lens_batch_wait_ms
        self.enable_dedup = config.lens_enable_similar_dedup
        
        # Processing queue
        self.pending_items: deque[BatchItem] = deque()
        self.processing_lock = asyncio.Lock()
        
        # Deduplication
        self.similarity_cache = VisualSimilarityCache(
            max_size=100,  # Small cache for batch dedup
            similarity_threshold=0.98  # High threshold for batch dedup
        )
        
        # Monitoring
        self.total_batches = 0
        self.total_items = 0
        self.total_api_calls = 0
        self.dedup_savings = 0
        
        # Background processing
        self.processing_task: Optional[asyncio.Task] = None
        self.running = False
        
        logger.info(f"Initialized LensBatchProcessor with batch_size={self.batch_size}, "
                   f"wait_time={self.batch_wait_ms}ms, dedup={self.enable_dedup}")
    
    async def start(self):
        """Start batch processing."""
        if self.running:
            return
        
        self.running = True
        self.processing_task = asyncio.create_task(self._process_batches())
        logger.info("Started batch processing")
    
    async def stop(self):
        """Stop batch processing."""
        self.running = False
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        # Process remaining items
        if self.pending_items:
            logger.info(f"Processing {len(self.pending_items)} remaining items...")
            await self._process_current_batch()
        
        logger.info("Stopped batch processing")
    
    async def add_item(self, item_id: str, object_id: str, image: np.ndarray,
                       metadata: Optional[Dict[str, Any]] = None,
                       priority: int = 0) -> None:
        """
        Add an item to the processing queue.
        
        Args:
            item_id: Unique item identifier
            object_id: Object/track identifier
            image: Image to process
            metadata: Optional metadata
            priority: Processing priority (higher = sooner)
        """
        item = BatchItem(
            item_id=item_id,
            object_id=object_id,
            image=image,
            timestamp=time.time(),
            metadata=metadata or {},
            priority=priority
        )
        
        async with self.processing_lock:
            # Add to queue based on priority
            if priority > 0:
                # Insert at appropriate position
                inserted = False
                for i, existing in enumerate(self.pending_items):
                    if existing.priority < priority:
                        self.pending_items.insert(i, item)
                        inserted = True
                        break
                if not inserted:
                    self.pending_items.append(item)
            else:
                self.pending_items.append(item)
            
            logger.debug(f"Added item {item_id} to batch queue (priority={priority}, "
                        f"queue_size={len(self.pending_items)})")
    
    async def _process_batches(self):
        """Background task to process batches."""
        last_batch_time = time.time()
        
        while self.running:
            try:
                async with self.processing_lock:
                    current_time = time.time()
                    time_since_last = (current_time - last_batch_time) * 1000
                    
                    should_process = (
                        len(self.pending_items) >= self.batch_size or
                        (len(self.pending_items) > 0 and time_since_last >= self.batch_wait_ms)
                    )
                    
                    if should_process:
                        await self._process_current_batch()
                        last_batch_time = current_time
                
                # Short sleep to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)
    
    async def _process_current_batch(self):
        """Process items currently in the queue."""
        if not self.pending_items:
            return
        
        start_time = time.time()
        batch_id = f"batch_{int(start_time)}_{self.total_batches}"
        
        # Extract batch items
        batch_items = []
        for _ in range(min(self.batch_size, len(self.pending_items))):
            batch_items.append(self.pending_items.popleft())
        
        logger.info(f"Processing batch {batch_id} with {len(batch_items)} items")
        
        # Deduplicate if enabled
        if self.enable_dedup:
            batch_items, dedup_map = await self._deduplicate_batch(batch_items)
            self.dedup_savings += len(dedup_map)
        else:
            dedup_map = {}
        
        # Process unique items
        results = {}
        items_succeeded = 0
        items_failed = 0
        items_cached = 0
        
        for item in batch_items:
            try:
                # Check main cache first
                cached_result = await self.lens_identifier.cache.get(item.image)
                if cached_result is not None:
                    results[item.item_id] = {
                        "object_id": item.object_id,
                        "identification": cached_result,
                        "from_cache": True
                    }
                    items_cached += 1
                    continue
                
                # Make API call
                identification = await self.lens_identifier._call_lens_api(item.image)
                
                # Cache result
                await self.lens_identifier.cache.put(item.image, identification)
                
                results[item.item_id] = {
                    "object_id": item.object_id,
                    "identification": identification,
                    "from_cache": False
                }
                items_succeeded += 1
                self.total_api_calls += 1
                
            except Exception as e:
                logger.error(f"Failed to process item {item.item_id}: {e}")
                item.last_error = str(e)
                item.retries += 1
                
                # Retry logic
                if item.retries < 3:
                    # Re-add to queue with lower priority
                    await self.add_item(
                        item.item_id,
                        item.object_id,
                        item.image,
                        item.metadata,
                        priority=item.priority - 1
                    )
                else:
                    results[item.item_id] = {
                        "object_id": item.object_id,
                        "identification": None,
                        "error": item.last_error
                    }
                    items_failed += 1
        
        # Apply deduplication results
        for dup_id, original_id in dedup_map.items():
            if original_id in results:
                results[dup_id] = results[original_id].copy()
                results[dup_id]["deduplicated_from"] = original_id
        
        # Update stats
        self.total_batches += 1
        self.total_items += len(batch_items) + len(dedup_map)
        
        # Create batch result
        processing_time = (time.time() - start_time) * 1000
        batch_result = BatchResult(
            batch_id=batch_id,
            items_processed=len(batch_items) + len(dedup_map),
            items_succeeded=items_succeeded,
            items_failed=items_failed,
            items_cached=items_cached + len(dedup_map),
            processing_time_ms=processing_time,
            results=results
        )
        
        # Log performance
        logger.info(f"Batch {batch_id} completed: "
                   f"{batch_result.items_processed} items "
                   f"({items_succeeded} succeeded, {items_failed} failed, "
                   f"{items_cached} cached, {len(dedup_map)} deduped) "
                   f"in {processing_time:.1f}ms")
        
        # Emit results (would typically use a callback or event system)
        await self._emit_batch_results(batch_result)
    
    async def _deduplicate_batch(self, items: List[BatchItem]) -> Tuple[List[BatchItem], Dict[str, str]]:
        """
        Deduplicate visually similar items within a batch.
        
        Args:
            items: List of batch items
            
        Returns:
            Tuple of (unique_items, dedup_map)
            where dedup_map maps duplicate_id -> original_id
        """
        if not items:
            return items, {}
        
        unique_items = []
        dedup_map = {}
        processed_hashes = {}
        
        for item in items:
            # Compute perceptual hash
            item_hash = self._compute_perceptual_hash(item.image)
            
            # Check for duplicates
            is_duplicate = False
            for unique_item in unique_items:
                unique_hash = processed_hashes.get(unique_item.item_id)
                if unique_hash and self._hashes_similar(item_hash, unique_hash):
                    dedup_map[item.item_id] = unique_item.item_id
                    is_duplicate = True
                    logger.debug(f"Item {item.item_id} is duplicate of {unique_item.item_id}")
                    break
            
            if not is_duplicate:
                unique_items.append(item)
                processed_hashes[item.item_id] = item_hash
        
        if dedup_map:
            logger.info(f"Deduplicated {len(dedup_map)} items in batch")
        
        return unique_items, dedup_map
    
    def _compute_perceptual_hash(self, image: np.ndarray) -> np.ndarray:
        """Compute perceptual hash for similarity comparison."""
        # Resize to 32x32 for more detailed comparison
        resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
        
        # Compute average
        avg = gray.mean()
        
        # Create binary hash
        hash_bits = (gray > avg).astype(np.uint8)
        
        return hash_bits
    
    def _hashes_similar(self, hash1: np.ndarray, hash2: np.ndarray, 
                       threshold: float = 0.95) -> bool:
        """Check if two perceptual hashes are similar."""
        if hash1.shape != hash2.shape:
            return False
        
        # Calculate similarity (1 - normalized Hamming distance)
        similarity = 1.0 - np.sum(hash1 != hash2) / hash1.size
        
        return similarity >= threshold
    
    async def _emit_batch_results(self, batch_result: BatchResult):
        """
        Emit batch results to listeners.
        
        Publishes results to RabbitMQ for downstream processing.
        """
        # Emit via callback if registered
        if hasattr(self, 'result_callback') and self.result_callback:
            try:
                await self.result_callback(batch_result)
            except Exception as e:
                logger.error(f"Error in result callback: {e}")
        
        # Also emit individual results for compatibility
        if hasattr(self, 'publisher') and self.publisher:
            for item_id, result in batch_result.results.items():
                if result.get("identification"):
                    try:
                        await self.publisher.publish_identification({
                            'object_id': result['object_id'],
                            'identification': result['identification'],
                            'timestamp': time.time(),
                            'batch_id': batch_result.batch_id,
                            'from_cache': result.get('from_cache', False)
                        })
                    except Exception as e:
                        logger.error(f"Error publishing result for {item_id}: {e}")
        
        logger.info(f"Batch {batch_result.batch_id} results emitted")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processor statistics."""
        return {
            "total_batches": self.total_batches,
            "total_items": self.total_items,
            "total_api_calls": self.total_api_calls,
            "dedup_savings": self.dedup_savings,
            "pending_items": len(self.pending_items),
            "avg_batch_size": self.total_items / max(1, self.total_batches),
            "dedup_rate": self.dedup_savings / max(1, self.total_items),
            "cache_hit_rate": self.lens_identifier.cache_hits / max(1, self.lens_identifier.total_queries)
        }
    
    async def flush(self) -> Optional[BatchResult]:
        """
        Force process all pending items immediately.
        
        Returns:
            BatchResult if items were processed, None otherwise
        """
        async with self.processing_lock:
            if self.pending_items:
                logger.info(f"Flushing {len(self.pending_items)} pending items")
                return await self._process_current_batch()
        return None
