"""
Google Lens API integration with visual similarity caching.

This module provides object identification using Google Lens API
with caching to reduce API calls and costs.
"""

import asyncio
from typing import Dict, List, Any, Optional
import numpy as np
import hashlib
from collections import OrderedDict
import time
import cv2

from core.utils import get_logger
from core.config import Config

logger = get_logger(__name__)


class LensIdentifier:
    """
    Manages Google Lens API calls for object identification.
    Includes caching and rate limiting for efficiency.
    """
    
    def __init__(self, config: Config, api_client=None):
        self.config = config
        self.api_client = api_client
        
        # Visual similarity cache
        self.cache = VisualSimilarityCache(
            max_size=config.lens_cache_size,
            similarity_threshold=config.lens_cache_similarity_threshold
        )
        
        # Rate limiter
        self.rate_limiter = RateLimiter(
            calls_per_second=config.lens_api_rate_limit
        )
        
        # Stats
        self.total_queries = 0
        self.cache_hits = 0
        
        logger.info(f"Initialized LensIdentifier with cache size={config.lens_cache_size}, "
                   f"rate limit={config.lens_api_rate_limit}/s")
        
    async def identify_objects(self, enhanced_crops: List[Dict]) -> List[Dict]:
        """
        Identify objects using Google Lens API with caching.
        Returns identification results for each crop.
        """
        results = []
        
        for crop_data in enhanced_crops:
            self.total_queries += 1
            
            # Check cache first
            cached_result = await self.cache.get(crop_data["enhanced_crop"])
            if cached_result is not None:
                self.cache_hits += 1
                logger.debug(f"Cache hit for object {crop_data['object_id']}")
                results.append({
                    "object_id": crop_data["object_id"],
                    "identification": cached_result,
                    "from_cache": True
                })
                continue
            
            # Rate-limited API call
            try:
                async with self.rate_limiter:
                    identification = await self._call_lens_api(
                        crop_data["enhanced_crop"]
                    )
                
                # Cache the result
                await self.cache.put(crop_data["enhanced_crop"], identification)
                
                results.append({
                    "object_id": crop_data["object_id"],
                    "identification": identification,
                    "from_cache": False
                })
                
            except Exception as e:
                logger.error(f"Lens API error for {crop_data['object_id']}: {e}")
                results.append({
                    "object_id": crop_data["object_id"],
                    "identification": None,
                    "error": str(e)
                })
        
        # Log cache performance periodically
        if self.total_queries > 0 and self.total_queries % 100 == 0:
            hit_rate = (self.cache_hits / self.total_queries) * 100
            logger.info(f"Lens cache hit rate: {hit_rate:.1f}% "
                       f"({self.cache_hits}/{self.total_queries})")
        
        return results
    
    async def _call_lens_api(self, image: np.ndarray) -> Dict[str, Any]:
        """Make actual API call to Google Lens via SerpAPI."""
        try:
            # First, we need to upload the image to GCS to get a URL
            image_url = None
            blob_name = None
            
            if self.api_client and hasattr(self.api_client, 'upload_image_to_gcs'):
                # Upload image and get URL
                upload_result = await self.api_client.upload_image_to_gcs(image)
                if upload_result:
                    image_url = upload_result["url"]
                    blob_name = upload_result["blob_name"]
            
            if not image_url:
                logger.error("Failed to upload image for Lens API")
                return self._empty_result()
            
            # Call Google Lens API via SerpAPI
            try:
                from serpapi import GoogleSearch
                import os
                
                api_key = os.environ.get("SERPAPI_API_KEY")
                if not api_key:
                    logger.error("SERPAPI_API_KEY not set")
                    return self._empty_result()
                
                # Setup parameters for Google Lens search
                params = {
                    "api_key": api_key,
                    "engine": "google_lens",
                    "url": image_url,
                    "hl": "en"
                }
                
                # Create the search
                search = GoogleSearch(params)
                
                # Get results
                results = search.get_dict()
                
                # Extract information from results
                extracted_info = self._extract_lens_info(results)
                
                return extracted_info
                
            finally:
                # Always cleanup GCS blob
                if blob_name and self.api_client:
                    await self.api_client.cleanup_gcs_blob(blob_name)
                    
        except Exception as e:
            logger.error(f"Error calling Lens API: {e}", exc_info=True)
            return self._empty_result()
    
    def _extract_lens_info(self, lens_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant information from Lens API results."""
        info = {
            "name": "Unknown",
            "confidence": 0.0,
            "category": "",
            "dimensions": {},
            "description": "",
            "products": [],
            "visual_matches": []
        }
        
        # Extract from visual matches
        if "visual_matches" in lens_results:
            visual_matches = lens_results["visual_matches"]
            if visual_matches:
                # Use first match as primary identification
                first_match = visual_matches[0]
                info["name"] = first_match.get("title", "Unknown")
                info["description"] = first_match.get("source", "")
                info["confidence"] = 0.9  # High confidence for visual matches
                
                # Store all visual matches
                for match in visual_matches[:5]:  # Limit to top 5
                    info["visual_matches"].append({
                        "title": match.get("title", ""),
                        "source": match.get("source", ""),
                        "link": match.get("link", ""),
                        "thumbnail": match.get("thumbnail", "")
                    })
        
        # Extract from products
        if "products" in lens_results:
            products = lens_results["products"]
            if products and not info["name"]:
                # Use product info if no visual match
                first_product = products[0]
                info["name"] = first_product.get("title", "Unknown")
                info["confidence"] = 0.7  # Lower confidence for products
            
            # Store product information
            for product in products[:5]:  # Limit to top 5
                info["products"].append({
                    "title": product.get("title", ""),
                    "source": product.get("source", ""),
                    "price": product.get("price", {}).get("value", ""),
                    "link": product.get("link", "")
                })
        
        # Determine category based on results
        if info["visual_matches"] or info["products"]:
            info["category"] = "identified"
        
        return info
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            "name": "Unknown",
            "confidence": 0.0,
            "category": "",
            "dimensions": {},
            "description": "",
            "products": [],
            "visual_matches": []
        }


class VisualSimilarityCache:
    """
    Cache for storing identification results based on visual similarity.
    Uses perceptual hashing to detect similar images.
    """
    
    def __init__(self, max_size: int = 1000, similarity_threshold: float = 0.95):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.cache: OrderedDict[str, Dict] = OrderedDict()
        self._lock = asyncio.Lock()
        
    def _compute_hash(self, image: np.ndarray) -> str:
        """Compute perceptual hash of image."""
        # Resize to 8x8 for DCT-based hashing
        resized = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale if needed
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
        
        # Compute DCT
        dct = cv2.dct(gray.astype(np.float32))
        
        # Use top-left 8x8 coefficients (excluding DC)
        dct_subset = dct[:8, :8].flatten()[1:]
        
        # Compute median
        median = np.median(dct_subset)
        
        # Generate binary hash
        hash_bits = (dct_subset > median).astype(np.uint8)
        
        # Convert to hex string
        hash_str = ''.join(str(b) for b in hash_bits)
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    async def get(self, image: np.ndarray) -> Optional[Dict]:
        """Get cached result for visually similar image."""
        async with self._lock:
            image_hash = self._compute_hash(image)
            
            # Direct hit
            if image_hash in self.cache:
                # Move to end (LRU)
                self.cache.move_to_end(image_hash)
                return self.cache[image_hash]
            
            # TODO: Implement similarity search for near matches
            # For now, return None for cache miss
            return None
    
    async def put(self, image: np.ndarray, result: Dict):
        """Store result in cache."""
        async with self._lock:
            image_hash = self._compute_hash(image)
            
            # Add to cache
            self.cache[image_hash] = result
            
            # Evict oldest if over capacity
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_second: int):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second if calls_per_second > 0 else 0
        self.last_call = 0
        self._lock = asyncio.Lock()
        
    async def __aenter__(self):
        async with self._lock:
            current_time = time.time()
            time_since_last = current_time - self.last_call
            
            if time_since_last < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last)
            
            self.last_call = time.time()
            
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass