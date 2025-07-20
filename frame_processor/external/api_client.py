"""
Unified API client for external services.

This module provides a clean interface to Google Cloud Storage, Google Lens (via SerpAPI),
and Perplexity AI, preserving all existing functionality while fitting into the new 
modular architecture.
"""

import os
import json
import time
import asyncio
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

import cv2
import numpy as np
import aiohttp
from google.cloud import storage
from google.oauth2 import service_account

from core.utils import get_logger, PerformanceTimer
from core.config import Config


# Global aiohttp session and lock
_aiohttp_session = None
_session_lock = asyncio.Lock()


async def get_aiohttp_session() -> aiohttp.ClientSession:
    """Get or create global aiohttp session with thread safety."""
    global _aiohttp_session
    async with _session_lock:
        if _aiohttp_session is None or _aiohttp_session.closed:
            _aiohttp_session = aiohttp.ClientSession()
        return _aiohttp_session


logger = get_logger(__name__)


class APIClient:
    """
    Unified client for all external API interactions.
    
    This preserves all functionality from the original modules/api_client.py
    while providing a cleaner interface for the refactored architecture.
    """
    
    def __init__(self, config: Config):
        """
        Initialize API client with configuration.
        
        Args:
            config: Application configuration
        """
        self.config = config
        
        # Feature flags
        self.use_gcs = config.use_gcs
        self.use_serpapi = config.use_serpapi
        self.use_perplexity = config.use_perplexity
        
        # API keys
        self.serpapi_key = config.serpapi_key
        self.perplexity_key = config.perplexity_key
        
        # GCS setup
        self.gcs_client = None
        self.gcs_bucket = None
        if self.use_gcs:
            self._initialize_gcs()
        
        # Cache setup
        self.cache_dir = self._setup_cache_dir()
        
        # Cache for API results
        self._lens_cache = {}
        self._perplexity_cache = {}
        
        # Rate limiting
        self._last_serpapi_call = 0
        self._last_perplexity_call = 0
        self._serpapi_min_interval = 1.0  # seconds
        self._perplexity_min_interval = 0.5  # seconds
        
        logger.info(
            f"APIClient initialized - GCS: {self.use_gcs}, "
            f"Lens: {self.use_serpapi}, Perplexity: {self.use_perplexity}"
        )
    
    def _initialize_gcs(self):
        """Initialize Google Cloud Storage client."""
        try:
            # Check for service account file
            service_account_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if service_account_path and os.path.exists(service_account_path):
                credentials = service_account.Credentials.from_service_account_file(
                    service_account_path
                )
                self.gcs_client = storage.Client(credentials=credentials)
            else:
                # Try default credentials
                self.gcs_client = storage.Client()
            
            # Get bucket
            bucket_name = self.config.gcs_bucket_name
            if bucket_name:
                self.gcs_bucket = self.gcs_client.bucket(bucket_name)
                logger.info(f"Connected to GCS bucket: {bucket_name}")
            else:
                logger.warning("GCS_BUCKET_NAME not set, GCS upload disabled")
                self.use_gcs = False
                
        except Exception as e:
            logger.error(f"Failed to initialize GCS: {e}")
            self.use_gcs = False
    
    def _setup_cache_dir(self) -> Path:
        """Setup cache directory, handling read-only filesystems."""
        # Try standard cache locations
        cache_locations = [
            Path("cache"),
            Path("/tmp/frame_processor_cache"),
            Path("/var/tmp/frame_processor_cache")
        ]
        
        for cache_dir in cache_locations:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                # Test write access
                test_file = cache_dir / ".write_test"
                test_file.write_text("test")
                test_file.unlink()
                logger.info(f"Using cache directory: {cache_dir}")
                return cache_dir
            except Exception as e:
                logger.debug(f"Cannot use {cache_dir}: {e}")
                continue
        
        # Fallback to memory-only cache
        logger.warning("No writable cache directory found, using memory-only cache")
        return None
    
    async def upload_image_to_gcs(self, image: np.ndarray) -> Optional[Dict[str, str]]:
        """
        Upload image to Google Cloud Storage asynchronously.
        
        Args:
            image: Image array to upload
            
        Returns:
            Dict with 'url' and 'blob_name' or None on failure
        """
        import uuid
        object_name = f"lens_objects/{int(time.time())}_{uuid.uuid4().hex}.jpg"
        
        # Use sync method in thread pool
        loop = asyncio.get_event_loop()
        url = await loop.run_in_executor(None, self.upload_to_gcs, image, object_name)
        
        if url:
            return {
                "url": url,
                "blob_name": object_name,
                "bucket": self.config.gcs_bucket_name
            }
        return None
    
    def upload_to_gcs(self, image: np.ndarray, object_name: str) -> Optional[str]:
        """
        Upload image to Google Cloud Storage.
        
        Args:
            image: Image array to upload
            object_name: Name for the object in GCS
            
        Returns:
            Public URL of uploaded image or None on failure
        """
        if not self.use_gcs or not self.gcs_bucket:
            return None
        
        with PerformanceTimer("gcs_upload", logger):
            try:
                # Generate unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{object_name}_{timestamp}.jpg"
                
                # Encode image
                _, buffer = cv2.imencode('.jpg', image, 
                                       [cv2.IMWRITE_JPEG_QUALITY, 90])
                
                # Upload to GCS
                blob = self.gcs_bucket.blob(f"frame_processor/{filename}")
                blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')
                
                # Make public if bucket allows
                try:
                    blob.make_public()
                    url = blob.public_url
                except Exception:
                    # Generate signed URL as fallback
                    url = blob.generate_signed_url(
                        version="v4",
                        expiration=timedelta(days=7),
                        method="GET"
                    )
                
                logger.debug(f"Uploaded to GCS: {filename}")
                return url
                
            except Exception as e:
                logger.error(f"GCS upload failed: {e}")
                return None
    
    async def cleanup_gcs_blob(self, blob_name: str) -> bool:
        """
        Delete a blob from GCS asynchronously.
        
        Args:
            blob_name: Name of the blob to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.use_gcs or not self.gcs_bucket:
            return False
        
        loop = asyncio.get_event_loop()
        
        def _delete_blob():
            try:
                blob = self.gcs_bucket.blob(blob_name)
                blob.delete()
                logger.debug(f"Deleted GCS blob: {blob_name}")
                return True
            except Exception as e:
                logger.error(f"Error deleting GCS blob {blob_name}: {e}")
                return False
        
        return await loop.run_in_executor(None, _delete_blob)
    
    async def identify_with_google_lens(self, image_url: str) -> Optional[Dict[str, Any]]:
        """
        Identify object using Google Lens via SerpAPI.
        
        Args:
            image_url: URL of image to analyze
            
        Returns:
            Lens results or None on failure
        """
        if not self.use_serpapi or not self.serpapi_key:
            return None
        
        # Check cache
        cache_key = hashlib.md5(image_url.encode()).hexdigest()
        if cache_key in self._lens_cache:
            logger.debug(f"Using cached Lens result for {cache_key}")
            return self._lens_cache[cache_key]
        
        # Rate limiting
        time_since_last = time.time() - self._last_serpapi_call
        if time_since_last < self._serpapi_min_interval:
            await asyncio.sleep(self._serpapi_min_interval - time_since_last)
        
        with PerformanceTimer("google_lens_api", logger):
            try:
                params = {
                    "engine": "google_lens",
                    "url": image_url,
                    "api_key": self.serpapi_key
                }
                
                session = await get_aiohttp_session()
                async with session.get(
                    "https://serpapi.com/search",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                self._last_serpapi_call = time.time()
                
                # Cache result
                self._lens_cache[cache_key] = result
                self._save_cache("lens_cache.json", self._lens_cache)
                
                logger.debug(f"Google Lens identified: {result.get('knowledge_graph', {}).get('title', 'Unknown')}")
                return result
                
            except Exception as e:
                logger.error(f"Google Lens API failed: {e}")
                return None
    
    def extract_products_from_lens(self, lens_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract product information from Google Lens results.
        
        This preserves the exact extraction logic from the original implementation.
        """
        if not lens_result:
            return {}
        
        product_info = {}
        
        # Extract from knowledge graph
        knowledge_graph = lens_result.get("knowledge_graph", {})
        if knowledge_graph:
            product_info["product_name"] = knowledge_graph.get("title", "")
            product_info["description"] = knowledge_graph.get("subtitle", "")
        
        # Extract from visual matches
        visual_matches = lens_result.get("visual_matches", [])
        if visual_matches:
            # Use first match as primary
            first_match = visual_matches[0]
            if not product_info.get("product_name"):
                product_info["product_name"] = first_match.get("title", "")
            
            # Collect all sources
            product_info["sources"] = [
                {
                    "title": match.get("title", ""),
                    "link": match.get("link", ""),
                    "source": match.get("source", "")
                }
                for match in visual_matches[:5]  # Limit to top 5
            ]
        
        # Extract from reverse image search
        reverse_image = lens_result.get("reverse_image_search", {})
        if reverse_image:
            link = reverse_image.get("link", "")
            if link and "detected_text" not in product_info:
                product_info["detected_text"] = link
        
        return product_info
    
    async def get_dimensions_with_perplexity(self, product_name: str, 
                                           product_description: str = "") -> Optional[Dict[str, float]]:
        """
        Query Perplexity AI for product dimensions.
        
        Args:
            product_name: Name of the product
            product_description: Additional description
            
        Returns:
            Dictionary with width, height, depth in meters or None
        """
        if not self.use_perplexity or not self.perplexity_key or not product_name:
            return None
        
        # Check cache
        cache_key = hashlib.md5(f"{product_name}_{product_description}".encode()).hexdigest()
        if cache_key in self._perplexity_cache:
            logger.debug(f"Using cached dimensions for {product_name}")
            return self._perplexity_cache[cache_key]
        
        # Rate limiting
        time_since_last = time.time() - self._last_perplexity_call
        if time_since_last < self._perplexity_min_interval:
            await asyncio.sleep(self._perplexity_min_interval - time_since_last)
        
        with PerformanceTimer("perplexity_api", logger):
            try:
                # Construct prompt (same as original)
                prompt = f"""What are the typical physical dimensions (width, height, depth) of a {product_name}?
                {f'Additional context: {product_description}' if product_description else ''}
                Please provide the dimensions in metric units (meters).
                Format the response as: width: X.XX m, height: X.XX m, depth: X.XX m"""
                
                headers = {
                    "Authorization": f"Bearer {self.perplexity_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "llama-3.1-sonar-small-128k-online",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that provides accurate product dimensions."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 150
                }
                
                session = await get_aiohttp_session()
                async with session.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                self._last_perplexity_call = time.time()
                
                # Parse dimensions from response
                dimensions = self._parse_dimensions_from_text(
                    result["choices"][0]["message"]["content"]
                )
                
                if dimensions:
                    # Cache result
                    self._perplexity_cache[cache_key] = dimensions
                    self._save_cache("perplexity_cache.json", self._perplexity_cache)
                    
                    logger.debug(f"Got dimensions for {product_name}: {dimensions}")
                
                return dimensions
                
            except Exception as e:
                logger.error(f"Perplexity API failed: {e}")
                return None
    
    def _parse_dimensions_from_text(self, text: str) -> Optional[Dict[str, float]]:
        """
        Parse dimensions from Perplexity response text.
        
        Looks for patterns like "width: 0.15 m, height: 0.20 m, depth: 0.10 m"
        """
        import re
        
        dimensions = {}
        
        # Patterns to match
        patterns = {
            'width': r'width[:\s]+(\d+\.?\d*)\s*(?:m|meter|metres?)',
            'height': r'height[:\s]+(\d+\.?\d*)\s*(?:m|meter|metres?)',
            'depth': r'depth[:\s]+(\d+\.?\d*)\s*(?:m|meter|metres?)'
        }
        
        for dim, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    dimensions[dim] = float(match.group(1))
                except ValueError:
                    pass
        
        # Only return if we got at least 2 dimensions
        if len(dimensions) >= 2:
            # Fill missing dimension with average of others
            if 'width' not in dimensions:
                dimensions['width'] = (dimensions['height'] + dimensions['depth']) / 2
            elif 'height' not in dimensions:
                dimensions['height'] = (dimensions['width'] + dimensions['depth']) / 2
            elif 'depth' not in dimensions:
                dimensions['depth'] = (dimensions['width'] + dimensions['height']) / 2
            
            return dimensions
        
        return None
    
    async def process_object_for_dimensions(self, image: np.ndarray, 
                                          object_name: str) -> Dict[str, Any]:
        """
        Complete pipeline: upload, identify, and get dimensions.
        
        Args:
            image: Object image (ROI)
            object_name: Initial object class name
            
        Returns:
            Dictionary with all API results
        """
        result = {
            "object_name": object_name,
            "gcs_url": None,
            "product_name": None,
            "product_description": None,
            "dimensions": None,
            "sources": []
        }
        
        # Step 1: Upload to GCS if enabled
        if self.use_gcs:
            url = self.upload_to_gcs(image, object_name)
            if url:
                result["gcs_url"] = url
            else:
                logger.warning("GCS upload failed, skipping API processing")
                return result
        else:
            # For testing without GCS, could save locally and use file:// URL
            logger.debug("GCS disabled, skipping upload")
            return result
        
        # Step 2: Identify with Google Lens
        if self.use_serpapi and result["gcs_url"]:
            lens_result = await self.identify_with_google_lens(result["gcs_url"])
            if lens_result:
                product_info = self.extract_products_from_lens(lens_result)
                result.update(product_info)
        
        # Step 3: Get dimensions with Perplexity
        if self.use_perplexity and result.get("product_name"):
            dimensions = await self.get_dimensions_with_perplexity(
                result["product_name"],
                result.get("product_description", "")
            )
            if dimensions:
                result["dimensions"] = dimensions
        
        return result
    
    def _save_cache(self, filename: str, cache_data: Dict):
        """Save cache to disk if possible."""
        if not self.cache_dir:
            return
        
        try:
            cache_file = self.cache_dir / filename
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to save cache: {e}")
    
    def _load_cache(self, filename: str) -> Dict:
        """Load cache from disk if available."""
        if not self.cache_dir:
            return {}
        
        try:
            cache_file = self.cache_dir / filename
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"Failed to load cache: {e}")
        
        return {}
    
    def load_caches(self):
        """Load all caches from disk."""
        self._lens_cache = self._load_cache("lens_cache.json")
        self._perplexity_cache = self._load_cache("perplexity_cache.json")
        logger.info(
            f"Loaded caches - Lens: {len(self._lens_cache)} entries, "
            f"Perplexity: {len(self._perplexity_cache)} entries"
        )
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "lens_cache_size": len(self._lens_cache),
            "perplexity_cache_size": len(self._perplexity_cache),
            "total_cached_results": len(self._lens_cache) + len(self._perplexity_cache)
        }
    
    async def close(self):
        """Cleanup resources including aiohttp session."""
        global _aiohttp_session
        async with _session_lock:
            if _aiohttp_session and not _aiohttp_session.closed:
                await _aiohttp_session.close()
                _aiohttp_session = None