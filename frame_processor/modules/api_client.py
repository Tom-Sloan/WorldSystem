print("[DEBUG api_client] Starting api_client imports...")
import sys
sys.stdout.flush()

import os
import cv2
import uuid
import time
import json
import hashlib
import requests
from typing import Dict, List, Optional, Tuple
from datetime import timedelta
print("[DEBUG api_client] Basic imports done, importing google.cloud.storage...")
sys.stdout.flush()

try:
    from google.cloud import storage
    print("[DEBUG api_client] google.cloud.storage imported successfully")
except Exception as e:
    print(f"[DEBUG api_client] ERROR importing google.cloud.storage: {e}")
    storage = None
sys.stdout.flush()

print("[DEBUG api_client] Importing serpapi...")
from serpapi import GoogleSearch
import numpy as np
print("[DEBUG api_client] All imports completed")
sys.stdout.flush()


class APIClient:
    """Handles all external API interactions."""
    
    def __init__(self, cache_dir="api_cache"):
        print("[DEBUG api_client] APIClient.__init__ called")
        sys.stdout.flush()
        
        # Try to use provided cache_dir, fallback to /tmp if needed
        self.cache_dir = cache_dir
        self.lens_cache_dir = os.path.join(cache_dir, "lens")
        self.perplexity_cache_dir = os.path.join(cache_dir, "perplexity")
        self.dimensions_cache_dir = os.path.join(cache_dir, "dimensions")
        
        print("[DEBUG api_client] Creating cache directories...")
        sys.stdout.flush()
        
        # Create cache directories with fallback
        cache_dirs = [self.lens_cache_dir, self.perplexity_cache_dir, self.dimensions_cache_dir]
        try:
            for dir in cache_dirs:
                os.makedirs(dir, exist_ok=True)
        except (OSError, PermissionError) as e:
            # Fallback to /tmp if main directory is read-only
            print(f"[DEBUG api_client] Warning: Cannot create cache in {cache_dir}, using /tmp: {e}")
            self.cache_dir = "/tmp/api_cache"
            self.lens_cache_dir = os.path.join(self.cache_dir, "lens")
            self.perplexity_cache_dir = os.path.join(self.cache_dir, "perplexity")
            self.dimensions_cache_dir = os.path.join(self.cache_dir, "dimensions")
            for dir in [self.lens_cache_dir, self.perplexity_cache_dir, self.dimensions_cache_dir]:
                os.makedirs(dir, exist_ok=True)
        
        print("[DEBUG api_client] Cache directories created")
        sys.stdout.flush()
        
        # Initialize Google Cloud Storage client with error handling
        print("[DEBUG api_client] Initializing Google Cloud Storage client...")
        sys.stdout.flush()
        
        if storage is None:
            print("[DEBUG api_client] Google Cloud Storage module not available, skipping initialization")
            self.storage_client = None
            self.bucket_name = None
        else:
            try:
                print("[DEBUG api_client] Creating storage.Client()...")
                sys.stdout.flush()
                self.storage_client = storage.Client()
                self.bucket_name = os.getenv("GCS_BUCKET_NAME", "worldsystem-frame-processor")
                print(f"[DEBUG api_client] Google Cloud Storage client initialized successfully, bucket: {self.bucket_name}")
            except Exception as e:
                print(f"[DEBUG api_client] Warning: Could not initialize Google Cloud Storage client: {e}")
                self.storage_client = None
                self.bucket_name = None
        sys.stdout.flush()
        
        # API keys and usage flags
        self.serpapi_key = os.getenv("SERPAPI_API_KEY")
        self.pplx_api_key = os.getenv("PPLX_API_KEY")
        
        # Check if APIs should be used
        self.use_serpapi = os.getenv("USE_SERPAPI", "false").lower() == "true"
        self.use_perplexity = os.getenv("USE_PERPLEXITY", "false").lower() == "true"
        self.use_gcs = os.getenv("USE_GCS", "false").lower() == "true"
        
        print(f"[DEBUG api_client] API usage: SERPAPI={self.use_serpapi}, Perplexity={self.use_perplexity}, GCS={self.use_gcs}")
        sys.stdout.flush()
        
        if self.use_serpapi and not self.serpapi_key:
            print("Warning: USE_SERPAPI=true but SERPAPI_API_KEY not set")
        if self.use_perplexity and not self.pplx_api_key:
            print("Warning: USE_PERPLEXITY=true but PPLX_API_KEY not set")
        
    def get_image_hash(self, image: np.ndarray) -> str:
        """Generate hash for an image array."""
        # Convert to bytes and hash
        _, buffer = cv2.imencode('.jpg', image)
        return hashlib.md5(buffer).hexdigest()
    
    def upload_to_gcs(self, image: np.ndarray, object_id: int) -> Optional[Dict]:
        """Upload image to Google Cloud Storage."""
        if not self.use_gcs:
            print("[DEBUG api_client] GCS upload skipped - USE_GCS=false")
            return None
            
        if not self.storage_client or not self.bucket_name:
            print("Warning: Google Cloud Storage not configured, skipping upload")
            return None
            
        try:
            # Save image temporarily
            temp_path = f"/tmp/object_{object_id}_{int(time.time())}.jpg"
            cv2.imwrite(temp_path, image)
            
            # Upload to GCS
            bucket = self.storage_client.bucket(self.bucket_name)
            blob_name = f"frame_processor/{int(time.time())}_{uuid.uuid4().hex}.jpg"
            blob = bucket.blob(blob_name)
            
            blob.upload_from_filename(temp_path)
            
            # Generate signed URL
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=15),
                method="GET"
            )
            
            # Clean up temp file
            os.remove(temp_path)
            
            return {
                "bucket": self.bucket_name,
                "blob_name": blob_name,
                "url": url
            }
            
        except Exception as e:
            print(f"Error uploading to GCS: {e}")
            return None
    
    def cleanup_gcs_blob(self, bucket_name: str, blob_name: str):
        """Delete blob after use."""
        try:
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.delete()
        except Exception as e:
            print(f"Error cleaning up blob: {e}")
    
    def identify_with_google_lens(self, image_url: str, cache_key: str) -> Optional[Dict]:
        """Use Google Lens API to identify objects."""
        # Skip if not enabled
        if not self.use_serpapi:
            print("[DEBUG api_client] Skipping Google Lens search - USE_SERPAPI=false")
            return None
            
        # Skip if API key not configured
        if not self.serpapi_key:
            print("[DEBUG api_client] Skipping Google Lens search - SERPAPI_API_KEY not set")
            return None
            
        # Check cache first
        cache_file = os.path.join(self.lens_cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        try:
            params = {
                "api_key": self.serpapi_key,
                "engine": "google_lens",
                "url": image_url,
                "hl": "en"
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            # Cache results
            with open(cache_file, 'w') as f:
                json.dump(results, f)
            
            return results
            
        except Exception as e:
            print(f"Error using Google Lens API: {e}")
            return None
    
    def extract_products_from_lens(self, lens_results: Dict) -> List[Dict]:
        """Extract product information from Lens results."""
        products = []
        
        # Extract from visual matches
        if "visual_matches" in lens_results:
            for match in lens_results["visual_matches"]:
                if "title" in match:
                    products.append({
                        "title": match["title"],
                        "source": match.get("source", ""),
                        "link": match.get("link", ""),
                        "confidence": match.get("score", 0.5),  # Add confidence score
                        "thumbnail": match.get("thumbnail", "")
                    })
        
        # Extract from products
        if "products" in lens_results:
            for product in lens_results["products"]:
                products.append({
                    "title": product.get("title", ""),
                    "source": product.get("source", ""),
                    "link": product.get("link", ""),
                    "confidence": 0.8,  # Higher confidence for direct products
                    "thumbnail": product.get("thumbnail", "")
                })
        
        return products
    
    def get_dimensions_with_perplexity(self, product_name: str) -> Optional[Dict]:
        """Get product dimensions using Perplexity AI."""
        # Skip if not enabled
        if not self.use_perplexity:
            print("[DEBUG api_client] Skipping Perplexity search - USE_PERPLEXITY=false")
            return None
            
        # Skip if API key not configured
        if not self.pplx_api_key:
            print("[DEBUG api_client] Skipping Perplexity search - PPLX_API_KEY not set")
            return None
            
        # Check cache first
        cache_key = hashlib.md5(product_name.encode()).hexdigest()
        cache_file = os.path.join(self.dimensions_cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            # Check if cache is still valid (30 days)
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < (30 * 86400):  # 30 days in seconds
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        try:
            headers = {
                "Authorization": f"Bearer {self.pplx_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "sonar-medium-online",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that provides accurate physical dimensions of objects. Always respond with exact measurements when available."
                    },
                    {
                        "role": "user", 
                        "content": f"What are the exact physical dimensions (width, height, depth) of the {product_name}? Please provide measurements in standard units (inches or cm). If this is a well-known product, provide the official specifications. Format your response as: Width: X inches/cm, Height: Y inches/cm, Depth: Z inches/cm"
                    }
                ]
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Parse dimensions from response
                dimensions = self._parse_dimensions_from_text(content)
                
                if dimensions:
                    # Cache the result
                    with open(cache_file, 'w') as f:
                        json.dump(dimensions, f)
                    
                    return dimensions
            
        except Exception as e:
            print(f"Error using Perplexity API: {e}")
        
        return None
    
    def _parse_dimensions_from_text(self, text: str) -> Optional[Dict]:
        """Parse dimensions from Perplexity response text."""
        import re
        
        # Look for patterns like "Width: 5.2 inches" or "5.2" x 3.1" x 3.1""
        dimensions = {}
        
        # Pattern 1: Width: X units, Height: Y units, Depth: Z units
        width_match = re.search(r'Width:\s*([\d.]+)\s*(inches?|cm|mm)', text, re.I)
        height_match = re.search(r'Height:\s*([\d.]+)\s*(inches?|cm|mm)', text, re.I)
        depth_match = re.search(r'Depth:\s*([\d.]+)\s*(inches?|cm|mm)', text, re.I)
        
        if width_match and height_match:
            dimensions['width'] = float(width_match.group(1))
            dimensions['height'] = float(height_match.group(1))
            dimensions['depth'] = float(depth_match.group(1)) if depth_match else dimensions['width'] * 0.6
            dimensions['unit'] = width_match.group(2).lower().rstrip('s')
            
            # Convert to meters for 3D scene
            if dimensions['unit'] == 'inches':
                factor = 0.0254
            elif dimensions['unit'] == 'cm':
                factor = 0.01
            elif dimensions['unit'] == 'mm':
                factor = 0.001
            else:
                factor = 1.0
                
            dimensions['width_m'] = dimensions['width'] * factor
            dimensions['height_m'] = dimensions['height'] * factor
            dimensions['depth_m'] = dimensions['depth'] * factor
            
            return dimensions
        
        # Pattern 2: X" x Y" x Z" or X x Y x Z inches
        pattern2 = re.search(r'([\d.]+)["\s]*x\s*([\d.]+)["\s]*x\s*([\d.]+)["s]?\s*(inches?|cm|mm)?', text, re.I)
        if pattern2:
            dimensions['width'] = float(pattern2.group(1))
            dimensions['height'] = float(pattern2.group(2))
            dimensions['depth'] = float(pattern2.group(3))
            dimensions['unit'] = pattern2.group(4).lower().rstrip('s') if pattern2.group(4) else 'inches'
            
            # Convert to meters
            if dimensions['unit'] == 'inches' or dimensions['unit'] == '"':
                factor = 0.0254
            elif dimensions['unit'] == 'cm':
                factor = 0.01
            else:
                factor = 1.0
                
            dimensions['width_m'] = dimensions['width'] * factor
            dimensions['height_m'] = dimensions['height'] * factor
            dimensions['depth_m'] = dimensions['depth'] * factor
            
            return dimensions
        
        return None
    
    def process_object_for_dimensions(self, image: np.ndarray, object_id: int, 
                                    class_name: str) -> Optional[Dict]:
        """Complete pipeline to get object dimensions."""
        # Upload image
        upload_result = self.upload_to_gcs(image, object_id)
        if not upload_result:
            return None
        
        try:
            # Get image hash for caching
            image_hash = self.get_image_hash(image)
            
            # Identify with Google Lens
            lens_results = self.identify_with_google_lens(upload_result["url"], image_hash)
            
            if lens_results:
                products = self.extract_products_from_lens(lens_results)
                
                if products:
                    # Get dimensions for top product
                    best_product = max(products, key=lambda p: p.get('confidence', 0))
                    dimensions = self.get_dimensions_with_perplexity(best_product['title'])
                    
                    if dimensions:
                        return {
                            'product_name': best_product['title'],
                            'confidence': best_product.get('confidence', 0.5),
                            'dimensions': dimensions,
                            'all_products': products[:5]  # Top 5 matches
                        }
            
        finally:
            # Always cleanup GCS
            self.cleanup_gcs_blob(upload_result["bucket"], upload_result["blob_name"])
        
        return None