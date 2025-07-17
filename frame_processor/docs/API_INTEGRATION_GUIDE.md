# API Integration Guide

This document provides comprehensive information about implementing the API integrations used in the ReverseLookup project in other applications.

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Google Cloud Storage API](#google-cloud-storage-api)
3. [Google Lens API (via SerpAPI)](#google-lens-api-via-serpapi)
4. [Perplexity AI API](#perplexity-ai-api)
5. [Apipie API](#apipie-api)
6. [Caching Strategy](#caching-strategy)

## Environment Setup

### Required API Keys

Create a `.env` file with the following keys:

```env
SERPAPI_API_KEY=your_serpapi_api_key_here
GOOGLE_APPLICATION_CREDENTIALS="path/to/your/google_credentials.json"
OPENAI_API_KEY=your_openai_api_key_here
PPLX_API_KEY=your_perplexity_api_key_here
```

### Python Dependencies

```bash
pip install python-dotenv google-cloud-storage google-cloud-vision serpapi requests pillow
```

### Load Environment Variables

```python
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
```

## Google Cloud Storage API

### Purpose
Used to upload images temporarily for Google Lens API processing.

### Setup
1. Create a Google Cloud project
2. Enable Cloud Storage API
3. Create a service account and download JSON credentials
4. Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable

### Implementation

```python
from google.cloud import storage
import uuid
import time
from datetime import timedelta

def upload_to_gcs(image_path, bucket_name="your-bucket-name"):
    """Upload image to Google Cloud Storage and return signed URL."""
    try:
        # Create a storage client
        storage_client = storage.Client()
        
        # Get the bucket
        bucket = storage_client.bucket(bucket_name)
        
        # Create a unique blob name
        blob_name = f"camera_analysis/{int(time.time())}_{uuid.uuid4().hex}.jpg"
        blob = bucket.blob(blob_name)
        
        # Upload the file
        print(f"Uploading image to GCS bucket: {bucket_name}/{blob_name}")
        blob.upload_from_filename(image_path)
        
        # Generate a signed URL that expires after 15 minutes
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=15),
            method="GET"
        )
        
        print(f"Image uploaded and available at: {url}")
        
        return {
            "bucket": bucket_name,
            "blob_name": blob_name,
            "url": url
        }
        
    except Exception as e:
        print(f"Error uploading to Google Cloud Storage: {e}")
        return None

def cleanup_gcs_blob(bucket_name, blob_name):
    """Delete blob after use."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()
        print(f"Deleted blob: {bucket_name}/{blob_name}")
        return True
    except Exception as e:
        print(f"Error cleaning up blob: {e}")
        return False
```

## Google Lens API (via SerpAPI)

### Purpose
Identifies objects in images using Google Lens visual search capabilities.

### Setup
1. Sign up for SerpAPI account at https://serpapi.com
2. Get your API key
3. Set `SERPAPI_API_KEY` environment variable

### Implementation

```python
from serpapi import GoogleSearch
import os

def identify_with_google_lens(image_url):
    """Use Google Lens API via SerpAPI to identify objects in image."""
    try:
        api_key = os.environ.get("SERPAPI_API_KEY")
        if not api_key:
            print("SerpAPI API key not set.")
            return None
            
        print("\nAnalyzing image using Google Lens API...")
        
        # Setup parameters for Google Lens search
        params = {
            "api_key": api_key,
            "engine": "google_lens",
            "url": image_url,
            "hl": "en"  # Language
        }
        
        # Create the search
        search = GoogleSearch(params)
        
        # Get results
        results = search.get_dict()
        
        return results
            
    except Exception as e:
        print(f"Error using Google Lens API: {e}")
        return None

def get_lens_products(page_token, api_key=None):
    """Get additional products using page token."""
    if not api_key:
        api_key = os.environ.get("SERPAPI_API_KEY")
    
    params = {
        "api_key": api_key,
        "engine": "google_lens",
        "page_token": page_token
    }
    
    search = GoogleSearch(params)
    return search.get_dict()

def extract_camera_info(lens_results):
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
                    "price": match.get("price", {}).get("value", ""),
                    "thumbnail": match.get("thumbnail", "")
                })
    
    # Extract from products
    if "products" in lens_results:
        for product in lens_results["products"]:
            products.append({
                "title": product.get("title", ""),
                "source": product.get("source", ""),
                "link": product.get("link", ""),
                "price": product.get("price", {}).get("value", ""),
                "thumbnail": product.get("thumbnail", "")
            })
    
    return products
```

### Response Structure
```json
{
  "visual_matches": [
    {
      "title": "Camera Model Name",
      "source": "example.com",
      "link": "https://example.com/camera",
      "price": {"value": "$999"},
      "thumbnail": "https://image.url"
    }
  ],
  "products": [...],
  "products_page_token": "token_for_more_products",
  "exact_matches_page_token": "token_for_exact_matches"
}
```

## Perplexity AI API

### Purpose
Retrieves detailed specifications for identified camera models.

### Setup
1. Sign up for Perplexity AI at https://www.perplexity.ai
2. Get API key from account settings
3. Set `PPLX_API_KEY` environment variable

### Implementation

```python
import requests
import os

def get_specs_with_perplexity(camera_model, model="sonar-medium-online"):
    """Get detailed camera specifications using Perplexity AI."""
    try:
        pplx_api_key = os.environ.get("PPLX_API_KEY")
        
        if not pplx_api_key:
            print("PPLX API key not set.")
            return None
        
        print(f"\nGetting specifications for {camera_model}...")
        
        headers = {
            "Authorization": f"Bearer {pplx_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that provides accurate information about cameras and electronic devices."
                },
                {
                    "role": "user", 
                    "content": f"Provide detailed specifications and information about the {camera_model}. Include details about its features, resolution, connectivity, and any other important technical specifications such as Field of View (FoV), camera frames per second (fps), and if it has movement capabilities."
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
            specs = result["choices"][0]["message"]["content"]
            return specs
        else:
            print(f"Error from PPLX API: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error using PPLX API: {e}")
        return None
```

### Available Models
- `sonar-small-online` - Fast, basic responses
- `sonar-medium-online` - Balanced performance (recommended)
- `sonar-large-online` - Most comprehensive responses

## Apipie API

### Purpose
Provides information about available AI models for dynamic model selection.

### Setup
No authentication required - it's a public API.

### Implementation

```python
import requests

def get_available_perplexity_models():
    """Get available Perplexity models from Apipie API."""
    try:
        print("Fetching available models from Apipie API...")
        response = requests.get("https://apipie.ai/v1/models")
        
        if response.status_code == 200:
            models_data = response.json()
            
            # Filter for Perplexity models
            perplexity_models = []
            for model in models_data.get("data", []):
                if ("sonar" in model.get("id", "").lower() and 
                    model.get("enabled") == 1 and 
                    model.get("available") == 1):
                    perplexity_models.append(model["id"])
            
            # Prefer sonar-medium-online
            if "sonar-medium-online" in perplexity_models:
                return "sonar-medium-online"
            elif perplexity_models:
                return perplexity_models[0]
                
        return None
        
    except Exception as e:
        print(f"Error getting available models: {e}")
        return None
```

### Response Structure
```json
{
  "data": [
    {
      "id": "sonar-medium-online",
      "name": "Perplexity Sonar Medium Online",
      "enabled": 1,
      "available": 1,
      "provider": "perplexity"
    }
  ]
}
```

## Caching Strategy

### Implementation

```python
import os
import hashlib
import json
import time

# Cache directories
CACHE_DIR = "api_cache"
LENS_CACHE_DIR = os.path.join(CACHE_DIR, "lens")
PERPLEXITY_CACHE_DIR = os.path.join(CACHE_DIR, "perplexity")
IMAGES_CACHE_DIR = os.path.join(CACHE_DIR, "images")
APIPIE_CACHE_DIR = os.path.join(CACHE_DIR, "apipie")

# Create directories
for dir in [LENS_CACHE_DIR, PERPLEXITY_CACHE_DIR, IMAGES_CACHE_DIR, APIPIE_CACHE_DIR]:
    os.makedirs(dir, exist_ok=True)

def get_file_hash(file_path):
    """Generate MD5 hash for file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def cache_api_response(cache_dir, cache_key, data, expiration_days=30):
    """Cache API response with expiration."""
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    
    # Check if cache exists and is still valid
    if os.path.exists(cache_file):
        file_modified_time = os.path.getmtime(cache_file)
        current_time = time.time()
        if current_time - file_modified_time < (expiration_days * 86400):
            with open(cache_file, 'r') as f:
                return json.load(f)
    
    # Cache new data
    with open(cache_file, 'w') as f:
        json.dump(data, f)
    
    return data

def load_cached_response(cache_dir, cache_key):
    """Load cached response if it exists."""
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None
```

### Cache Expiration Times
- **Google Lens results**: 30 days
- **Perplexity specifications**: 30 days
- **Apipie models list**: 1 day
- **Downloaded images**: No expiration

## Complete Example Integration

```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def identify_and_get_specs(image_path):
    """Complete pipeline to identify camera and get specifications."""
    
    # 1. Upload image to Google Cloud Storage
    upload_result = upload_to_gcs(image_path)
    if not upload_result:
        return None
    
    # 2. Use Google Lens to identify camera
    lens_results = identify_with_google_lens(upload_result["url"])
    if not lens_results:
        cleanup_gcs_blob(upload_result["bucket"], upload_result["blob_name"])
        return None
    
    # 3. Extract camera information
    products = extract_camera_info(lens_results)
    
    # 4. Clean up GCS blob
    cleanup_gcs_blob(upload_result["bucket"], upload_result["blob_name"])
    
    # 5. Get specifications for first identified camera
    if products:
        camera_model = products[0]["title"]
        
        # Get best available model
        model = get_available_perplexity_models()
        if not model:
            model = "sonar-medium-online"
        
        # Get specifications
        specs = get_specs_with_perplexity(camera_model, model)
        
        return {
            "camera": camera_model,
            "specifications": specs,
            "all_products": products
        }
    
    return None

# Usage
result = identify_and_get_specs("path/to/camera/image.jpg")
if result:
    print(f"Camera: {result['camera']}")
    print(f"Specifications: {result['specifications']}")
```

## Error Handling Best Practices

1. Always check for API keys before making requests
2. Implement comprehensive try-except blocks
3. Cache responses to minimize API costs
4. Clean up temporary resources (GCS blobs)
5. Provide fallback options for model selection
6. Validate cache expiration times
7. Handle rate limiting gracefully

## Rate Limits and Costs

### SerpAPI (Google Lens)
- Rate limit: Depends on plan
- Cost: ~$0.01 per search

### Perplexity AI
- Rate limit: Varies by plan
- Cost: Based on tokens used

### Google Cloud Storage
- Storage: ~$0.02 per GB per month
- Operations: ~$0.005 per 10,000 operations
- Bandwidth: ~$0.08 per GB

### Apipie API
- Free public API
- No authentication required

## Security Considerations

1. Never commit API keys to version control
2. Use environment variables for all secrets
3. Implement proper access controls for GCS buckets
4. Delete temporary files after processing
5. Validate and sanitize all user inputs
6. Use signed URLs with expiration for GCS
7. Regularly rotate API keys