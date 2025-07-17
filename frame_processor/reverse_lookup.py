"""
File: reverse_lookup.py
Purpose: Process HEIC images, detect objects with YOLO, crop detected cameras, and identify camera models
         using Google Cloud Vision API and web search.

Prompts:
1. Read HEIC image and use YOLO to determine what is in the image
2. Crop image to only have the camera and determine what type of camera it is
3. Get specifications for camera

Author: Tom Sloan
Created: 2025-03-20
"""

import os
import io
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from pillow_heif import register_heif_opener
import requests
from ultralytics import YOLO
from google.cloud import vision
from google.cloud.vision_v1 import types
import json
from serpapi import GoogleSearch
import openai
from dotenv import load_dotenv
from google.cloud import storage
import uuid
import time
from datetime import datetime, timedelta
import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                            QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, 
                            QScrollArea, QFrame, QTextEdit, QSizePolicy)
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
import hashlib
import os.path
import threading
from io import BytesIO
from urllib.request import urlopen
import re

# Load environment variables from .env file
load_dotenv()

# Register HEIF file format with PIL
register_heif_opener()

# Constants for cache directories
CACHE_DIR = "api_cache"
LENS_CACHE_DIR = os.path.join(CACHE_DIR, "lens")
PERPLEXITY_CACHE_DIR = os.path.join(CACHE_DIR, "perplexity")
IMAGES_CACHE_DIR = os.path.join(CACHE_DIR, "images")
APIPIE_CACHE_DIR = os.path.join(CACHE_DIR, "apipie")

# Create cache directories if they don't exist
os.makedirs(LENS_CACHE_DIR, exist_ok=True)
os.makedirs(PERPLEXITY_CACHE_DIR, exist_ok=True)
os.makedirs(IMAGES_CACHE_DIR, exist_ok=True)
os.makedirs(APIPIE_CACHE_DIR, exist_ok=True)

# Global variable to store specification results
global_specs_result = None

class ImageProcessor:
    def __init__(self, image_path):
        """Initialize the image processor with an image path."""
        self.image_path = image_path
        self.original_image = None
        self.yolo_model = YOLO("yolo11x.pt")  # Load the YOLO model
        self.detected_objects = []
        self.camera_crop = None
        
    def load_image(self):
        """Load an image (HEIC or JPEG) and convert it to a format suitable for processing."""
        try:
            # Open the image using PIL
            img = Image.open(self.image_path)
            # Convert PIL image to a format suitable for OpenCV/YOLO
            self.original_image = np.array(img)
            
            # Check if we need to convert from RGB to BGR (for HEIC files)
            if self.image_path.lower().endswith(('.heic')):
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)
            # Add conversion for JPEG files as well - PIL loads as RGB but OpenCV expects BGR
            elif self.image_path.lower().endswith(('.jpg', '.jpeg')):
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)
            
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
            
    def detect_objects(self):
        """Use YOLO to detect objects in the image."""
        if self.original_image is None:
            print("No image loaded. Please load an image first.")
            return False
            
        # Run YOLO model on the image
        results = self.yolo_model(self.original_image)
        
        # Process the results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                name = result.names[cls]
                
                self.detected_objects.append({
                    'name': name,
                    'confidence': conf,
                    'box': (int(x1), int(y1), int(x2), int(y2))
                })
                
        return len(self.detected_objects) > 0
    
    def crop_camera(self):
        """Crop the camera from the image based on YOLO detection."""
        if not self.detected_objects:
            print("No objects detected. Run detect_objects() first.")
            return False
            
        # Find camera in detected objects
        for obj in self.detected_objects:
            if obj['name'].lower() in ['camera', 'cell phone', 'device', 'tv', 'remote']:
                x1, y1, x2, y2 = obj['box']
                
                # Add a small margin
                margin = 20
                height, width = self.original_image.shape[:2]
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(width, x2 + margin)
                y2 = min(height, y2 + margin)
                
                # Crop the image
                self.camera_crop = self.original_image[y1:y2, x1:x2]
                return True
                
        print("No camera found in the image.")
        return False
        
    def show_results(self):
        """Display the original image with bounding boxes and the cropped camera."""
        if self.original_image is None:
            print("No image loaded.")
            return
            
        # Create a copy for drawing
        img_with_boxes = self.original_image.copy()
        
        # Draw bounding boxes
        for obj in self.detected_objects:
            x1, y1, x2, y2 = obj['box']
            name = obj['name']
            conf = obj['confidence']
            
            # Drawing the bounding box
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"{name} {conf:.2f}"
            cv2.putText(img_with_boxes, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert from BGR to RGB for matplotlib
        img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Display the original image with boxes
        plt.subplot(1, 2, 1)
        plt.imshow(img_with_boxes_rgb)
        plt.title("Detected Objects")
        plt.axis("off")
        
        # Display the cropped camera if available
        if self.camera_crop is not None:
            camera_crop_rgb = cv2.cvtColor(self.camera_crop, cv2.COLOR_BGR2RGB)
            plt.subplot(1, 2, 2)
            plt.imshow(camera_crop_rgb)
            plt.title("Cropped Camera")
            plt.axis("off")
            
        plt.tight_layout()
        plt.show()
        
    def save_cropped_camera(self, output_path="camera_crop.jpg"):
        """Save the cropped camera image to a file."""
        if self.camera_crop is None:
            print("No camera crop available.")
            return False
            
        cv2.imwrite(output_path, self.camera_crop)
        print(f"Cropped camera saved to {output_path}")
        return True

class CameraIdentifier:
    def __init__(self):
        """Initialize the camera identifier."""
        pass
        
    def upload_to_gcs(self, image_path, bucket_name="worldsystembucket"):
        """Upload image to Google Cloud Storage and return public URL."""
        # Check cache based on image hash
        image_hash = self._get_file_hash(image_path)
        cache_file = os.path.join(LENS_CACHE_DIR, f"gcs_{image_hash}.json")
        
        # If we have a cached result, return it
        if os.path.exists(cache_file):
            print(f"Using cached GCS upload result for {image_path}")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        try:
            # Create a storage client
            storage_client = storage.Client()
            
            # Get the bucket
            bucket = storage_client.bucket(bucket_name)
            
            # Create a unique blob name using timestamp and UUID
            blob_name = f"camera_analysis/{int(time.time())}_{uuid.uuid4().hex}.jpg"
            blob = bucket.blob(blob_name)
            
            # Upload the file
            print(f"Uploading image to GCS bucket: {bucket_name}/{blob_name}")
            blob.upload_from_filename(image_path)
            
            # Instead, generate a signed URL that expires after some time
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=15),
                method="GET"
            )
            
            print(f"Image uploaded and available at: {url}")
            
            result = {
                "bucket": bucket_name,
                "blob_name": blob_name,
                "url": url
            }
            
            # Cache the result
            with open(cache_file, 'w') as f:
                json.dump(result, f)
                
            return result
            
        except Exception as e:
            print(f"Error uploading to Google Cloud Storage: {e}")
            return None
            
    def _get_file_hash(self, file_path):
        """Generate a hash for a file to use as a cache key."""
        if not os.path.exists(file_path):
            return None
            
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
        
    def make_private(self, bucket_name, blob_name):
        """Clean up blob after analysis."""
        try:
            # Create a storage client
            storage_client = storage.Client()
            
            # Get the bucket
            bucket = storage_client.bucket(bucket_name)
            
            # Get the blob
            blob = bucket.blob(blob_name)
            
            # Instead of making private (which uses ACLs), just delete it
            blob.delete()
            
            print(f"Deleted blob: {bucket_name}/{blob_name}")
            return True
            
        except Exception as e:
            print(f"Error cleaning up blob: {e}")
            return False
        
    def identify_with_google_lens(self, image_path):
        """Use Google Lens API via SerpAPI to identify the camera directly from the image."""
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return None
        
        # Generate hash of the image for caching
        image_hash = self._get_file_hash(image_path)
        cache_file = os.path.join(LENS_CACHE_DIR, f"{image_hash}.json")
        
        # If we have a cached result, return it
        if os.path.exists(cache_file):
            print(f"Using cached Google Lens result for {image_path}")
            with open(cache_file, 'r') as f:
                return json.load(f)
            
        try:
            api_key = os.environ.get("SERPAPI_API_KEY")
            if not api_key:
                print("SerpAPI API key not set. Please set SERPAPI_API_KEY environment variable.")
                return None
                
            print("\nAnalyzing camera image using Google Lens API via SerpAPI...")
            
            # First, upload the image to GCS to get a public URL
            upload_result = self.upload_to_gcs(image_path)
            
            if not upload_result:
                print("Failed to upload image to Google Cloud Storage.")
                return None
                
            # Use SerpAPI with the public URL
            params = {
                "api_key": api_key,
                "engine": "google_lens",
                "url": upload_result["url"],
                "hl": "en"  # Set language to English
            }
            
            # Create the SerpAPI client
            search = GoogleSearch(params)
            
            # Get the results first
            print("Fetching results from Google Lens API...")
            results = search.get_dict()
            
            # Now write the results (not the search object) to a file
            with open("search.json", "w") as f:
                json.dump(results, f, indent=2)
            
            # Cache the result
            with open(cache_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Make the blob private after analysis
            self.make_private(upload_result["bucket"], upload_result["blob_name"])
            
            return results
                
        except Exception as e:
            print(f"Error using Google Lens API via SerpAPI: {e}")
            return None
            
    def get_lens_products(self, page_token=None):
        """Get products from Google Lens API using a page token."""
        # Check cache
        if page_token:
            cache_file = os.path.join(LENS_CACHE_DIR, f"products_{hashlib.md5(page_token.encode()).hexdigest()}.json")
            if os.path.exists(cache_file):
                print("Using cached Google Lens products result")
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        try:
            api_key = os.environ.get("SERPAPI_API_KEY")
            if not api_key:
                print("SerpAPI API key not set. Please set SERPAPI_API_KEY environment variable.")
                return None
                
            if not page_token:
                print("No page token provided for products.")
                return None
                
            print("\nFetching products using page token...")
            
            # Create the SerpAPI client for products
            params = {
                "api_key": api_key,
                "engine": "google_lens",
                "page_token": page_token
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            # Cache the results
            if page_token:
                with open(cache_file, 'w') as f:
                    json.dump(results, f, indent=2)
            
            return results
                
        except Exception as e:
            print(f"Error fetching Google Lens products: {e}")
            return None
            
    def get_lens_exact_matches(self, page_token=None):
        """Get exact matches from Google Lens API using a page token."""
        # Check cache
        if page_token:
            cache_file = os.path.join(LENS_CACHE_DIR, f"exact_matches_{hashlib.md5(page_token.encode()).hexdigest()}.json")
            if os.path.exists(cache_file):
                print("Using cached Google Lens exact matches result")
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        try:
            api_key = os.environ.get("SERPAPI_API_KEY")
            if not api_key:
                print("SerpAPI API key not set. Please set SERPAPI_API_KEY environment variable.")
                return None
                
            if not page_token:
                print("No page token provided for exact matches.")
                return None
                
            print("\nFetching exact matches using page token...")
            
            # Create the SerpAPI client for exact matches
            params = {
                "api_key": api_key,
                "engine": "google_lens",
                "page_token": page_token
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            # Cache the results
            if page_token:
                with open(cache_file, 'w') as f:
                    json.dump(results, f, indent=2)
            
            return results
                
        except Exception as e:
            print(f"Error fetching Google Lens exact matches: {e}")
            return None

    def extract_camera_info(self, lens_results):
        """Extract camera information from Google Lens results."""
        if not lens_results or 'visual_matches' not in lens_results:
            print("No visual matches found in Google Lens results.")
            return None

        # Extract product names, thumbnails and links from visual matches
        products = []
        
        for match in lens_results['visual_matches']:
            product = {}
            if 'title' in match:
                product['title'] = match['title']
            if 'thumbnail' in match:
                product['thumbnail'] = match['thumbnail']
            if 'link' in match:
                product['link'] = match['link']
            if 'price' in match and isinstance(match['price'], dict) and 'value' in match['price']:
                product['price'] = match['price']['value']
            if len(product) > 0:
                products.append(product)
                
        return products
    
    def get_specs_with_perplexity(self, camera_model=None):
        """Use Perplexity API to get detailed camera specifications."""
        if not camera_model:
            print("No camera model available for specification lookup.")
            return None
        
        # Create a cache key from the camera model
        cache_key = hashlib.md5(camera_model.encode()).hexdigest()
        cache_file = os.path.join(PERPLEXITY_CACHE_DIR, f"{cache_key}.txt")
        
        # Check if we have a cached result
        if os.path.exists(cache_file):
            print(f"Using cached specifications for {camera_model}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
            
        try:
            # Use PPLX API to get information
            pplx_api_key = os.environ.get("PPLX_API_KEY")
            
            if not pplx_api_key:
                print("PPLX API key not set. Please set PPLX_API_KEY environment variable.")
                return None
            
            # Get available models from Apipie API
            model_to_use = self._get_available_model()
            if not model_to_use:
                model_to_use = "sonar-medium-online"  # Fallback model if API fetch fails
                
            print(f"\nGetting detailed specifications for {camera_model} using Perplexity AI with model: {model_to_use}...")
            
            headers = {
                "Authorization": f"Bearer {pplx_api_key}",
                "Content-Type": "application/json"
            }
            # 68, 4.5, 2
            # Use the selected model from Apipie API
            payload = {
                "model": model_to_use,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that provides accurate information about cameras and electronic devices."},
                    {"role": "user", "content": f"Provide detailed specifications and information about the {camera_model}. Include details about its features, resolution, connectivity, and any other important technical specifications such as Field of View (FoV), camera frames per second (fps), and if it has movement capabilities."}
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
                
                # Cache the specifications
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(specs)
                
                return specs
            else:
                print(f"Error from PPLX API: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Error using PPLX API: {e}")
            return None
            
    def _get_available_model(self):
        """Get available models from Apipie API."""
        try:
            # Check cache for models (cache for 1 day)
            cache_file = os.path.join(APIPIE_CACHE_DIR, "models.json")
            
            use_cached = False
            if os.path.exists(cache_file):
                file_modified_time = os.path.getmtime(cache_file)
                current_time = time.time()
                # If cache is less than 1 day old, use it
                if current_time - file_modified_time < 86400:  # 86400 seconds = 1 day
                    use_cached = True
            
            if use_cached:
                print("Using cached model information from Apipie")
                with open(cache_file, 'r') as f:
                    models_data = json.load(f)
            else:
                print("Fetching available models from Apipie API...")
                response = requests.get("https://apipie.ai/v1/models")
                
                if response.status_code == 200:
                    models_data = response.json()
                    # Cache the result
                    with open(cache_file, 'w') as f:
                        json.dump(models_data, f)
                else:
                    print(f"Error fetching models from Apipie: {response.status_code} - {response.text}")
                    return None
            
            # Find appropriate perplexity models
            perplexity_models = []
            for model in models_data.get("data", []):
                # Check if the model is from perplexity
                if "sonar" in model.get("id", "").lower() and model.get("enabled") == 1 and model.get("available") == 1:
                    perplexity_models.append(model["id"])
            
            # Prefer sonar-medium-online if available
            if "sonar-medium-online" in perplexity_models:
                return "sonar-medium-online"
            # Otherwise return the first available model
            elif perplexity_models:
                return perplexity_models[0]
                
            return None
            
        except Exception as e:
            print(f"Error getting available models: {e}")
            return None

    # Keep these methods for backward compatibility
    def identify_with_search_api(self, image_description=None):
        """Use search API to find information about the camera - DEPRECATED."""
        print("Using deprecated method. Please use get_camera_with_serpapi and get_specs_with_perplexity instead.")
        return self.fallback_to_serpapi(image_description)
    
    def fallback_to_openai(self, image_description):
        """Fallback to OpenAI if PPLX API fails - DEPRECATED."""
        print("Using deprecated method. Please use get_camera_with_serpapi and get_specs_with_perplexity instead.")
        return None
    
    def fallback_to_serpapi(self, image_description):
        """Final fallback to SerpAPI - DEPRECATED."""
        print("Using deprecated method. Please use get_camera_with_serpapi instead.")
        return self.get_camera_with_serpapi(image_description)

def extract_product_names(serp_results):
    """Extract only the product names from SerpAPI results."""
    product_names = []
    
    # Check organic results
    if 'organic_results' in serp_results:
        for result in serp_results['organic_results']:
            if 'title' in result:
                # Extract just the product name portion from the title
                title = result['title']
                # Look for common camera brand names and extract product portion
                product_names.append(title)
    
    # Check shopping results
    if 'shopping_results' in serp_results:
        for result in serp_results['shopping_results']:
            if 'title' in result:
                product_names.append(result['title'])
    
    # Check knowledge graph
    if 'knowledge_graph' in serp_results and 'title' in serp_results['knowledge_graph']:
        product_names.append(serp_results['knowledge_graph']['title'])
    
    return product_names

def print_lens_results(lens_results):
    """Print all information from Google Lens API in a structured format."""
    if not lens_results:
        print("No results available to print.")
        return
        
    # Print visual matches with all available information
    if 'visual_matches' in lens_results:
        print("\nVISUAL MATCHES:")
        print("="*100)
        
        for i, match in enumerate(lens_results['visual_matches']):
            print(f"\nMatch #{i+1}:")
            print("-"*100)
            
            # Print all available fields for each match
            if 'position' in match:
                print(f"Position: {match['position']}")
                
            if 'title' in match:
                print(f"Title: {match['title']}")
                
            if 'link' in match:
                print(f"Link: {match['link']}")
                
            if 'source' in match:
                print(f"Source: {match['source']}")
                
            if 'source_icon' in match:
                print(f"Source Icon: {match['source_icon']}")
                
            if 'rating' in match:
                print(f"Rating: {match['rating']}")
                
            if 'reviews' in match:
                print(f"Reviews: {match['reviews']}")
                
            if 'price' in match and isinstance(match['price'], dict):
                print("Price:")
                price = match['price']
                if 'value' in price:
                    print(f"  Value: {price['value']}")
                if 'extracted_value' in price:
                    print(f"  Extracted Value: {price['extracted_value']}")
                if 'currency' in price:
                    print(f"  Currency: {price['currency']}")
                    
            if 'in_stock' in match:
                print(f"In Stock: {match['in_stock']}")
                
            if 'condition' in match:
                print(f"Condition: {match['condition']}")
                
            if 'thumbnail' in match:
                print(f"Thumbnail: {match['thumbnail']}")
                
            if 'thumbnail_width' in match:
                print(f"Thumbnail Width: {match['thumbnail_width']}")
                
            if 'thumbnail_height' in match:
                print(f"Thumbnail Height: {match['thumbnail_height']}")
                
            if 'image' in match:
                print(f"Image: {match['image']}")
                
            if 'image_width' in match:
                print(f"Image Width: {match['image_width']}")
                
            if 'image_height' in match:
                print(f"Image Height: {match['image_height']}")
    
    # Print related content
    if 'related_content' in lens_results:
        print("\nRELATED CONTENT:")
        print("="*100)
        
        for i, content in enumerate(lens_results['related_content']):
            print(f"\nRelated Content #{i+1}:")
            print("-"*100)
            
            if 'query' in content:
                print(f"Query: {content['query']}")
                
            if 'link' in content:
                print(f"Link: {content['link']}")
                
            if 'thumbnail' in content:
                print(f"Thumbnail: {content['thumbnail']}")
                
            if 'serpapi_link' in content:
                print(f"SerpAPI Link: {content['serpapi_link']}")
    
    # Print page tokens and navigation links
    print("\nNAVIGATION AND PAGINATION:")
    print("="*100)
    
    if 'products_page_token' in lens_results:
        print(f"Products Page Token: {lens_results['products_page_token']}")
        
    if 'serpapi_products_link' in lens_results:
        print(f"SerpAPI Products Link: {lens_results['serpapi_products_link']}")
        
    if 'exact_matches_page_token' in lens_results:
        print(f"Exact Matches Page Token: {lens_results['exact_matches_page_token']}")
        
    if 'serpapi_exact_matches_link' in lens_results:
        print(f"SerpAPI Exact Matches Link: {lens_results['serpapi_exact_matches_link']}")
    
    # Print other top-level keys that might be present
    other_keys = [key for key in lens_results.keys() if key not in 
                 ['visual_matches', 'related_content', 'products_page_token', 
                  'serpapi_products_link', 'exact_matches_page_token', 'serpapi_exact_matches_link']]
    
    if other_keys:
        print("\nOTHER INFORMATION:")
        print("="*100)
        
        for key in other_keys:
            print(f"{key}: {lens_results[key]}")

def load_image_from_url(url, cache=True):
    """Load an image from a URL with caching."""
    if cache:
        # Create a hash of the URL for caching
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_path = os.path.join(IMAGES_CACHE_DIR, f"{url_hash}.jpg")
        
        # Check if cached image exists
        if os.path.exists(cache_path):
            try:
                return cache_path  # Return path instead of PIL image
            except Exception as e:
                print(f"Error loading cached image: {e}")
                # Continue to download if cached image can't be loaded
    
    try:
        # Open the URL
        with urlopen(url) as response:
            image_data = response.read()
        
        # Create cache path if needed
        if cache:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                f.write(image_data)
            return cache_path
        else:
            # Create a temporary file
            temp_path = os.path.join(IMAGES_CACHE_DIR, f"temp_{uuid.uuid4().hex}.jpg")
            with open(temp_path, 'wb') as f:
                f.write(image_data)
            return temp_path
            
    except Exception as e:
        print(f"Error loading image from {url}: {e}")
        return None

# Thread for loading images asynchronously in PyQt5
class ImageLoaderThread(QThread):
    image_loaded = pyqtSignal(int, str)  # Signal: (product_index, image_path)
    
    def __init__(self, products):
        super().__init__()
        self.products = products
        
    def run(self):
        for i, product in enumerate(self.products):
            if 'thumbnail' in product:
                try:
                    # Load the image and get the path
                    img_path = load_image_from_url(product['thumbnail'])
                    if img_path:
                        # Emit the signal with the product index and image path
                        self.image_loaded.emit(i, img_path)
                except Exception as e:
                    print(f"Error loading image for product {i}: {e}")

class ProductFrame(QFrame):
    """Frame to display a camera product with image, name, price and spec button."""
    
    def __init__(self, product, index, specs_callback):
        super().__init__()
        self.product = product
        self.index = index
        self.specs_callback = specs_callback
        
        # Set up the frame style
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(1)
        self.setMidLineWidth(0)
        self.setStyleSheet("background-color: white;")
        
        # Create layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Image placeholder
        self.image_label = QLabel()
        self.image_label.setFixedSize(200, 200)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #f0f0f0;")
        layout.addWidget(self.image_label)
        
        # Product name
        name = product.get('title', 'Unknown Camera')
        if len(name) > 30:
            name = name[:27] + "..."
        self.name_label = QLabel(name)
        self.name_label.setWordWrap(True)
        self.name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.name_label)
        
        # Price if available
        if 'price' in product:
            price_label = QLabel(f"Price: {product['price']}")
            price_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(price_label)
        
        # Specifications button
        specs_button = QPushButton("Get Specifications")
        specs_button.setStyleSheet(
            "background-color: #4CAF50; color: white; border-radius: 4px; padding: 6px;"
        )
        specs_button.clicked.connect(self.get_specs)
        layout.addWidget(specs_button)
        
    def get_specs(self):
        """Handle the button click to get specifications."""
        camera_model = self.product.get('title', 'Unknown Camera')
        self.specs_callback(camera_model)
        
    def set_image(self, image_path):
        """Set the product image from a file path."""
        try:
            pixmap = QPixmap(image_path)
            pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
        except Exception as e:
            print(f"Error setting image for product {self.index}: {e}")

class SpecificationsWindow(QMainWindow):
    """Window to display camera specifications."""
    
    def __init__(self, camera_model, specs):
        super().__init__()
        self.setWindowTitle(f"Specifications for {camera_model}")
        self.setGeometry(100, 100, 900, 700)
        self.camera_model = camera_model
        self.specs_text = specs
        
        # Set up window style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QLabel {
                color: #333333;
            }
            QLabel#title {
                font-size: 18pt;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
            }
            QLabel#section {
                font-size: 14pt;
                font-weight: bold;
                color: #34495e;
                padding: 5px;
            }
            QFrame#key_spec {
                background-color: #e8f4f8;
                border-radius: 8px;
                border: 1px solid #bbd8e7;
                padding: 5px;
            }
            QLabel#key_title {
                font-size: 12pt;
                font-weight: bold;
                color: #3498db;
            }
            QLabel#key_value {
                font-size: 14pt;
                font-weight: bold;
                color: #2980b9;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 10px;
                font-size: 11pt;
            }
        """)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel(f"{camera_model}")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        # Extract key specifications
        key_specs = self._extract_key_specs(specs)
        
        # Key specs section
        key_section = QLabel("Key Specifications")
        key_section.setObjectName("section")
        main_layout.addWidget(key_section)
        
        # Key specs grid
        key_specs_grid = QGridLayout()
        key_specs_grid.setSpacing(10)
        
        # Add key specs to grid
        self._add_key_spec(key_specs_grid, 0, 0, "Resolution", key_specs.get("resolution", "N/A"))
        self._add_key_spec(key_specs_grid, 0, 1, "Field of View", key_specs.get("fov", "N/A"))
        self._add_key_spec(key_specs_grid, 0, 2, "Frame Rate", key_specs.get("fps", "N/A"))
        self._add_key_spec(key_specs_grid, 1, 0, "Connectivity", key_specs.get("connectivity", "N/A"))
        self._add_key_spec(key_specs_grid, 1, 1, "Power", key_specs.get("power", "N/A"))
        self._add_key_spec(key_specs_grid, 1, 2, "Storage", key_specs.get("storage", "N/A"))
        
        main_layout.addLayout(key_specs_grid)
        
        # Detailed specs section
        details_section = QLabel("Detailed Specifications")
        details_section.setObjectName("section")
        main_layout.addWidget(details_section)
        
        # Text edit for detailed specifications
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setText(specs)
        self.details_text.setMinimumHeight(300)
        main_layout.addWidget(self.details_text)
        
    def _extract_key_specs(self, specs_text):
        """Extract key specifications from the text."""
        key_specs = {
            "resolution": "Not specified",
            "fov": "Not specified",
            "fps": "Not specified",
            "connectivity": "Not specified",
            "power": "Not specified",
            "storage": "Not specified"
        }
        
        if not specs_text:
            return key_specs
            
        # Extract resolution
        resolution_patterns = [
            r"(\d+p|\d+K|UHD|\d+x\d+|HD|Full HD|4K|1080p|720p|2K)",
            r"resolution.{1,20}?(\d+p|\d+K|UHD|\d+x\d+|HD|Full HD|4K|1080p|720p|2K)",
            r"(\d+ ?MP)"
        ]
        for pattern in resolution_patterns:
            matches = re.findall(pattern, specs_text, re.IGNORECASE)
            if matches:
                key_specs["resolution"] = matches[0]
                break
                
        # Extract field of view
        fov_patterns = [
            r"field of view.{1,10}?(\d+[°º])",
            r"FOV.{1,10}?(\d+[°º])",
            r"(\d+[°º]).{1,10}?field of view",
            r"(\d+[°º]).{1,10}?FOV",
            r"diagonal.{1,20}?(\d+[°º])"
        ]
        for pattern in fov_patterns:
            matches = re.findall(pattern, specs_text, re.IGNORECASE)
            if matches:
                key_specs["fov"] = matches[0]
                break
                
        # Extract frame rate
        fps_patterns = [
            r"(\d+.?fps)",
            r"frame.?rate.{1,15}?(\d+.?fps|\d+ frames per second)",
            r"(\d+.?Hz|\d+ frames per second)",
            r"(\d+.?fps|\d+ frames per second)"
        ]
        for pattern in fps_patterns:
            matches = re.findall(pattern, specs_text, re.IGNORECASE)
            if matches:
                key_specs["fps"] = matches[0]
                break
                
        # Extract connectivity
        connectivity_patterns = [
            r"(Wi-Fi|WiFi|Bluetooth|2\.4GHz|5GHz|802\.11)",
            r"connectivity.{1,30}?(Wi-Fi|WiFi|Bluetooth|2\.4GHz|5GHz|802\.11)"
        ]
        for pattern in connectivity_patterns:
            matches = re.findall(pattern, specs_text, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple):
                    key_specs["connectivity"] = matches[0][0]
                else:
                    key_specs["connectivity"] = matches[0]
                break
                
        # Extract power information
        power_patterns = [
            r"power.{1,30}?([\d.]+ ?V|battery|AC|DC)",
            r"battery.{1,30}?([\d.]+ ?mAh|[\d.]+ ?hours|rechargeable)"
        ]
        for pattern in power_patterns:
            matches = re.findall(pattern, specs_text, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple):
                    key_specs["power"] = matches[0][0]
                else:
                    key_specs["power"] = matches[0]
                break
                
        # Extract storage
        storage_patterns = [
            r"storage.{1,20}?([\d.]+ ?GB|cloud|microSD|SD card)",
            r"memory.{1,20}?([\d.]+ ?GB|[\d.]+ ?MB)"
        ]
        for pattern in storage_patterns:
            matches = re.findall(pattern, specs_text, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple):
                    key_specs["storage"] = matches[0][0]
                else:
                    key_specs["storage"] = matches[0]
                break
                
        return key_specs
        
    def _add_key_spec(self, layout, row, col, title, value):
        """Add a key specification widget to the layout."""
        frame = QFrame()
        frame.setObjectName("key_spec")
        frame.setMinimumSize(QSize(180, 90))
        frame.setMaximumSize(QSize(280, 120))
        
        frame_layout = QVBoxLayout(frame)
        frame_layout.setSpacing(5)
        
        # Spec title
        title_label = QLabel(title)
        title_label.setObjectName("key_title")
        title_label.setAlignment(Qt.AlignCenter)
        frame_layout.addWidget(title_label)
        
        # Spec value
        value_label = QLabel(value)
        value_label.setObjectName("key_value")
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setWordWrap(True)
        frame_layout.addWidget(value_label)
        
        layout.addWidget(frame, row, col)

class CameraSelectionWindow(QMainWindow):
    """Main window for camera selection grid."""
    
    def __init__(self, products):
        super().__init__()
        self.products = products
        self.setWindowTitle("Select Camera")
        self.setGeometry(100, 100, 900, 600)
        
        # Create central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout
        main_layout = QVBoxLayout(central)
        
        # Title
        title_label = QLabel("Select a Camera to Get Specifications")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        main_layout.addWidget(title_label)
        
        # Scroll area for products
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)
        
        # Container for grid
        scroll_content = QWidget()
        scroll_area.setWidget(scroll_content)
        
        # Grid layout for products
        self.grid_layout = QGridLayout(scroll_content)
        
        # Add product frames to grid
        self.product_frames = []
        max_cols = 3
        for i, product in enumerate(products):
            row = i // max_cols
            col = i % max_cols
            
            product_frame = ProductFrame(product, i, self.show_specifications)
            self.grid_layout.addWidget(product_frame, row, col)
            self.product_frames.append(product_frame)
        
        # Add close button at the bottom
        close_button = QPushButton("Close")
        close_button.setStyleSheet(
            "background-color: #f44336; color: white; border-radius: 4px; padding: 8px; font-weight: bold;"
        )
        close_button.clicked.connect(self.close)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        button_layout.addStretch()
        main_layout.addLayout(button_layout)
        
        # Start loading images
        self.start_image_loading()
        
    def start_image_loading(self):
        """Start the image loading thread."""
        self.thread = ImageLoaderThread(self.products)
        self.thread.image_loaded.connect(self.update_product_image)
        self.thread.start()
        
    def update_product_image(self, index, image_path):
        """Update a product's image when loaded."""
        if 0 <= index < len(self.product_frames):
            self.product_frames[index].set_image(image_path)
            
    def show_specifications(self, camera_model):
        """Show specifications for the selected camera."""
        global global_specs_result
        
        # Get specifications
        identifier = CameraIdentifier()
        specs = identifier.get_specs_with_perplexity(camera_model)
        global_specs_result = specs
        
        # Show specifications window
        self.specs_window = SpecificationsWindow(camera_model, specs)
        self.specs_window.show()

def select_camera_with_grid(products):
    """Display a grid of camera options with images and get specifications buttons using PyQt5."""
    if not products:
        return None
    
    # Create PyQt application
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    
    # Create and show the selection window
    window = CameraSelectionWindow(products)
    window.show()
    
    # Run the application event loop
    app.exec_()
    
    # Return global specs result
    return global_specs_result

def main():
    """Main function to process the image and identify the camera."""
    # Update the image path to use the new JPEG file
    # image_path = "54CD8710-E88F-4480-9134-30B75557939E_1_102_o.jpeg"
    image_path = "drone_pic.jpeg"
    # image_path = "1813FC73-B284-431E-B3A5-455AB11FD2A9.heic"
    
    print("\n" + "="*80)
    print("STEP 1: LOADING AND PROCESSING IMAGE")
    print("="*80)
    print(f"Loading image from: {image_path}")
    
    # Process the image
    processor = ImageProcessor(image_path)
    if not processor.load_image():
        print("Failed to load image. Exiting.")
        return
    else:
        print(f"✓ Successfully loaded image with shape: {processor.original_image.shape}")
    
    print("\n" + "="*80)
    print("STEP 2: DETECTING OBJECTS WITH YOLO")
    print("="*80)
    print("Running YOLO model on the image...")
    
    if processor.detect_objects():
        print(f"✓ YOLO Detection successful. Found {len(processor.detected_objects)} objects:")
        print("\nDetected Objects:")
        print("-" * 50)
        print(f"{'Object':<20} {'Confidence':<15} {'Position (x1,y1,x2,y2)'}")
        print("-" * 50)
        for obj in processor.detected_objects:
            print(f"{obj['name']:<20} {obj['confidence']:.4f}{'':<8} {obj['box']}")
    else:
        print("No objects detected. Exiting.")
        return
    
    print("\n" + "="*80)
    print("STEP 3: CROPPING CAMERA FROM IMAGE")
    print("="*80)
    
    if processor.crop_camera():
        print("✓ Camera successfully identified and cropped")
        print(f"Cropped camera shape: {processor.camera_crop.shape}")
        crop_path = "camera_crop.jpg"
        processor.save_cropped_camera(crop_path)
        print(f"Saved cropped image to: {crop_path}")
    else:
        print("Failed to crop camera. Exiting.")
        return
    
    # Display results - optional, can be commented out if not needed
    print("\nGenerating visualization of detected objects and camera crop...")
    processor.show_results()
    
    print("\n" + "="*80)
    print("STEP 4: IDENTIFYING CAMERA WITH GOOGLE LENS API")
    print("="*80)
    
    identifier = CameraIdentifier()
    
    # Use Google Lens API via SerpAPI
    lens_results = identifier.identify_with_google_lens(crop_path)
    
    if lens_results:
        print("\n✓ Successfully received response from Google Lens API")
        
        # Print all information from Google Lens results
        print_lens_results(lens_results)
        
        # Check if there are additional products to fetch
        if 'products_page_token' in lens_results:
            print("\n" + "="*80)
            print("FETCHING ADDITIONAL PRODUCTS")
            print("="*80)
            products_results = identifier.get_lens_products(lens_results['products_page_token'])
            if products_results:
                print("\n✓ Successfully received products response")
                print_lens_results(products_results)
        
        # Check if there are exact matches to fetch
        if 'exact_matches_page_token' in lens_results:
            print("\n" + "="*80)
            print("FETCHING EXACT MATCHES")
            print("="*80)
            exact_matches_results = identifier.get_lens_exact_matches(lens_results['exact_matches_page_token'])
            if exact_matches_results:
                print("\n✓ Successfully received exact matches response")
                print_lens_results(exact_matches_results)
        
        # Extract product information (now includes images and more details)
        products = identifier.extract_camera_info(lens_results)
        
        if products:
            print("\n" + "="*80)
            print("STEP 5: SELECTING CAMERA MODEL")
            print("="*80)
            
            print(f"\nFound {len(products)} possible camera models.")
            print("Opening product selection grid...")
            
            # Display grid UI and get specs for the selected model
            specs = select_camera_with_grid(products)
            
            if specs:
                print("\n✓ Successfully retrieved specifications")
                print("\nDetailed Camera Specifications:")
                print("-" * 100)
                print(specs)
            else:
                print("\nNo specifications retrieved.")
        else:
            print("No camera models detected.")
    else:
        print("✗ Failed to get results from Google Lens API.")

    print("\n" + "="*80)
    print("PROCESS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
