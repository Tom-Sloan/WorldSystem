# Enhanced Frame Processor with Real-World Scale Estimation

## Overview

This enhanced frame processor integrates object detection, tracking, and real-world dimension estimation to provide accurate scaling for 3D scene reconstruction. By identifying objects in the scene and retrieving their real-world dimensions, the system can automatically determine the correct scale for the entire 3D reconstruction.

### Key Features:
- **Object Detection & Tracking** - SAM2-based segmentation with IOU tracking
- **Dimension Estimation** - Google Lens + Perplexity API for real-world object sizes
- **Weighted Scale Calculation** - Confidence-based averaging of multiple object dimensions
- **Real-time Visualization** - Rerun.io integration for monitoring and debugging
- **Automatic Scene Scaling** - Publishes scale factor to mesh service for proper 3D reconstruction

### Why Real-World Scaling Matters:

Without real-world scale information, 3D reconstructions are dimensionless - a room could be 1 unit or 1000 units wide. By identifying common objects (phones, monitors, keyboards, etc.) and knowing their real dimensions, we can:

1. **Accurate Room Measurements** - Determine actual room dimensions in meters
2. **Proper Object Placement** - Place virtual objects at correct real-world scales
3. **Distance Measurements** - Calculate real distances between points in the scene
4. **AR/VR Applications** - Ensure virtual content matches real-world scale
5. **Architectural Planning** - Generate floor plans with accurate dimensions
6. **Navigation & Path Planning** - Calculate real-world distances for drone navigation

## Rerun Visualization Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                  Drone Object Enhancement Pipeline               │
├─────────────────────────────┬───────────────────────────────────┤
│                             │                                   │
│      Camera Feed            │        Enhancement Results        │
│   ┌─────────────────┐       │   ┌─────────────┬─────────────┐ │
│   │                 │       │   │  Original   │  Enhanced   │ │
│   │   Live Video    │       │   │             │             │ │
│   │   + SAM2        │       │   │             │             │ │
│   │   Segmentation  │       │   └─────────────┴─────────────┘ │
│   │                 │       │                                   │
│   └─────────────────┘       │   Google Lens Results            │
│                             │   ┌───────────────────────────┐ │
│   Active Object Tracking    │   │ Product: Arlo Pro 4       │ │
│   ┌─────────────────┐       │   │ Dims: 5.2"x3.1"x3.1"     │ │
│   │ ID #42: Camera  │       │   │ Confidence: 94%          │ │
│   │ Frames: 28     │       │   └───────────────────────────┘ │
│   │ Best Score: 0.87│       │                                   │
│   │ Status: Process │       │   Processing Metrics             │
│   └─────────────────┘       │   ┌───────────────────────────┐ │
│                             │   │ Frame Time: 45ms         │ │
│   Quality Score Timeline    │   │ Enhancement: 185ms       │ │
│   ┌─────────────────┐       │   │ API Call: 1.2s           │ │
│   │ ▁▃▅▇█▇▅▃▁ Score │       │   │ GPU Memory: 3.2GB        │ │
│   │ ━━━━━━━━ Time   │       │   └───────────────────────────┘ │
│   └─────────────────┘       │                                   │
├─────────────────────────────┴───────────────────────────────────┤
│ System Logs                                                      │
│ [INFO] Object 42 detected - starting quality tracking            │
│ [INFO] Processing triggered after 1.5s (22 frames collected)     │
│ [SUCCESS] Google Lens identified: "Arlo Pro 4 Camera"           │
│ [INFO] Dimensions: 5.2" x 3.1" x 3.1" - Reprocessing in 3s      │
└──────────────────────────────────────────────────────────────────┘
```

## Implementation Code

**Note**: This implementation builds on the base classes from the previous drone pipeline implementation (SimpleTracker, EnhancedTracker, FrameQualityScorer, LightEnhancement). The code below shows the Rerun-specific additions and modifications.

### 1. Initialize Rerun with Custom Layout

```python
import rerun as rr
import rerun.blueprint as bp
import numpy as np
import cv2
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import threading
import queue

# Note: These base classes come from your existing implementation
# from tracker import SimpleTracker, EnhancedTracker
# from frame_scorer import FrameQualityScorer  
# from enhancement import LightEnhancement
# from complete_pipeline import DroneEnhancementPipeline
# from google_lens_api import GoogleLensAPI

class DroneRerunVisualizer:
    def __init__(self, recording_name="drone_object_enhancement"):
        # Initialize Rerun
        rr.init(recording_name, spawn=True)
        
        # Set up the blueprint (layout)
        self.setup_blueprint()
        
        # Entity paths
        self.CAMERA_PATH = "camera"
        self.DETECTIONS_PATH = "camera/detections"
        self.ENHANCEMENT_PATH = "enhancement"
        self.METRICS_PATH = "metrics"
        self.LOGS_PATH = "logs"
        
    def setup_blueprint(self):
        """Configure the Rerun viewer layout"""
        blueprint = bp.Blueprint(
            bp.Grid(
                bp.Horizontal(
                    bp.Vertical(
                        bp.SpaceView(
                            name="Camera Feed",
                            origin="/camera",
                            contents=[
                                "/camera/**",
                            ]
                        ),
                        bp.Horizontal(
                            bp.TextDocumentView(
                                name="Active Object Tracking",
                                origin="/metrics/tracking",
                                contents=["/metrics/tracking/**"]
                            ),
                            bp.TimeSeriesView(
                                name="Quality Score Timeline",
                                origin="/metrics/quality",
                                contents=["/metrics/quality/**"]
                            ),
                            column_shares=[1, 1]
                        ),
                        row_shares=[3, 1]
                    ),
                    bp.Vertical(
                        bp.SpaceView(
                            name="Enhancement Results",
                            origin="/enhancement",
                            contents=["/enhancement/**"]
                        ),
                        bp.TextDocumentView(
                            name="Google Lens Results",
                            origin="/metrics/lens_results",
                            contents=["/metrics/lens_results/**"]
                        ),
                        bp.TextDocumentView(
                            name="Processing Metrics",
                            origin="/metrics/performance",
                            contents=["/metrics/performance/**"]
                        ),
                        row_shares=[2, 1, 1]
                    ),
                    column_shares=[1, 1]
                ),
                bp.TextLogView(
                    name="System Logs",
                    origin="/logs",
                    contents=["/logs/**"]
                ),
                row_shares=[5, 1]
            )
        )
        rr.send_blueprint(blueprint)
```

### 2. Enhanced Tracker with Timed Processing

```python
# Assuming these imports from your base implementation:
# from tracker import SimpleTracker, EnhancedTracker

@dataclass
class TrackedObjectV2:
    """Enhanced tracked object with processing timers"""
    id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_name: str
    confidence: float
    frame_history: List[Dict]
    best_frame: Optional[np.ndarray] = None
    best_score: float = 0.0
    last_seen_frame: int = 0
    first_seen_time: float = 0.0
    last_processed_time: float = 0.0
    processing_count: int = 0
    is_being_processed: bool = False
    best_bbox: Optional[Tuple[int, int, int, int]] = None
    best_frame_number: int = 0
    score_components: Optional[Dict] = None

# Extend the existing tracker to add Rerun visualization
class TimedProcessingTracker(EnhancedTracker):
    """Tracker that triggers processing after time threshold"""
    
    def __init__(self, 
                 process_after_seconds=1.5,
                 reprocess_interval_seconds=3.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.process_after_seconds = process_after_seconds
        self.reprocess_interval_seconds = reprocess_interval_seconds
        self.visualizer = DroneRerunVisualizer()
        # Inherited from EnhancedTracker:
        # self.frame_scorer = FrameQualityScorer()
        
    def update(self, detections, frame, frame_number):
        """Enhanced update with timed processing triggers"""
        current_time = time.time()
        
        # Regular tracking update
        matched_detections = set()
        
        for detection in detections:
            bbox, class_name, confidence = detection
            best_iou = 0
            best_track_id = None
            
            # Find best matching track
            for track_id, track in self.tracked_objects.items():
                if track.class_name == class_name:
                    iou = self.calculate_iou(bbox, track.bbox)
                    if iou > best_iou and iou > self.iou_threshold:
                        best_iou = iou
                        best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                track = self.tracked_objects[best_track_id]
                track.bbox = bbox
                track.confidence = confidence
                track.last_seen_frame = frame_number
                track.frame_history.append({
                    'frame': frame,
                    'bbox': bbox,
                    'frame_number': frame_number,
                    'confidence': confidence,
                    'timestamp': current_time
                })
                matched_detections.add(detection)
                
                # Visualize tracking update
                self.log_tracking_update(track, frame, frame_number)
                
            else:
                # Create new track
                new_track = TrackedObjectV2(
                    id=self.next_id,
                    bbox=bbox,
                    class_name=class_name,
                    confidence=confidence,
                    frame_history=[{
                        'frame': frame,
                        'bbox': bbox,
                        'frame_number': frame_number,
                        'confidence': confidence,
                        'timestamp': current_time
                    }],
                    last_seen_frame=frame_number,
                    first_seen_time=current_time,
                    last_processed_time=0.0,
                    processing_count=0,
                    is_being_processed=False
                )
                self.tracked_objects[self.next_id] = new_track
                self.next_id += 1
                
                # Log new object detection
                rr.log(
                    self.visualizer.LOGS_PATH,
                    rr.TextLog(
                        f"New object detected - ID: {new_track.id}, "
                        f"Class: {class_name}, Confidence: {confidence:.2f}",
                        level="INFO"
                    )
                )
        
        # Check for objects ready for processing
        objects_to_process = []
        
        for track_id, track in self.tracked_objects.items():
            time_since_first_seen = current_time - track.first_seen_time
            time_since_last_processed = current_time - track.last_processed_time
            
            # Process if:
            # 1. Seen for minimum time AND not being processed AND
            # 2. Either never processed OR reprocess interval exceeded
            if (time_since_first_seen >= self.process_after_seconds and 
                not track.is_being_processed and
                (track.processing_count == 0 or 
                 time_since_last_processed >= self.reprocess_interval_seconds)):
                
                objects_to_process.append(track)
                track.is_being_processed = True
                track.last_processed_time = current_time
                track.processing_count += 1
                
                rr.log(
                    self.visualizer.LOGS_PATH,
                    rr.TextLog(
                        f"Processing triggered for object {track.id} after "
                        f"{time_since_first_seen:.1f}s ({len(track.frame_history)} frames) "
                        f"- Processing count: {track.processing_count}",
                        level="INFO"
                    )
                )
        
        # Check for lost tracks
        lost_tracks = []
        for track_id, track in self.tracked_objects.items():
            if frame_number - track.last_seen_frame > self.max_lost_frames:
                lost_tracks.append(track_id)
                
                # Log object leaving scene
                rr.log(
                    self.visualizer.LOGS_PATH,
                    rr.TextLog(
                        f"Object {track.id} left scene after "
                        f"{len(track.frame_history)} frames",
                        level="INFO"
                    )
                )
        
        # Clean up lost tracks
        for track_id in lost_tracks:
            del self.tracked_objects[track_id]
        
        return objects_to_process
    
    def log_tracking_update(self, track, frame, frame_number):
        """Log tracking data to Rerun"""
        x1, y1, x2, y2 = track.bbox
        
        # Log bounding box on camera feed
        rr.log(
            f"{self.visualizer.DETECTIONS_PATH}/box_{track.id}",
            rr.Boxes2D(
                array=[[x1, y1, x2-x1, y2-y1]],
                array_format=rr.Box2DFormat.XYWH,
                labels=[f"{track.class_name} #{track.id}"],
                colors=[[0, 255, 0]]
            )
        )
        
        # Log tracking info
        tracking_info = f"""### Object #{track.id}
**Class:** {track.class_name}
**Confidence:** {track.confidence:.2f}
**Frames tracked:** {len(track.frame_history)}
**Best score:** {track.best_score:.3f}
**Processing count:** {track.processing_count}
"""
        rr.log(
            f"{self.visualizer.METRICS_PATH}/tracking",
            rr.TextDocument(tracking_info, media_type=rr.MediaType.MARKDOWN)
        )
```

### 3. Rerun-Integrated Enhancement Pipeline

```python
import torch

class RerunEnhancementPipeline:
    """Enhancement pipeline with Rerun visualization"""
    
    def __init__(self, sam2_model, google_lens_api_key, gcs_bucket_name="your-bucket-name"):
        # Initialize base components
        self.sam2_model = sam2_model
        self.enhancer = LightEnhancement(device='cuda')
        
        # Initialize tracker with visualizer
        self.tracker = TimedProcessingTracker(
            process_after_seconds=1.5,
            reprocess_interval_seconds=3.0,
            iou_threshold=0.3,
            max_lost_frames=10
        )
        self.visualizer = self.tracker.visualizer
        
        # Initialize Google Lens with Rerun integration
        self.google_lens = RerunGoogleLensAPI(
            api_key=google_lens_api_key,
            gcs_bucket_name=gcs_bucket_name,
            visualizer=self.visualizer
        )
        
        # Processing queue and storage
        self.enhancement_queue = queue.Queue()
        self.results_storage = {}
        
        # Start enhancement thread
        self.enhancement_thread = threading.Thread(target=self.enhancement_worker)
        self.enhancement_thread.daemon = True
        self.enhancement_thread.start()
        
        # Metrics
        self.frame_count = 0
        self.objects_processed = 0
        self.total_processing_time = 0
        self.start_time = time.time()
        self.last_frame_time = 0
        
    def process_frame(self, frame, frame_number):
        """Process frame with Rerun logging"""
        start_time = time.time()
        
        # Log camera frame
        rr.log(
            self.visualizer.CAMERA_PATH,
            rr.Image(frame).compress(jpeg_quality=80)
        )
        
        # Run SAM2 segmentation
        results = self.sam2_model(frame)
        
        # Extract detections
        detections = []
        all_boxes = []
        labels = []
        colors = []
        
        for detection in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = detection
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            class_name = self.get_object_class(detection)  # SAM2 doesn't provide classes
            detections.append((bbox, class_name, float(conf)))
            
            # Prepare for batch logging
            all_boxes.append([x1, y1, x2-x1, y2-y1])
            labels.append(f"{class_name} ({conf:.2f})")
            colors.append([255, 0, 0])  # Red for new detections
        
        # Log all detections at once
        if all_boxes:
            rr.log(
                f"{self.visualizer.DETECTIONS_PATH}/sam2",
                rr.Boxes2D(
                    array=all_boxes,
                    array_format=rr.Box2DFormat.XYWH,
                    labels=labels,
                    colors=colors
                )
            )
        
        # Update tracker - returns objects ready for processing
        objects_to_process = self.tracker.update(detections, frame, frame_number)
        
        # Update quality scores for active tracks
        for track_id, track in self.tracker.tracked_objects.items():
            if track.last_seen_frame == frame_number:
                self.update_quality_visualization(track, frame, frame_number)
        
        # Queue objects for enhancement
        for obj in objects_to_process:
            if obj.best_frame is not None:
                self.enhancement_queue.put(obj)
                self.objects_processed += 1
        
        # Log frame processing time
        frame_time = time.time() - start_time
        self.last_frame_time = frame_time
        rr.log(
            f"{self.visualizer.METRICS_PATH}/frame_time",
            rr.Scalar(frame_time * 1000)  # Convert to ms
        )
        
    def update_quality_visualization(self, track, frame, frame_number):
        """Update quality score visualization"""
        frame_shape = frame.shape
        score, components = self.tracker.frame_scorer.score_frame(
            frame, track.bbox, frame_shape
        )
        
        # Update best frame if needed
        if score > track.best_score:
            track.best_score = score
            track.best_frame = frame.copy()
            track.best_bbox = track.bbox
            track.best_frame_number = frame_number
            track.score_components = components
            
            # Log quality update
            rr.log(
                self.visualizer.LOGS_PATH,
                rr.TextLog(
                    f"New best frame for object {track.id} "
                    f"(score: {score:.3f})",
                    level="DEBUG"
                )
            )
        
        # Log quality metrics
        quality_text = f"""### Quality Scores - Object #{track.id}

**Overall Score:** {score:.3f}

**Components:**
- **Sharpness:** {components.get('sharpness', 0):.3f}
- **Exposure:** {components.get('exposure', 0):.3f}
- **Size:** {components.get('size', 0):.3f}
- **Centering:** {components.get('centering', 0):.3f}

**Best Score:** {track.best_score:.3f}
"""
        rr.log(
            f"{self.visualizer.METRICS_PATH}/quality",
            rr.TextDocument(quality_text, media_type=rr.MediaType.MARKDOWN)
        )
        
        # Log individual metrics as scalars
        for metric, value in components.items():
            rr.log(
                f"{self.visualizer.METRICS_PATH}/quality/{metric}/{track.id}",
                rr.Scalar(value)
            )
    
    def enhancement_worker(self):
        """Enhanced worker with Rerun visualization"""
        while True:
            try:
                obj = self.enhancement_queue.get(timeout=1)
                start_time = time.time()
                
                # Log original image
                x1, y1, x2, y2 = obj.best_bbox
                original_crop = obj.best_frame[y1:y2, x1:x2]
                
                rr.log(
                    f"{self.visualizer.ENHANCEMENT_PATH}/original/{obj.id}",
                    rr.Image(original_crop).compress(jpeg_quality=90)
                )
                
                # Enhance the best frame
                enhanced = self.enhancer.enhance_frame(
                    obj.best_frame, 
                    obj.best_bbox
                )
                
                # Log enhanced image
                enhanced_crop = enhanced[y1:y2, x1:x2]
                rr.log(
                    f"{self.visualizer.ENHANCEMENT_PATH}/enhanced/{obj.id}",
                    rr.Image(enhanced_crop).compress(jpeg_quality=90)
                )
                
                # Send to Google Lens
                api_start_time = time.time()
                results = self.google_lens.search_product(enhanced_crop, obj)
                api_time = time.time() - api_start_time
                
                # Mark processing complete
                obj.is_being_processed = False
                
                # Store results
                processing_time = time.time() - start_time
                self.results_storage[obj.id] = {
                    'object': obj,
                    'enhanced_image': enhanced,
                    'google_lens_results': results,
                    'processing_time': processing_time
                }
                
                # Log results
                rr.log(
                    self.visualizer.LOGS_PATH,
                    rr.TextLog(
                        f"Object {obj.id} ({obj.class_name}) processed in "
                        f"{processing_time:.2f}s - Processing #{obj.processing_count}",
                        level="INFO"
                    )
                )
                
                # Log Google Lens results
                lens_results_text = f"### Object #{obj.id} - {obj.class_name}\n\n"
                
                if results and results['products']:
                    lens_results_text += f"**Found {len(results['products'])} matches**\n\n"
                    
                    for i, product in enumerate(results['products'][:3]):
                        lens_results_text += f"**Match {i+1}:**\n"
                        lens_results_text += f"- **Product:** {product.get('title', 'Unknown')}\n"
                        lens_results_text += f"- **Source:** {product.get('source', 'N/A')}\n"
                        lens_results_text += f"- **Price:** {product.get('price', 'N/A')}\n"
                        
                        if 'dimensions' in product:
                            dims = product['dimensions']
                            lens_results_text += f"- **Dimensions:** {dims['width']} x {dims['height']} x {dims['depth']} {dims['unit']}\n"
                            
                            rr.log(
                                self.visualizer.LOGS_PATH,
                                rr.TextLog(
                                    f"Found: {product['title']} - "
                                    f"Dimensions: {dims['width']} x {dims['height']} x {dims['depth']} {dims['unit']}",
                                    level="SUCCESS"
                                )
                            )
                        lens_results_text += "\n"
                else:
                    lens_results_text += "**No matches found**\n"
                
                rr.log(
                    f"{self.visualizer.METRICS_PATH}/lens_results",
                    rr.TextDocument(lens_results_text, media_type=rr.MediaType.MARKDOWN)
                )
                
                # Update performance metrics
                self.update_performance_metrics(processing_time, api_time)
                
            except queue.Empty:
                continue
            except Exception as e:
                rr.log(
                    self.visualizer.LOGS_PATH,
                    rr.TextLog(f"Enhancement error: {e}", level="ERROR")
                )
    
    def update_performance_metrics(self, total_time, api_time):
        """Update performance metrics display"""
        enhancement_time = total_time - api_time
        gpu_memory_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        
        metrics_text = f"""### Processing Performance

**Latest Timings:**
- **Frame Processing:** {self.last_frame_time*1000:.1f} ms
- **Enhancement:** {enhancement_time*1000:.1f} ms
- **API Call:** {api_time:.2f} s
- **Total:** {total_time:.2f} s

**System Resources:**
- **GPU Memory:** {gpu_memory_mb:.1f} MB / {torch.cuda.get_device_properties(0).total_memory/1024**2:.1f} MB
- **Objects Processed:** {self.objects_processed}
- **Average FPS:** {self.frame_count / (time.time() - self.start_time):.1f}
"""
        
        rr.log(
            f"{self.visualizer.METRICS_PATH}/performance",
            rr.TextDocument(metrics_text, media_type=rr.MediaType.MARKDOWN)
        )
        
        # Log time series data
        rr.log(
            f"{self.visualizer.METRICS_PATH}/processing_time",
            rr.Scalar(total_time)
        )
        rr.log(
            f"{self.visualizer.METRICS_PATH}/gpu_memory",
            rr.Scalar(gpu_memory_mb)
        )
```

### 4. Google Lens Integration with Your API Guide

```python
from google.cloud import storage
from serpapi import GoogleSearch
import uuid
import time
from datetime import timedelta
import cv2
import os
import rerun as rr

# From your API guide implementation:
# from google_lens_api import GoogleLensAPI

class RerunGoogleLensAPI(GoogleLensAPI):
    """Extended Google Lens API with Rerun logging"""
    
    def __init__(self, api_key, gcs_bucket_name, visualizer):
        super().__init__(api_key)
        self.gcs_bucket_name = gcs_bucket_name
        self.visualizer = visualizer
        self.storage_client = storage.Client()
        
    def search_product(self, image, object_info):
        """Enhanced product search with GCS upload and Rerun logging"""
        try:
            # Save image temporarily
            temp_path = f"/tmp/object_{object_info.id}_{int(time.time())}.jpg"
            cv2.imwrite(temp_path, image)
            
            # Upload to GCS
            upload_result = self.upload_to_gcs(temp_path)
            if not upload_result:
                return None
            
            # Log upload status
            rr.log(
                self.visualizer.LOGS_PATH,
                rr.TextLog(
                    f"Uploaded object {object_info.id} to GCS: {upload_result['blob_name']}",
                    level="DEBUG"
                )
            )
            
            # Use Google Lens API
            lens_results = self.identify_with_google_lens(upload_result["url"])
            
            # Clean up GCS
            self.cleanup_gcs_blob(upload_result["bucket"], upload_result["blob_name"])
            
            # Parse results
            if lens_results:
                products = self.extract_camera_info(lens_results)
                
                # Get additional products if available
                if "products_page_token" in lens_results:
                    more_products = self.get_lens_products(
                        lens_results["products_page_token"]
                    )
                    if "products" in more_products:
                        products.extend(self.extract_camera_info(more_products))
                
                # Log found products
                rr.log(
                    self.visualizer.LOGS_PATH,
                    rr.TextLog(
                        f"Google Lens found {len(products)} matches for object {object_info.id}",
                        level="INFO"
                    )
                )
                
                return {
                    'object_id': object_info.id,
                    'class_name': object_info.class_name,
                    'products': products[:5]  # Top 5 matches
                }
            
            return None
            
        except Exception as e:
            rr.log(
                self.visualizer.LOGS_PATH,
                rr.TextLog(f"Google Lens API error: {e}", level="ERROR")
            )
            return None
    
    def upload_to_gcs(self, image_path):
        """Upload image to Google Cloud Storage"""
        try:
            bucket = self.storage_client.bucket(self.gcs_bucket_name)
            blob_name = f"drone_objects/{int(time.time())}_{uuid.uuid4().hex}.jpg"
            blob = bucket.blob(blob_name)
            
            blob.upload_from_filename(image_path)
            
            # Generate signed URL
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=15),
                method="GET"
            )
            
            return {
                "bucket": self.gcs_bucket_name,
                "blob_name": blob_name,
                "url": url
            }
            
        except Exception as e:
            rr.log(
                self.visualizer.LOGS_PATH,
                rr.TextLog(f"GCS upload error: {e}", level="ERROR")
            )
            return None
```

### 5. Usage Example

```python
# main.py
import torch
from dotenv import load_dotenv
import os
import rerun as rr

# Import from your existing implementation
# from config import PipelineConfig
# from rerun_pipeline import RerunEnhancementPipeline

# Load environment variables
load_dotenv()

def main():
    # Initialize SAM2 model
    sam2_model = initialize_sam2_model()  # Your SAM2 initialization
    
    # Create pipeline with Rerun visualization
    pipeline = RerunEnhancementPipeline(
        sam2_model=sam2_model,
        google_lens_api_key=os.environ.get("SERPAPI_API_KEY"),
        gcs_bucket_name="your-gcs-bucket-name"  # Replace with your bucket
    )
    
    # Configure GPU
    PipelineConfig.configure_gpu()
    
    # Process video stream
    video_source = "drone_video.mp4"  # or camera index for live feed
    
    # Optional: Set Rerun recording
    rr.save("drone_recording.rrd")
    
    # Start processing
    pipeline.process_video_stream(video_source)

if __name__ == "__main__":
    main()
```

## Key Features of This Design

### 1. **Timed Processing Strategy**
- Objects are processed after 1.5 seconds of tracking (configurable)
- Reprocessing happens every 3 seconds if the object remains in view
- No need to wait for objects to leave the scene
- Processing count tracks how many times each object has been analyzed

### 2. **Focused 2D Visualization** (3D handled by separate service)
- **Camera Feed**: Live video with SAM2 segmentations overlaid
- **Enhancement Results**: Side-by-side original vs enhanced images
- **Active Object Tracking**: Real-time list of tracked objects with scores
- **Quality Score Timeline**: Temporal view of quality metrics
- **Google Lens Results**: Structured display of product matches and dimensions
- **Processing Metrics**: Real-time performance monitoring
- **System Logs**: Color-coded logs with levels (INFO, DEBUG, ERROR, SUCCESS)

### 3. **Performance Monitoring**
- Frame processing time graphs
- Enhancement processing time tracking
- API call time measurement
- GPU memory usage monitoring
- Objects per second metrics
- Average FPS calculation

### 4. **Google Lens Integration**
- Uses your existing API guide implementation
- Automatic GCS upload/cleanup
- Caches results to avoid duplicate API calls
- Extracts and displays dimensions in structured format
- Shows top 3 product matches with details

### 5. **Robustness Features**
- Graceful error handling with logging
- Non-blocking processing queue
- Automatic cleanup of lost tracks
- GPU memory management
- Processing state tracking to avoid duplicate work

## Benefits of Rerun Integration

1. **Real-time Debugging**: See exactly why objects are being tracked/lost
2. **Quality Visualization**: Understand which frames are selected and why
3. **Performance Analysis**: Identify bottlenecks in processing
4. **Result Validation**: Verify Google Lens results visually
5. **Recording Capability**: Save entire sessions for later analysis
6. **Web-based Viewing**: Share recordings with team members

## How Scene Scaling Works

### 1. **Object Detection & Tracking**
The system continuously segments objects using SAM2 and tracks them across frames. Common household objects are ideal references:
- Monitors (typically 21-27 inches diagonal)
- Keyboards (standard ~45cm width)
- Smartphones (5-7 inches)
- Laptops (13-17 inches)
- Books, furniture, appliances

### 2. **Dimension Lookup Pipeline**
```
Frame → SAM2 Segmentation → Track for 1.5s → Select Best Frame → Enhance
  ↓
Google Lens API (Object Identification)
  ↓
Perplexity API (Dimension Lookup)
  ↓
Weighted Average Scale Calculation → Publish to Mesh Service
```

### 3. **Scale Calculation Example**
If the system detects:
- iPhone 13 (146.7mm × 71.5mm) with 85% confidence
- Dell 24" Monitor (539.5mm × 323.5mm) with 92% confidence
- Logitech Keyboard (430mm × 137mm) with 78% confidence

The weighted average considers both API confidence and segmentation quality to determine the most reliable scale factor.

### 4. **Integration with Mesh Service**
The calculated scale is published to the `scene_scaling_exchange` with:
```json
{
  "scale_factor": 0.0254,        // Example: 1 unit = 0.0254 meters
  "units_per_meter": 39.37,      // Inverse for convenience
  "confidence": 0.87,            // Combined confidence score
  "num_estimates": 3,            // Number of objects used
  "timestamp_ns": 1234567890
}
```

## Real-World Applications

### 1. **Accurate Room Dimensions**
- Measure room width, height, and depth in meters
- Calculate floor area and volume
- Verify against building specifications

### 2. **Virtual Object Placement**
- Place furniture at correct scale before purchasing
- Visualize renovations with accurate dimensions
- Design room layouts with proper spacing

### 3. **Safety & Navigation**
- Calculate safe drone flight paths
- Maintain proper distance from obstacles
- Plan emergency exit routes with real distances

### 4. **Professional Applications**
- **Real Estate**: Generate accurate floor plans
- **Architecture**: Verify as-built vs. design dimensions
- **Insurance**: Document room sizes for claims
- **Gaming/AR**: Place virtual content at realistic scales

### 5. **Measurement Without Rulers**
Once calibrated, the system becomes a 3D measuring tool:
- Measure furniture dimensions
- Check ceiling height
- Calculate distances between objects
- Verify doorway widths

## Configuration & Tuning

### Scale Confidence Thresholds
```env
MIN_CONFIDENCE_FOR_SCALING=0.7  # Minimum combined confidence
DIMENSION_CACHE_EXPIRY_DAYS=30  # Cache known object dimensions
```

### Processing Parameters
```env
PROCESS_AFTER_SECONDS=1.5       # Time to track before processing
REPROCESS_INTERVAL_SECONDS=3.0  # Re-check if object remains
IOU_THRESHOLD=0.3              # Object matching threshold
```

## Troubleshooting Scale Estimation

1. **Low Confidence Scores**
   - Ensure good lighting for clear object detection
   - Position camera to capture objects fully
   - Include multiple known objects in scene

2. **Inconsistent Scales**
   - Check if detected objects are standard sizes
   - Verify API responses for dimension accuracy
   - Review confidence thresholds

3. **No Scale Updates**
   - Confirm API keys are set correctly
   - Check network connectivity
   - Verify objects are being tracked long enough

## Future Enhancements

1. **Custom Object Database**: Add organization-specific objects
2. **Multiple Scale Validation**: Cross-check using multiple methods
3. **Temporal Smoothing**: Average scales over time for stability
4. **Scene-Specific Calibration**: Save scale profiles per location
5. **Manual Override**: Allow user to specify known object dimensions