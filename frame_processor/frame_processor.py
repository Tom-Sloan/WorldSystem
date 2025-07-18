print("[DEBUG] Starting frame_processor.py imports...")
import sys
sys.stdout.flush()

print("[DEBUG] Importing standard libraries...")
import pika
import os
import time
import json
import cv2
import numpy as np
import torch
import ntplib
import socket
import threading
import queue
from typing import Dict, List, Optional, Tuple
print("[DEBUG] Standard libraries imported successfully")
sys.stdout.flush()

print("[DEBUG] Importing ultralytics YOLO...")
from ultralytics import YOLO
print("[DEBUG] YOLO imported successfully")
sys.stdout.flush()

print("[DEBUG] Importing rerun...")
import rerun as rr
# Check if Box2DFormat is available (for debugging)
try:
    from rerun.datatypes import Box2DFormat
    print("[DEBUG] Box2DFormat available from rerun.datatypes")
except ImportError:
    try:
        Box2DFormat = rr.Box2DFormat
        print("[DEBUG] Box2DFormat available from rr.Box2DFormat")
    except AttributeError:
        print("[DEBUG] Box2DFormat not available in this Rerun version")
        Box2DFormat = None
print("[DEBUG] Rerun imported successfully")
sys.stdout.flush()

print("[DEBUG] Importing prometheus_client...")
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
from collections import deque
from datetime import datetime
print("[DEBUG] Prometheus client imported successfully")
sys.stdout.flush()

print("[DEBUG] Importing dotenv...")
from dotenv import load_dotenv
print("[DEBUG] Dotenv imported successfully")
sys.stdout.flush()

# Import our modules
print("[DEBUG] Starting custom module imports...")
sys.stdout.flush()

print("[DEBUG] Importing modules.tracker...")
from modules.tracker import ObjectTracker, TrackedObject
print("[DEBUG] modules.tracker imported successfully")
sys.stdout.flush()

print("[DEBUG] Importing modules.frame_scorer...")
from modules.frame_scorer import FrameQualityScorer
print("[DEBUG] modules.frame_scorer imported successfully")
sys.stdout.flush()

print("[DEBUG] Importing modules.enhancement...")
from modules.enhancement import ImageEnhancer
print("[DEBUG] modules.enhancement imported successfully")
sys.stdout.flush()

print("[DEBUG] Importing modules.api_client...")
from modules.api_client import APIClient
print("[DEBUG] modules.api_client imported successfully")
sys.stdout.flush()

print("[DEBUG] Importing modules.scene_scaler...")
from modules.scene_scaler import SceneScaler
print("[DEBUG] modules.scene_scaler imported successfully")
sys.stdout.flush()

print("[DEBUG] Importing modules.file_logger...")
from modules.file_logger import FileLogger
print("[DEBUG] modules.file_logger imported successfully")
sys.stdout.flush()

print("[DEBUG] Importing enhanced_rerun_visualizer...")
from enhanced_rerun_visualizer import EnhancedRerunVisualizer, show_both_pages, show_live_page_only, show_process_page_only
from integrate_enhanced_visualizer import integrate_enhanced_visualization
print("[DEBUG] enhanced_rerun_visualizer imported successfully")
sys.stdout.flush()

# Load environment variables
print("[DEBUG] Loading environment variables...")
load_dotenv()
print("[DEBUG] Environment variables loaded")
sys.stdout.flush()

# Prometheus metrics
PROCESSED_FRAMES = Counter('frame_processor_frames_processed_total', 'Total number of frames processed')
YOLO_DETECTIONS = Counter('frame_processor_yolo_detections_total', 'Total number of YOLO detections')
PROCESSING_TIME = Histogram('frame_processor_processing_time_ms', 'Processing time in milliseconds',
                            buckets=(1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000))
FRAME_SIZE = Gauge('frame_processor_frame_size_bytes', 'Size of the processed frame in bytes')
CONNECTION_STATUS = Gauge('frame_processor_connection_status', 'Connection status (1=connected, 0=disconnected)')
NTP_TIME_OFFSET = Gauge('frame_processor_ntp_time_offset_seconds', 'NTP time offset in seconds')
OBJECTS_TRACKED = Gauge('frame_processor_objects_tracked', 'Number of objects currently being tracked')
OBJECTS_PROCESSED = Counter('frame_processor_objects_processed_total', 'Total objects processed for dimensions')
SCENE_SCALE_CONFIDENCE = Gauge('frame_processor_scene_scale_confidence', 'Confidence of scene scale estimation')

# Additional metrics
FRAME_WIDTH = Gauge('frame_processor_frame_width_pixels', 'Width of the processed frame in pixels')
FRAME_HEIGHT = Gauge('frame_processor_frame_height_pixels', 'Height of the processed frame in pixels')
FRAME_TIMESTAMP = Gauge('frame_processor_latest_timestamp_ms', 'Latest frame timestamp in milliseconds')

# NTP client and time offset tracking
ntp_client = ntplib.NTPClient()
ntp_time_offset = 0.0
last_ntp_sync = 0
NTP_SYNC_INTERVAL = 60
NTP_SERVER = os.getenv("NTP_SERVER", "pool.ntp.org")

# Image statistics tracking
MAX_STATS_ENTRIES = 20
image_stats = deque(maxlen=MAX_STATS_ENTRIES)
stats_lock = threading.Lock()


class EnhancedFrameProcessor:
    """Enhanced frame processor with tracking, quality scoring, and dimension estimation."""
    
    def __init__(self):
        print("[DEBUG EnhancedFrameProcessor] Initializing EnhancedFrameProcessor...")
        sys.stdout.flush()
        
        # Configuration
        self.process_after_seconds = float(os.getenv("PROCESS_AFTER_SECONDS", "1.5"))
        self.reprocess_interval_seconds = float(os.getenv("REPROCESS_INTERVAL_SECONDS", "3.0"))
        self.iou_threshold = float(os.getenv("IOU_THRESHOLD", "0.3"))
        self.max_lost_frames = int(os.getenv("MAX_LOST_FRAMES", "10"))
        self.enhancement_enabled = os.getenv("ENHANCEMENT_ENABLED", "true").lower() == "true"
        
        print(f"[DEBUG EnhancedFrameProcessor] Config loaded - process_after: {self.process_after_seconds}s, enhancement: {self.enhancement_enabled}")
        sys.stdout.flush()
        
        # Initialize file logger
        log_dir = os.getenv("LOG_DIR", "/app/logs")
        self.logger = FileLogger(log_dir=log_dir)
        self.logger.log("general", "PROCESSOR_INIT", {
            "process_after_seconds": self.process_after_seconds,
            "reprocess_interval_seconds": self.reprocess_interval_seconds,
            "enhancement_enabled": self.enhancement_enabled
        })
        
        # Initialize components
        print("[DEBUG EnhancedFrameProcessor] Creating ObjectTracker...")
        sys.stdout.flush()
        self.tracker = ObjectTracker(
            iou_threshold=self.iou_threshold,
            max_lost_frames=self.max_lost_frames,
            process_after_seconds=self.process_after_seconds,
            reprocess_interval_seconds=self.reprocess_interval_seconds
        )
        print("[DEBUG EnhancedFrameProcessor] ObjectTracker created")
        sys.stdout.flush()
        
        print("[DEBUG EnhancedFrameProcessor] Creating FrameQualityScorer...")
        sys.stdout.flush()
        self.frame_scorer = FrameQualityScorer()
        print("[DEBUG EnhancedFrameProcessor] FrameQualityScorer created")
        sys.stdout.flush()
        
        print("[DEBUG EnhancedFrameProcessor] Creating ImageEnhancer...")
        sys.stdout.flush()
        self.enhancer = ImageEnhancer(
            gamma=float(os.getenv("ENHANCEMENT_GAMMA", "1.2")),
            alpha=float(os.getenv("ENHANCEMENT_ALPHA", "1.3")),
            beta=int(os.getenv("ENHANCEMENT_BETA", "20"))
        )
        print("[DEBUG EnhancedFrameProcessor] ImageEnhancer created")
        sys.stdout.flush()
        
        print("[DEBUG EnhancedFrameProcessor] Creating APIClient...")
        sys.stdout.flush()
        self.api_client = APIClient()
        print("[DEBUG EnhancedFrameProcessor] APIClient created")
        sys.stdout.flush()
        
        print("[DEBUG EnhancedFrameProcessor] Creating SceneScaler...")
        sys.stdout.flush()
        self.scene_scaler = SceneScaler(
            min_confidence=float(os.getenv("MIN_CONFIDENCE_FOR_SCALING", "0.7"))
        )
        print("[DEBUG EnhancedFrameProcessor] SceneScaler created")
        sys.stdout.flush()
        
        # Processing queue
        self.enhancement_queue = queue.Queue()
        self.results_storage = {}
        
        # Start enhancement thread
        self.enhancement_thread = threading.Thread(target=self.enhancement_worker)
        self.enhancement_thread.daemon = True
        self.enhancement_thread.start()
        
        # Rerun is already initialized globally, no need for setup here
        
        # Metrics
        self.frame_count = 0
        self.objects_processed = 0
        self.start_time = time.time()
        self.last_detection_count = 0
        
        
    def process_frame(self, frame: np.ndarray, properties, frame_number: int):
        """Process a frame with enhanced tracking and quality scoring."""
        start_time_ns = get_ntp_time_ns()
        
        # Get timestamps from headers
        timestamp_ns = properties.headers.get("timestamp_ns") if properties.headers else None
        if timestamp_ns and isinstance(timestamp_ns, str):
            timestamp_ns = int(timestamp_ns)
        
        # Camera frame logging is now handled by enhanced visualizer
        
        # Run YOLO detection
        try:
            results = model(frame, conf=0.5)
            detections = []
            detection_count = 0
            
            for result in results:
                try:
                    boxes = result.boxes
                    if boxes is None:
                        self.logger.log("general", "YOLO_NO_BOXES", 
                            {"frame_number": frame_number})
                        continue
                        
                    detection_count += len(boxes)
                    for box in boxes:
                        try:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Validate coordinates
                            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                                self.logger.log("general", "YOLO_BBOX_OUT_OF_BOUNDS",
                                    {"bbox": [x1, y1, x2, y2], "frame_shape": frame.shape[:2]})
                                # Clip to frame bounds
                                x1 = max(0, x1)
                                y1 = max(0, y1)
                                x2 = min(frame.shape[1], x2)
                                y2 = min(frame.shape[0], y2)
                            
                            class_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = result.names[class_id]
                            bbox = (x1, y1, x2, y2)
                            detections.append((bbox, class_name, conf))
                            
                        except Exception as e:
                            self.logger.log_error("YOLO_BOX_PARSE_ERROR",
                                f"Failed to parse YOLO box: {str(e)}",
                                {"frame_number": frame_number})
                            continue
                            
                except Exception as e:
                    self.logger.log_error("YOLO_RESULT_PARSE_ERROR",
                        f"Failed to parse YOLO result: {str(e)}",
                        {"frame_number": frame_number})
                    continue
                    
        except Exception as e:
            self.logger.log_error("YOLO_INFERENCE_ERROR",
                f"YOLO inference failed: {str(e)}",
                {"frame_number": frame_number})
            detections = []
            detection_count = 0
        
        # YOLO detections logging is now handled by enhanced visualizer
        if False and RERUN_ENABLED and detections:  # Disabled - handled by enhanced visualizer
            try:
                boxes_2d = []
                labels = []
                class_ids = []
                colors = []
                
                for bbox, class_name, conf in detections:
                    try:
                        x1, y1, x2, y2 = bbox
                        
                        # Validate bbox
                        if x1 >= x2 or y1 >= y2:
                            self.logger.log_error("YOLO_BBOX_VALIDATION_ERROR",
                                f"Invalid YOLO bbox: ({x1},{y1},{x2},{y2}) for {class_name}",
                                {"frame_number": frame_number, "class_name": class_name})
                            continue
                            
                        # Convert to center + half-size format for Rerun
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        half_width = (x2 - x1) / 2
                        half_height = (y2 - y1) / 2
                        
                        if half_width <= 0 or half_height <= 0:
                            self.logger.log_error("YOLO_DIMENSION_ERROR",
                                f"Invalid YOLO dimensions: width={half_width}, height={half_height}",
                                {"frame_number": frame_number, "class_name": class_name})
                            continue
                        
                        boxes_2d.append([center_x, center_y, half_width, half_height])
                        labels.append(f"{class_name} ({conf:.2f})")
                        class_ids.append(class_name)
                        # Color based on class (using hash for consistency)
                        color_hash = hash(class_name) % 360
                        colors.append([color_hash, 255, 255])  # HSV color
                        
                    except Exception as e:
                        self.logger.log_error("YOLO_DETECTION_PROCESSING_ERROR",
                            f"Error processing YOLO detection: {str(e)}",
                            {"frame_number": frame_number, "bbox": str(bbox)})
                        continue
                
                if boxes_2d:  # Only log if we have valid boxes
                    try:
                        # Convert to numpy arrays for Rerun 0.23.2
                        import numpy as np
                        boxes_array = np.array(boxes_2d, dtype=np.float32)
                        
                        # For Rerun 0.23.2, use centers and half_sizes
                        # boxes_array is in format [center_x, center_y, half_width, half_height]
                        centers = boxes_array[:, :2]  # First two columns: center positions
                        half_sizes = boxes_array[:, 2:]  # Last two columns: half dimensions
                        
                        try:
                            rr.log(
                                "detections/yolo",
                                rr.Boxes2D(
                                    centers=centers,
                                    half_sizes=half_sizes,
                                    labels=labels,
                                    class_ids=class_ids,
                                    colors=colors
                                )
                            )
                        except Exception as e:
                            # Fallback - try array approach
                            self.logger.log("rerun", "YOLO_RERUN_FALLBACK",
                                {"message": "Using array fallback for YOLO", "error": str(e)})
                            try:
                                rr.log(
                                    "detections/yolo",
                                    rr.Boxes2D(
                                        array=boxes_array,
                                        labels=labels,
                                        class_ids=class_ids,
                                        colors=colors
                                    )
                                )
                            except Exception as e2:
                                self.logger.log_error("YOLO_BOXES2D_FAILED",
                                    f"All YOLO Boxes2D approaches failed: {str(e2)}",
                                    {"initial_error": str(e)})
                        
                        # Log successful Rerun message
                        self.logger.log_rerun_message("detections/yolo", "YOLO_DETECTIONS_LOGGED", {
                            "num_detections": len(boxes_2d),
                            "frame_number": frame_number
                        })
                        
                    except Exception as e:
                        self.logger.log_error("RERUN_YOLO_BOXES2D_ERROR",
                            f"Failed to log YOLO boxes to Rerun: {str(e)}",
                            {"frame_number": frame_number, "num_boxes": len(boxes_2d)})
                        import traceback
                        self.logger.log("errors", "YOLO_TRACEBACK", {"traceback": traceback.format_exc()})
                        
            except Exception as e:
                self.logger.log_error("LOG_YOLO_DETECTIONS_ERROR",
                    f"Failed to log YOLO detections to Rerun: {str(e)}",
                    {"frame_number": frame_number})
        
        # Update YOLO detection counter
        YOLO_DETECTIONS.inc(detection_count)
        
        # Store last detection count for logging
        self.last_detection_count = detection_count
        
        # Log detections to file
        for bbox, class_name, conf in detections:
            self.logger.log_detection(frame_number, class_name, conf, bbox)
        
        # Update tracker
        try:
            objects_to_process = self.tracker.update(detections, frame, frame_number)
        except Exception as e:
            self.logger.log_error("TRACKER_UPDATE_ERROR",
                f"Failed to update tracker: {str(e)}",
                {"frame_number": frame_number, "num_detections": len(detections)})
            objects_to_process = []
        
        # Update tracking metrics
        OBJECTS_TRACKED.set(len(self.tracker.tracked_objects))
        
        # Log objects to process
        for obj in objects_to_process:
            self.logger.log_tracking_event("OBJECT_READY_FOR_PROCESSING", obj.id, obj.class_name, {
                "best_score": obj.best_score,
                "frame_count": len(obj.frame_history)
            })
        
        # Tracked objects logging is now handled by enhanced visualizer
        
        # Update quality scores for active tracks
        for track in self.tracker.get_active_tracks():
            if track.last_seen_frame == frame_number:
                self.update_track_quality(track, frame)
        
        # Queue objects for enhancement and API processing
        for obj in objects_to_process:
            if obj.best_frame is not None:
                self.enhancement_queue.put(obj)
                self.objects_processed += 1
                OBJECTS_PROCESSED.inc()
        
        # Log tracking information
        try:
            self.log_tracking_status()
        except Exception as e:
            self.logger.log_error("LOG_TRACKING_STATUS_CALL_ERROR",
                f"Failed to call log_tracking_status: {str(e)}",
                {"frame_number": frame_number})
        
        # Return processed frame for standard pipeline
        return self.annotate_frame(frame, detections)
    
    def log_tracked_objects_to_rerun(self, frame_number: int):
        """Log tracked objects with their IDs to Rerun for visualization."""
        try:
            active_tracks = self.tracker.get_active_tracks()
            if not active_tracks:
                return
            
            boxes_2d = []
            labels = []
            class_ids = []
            colors = []
            
            for track in active_tracks:
                try:
                    x1, y1, x2, y2 = track.bbox
                    
                    # Validate bbox values
                    if x1 >= x2 or y1 >= y2:
                        self.logger.log_error("BBOX_VALIDATION_ERROR", 
                            f"Invalid bbox for track {track.id}: ({x1},{y1},{x2},{y2})",
                            {"track_id": track.id, "frame_number": frame_number})
                        continue
                        
                    # Convert to center + half-size format for Rerun
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    half_width = (x2 - x1) / 2
                    half_height = (y2 - y1) / 2
                    
                    # Validate converted values
                    if half_width <= 0 or half_height <= 0:
                        self.logger.log_error("BBOX_DIMENSION_ERROR",
                            f"Invalid bbox dimensions for track {track.id}: width={half_width}, height={half_height}",
                            {"track_id": track.id, "frame_number": frame_number})
                        continue
                    
                    boxes_2d.append([center_x, center_y, half_width, half_height])
                    
                    # Label with track ID and dimensions if available
                    label = f"#{track.id} {track.class_name}"
                    if track.estimated_dimensions:
                        dims = track.estimated_dimensions
                        label += f" ({dims.get('width_m', 0):.2f}m)"
                    labels.append(label)
                    
                    class_ids.append(f"track_{track.id}")
                    
                    # Color based on whether we have dimensions
                    if track.estimated_dimensions:
                        colors.append([120, 255, 255])  # Green in HSV
                    else:
                        colors.append([60, 255, 255])   # Yellow in HSV
                        
                except Exception as e:
                    self.logger.log_error("TRACK_PROCESSING_ERROR",
                        f"Error processing track {track.id}: {str(e)}",
                        {"track_id": track.id, "frame_number": frame_number})
                    continue
            
            if boxes_2d:  # Only log if we have valid boxes
                try:
                    # Convert to numpy arrays for Rerun
                    import numpy as np
                    boxes_array = np.array(boxes_2d, dtype=np.float32)
                    
                    # For Rerun 0.23.2, use centers and half_sizes instead of array format
                    # boxes_array is in format [center_x, center_y, half_width, half_height]
                    centers = boxes_array[:, :2]  # First two columns
                    half_sizes = boxes_array[:, 2:]  # Last two columns
                    
                    try:
                        rr.log(
                            "tracking/objects",
                            rr.Boxes2D(
                                centers=centers,
                                half_sizes=half_sizes,
                                labels=labels,
                                class_ids=class_ids,
                                colors=colors
                            )
                        )
                    except Exception as e:
                        # Fallback - try the array approach without format
                        self.logger.log("rerun", "RERUN_BOXES2D_FALLBACK", 
                            {"message": "Using array fallback", "error": str(e)})
                        try:
                            rr.log(
                                "tracking/objects",
                                rr.Boxes2D(
                                    array=boxes_array,
                                    labels=labels,
                                    class_ids=class_ids,
                                    colors=colors
                                )
                            )
                        except Exception as e2:
                            self.logger.log_error("RERUN_BOXES2D_FAILED",
                                f"All Boxes2D approaches failed: {str(e2)}",
                                {"initial_error": str(e)})
                    
                    # Log successful Rerun message
                    self.logger.log_rerun_message("tracking/objects", "TRACKED_OBJECTS_LOGGED", {
                        "num_tracks": len(boxes_2d),
                        "frame_number": frame_number
                    })
                    
                except Exception as e:
                    self.logger.log_error("RERUN_BOXES2D_ERROR", 
                        f"Failed to log boxes to Rerun: {str(e)}",
                        {"frame_number": frame_number, "num_boxes": len(boxes_2d)})
                    # Log individual components for debugging
                    self.logger.log("rerun", "RERUN_DEBUG_INFO", {
                        "boxes_2d_sample": str(boxes_2d[:2]) if boxes_2d else "empty",
                        "labels_sample": str(labels[:2]) if labels else "empty",
                        "error_type": type(e).__name__
                    })
                    
        except Exception as e:
            self.logger.log_error("LOG_TRACKED_OBJECTS_ERROR",
                f"Failed to log tracked objects to Rerun: {str(e)}",
                {"frame_number": frame_number})
            import traceback
            self.logger.log("errors", "TRACEBACK", {"traceback": traceback.format_exc()})
        
        # Also log tracking statistics
        try:
            rr.log("tracking/stats/active_objects", rr.Scalar(len(active_tracks)))
            rr.log("tracking/stats/frame_number", rr.Scalar(frame_number))
        except Exception as e:
            self.logger.log_error("RERUN_STATS_LOG_ERROR",
                f"Failed to log tracking stats: {str(e)}",
                {"frame_number": frame_number})
    
    def update_track_quality(self, track: TrackedObject, frame: np.ndarray):
        """Update quality score for a tracked object."""
        score, components = self.frame_scorer.score_frame(
            frame, track.bbox, frame.shape[:2]
        )
        
        # Update best frame if needed
        if score > track.best_score:
            track.best_score = score
            track.best_frame = frame.copy()
            track.best_bbox = track.bbox
            track.best_frame_number = self.frame_count
            track.score_components = components
            
            # Best frame logging is now handled by enhanced visualizer
            
            self.logger.log_tracking_event("NEW_BEST_FRAME", track.id, track.class_name, {
                "score": score,
                "components": components
            })
    
    def enhancement_worker(self):
        """Worker thread for enhancement and API processing."""
        while True:
            try:
                track = self.enhancement_queue.get(timeout=1)
                
                # Log processing start
                if RERUN_ENABLED:
                    rr.log(
                        "/logs",
                        rr.TextLog(
                            f"Processing object {track.id} ({track.class_name})",
                            level="INFO"
                        )
                    )
                
                self.logger.log_tracking_event("PROCESSING_START", track.id, track.class_name)
                
                # Extract best frame region
                x1, y1, x2, y2 = track.best_bbox
                roi = track.best_frame[y1:y2, x1:x2]
                
                # Enhance if enabled
                if self.enhancement_enabled:
                    enhanced_roi = self.enhancer.enhance_frame(roi)
                else:
                    enhanced_roi = roi
                
                # Log enhanced image
                if RERUN_ENABLED:
                    try:
                        rr.log(
                            f"/enhancement/object_{track.id}",
                            rr.Image(enhanced_roi).compress(jpeg_quality=90)
                        )
                    except AttributeError:
                        # Fallback for older versions without compress method
                        rr.log(
                            f"/enhancement/object_{track.id}",
                            rr.Image(enhanced_roi)
                        )
                
                # Process with API for dimensions
                dimension_result = self.api_client.process_object_for_dimensions(
                    enhanced_roi, track.id, track.class_name
                )
                
                if dimension_result:
                    # Update track with dimension info
                    track.identified_products = dimension_result.get('all_products', [])
                    track.estimated_dimensions = dimension_result.get('dimensions')
                    
                    # Add to scene scaler
                    self.scene_scaler.add_dimension_estimate(
                        track.id, track.class_name, dimension_result, track.confidence
                    )
                    
                    # Log dimension results
                    self.log_dimension_results(track, dimension_result)
                    
                    # Log to file
                    self.logger.log_dimension_result(
                        track.id,
                        dimension_result.get('product_name', 'Unknown'),
                        dimension_result.get('dimensions', {}),
                        dimension_result.get('confidence', 0)
                    )
                    
                    # Update scene scale and publish
                    self.update_and_publish_scene_scale()
                
                # Mark processing complete
                track.is_being_processed = False
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Enhancement worker error: {e}")
                if RERUN_ENABLED:
                    rr.log("/logs", rr.TextLog(f"Processing error: {e}", level="ERROR"))
                
                self.logger.log_error("ENHANCEMENT_ERROR", str(e), {
                    "track_id": track.id if 'track' in locals() else None
                })
    
    def log_dimension_results(self, track: TrackedObject, result: Dict):
        """Log dimension results to Rerun."""
        if not RERUN_ENABLED:
            return
            
        dims = result.get('dimensions', {})
        text = f"""### Object #{track.id} - {track.class_name}

**Product:** {result.get('product_name', 'Unknown')}
**Confidence:** {result.get('confidence', 0):.2%}

**Dimensions:**
- Width: {dims.get('width', 0):.1f} {dims.get('unit', '')}
- Height: {dims.get('height', 0):.1f} {dims.get('unit', '')}
- Depth: {dims.get('depth', 0):.1f} {dims.get('unit', '')}

**Metric:**
- Width: {dims.get('width_m', 0):.3f} m
- Height: {dims.get('height_m', 0):.3f} m
- Depth: {dims.get('depth_m', 0):.3f} m
"""
        
        rr.log(f"/dimensions", rr.TextDocument(text, media_type=rr.MediaType.MARKDOWN))
        
        self.logger.log_rerun_message("/dimensions", "DIMENSION_DOCUMENT", {
            "track_id": track.id,
            "product_name": result.get('product_name')
        })
        
        rr.log(
            "/logs",
            rr.TextLog(
                f"Identified object {track.id} as {result.get('product_name')} "
                f"({dims.get('width_m', 0):.2f}m x {dims.get('height_m', 0):.2f}m x {dims.get('depth_m', 0):.2f}m)",
                level="SUCCESS"
            )
        )
    
    def update_and_publish_scene_scale(self):
        """Calculate and publish scene scale to RabbitMQ."""
        scale_info = self.scene_scaler.calculate_weighted_scale()
        
        # Update metrics
        SCENE_SCALE_CONFIDENCE.set(scale_info['confidence'])
        
        # Log to Rerun
        if RERUN_ENABLED:
            scale_text = f"""### Scene Scale Estimation

**Scale Factor:** {scale_info['scale_factor']:.4f} m/unit
**Confidence:** {scale_info['confidence']:.2%}
**Based on:** {scale_info['num_estimates']} objects

**Average Dimensions:**
- Width: {scale_info['avg_dimensions_m']['width']:.3f} m
- Height: {scale_info['avg_dimensions_m']['height']:.3f} m  
- Depth: {scale_info['avg_dimensions_m']['depth']:.3f} m

**Contributing Objects:**
"""
            for est in scale_info['estimates']:
                scale_text += f"\n- {est['product']} (conf: {est['confidence']:.2%})"
                
            rr.log("/scale", rr.TextDocument(scale_text, media_type=rr.MediaType.MARKDOWN))
            
            self.logger.log("general", "SCENE_SCALE_UPDATE", {
                "scale_factor": scale_info['scale_factor'],
                "confidence": scale_info['confidence'],
                "num_estimates": scale_info['num_estimates']
            })
        
        # Publish to RabbitMQ
        if scale_info['confidence'] > 0:
            self.publish_scene_scale(scale_info)
    
    def publish_scene_scale(self, scale_info: Dict):
        """Publish scene scale to RabbitMQ for mesh service."""
        try:
            message = {
                'scale_factor': scale_info['scale_factor'],
                'units_per_meter': 1.0 / scale_info['scale_factor'],
                'confidence': scale_info['confidence'],
                'num_estimates': scale_info['num_estimates'],
                'timestamp_ns': get_ntp_time_ns()
            }
            
            channel.basic_publish(
                exchange=SCENE_SCALING_EXCHANGE,
                routing_key='',
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    content_type="application/json"
                )
            )
            
        except Exception as e:
            print(f"Error publishing scene scale: {e}")
            self.logger.log_error("SCENE_SCALE_PUBLISH_ERROR", str(e), {
                "scale_factor": scale_info.get('scale_factor'),
                "confidence": scale_info.get('confidence')
            })
    
    def log_tracking_status(self):
        """Log current tracking status to Rerun."""
        if not RERUN_ENABLED:
            return
            
        try:
            active_tracks = self.tracker.get_active_tracks()
            if not active_tracks:
                return
                
            status_text = "### Active Objects\n\n"
            for track in active_tracks:
                try:
                    status_text += f"""**Object #{track.id} - {track.class_name}**
- Confidence: {track.confidence:.2%}
- Frames: {len(track.frame_history)}
- Best Score: {track.best_score:.3f}
- Processing: {track.processing_count} times
- Has Dimensions: {'Yes' if track.estimated_dimensions else 'No'}

"""
                except Exception as e:
                    self.logger.log_error("TRACKING_STATUS_FORMAT_ERROR",
                        f"Failed to format track status: {str(e)}",
                        {"track_id": track.id if hasattr(track, 'id') else None})
                    continue
            
            rr.log("/tracking/active", rr.TextDocument(status_text, media_type=rr.MediaType.MARKDOWN))
            
        except Exception as e:
            self.logger.log_error("LOG_TRACKING_STATUS_ERROR",
                f"Failed to log tracking status: {str(e)}")
            import traceback
            self.logger.log("errors", "TRACKING_STATUS_TRACEBACK", {"traceback": traceback.format_exc()})
    
    def annotate_frame(self, frame: np.ndarray, detections: List) -> np.ndarray:
        """Annotate frame with tracking information."""
        annotated = frame.copy()
        
        for track in self.tracker.get_active_tracks():
            x1, y1, x2, y2 = track.bbox
            
            # Color based on whether we have dimensions
            if track.estimated_dimensions:
                color = (0, 255, 0)  # Green for dimensioned objects
            else:
                color = (255, 255, 0)  # Yellow for tracking
                
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Label with dimensions if available
            label = f"{track.class_name} #{track.id}"
            if track.estimated_dimensions:
                dims = track.estimated_dimensions
                label += f" ({dims.get('width_m', 0):.2f}m)"
                
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated


# Initialize NTP synchronization
def sync_ntp_time():
    """Update the NTP time offset by querying an NTP server."""
    global ntp_time_offset, last_ntp_sync
    
    try:
        response = ntp_client.request(NTP_SERVER, timeout=5)
        ntp_time_offset = response.offset
        last_ntp_sync = time.time()
        print(f"[NTP] Synchronized time, offset: {ntp_time_offset:.6f} seconds")
        NTP_TIME_OFFSET.set(ntp_time_offset)
        return True
    except (ntplib.NTPException, socket.gaierror, socket.timeout) as e:
        print(f"[NTP] Synchronization failed: {str(e)}")
        if 'processor' in globals() and hasattr(processor, 'logger'):
            processor.logger.log_error("NTP_SYNC_ERROR", str(e))
        return False

def get_ntp_time_ns():
    """Get current time in nanoseconds, synchronized with NTP."""
    current_time = time.time() + ntp_time_offset
    if time.time() - last_ntp_sync > NTP_SYNC_INTERVAL:
        threading.Thread(target=sync_ntp_time, daemon=True).start()
    return int(current_time * 1e9)

# Get metrics port from environment
print("[DEBUG] Getting metrics port from environment...")
sys.stdout.flush()
METRICS_PORT = int(os.getenv("METRICS_PORT", 8003))
print(f"[DEBUG] Metrics port: {METRICS_PORT}")
sys.stdout.flush()

# Start Prometheus HTTP server
print(f"[Prometheus] Starting metrics server on port {METRICS_PORT}...")
sys.stdout.flush()
start_http_server(METRICS_PORT)
print(f"[DEBUG] Prometheus server started successfully")
sys.stdout.flush()

# Initialize NTP synchronization
print("[NTP] Initializing time synchronization...")
sys.stdout.flush()
sync_ntp_time()
print("[DEBUG] NTP sync completed")
sys.stdout.flush()

# Initialize Rerun
print("[DEBUG] Checking if Rerun is enabled...")
sys.stdout.flush()
RERUN_ENABLED = os.getenv("RERUN_ENABLED", "true").lower() == "true"
print(f"[DEBUG] Rerun enabled: {RERUN_ENABLED}")
sys.stdout.flush()

if RERUN_ENABLED:
    print("[Rerun] Initializing...")
    sys.stdout.flush()
    # Get environment variables for Rerun configuration
    viewer_address = os.environ.get("RERUN_VIEWER_ADDRESS", "0.0.0.0:9090")
    print(f"[Rerun] Viewer address: {viewer_address}")
    sys.stdout.flush()
    
    # Initialize Rerun with application name - don't automatically spawn a viewer
    print("[DEBUG] Calling rr.init()...")
    sys.stdout.flush()
    rr.init("frame_processor", spawn=False)
    print("[DEBUG] rr.init() completed")
    sys.stdout.flush()
    
    print(f"[Rerun] Initialized with viewer address: {viewer_address}")
    sys.stdout.flush()
    
    # Connect to the viewer using gRPC
    rerun_connect_url = os.getenv(
        "RERUN_CONNECT_URL",
        "rerun+http://localhost:9876/proxy"  # sensible fallback
    )
    try:
        print(f"[Rerun] Connecting to viewer via gRPC at {rerun_connect_url}")
        print(f"[DEBUG] Attempting to connect to Rerun gRPC at: {rerun_connect_url}")
        sys.stdout.flush()
        rr.connect_grpc(rerun_connect_url)
        print(f"[Rerun] Connected to viewer via gRPC")
        sys.stdout.flush()
        
        # Send test data to verify connection
        print("[Rerun] Sending test data...")
        sys.stdout.flush()
        positions = np.zeros((10, 3))
        positions[:,0] = np.linspace(-10,10,10)
        
        colors = np.zeros((10,3), dtype=np.uint8)
        colors[:,0] = np.linspace(0,255,10)
        
        # Test data logging disabled - using enhanced visualizer instead
        # rr.log(
        #     "test/my_points",
        #     rr.Points3D(positions, colors=colors, radii=0.5)
        # )
        print("[Rerun] Test data sent successfully")
        sys.stdout.flush()
    except Exception as e:
        print(f"[Rerun] Error connecting to viewer via gRPC: {e}")
        print("[Rerun] Will continue sending data regardless of connection status")

# RabbitMQ setup
print("[DEBUG] Setting up RabbitMQ connection parameters...")
sys.stdout.flush()
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://rabbitmq")
print(f"[DEBUG] RabbitMQ URL: {RABBITMQ_URL}")
sys.stdout.flush()
params = pika.URLParameters(RABBITMQ_URL)
params.heartbeat = 3600
print("[DEBUG] RabbitMQ parameters configured")
sys.stdout.flush()

def connect_rabbitmq_with_retry(max_retries=10, delay=2):
    for attempt in range(1, max_retries+1):
        print(f"Attempting to connect to RabbitMQ (attempt {attempt}/{max_retries})...")
        try:
            connection = pika.BlockingConnection(params)
            print("Successfully connected to RabbitMQ.")
            CONNECTION_STATUS.set(1)
            return connection
        except pika.exceptions.AMQPConnectionError as e:
            print(f"Connection failed: {e}")
            CONNECTION_STATUS.set(0)
            if attempt == max_retries:
                raise RuntimeError("Could not connect to RabbitMQ after multiple retries.")
            time.sleep(delay)

print("[DEBUG] Starting RabbitMQ connection...")
sys.stdout.flush()
print("Connecting to RabbitMQ...")
sys.stdout.flush()
connection = connect_rabbitmq_with_retry()
print("[DEBUG] RabbitMQ connection established")
sys.stdout.flush()
channel = connection.channel()
print("[DEBUG] RabbitMQ channel created")
sys.stdout.flush()

# Exchange names
print("[DEBUG] Getting exchange names from environment...")
sys.stdout.flush()
VIDEO_FRAMES_EXCHANGE = os.getenv("VIDEO_FRAMES_EXCHANGE", "video_frames_exchange")
PROCESSED_FRAMES_EXCHANGE = os.getenv("PROCESSED_FRAMES_EXCHANGE", "processed_frames_exchange")
ANALYSIS_MODE_EXCHANGE = os.getenv("ANALYSIS_MODE_EXCHANGE", "analysis_mode_exchange")
SCENE_SCALING_EXCHANGE = os.getenv("SCENE_SCALING_EXCHANGE", "scene_scaling_exchange")
print(f"[DEBUG] Exchange names loaded: video={VIDEO_FRAMES_EXCHANGE}, processed={PROCESSED_FRAMES_EXCHANGE}")
sys.stdout.flush()

# Initialize YOLO model with error handling
print("[DEBUG] Initializing YOLO model...")
sys.stdout.flush()
try:
    print("[DEBUG] Loading yolov11l.pt...")
    sys.stdout.flush()
    model = YOLO("yolov11l.pt")
    print("[DEBUG] YOLO model loaded, checking CUDA availability...")
    sys.stdout.flush()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
    sys.stdout.flush()
    model.to(device)
    print(f"Using device: {device}")
    sys.stdout.flush()
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Attempting to download yolov8n.pt...")
    try:
        model = YOLO("yolov11l.pt")  # This will trigger download
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"Model downloaded successfully. Using device: {device}")
    except Exception as e2:
        print(f"Failed to download YOLO model: {e2}")
        raise

# Current analysis mode
current_mode = os.getenv("INITIAL_ANALYSIS_MODE", "none").lower()
print(f"[DEBUG] Initial analysis mode: {current_mode}")
sys.stdout.flush()

# Declare exchanges
print("[DEBUG] Declaring RabbitMQ exchanges...")
sys.stdout.flush()
channel.exchange_declare(exchange=VIDEO_FRAMES_EXCHANGE, exchange_type="fanout", durable=True)
print(f"[DEBUG] Declared exchange: {VIDEO_FRAMES_EXCHANGE}")
sys.stdout.flush()
channel.exchange_declare(exchange=PROCESSED_FRAMES_EXCHANGE, exchange_type="fanout", durable=True)
print(f"[DEBUG] Declared exchange: {PROCESSED_FRAMES_EXCHANGE}")
sys.stdout.flush()
channel.exchange_declare(exchange=ANALYSIS_MODE_EXCHANGE, exchange_type="fanout", durable=True)
print(f"[DEBUG] Declared exchange: {ANALYSIS_MODE_EXCHANGE}")
sys.stdout.flush()
channel.exchange_declare(exchange=SCENE_SCALING_EXCHANGE, exchange_type="fanout", durable=True)
print(f"[DEBUG] Declared exchange: {SCENE_SCALING_EXCHANGE}")
sys.stdout.flush()

# Create queues
print("[DEBUG] Creating queues...")
sys.stdout.flush()
q = channel.queue_declare(queue='frame_processor_video_input', exclusive=True)
queue_name = q.method.queue
print(f"[DEBUG] Created queue: {queue_name}")
sys.stdout.flush()
channel.queue_bind(exchange=VIDEO_FRAMES_EXCHANGE, queue=queue_name)
print(f"[DEBUG] Bound queue {queue_name} to exchange {VIDEO_FRAMES_EXCHANGE}")
sys.stdout.flush()

analysis_queue = channel.queue_declare(queue='frame_processor_analysis_mode', exclusive=True)
analysis_queue_name = analysis_queue.method.queue
print(f"[DEBUG] Created analysis queue: {analysis_queue_name}")
sys.stdout.flush()
channel.queue_bind(exchange=ANALYSIS_MODE_EXCHANGE, queue=analysis_queue_name)
print(f"[DEBUG] Bound queue {analysis_queue_name} to exchange {ANALYSIS_MODE_EXCHANGE}")
sys.stdout.flush()

# Initialize enhanced processor
print("[DEBUG] Creating EnhancedFrameProcessor instance...")
sys.stdout.flush()
processor = EnhancedFrameProcessor()
print("[DEBUG] EnhancedFrameProcessor instance created successfully")
sys.stdout.flush()

# Integrate enhanced Rerun visualizer
if RERUN_ENABLED:
    print("[DEBUG] Integrating enhanced Rerun visualizer...")
    sys.stdout.flush()
    visualizer = integrate_enhanced_visualization(processor)
    print("[DEBUG] Enhanced Rerun visualizer integrated successfully")
    print("[DEBUG] You can switch view modes with:")
    print("  - show_both_pages(processor.visualizer)     # Both pages (default)")
    print("  - show_live_page_only(processor.visualizer) # Live monitoring only")
    print("  - show_process_page_only(processor.visualizer) # Gallery only")
    sys.stdout.flush()

print("Enhanced frame_processor: Ready to process frames...")
sys.stdout.flush()

def analysis_mode_callback(ch, method, properties, body):
    """Handle analysis mode changes."""
    global current_mode
    try:
        msg = json.loads(body)
        current_mode = msg.get("data", "none").lower()
        print(f"Analysis mode changed to: {current_mode}")
    except Exception as e:
        print(f"Error processing analysis mode update: {e}")

def frame_callback(ch, method, properties, body):
    """Process a raw frame with enhanced tracking and dimension estimation."""
    global processor
    
    print(f"[DEBUG frame_callback] Received frame, body size: {len(body)} bytes")
    sys.stdout.flush()
    
    try:
        # Record start time
        start_time_ns = get_ntp_time_ns()
        
        # Decode frame
        print("[DEBUG frame_callback] Decoding frame...")
        sys.stdout.flush()
        np_arr = np.frombuffer(body, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            print("[ERROR frame_callback] Invalid frame data - cv2.imdecode returned None")
            sys.stdout.flush()
            return
        
        print(f"[DEBUG frame_callback] Frame decoded successfully, shape: {frame.shape}, current_mode: {current_mode}")
        sys.stdout.flush()
        
        # Get timestamp from headers
        timestamp_ns = properties.headers.get("timestamp_ns") if properties.headers else None
        if timestamp_ns and isinstance(timestamp_ns, str):
            timestamp_ns = int(timestamp_ns)
        
        # Raw frame logging is now handled by enhanced visualizer
        
        # Process frame if mode is active
        if current_mode == "yolo":
            print("[DEBUG frame_callback] Processing frame with YOLO...")
            sys.stdout.flush()
            try:
                processed_frame = processor.process_frame(frame, properties, processor.frame_count)
            except Exception as e:
                print(f"[ERROR frame_callback] process_frame failed: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                
                if hasattr(processor, 'logger'):
                    processor.logger.log_error("PROCESS_FRAME_ERROR", str(e), {
                        "frame_count": processor.frame_count,
                        "mode": current_mode
                    })
                    processor.logger.log("errors", "PROCESS_FRAME_TRACEBACK", 
                        {"traceback": traceback.format_exc()})
                
                # Return original frame if processing fails
                processed_frame = frame
        else:
            print(f"[DEBUG frame_callback] Skipping YOLO processing (mode='{current_mode}')")
            sys.stdout.flush()
            # Still log raw frame to Rerun even in 'none' mode
            processed_frame = frame
        
        processor.frame_count += 1
        
        # Get frame dimensions
        height, width = processed_frame.shape[:2]
        
        # Calculate processing time
        end_time_ns = get_ntp_time_ns()
        processing_time_ms = (end_time_ns - start_time_ns) / 1_000_000
        
        # Update Prometheus metrics
        PROCESSED_FRAMES.inc()
        PROCESSING_TIME.observe(processing_time_ms)
        FRAME_WIDTH.set(width)
        FRAME_HEIGHT.set(height)
        
        # Log frame processing metrics
        if current_mode == "yolo" and hasattr(processor, 'logger'):
            # Count detections from processed frame (YOLO mode)
            detection_count = 0
            if hasattr(processor, 'frame_count'):
                # Get detection count from last YOLO run (stored in processor)
                detection_count = getattr(processor, 'last_detection_count', 0)
            
            tracked_count = len(processor.tracker.tracked_objects) if hasattr(processor, 'tracker') else 0
            processor.logger.log_frame_processing(
                processor.frame_count,
                processing_time_ms,
                detection_count,
                tracked_count
            )
        
        # Encode and publish
        _, encoded = cv2.imencode(".jpg", processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        annotated_bytes = encoded.tobytes()
        
        FRAME_SIZE.set(len(annotated_bytes))
        
        # Get original headers
        timestamp_ns = properties.headers.get("timestamp_ns") if properties.headers else None
        if timestamp_ns and isinstance(timestamp_ns, str):
            timestamp_ns = int(timestamp_ns)
        server_received = properties.headers.get("server_received") if properties.headers else None
        ntp_time = properties.headers.get("ntp_time") if properties.headers else server_received
        
        # Include all metadata in headers
        headers = {
            "timestamp_ns": timestamp_ns,
            "server_received": server_received,
            "ntp_time": ntp_time, 
            "processing_time_ms": str(processing_time_ms),
            "width": width,
            "height": height,
            "ntp_offset": str(ntp_time_offset),
            "objects_tracked": str(len(processor.tracker.tracked_objects) if hasattr(processor.tracker, 'tracked_objects') else 0),
            "scene_scale_confidence": str(processor.scene_scaler.calculate_weighted_scale()['confidence'] if hasattr(processor, 'scene_scaler') else 0)
        }
        
        print(f"[DEBUG frame_callback] Publishing processed frame to exchange: {PROCESSED_FRAMES_EXCHANGE}")
        sys.stdout.flush()
        
        channel.basic_publish(
            exchange=PROCESSED_FRAMES_EXCHANGE,
            routing_key='',
            body=annotated_bytes,
            properties=pika.BasicProperties(
                content_type="application/octet-stream",
                headers=headers
            )
        )
        
        print(f"[DEBUG frame_callback] Frame {processor.frame_count} processed and published successfully")
        sys.stdout.flush()
        
    except Exception as e:
        print(f"Error in enhanced frame_processor callback: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        
        if 'processor' in locals() and hasattr(processor, 'logger'):
            processor.logger.log_error("FRAME_CALLBACK_ERROR", str(e), {
                "frame_count": processor.frame_count if hasattr(processor, 'frame_count') else None
            })

# Set up consumers
print("[DEBUG] Setting up RabbitMQ consumers...")
sys.stdout.flush()
channel.basic_consume(queue=analysis_queue_name, on_message_callback=analysis_mode_callback, auto_ack=True)
print(f"[DEBUG] Consumer set up for analysis mode queue: {analysis_queue_name}")
sys.stdout.flush()
channel.basic_consume(queue=queue_name, on_message_callback=frame_callback, auto_ack=True)
print(f"[DEBUG] Consumer set up for frame queue: {queue_name}")
sys.stdout.flush()

print("Enhanced frame_processor: Processing frames with tracking and dimension estimation...")
sys.stdout.flush()
print("[DEBUG] Starting to consume messages from RabbitMQ...")
sys.stdout.flush()
try:
    channel.start_consuming()
except KeyboardInterrupt:
    print("\nShutting down gracefully...")
    if hasattr(processor, 'logger'):
        processor.logger.close()
    channel.stop_consuming()
    connection.close()
except Exception as e:
    print(f"Fatal error: {e}")
    if hasattr(processor, 'logger'):
        processor.logger.log_error("FATAL_ERROR", str(e))
        processor.logger.close()
    raise