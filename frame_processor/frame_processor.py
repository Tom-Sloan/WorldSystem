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
from ultralytics import YOLO
import rerun as rr
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
from collections import deque
from datetime import datetime
from dotenv import load_dotenv

# Import our modules
from modules.tracker import ObjectTracker, TrackedObject
from modules.frame_scorer import FrameQualityScorer
from modules.enhancement import ImageEnhancer
from modules.api_client import APIClient
from modules.scene_scaler import SceneScaler

# Load environment variables
load_dotenv()

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
        # Configuration
        self.process_after_seconds = float(os.getenv("PROCESS_AFTER_SECONDS", "1.5"))
        self.reprocess_interval_seconds = float(os.getenv("REPROCESS_INTERVAL_SECONDS", "3.0"))
        self.iou_threshold = float(os.getenv("IOU_THRESHOLD", "0.3"))
        self.max_lost_frames = int(os.getenv("MAX_LOST_FRAMES", "10"))
        self.enhancement_enabled = os.getenv("ENHANCEMENT_ENABLED", "true").lower() == "true"
        
        # Initialize components
        self.tracker = ObjectTracker(
            iou_threshold=self.iou_threshold,
            max_lost_frames=self.max_lost_frames,
            process_after_seconds=self.process_after_seconds,
            reprocess_interval_seconds=self.reprocess_interval_seconds
        )
        
        self.frame_scorer = FrameQualityScorer()
        self.enhancer = ImageEnhancer(
            gamma=float(os.getenv("ENHANCEMENT_GAMMA", "1.2")),
            alpha=float(os.getenv("ENHANCEMENT_ALPHA", "1.3")),
            beta=int(os.getenv("ENHANCEMENT_BETA", "20"))
        )
        
        self.api_client = APIClient()
        self.scene_scaler = SceneScaler(
            min_confidence=float(os.getenv("MIN_CONFIDENCE_FOR_SCALING", "0.7"))
        )
        
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
        
        
    def process_frame(self, frame: np.ndarray, properties, frame_number: int):
        """Process a frame with enhanced tracking and quality scoring."""
        start_time_ns = get_ntp_time_ns()
        
        # Get timestamps from headers
        timestamp_ns = properties.headers.get("timestamp_ns") if properties.headers else None
        
        # Log camera frame
        if RERUN_ENABLED:
            try:
                rr.log("/camera", rr.Image(frame).compress(jpeg_quality=80))
            except AttributeError:
                # Fallback for older versions without compress method
                rr.log("/camera", rr.Image(frame))
        
        # Run YOLO detection
        results = model(frame, conf=0.5)
        detections = []
        detection_count = 0
        
        for result in results:
            boxes = result.boxes
            detection_count += len(boxes)
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = result.names[class_id]
                bbox = (x1, y1, x2, y2)
                detections.append((bbox, class_name, conf))
        
        # Update YOLO detection counter
        YOLO_DETECTIONS.inc(detection_count)
        
        # Update tracker
        objects_to_process = self.tracker.update(detections, frame, frame_number)
        
        # Update tracking metrics
        OBJECTS_TRACKED.set(len(self.tracker.tracked_objects))
        
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
        self.log_tracking_status()
        
        # Return processed frame for standard pipeline
        return self.annotate_frame(frame, detections)
    
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
            
            if RERUN_ENABLED:
                rr.log(
                    "/logs",
                    rr.TextLog(
                        f"New best frame for object {track.id} (score: {score:.3f})",
                        level="DEBUG"
                    )
                )
    
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
    
    def log_tracking_status(self):
        """Log current tracking status to Rerun."""
        if not RERUN_ENABLED:
            return
            
        active_tracks = self.tracker.get_active_tracks()
        if not active_tracks:
            return
            
        status_text = "### Active Objects\n\n"
        for track in active_tracks:
            status_text += f"""**Object #{track.id} - {track.class_name}**
- Confidence: {track.confidence:.2%}
- Frames: {len(track.frame_history)}
- Best Score: {track.best_score:.3f}
- Processing: {track.processing_count} times
- Has Dimensions: {'Yes' if track.estimated_dimensions else 'No'}

"""
        
        rr.log("/tracking/active", rr.TextDocument(status_text, media_type=rr.MediaType.MARKDOWN))
    
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
        return False

def get_ntp_time_ns():
    """Get current time in nanoseconds, synchronized with NTP."""
    current_time = time.time() + ntp_time_offset
    if time.time() - last_ntp_sync > NTP_SYNC_INTERVAL:
        threading.Thread(target=sync_ntp_time, daemon=True).start()
    return int(current_time * 1e9)

# Get metrics port from environment
METRICS_PORT = int(os.getenv("METRICS_PORT", 8003))

# Start Prometheus HTTP server
print(f"[Prometheus] Starting metrics server on port {METRICS_PORT}...")
start_http_server(METRICS_PORT)

# Initialize NTP synchronization
print("[NTP] Initializing time synchronization...")
sync_ntp_time()

# Initialize Rerun
RERUN_ENABLED = os.getenv("RERUN_ENABLED", "true").lower() == "true"
if RERUN_ENABLED:
    print("[Rerun] Initializing...")
    # Get environment variables for Rerun configuration
    viewer_address = os.environ.get("RERUN_VIEWER_ADDRESS", "0.0.0.0:9090")
    print(f"[Rerun] Viewer address: {viewer_address}")
    
    # Initialize Rerun with application name - don't automatically spawn a viewer
    rr.init("frame_processor", spawn=False)
    
    print(f"[Rerun] Initialized with viewer address: {viewer_address}")
    
    # Connect to the viewer using gRPC
    rerun_connect_url = os.getenv(
        "RERUN_CONNECT_URL",
        "rerun+http://localhost:9876/proxy"  # sensible fallback
    )
    try:
        print(f"[Rerun] Connecting to viewer via gRPC at {rerun_connect_url}")
        rr.connect_grpc(rerun_connect_url)
        print(f"[Rerun] Connected to viewer via gRPC")
        
        # Send test data to verify connection
        print("[Rerun] Sending test data...")
        positions = np.zeros((10, 3))
        positions[:,0] = np.linspace(-10,10,10)
        
        colors = np.zeros((10,3), dtype=np.uint8)
        colors[:,0] = np.linspace(0,255,10)
        
        rr.log(
            "test/my_points",
            rr.Points3D(positions, colors=colors, radii=0.5)
        )
        print("[Rerun] Test data sent successfully")
    except Exception as e:
        print(f"[Rerun] Error connecting to viewer via gRPC: {e}")
        print("[Rerun] Will continue sending data regardless of connection status")

# RabbitMQ setup
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://rabbitmq")
params = pika.URLParameters(RABBITMQ_URL)
params.heartbeat = 3600

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

print("Connecting to RabbitMQ...")
connection = connect_rabbitmq_with_retry()
channel = connection.channel()

# Exchange names
VIDEO_FRAMES_EXCHANGE = os.getenv("VIDEO_FRAMES_EXCHANGE", "video_frames_exchange")
PROCESSED_FRAMES_EXCHANGE = os.getenv("PROCESSED_FRAMES_EXCHANGE", "processed_frames_exchange")
ANALYSIS_MODE_EXCHANGE = os.getenv("ANALYSIS_MODE_EXCHANGE", "analysis_mode_exchange")
SCENE_SCALING_EXCHANGE = os.getenv("SCENE_SCALING_EXCHANGE", "scene_scaling_exchange")

# Initialize YOLO model with error handling
try:
    model = YOLO("yolov8n.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Attempting to download yolov8n.pt...")
    try:
        model = YOLO("yolov8n.pt")  # This will trigger download
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"Model downloaded successfully. Using device: {device}")
    except Exception as e2:
        print(f"Failed to download YOLO model: {e2}")
        raise

# Current analysis mode
current_mode = os.getenv("INITIAL_ANALYSIS_MODE", "none").lower()

# Declare exchanges
channel.exchange_declare(exchange=VIDEO_FRAMES_EXCHANGE, exchange_type="fanout", durable=True)
channel.exchange_declare(exchange=PROCESSED_FRAMES_EXCHANGE, exchange_type="fanout", durable=True)
channel.exchange_declare(exchange=ANALYSIS_MODE_EXCHANGE, exchange_type="fanout", durable=True)
channel.exchange_declare(exchange=SCENE_SCALING_EXCHANGE, exchange_type="fanout", durable=True)

# Create queues
q = channel.queue_declare(queue='frame_processor_video_input', exclusive=True)
queue_name = q.method.queue
channel.queue_bind(exchange=VIDEO_FRAMES_EXCHANGE, queue=queue_name)

analysis_queue = channel.queue_declare(queue='frame_processor_analysis_mode', exclusive=True)
analysis_queue_name = analysis_queue.method.queue
channel.queue_bind(exchange=ANALYSIS_MODE_EXCHANGE, queue=analysis_queue_name)

# Initialize enhanced processor
processor = EnhancedFrameProcessor()

print("Enhanced frame_processor: Ready to process frames...")

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
    
    try:
        # Record start time
        start_time_ns = get_ntp_time_ns()
        
        # Decode frame
        np_arr = np.frombuffer(body, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            print("Invalid frame data.")
            return
        
        # Process frame if mode is active
        if current_mode == "yolo":
            processed_frame = processor.process_frame(frame, properties, processor.frame_count)
        else:
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
        
        # Encode and publish
        _, encoded = cv2.imencode(".jpg", processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        annotated_bytes = encoded.tobytes()
        
        FRAME_SIZE.set(len(annotated_bytes))
        
        # Get original headers
        timestamp_ns = properties.headers.get("timestamp_ns") if properties.headers else None
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
            "objects_tracked": str(len(processor.tracker.tracked_objects)),
            "scene_scale_confidence": str(processor.scene_scaler.calculate_weighted_scale()['confidence'])
        }
        
        channel.basic_publish(
            exchange=PROCESSED_FRAMES_EXCHANGE,
            routing_key='',
            body=annotated_bytes,
            properties=pika.BasicProperties(
                content_type="application/octet-stream",
                headers=headers
            )
        )
        
    except Exception as e:
        print(f"Error in enhanced frame_processor callback: {e}")

# Set up consumers
channel.basic_consume(queue=analysis_queue_name, on_message_callback=analysis_mode_callback, auto_ack=True)
channel.basic_consume(queue=queue_name, on_message_callback=frame_callback, auto_ack=True)

print("Enhanced frame_processor: Processing frames with tracking and dimension estimation...")
channel.start_consuming()