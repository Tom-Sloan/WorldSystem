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
from ultralytics import YOLO
import rerun as rr  # Import Rerun SDK
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
import threading
from collections import deque
from datetime import datetime
from rabbitmq_config import EXCHANGES, ROUTING_KEYS

# Prometheus metrics
PROCESSED_FRAMES = Counter('frame_processor_frames_processed_total', 'Total number of frames processed')
YOLO_DETECTIONS = Counter('frame_processor_yolo_detections_total', 'Total number of YOLO detections')
PROCESSING_TIME = Histogram('frame_processor_processing_time_ms', 'Processing time in milliseconds',
                            buckets=(1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000))
FRAME_SIZE = Gauge('frame_processor_frame_size_bytes', 'Size of the processed frame in bytes')
CONNECTION_STATUS = Gauge('frame_processor_connection_status', 'Connection status (1=connected, 0=disconnected)')
NTP_TIME_OFFSET = Gauge('frame_processor_ntp_time_offset_seconds', 'NTP time offset in seconds')

# Additional image statistics metrics
FRAME_WIDTH = Gauge('frame_processor_frame_width_pixels', 'Width of the processed frame in pixels')
FRAME_HEIGHT = Gauge('frame_processor_frame_height_pixels', 'Height of the processed frame in pixels')
FRAME_TIMESTAMP = Gauge('frame_processor_latest_timestamp_ms', 'Latest frame timestamp in milliseconds')

# Add NTP client and time offset tracking
ntp_client = ntplib.NTPClient()
ntp_time_offset = 0.0  # Offset between system time and NTP time in seconds
last_ntp_sync = 0
NTP_SYNC_INTERVAL = 60  # Sync NTP every 60 seconds
NTP_SERVER = os.getenv("NTP_SERVER", "pool.ntp.org")

# Image statistics tracking
MAX_STATS_ENTRIES = 20  # Keep last 20 entries like the React component
image_stats = deque(maxlen=MAX_STATS_ENTRIES)
stats_lock = threading.Lock()

def sync_ntp_time():
    """Update the NTP time offset by querying an NTP server."""
    global ntp_time_offset, last_ntp_sync
    
    try:
        response = ntp_client.request(NTP_SERVER, timeout=5)
        # Calculate offset: NTP time - local time
        ntp_time_offset = response.offset
        last_ntp_sync = time.time()
        print(f"[NTP] Synchronized time, offset: {ntp_time_offset:.6f} seconds")
        # Update Prometheus metric
        NTP_TIME_OFFSET.set(ntp_time_offset)
        return True
    except (ntplib.NTPException, socket.gaierror, socket.timeout) as e:
        print(f"[NTP] Synchronization failed: {str(e)}")
        return False

def get_ntp_time_ns():
    """Get current time in nanoseconds, synchronized with NTP."""
    current_time = time.time() + ntp_time_offset
    # Check if we need to resync
    if time.time() - last_ntp_sync > NTP_SYNC_INTERVAL:
        # Start a thread to sync NTP time without blocking
        threading.Thread(target=sync_ntp_time, daemon=True).start()
    return int(current_time * 1e9)  # Convert to nanoseconds

# Get metrics port from environment with default of 8003
METRICS_PORT = int(os.getenv("METRICS_PORT", 8003))

# Start Prometheus HTTP server on configured port
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
    except Exception as e:
        print(f"[Rerun] Error connecting to viewer via gRPC: {e}")
        print("[Rerun] Will continue sending data regardless of connection status")

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://rabbitmq")
params = pika.URLParameters(RABBITMQ_URL)
params.heartbeat = 3600

def connect_rabbitmq_with_retry(max_retries=10, delay=2):
    for attempt in range(1, max_retries+1):
        print(f"Attempting to connect to RabbitMQ (attempt {attempt}/{max_retries})...")
        try:
            connection = pika.BlockingConnection(params)
            print("Successfully connected to RabbitMQ.")
            # Update connection status in Prometheus
            CONNECTION_STATUS.set(1)
            return connection
        except pika.exceptions.AMQPConnectionError as e:
            print(f"Connection failed: {e}")
            # Update connection status in Prometheus
            CONNECTION_STATUS.set(0)
            if attempt == max_retries:
                raise RuntimeError("Could not connect to RabbitMQ after multiple retries.")
            time.sleep(delay)

print("Connecting to RabbitMQ...")
connection = connect_rabbitmq_with_retry()
channel = connection.channel()

# RabbitMQ configuration
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://rabbitmq")
# We assume YOLO is installed
model = YOLO("yolov8n.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

current_mode = os.getenv("INITIAL_ANALYSIS_MODE", "none").lower()

# Declare new topic exchanges
for exchange_config in EXCHANGES.values():
    channel.exchange_declare(
        exchange=exchange_config['name'],
        exchange_type=exchange_config['type'],
        durable=exchange_config['durable'],
        auto_delete=exchange_config['auto_delete']
    )

# Create queue for video frames with meaningful name
q = channel.queue_declare(queue='frame_processor_video_input', exclusive=True)
queue_name = q.method.queue
channel.queue_bind(exchange='sensor_data', queue=queue_name, routing_key=ROUTING_KEYS['VIDEO_FRAMES'])

# Create queue for analysis mode updates
analysis_queue = channel.queue_declare(queue='frame_processor_analysis_mode', exclusive=True)
analysis_queue_name = analysis_queue.method.queue
channel.queue_bind(exchange='control_commands', queue=analysis_queue_name, routing_key=ROUTING_KEYS['ANALYSIS_MODE'])

print("frame_processor: Listening for raw frames...")

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
    """Process a raw frame, run YOLO if requested, and publish the annotated frame."""
    try:
        # Record start time using NTP-synchronized clock
        start_time_ns = get_ntp_time_ns()
        
        # Get timestamps from headers
        timestamp_ns = properties.headers.get("timestamp_ns") if properties.headers else None
        server_received = properties.headers.get("server_received") if properties.headers else None
        ntp_time = properties.headers.get("ntp_time") if properties.headers else server_received
        
        # Ensure timestamp_ns is an integer if provided
        if timestamp_ns is not None:
            try:
                timestamp_ns = int(timestamp_ns)
            except (ValueError, TypeError):
                timestamp_ns = None
        
        np_arr = np.frombuffer(body, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            print("Invalid frame data.")
            return
        
        processed_frame = frame.copy()
        yolo_results = None
        detection_count = 0
        
        if current_mode == "yolo":
            yolo_results = model(frame, conf=0.5)
            for result in yolo_results:
                boxes = result.boxes
                detection_count += len(boxes)
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = result.names[class_id]
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(processed_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Update YOLO detection counter
            YOLO_DETECTIONS.inc(detection_count)
        
        # Get image dimensions
        height, width = processed_frame.shape[:2]
        
        # Calculate processing time using NTP-synchronized time
        end_time_ns = get_ntp_time_ns()
        processing_time_ms = (end_time_ns - start_time_ns) / 1_000_000  # Convert ns to ms
        
        # Update Prometheus metrics
        PROCESSED_FRAMES.inc()
        PROCESSING_TIME.observe(processing_time_ms)
        FRAME_WIDTH.set(width)
        FRAME_HEIGHT.set(height)
        # Ensure timestamp is int before division
        ts_for_metric = timestamp_ns if isinstance(timestamp_ns, int) else (int(timestamp_ns) if timestamp_ns else start_time_ns)
        FRAME_TIMESTAMP.set(ts_for_metric // 1_000_000)
        
        # Log to Rerun if enabled
        if RERUN_ENABLED:
            timestamp = int(timestamp_ns) if timestamp_ns is not None else start_time_ns
            
            # Convert from BGR (OpenCV) to RGB (Rerun)
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Log the frame to Rerun
            rr.set_time("sensor_time", timestamp=1e-9 * timestamp)
            rr.log("processed_frame", rr.Image(rgb_frame).compress(jpeg_quality=85))
            
            # If we're using YOLO, also log the detections
            if current_mode == "yolo" and yolo_results is not None:
                for result in yolo_results:
                    boxes = result.boxes
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        class_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = result.names[class_id]
                        
                        # Calculate for Rerun's format
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        box_width = x2 - x1
                        box_height = y2 - y1
                        
                        # Log bounding box to Rerun
                        rr.log(
                            f"detections/{class_name}/{i}",
                            rr.Boxes2D(
                                array=[[center_x, center_y, box_width, box_height]],
                                array_format=rr.Box2DFormat.XYWH,
                                labels=[f"{class_name} {conf:.2f}"],
                                colors=[[0, 255, 0, 255]]  # RGBA
                            )
                        )
        
        _, encoded = cv2.imencode(".jpg", processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        annotated_bytes = encoded.tobytes()
        
        # Update frame size metric
        FRAME_SIZE.set(len(annotated_bytes))
        
        # Get original resolution if available
        original_width = properties.headers.get("original_width") if properties.headers else width
        original_height = properties.headers.get("original_height") if properties.headers else height
        resolution = properties.headers.get("resolution") if properties.headers else "Unknown"
        
        # Ensure original dimensions are integers
        try:
            original_width = int(original_width) if original_width is not None else width
            original_height = int(original_height) if original_height is not None else height
        except (ValueError, TypeError):
            original_width = width
            original_height = height
        
        # Ensure timestamp_ns is an integer for calculations
        timestamp_ns_int = timestamp_ns if isinstance(timestamp_ns, int) else (int(timestamp_ns) if timestamp_ns else start_time_ns)
        
        # Collect image statistics
        image_stat = {
            "timestamp_ns": timestamp_ns_int,
            "timestamp_ms": timestamp_ns_int // 1_000_000,  # Convert to ms
            "size_bytes": len(annotated_bytes),
            "width": width,
            "height": height,
            "original_width": original_width,
            "original_height": original_height,
            "processing_time_ms": processing_time_ms,
            "detection_count": detection_count,
            "resolution": resolution
        }
        
        # Add to stats deque (thread-safe)
        with stats_lock:
            image_stats.append(image_stat)
            
        # Log statistics to Rerun
        if RERUN_ENABLED:
            # Create arrays for the table visualization
            with stats_lock:
                stats_list = list(image_stats)
            
            if stats_list:
                # Extract data for visualization
                timestamps_ms = [s["timestamp_ms"] for s in stats_list]
                sizes_kb = [s["size_bytes"] / 1024 for s in stats_list]  # Convert to KB
                processing_times = [s["processing_time_ms"] for s in stats_list]
                detection_counts = [s["detection_count"] for s in stats_list]
                
                # Log as time series data
                rr.set_time("sensor_time", timestamp=1e-9 * timestamp)
                
                # Log image size over time
                rr.log("stats/image_size_kb", rr.Scalars(sizes_kb[-1]))
                
                # Log processing time
                rr.log("stats/processing_time_ms", rr.Scalars(processing_times[-1]))
                
                # Log detection count if YOLO is active
                if current_mode == "yolo":
                    rr.log("stats/detection_count", rr.Scalars(float(detection_counts[-1])))
                
                # Log frame dimensions
                rr.log("stats/frame_width", rr.Scalars(float(width)))
                rr.log("stats/frame_height", rr.Scalars(float(height)))
                
                # Create a text log with the latest stats
                latest_stat = stats_list[-1]
                stats_text = f"""Frame Statistics:
Timestamp: {latest_stat['timestamp_ms']} ms
Size: {latest_stat['size_bytes'] / 1024:.1f} KB
Dimensions: {latest_stat['width']}x{latest_stat['height']}
Processing Time: {latest_stat['processing_time_ms']:.1f} ms
Detections: {latest_stat['detection_count']}
Resolution: {latest_stat['resolution']}"""
                
                rr.log("stats/latest_info", rr.TextLog(stats_text))
                
                # Log a table view of recent stats (similar to React component)
                if len(stats_list) > 1:
                    table_text = "Recent Frame Statistics:\n"
                    table_text += "Timestamp (ms) | Size (KB) | Dimensions | Proc Time (ms) | Detections\n"
                    table_text += "-" * 70 + "\n"
                    for stat in stats_list[-10:]:  # Show last 10 entries
                        table_text += f"{stat['timestamp_ms']:14} | {stat['size_bytes']/1024:9.1f} | "
                        table_text += f"{stat['width']:4}x{stat['height']:<4} | {stat['processing_time_ms']:14.1f} | "
                        table_text += f"{stat['detection_count']:10}\n"
                    
                    rr.log("stats/recent_table", rr.TextLog(table_text))
        
        # Include all metadata in the headers including NTP information
        headers = {
            "timestamp_ns": timestamp_ns,
            "server_received": server_received,
            "ntp_time": ntp_time, 
            "processing_time_ms": str(processing_time_ms),
            "width": width,
            "height": height,
            "resolution": resolution,
            "original_width": original_width,
            "original_height": original_height,
            "ntp_offset": str(ntp_time_offset)
        }
        
        channel.basic_publish(
            exchange='processing_results',
            routing_key=ROUTING_KEYS['YOLO_FRAMES'],
            body=annotated_bytes,
            properties=pika.BasicProperties(
                content_type="application/octet-stream",
                headers=headers
            )
        )
    except Exception as e:
        print(f"Error in frame_processor callback: {e}")

# Set up consumers using the new exchange names
channel.basic_consume(queue=analysis_queue_name, on_message_callback=analysis_mode_callback, auto_ack=True)
channel.basic_consume(queue=queue_name, on_message_callback=frame_callback, auto_ack=True)

print("frame_processor: Ready to process frames...")
channel.start_consuming()