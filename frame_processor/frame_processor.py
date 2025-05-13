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

# Prometheus metrics
PROCESSED_FRAMES = Counter('frame_processor_frames_processed_total', 'Total number of frames processed')
YOLO_DETECTIONS = Counter('frame_processor_yolo_detections_total', 'Total number of YOLO detections')
PROCESSING_TIME = Histogram('frame_processor_processing_time_ms', 'Processing time in milliseconds',
                            buckets=(1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000))
FRAME_SIZE = Gauge('frame_processor_frame_size_bytes', 'Size of the processed frame in bytes')
CONNECTION_STATUS = Gauge('frame_processor_connection_status', 'Connection status (1=connected, 0=disconnected)')
NTP_TIME_OFFSET = Gauge('frame_processor_ntp_time_offset_seconds', 'NTP time offset in seconds')

# Add NTP client and time offset tracking
ntp_client = ntplib.NTPClient()
ntp_time_offset = 0.0  # Offset between system time and NTP time in seconds
last_ntp_sync = 0
NTP_SYNC_INTERVAL = 60  # Sync NTP every 60 seconds
NTP_SERVER = os.getenv("NTP_SERVER", "pool.ntp.org")

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

# Read exchange names from environment variables (with defaults)
VIDEO_FRAMES_EXCHANGE   = os.getenv("VIDEO_FRAMES_EXCHANGE", "video_frames_exchange")
PROCESSED_FRAMES_EXCHANGE = os.getenv("PROCESSED_FRAMES_EXCHANGE", "processed_frames_exchange")
ANALYSIS_MODE_EXCHANGE  = os.getenv("ANALYSIS_MODE_EXCHANGE", "analysis_mode_exchange")
# We assume YOLO is installed
model = YOLO("yolov8n.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

current_mode = os.getenv("INITIAL_ANALYSIS_MODE", "none").lower()

# Declare exchanges using the variables
channel.exchange_declare(exchange=VIDEO_FRAMES_EXCHANGE, exchange_type="fanout", durable=True)
channel.exchange_declare(exchange=PROCESSED_FRAMES_EXCHANGE, exchange_type="fanout", durable=True)
channel.exchange_declare(exchange=ANALYSIS_MODE_EXCHANGE, exchange_type="fanout", durable=True)

# Create queue for video frames with meaningful name
q = channel.queue_declare(queue='frame_processor_video_input', exclusive=True)
queue_name = q.method.queue
channel.queue_bind(exchange=VIDEO_FRAMES_EXCHANGE, queue=queue_name)

# Create queue for analysis mode updates
analysis_queue = channel.queue_declare(queue='frame_processor_analysis_mode', exclusive=True)
analysis_queue_name = analysis_queue.method.queue
channel.queue_bind(exchange=ANALYSIS_MODE_EXCHANGE, queue=analysis_queue_name)

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
        
        # Log to Rerun if enabled
        if RERUN_ENABLED:
            timestamp = int(timestamp_ns) if timestamp_ns is not None else start_time_ns
            
            # Convert from BGR (OpenCV) to RGB (Rerun)
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Log the frame to Rerun
            rr.set_time_nanos("capture_time", timestamp)
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
            exchange=PROCESSED_FRAMES_EXCHANGE,
            routing_key='',
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