import pika
import os
import time
import json
import cv2
import numpy as np
import torch
from ultralytics import YOLO

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://rabbitmq")
params = pika.URLParameters(RABBITMQ_URL)
params.heartbeat = 3600

def connect_rabbitmq_with_retry(max_retries=10, delay=2):
    for attempt in range(1, max_retries+1):
        print(f"Attempting to connect to RabbitMQ (attempt {attempt}/{max_retries})...")
        try:
            connection = pika.BlockingConnection(params)
            print("Successfully connected to RabbitMQ.")
            return connection
        except pika.exceptions.AMQPConnectionError as e:
            print(f"Connection failed: {e}")
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

current_mode = "none"

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
        timestamp_ns = properties.headers.get("timestamp_ns") if properties.headers else None
        np_arr = np.frombuffer(body, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            print("Invalid frame data.")
            return
        
        processed_frame = frame.copy()
        if current_mode == "yolo":
            results = model(frame, conf=0.5)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = result.names[class_id]
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(processed_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        _, encoded = cv2.imencode(".jpg", processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        annotated_bytes = encoded.tobytes()
        
        channel.basic_publish(
            exchange=PROCESSED_FRAMES_EXCHANGE,
            routing_key='',
            body=annotated_bytes,
            properties=pika.BasicProperties(
                content_type="application/octet-stream",
                headers={"timestamp_ns": timestamp_ns or "0"}
            )
        )
    except Exception as e:
        print(f"Error in frame_processor callback: {e}")

# Set up consumers using the new exchange names
channel.basic_consume(queue=analysis_queue_name, on_message_callback=analysis_mode_callback, auto_ack=True)
channel.basic_consume(queue=queue_name, on_message_callback=frame_callback, auto_ack=True)

print("frame_processor: Ready to process frames...")
channel.start_consuming()
