# frame_processor/frame_processor.py

import pika
import os
import time
import io
import cv2
import numpy as np
import torch
from ultralytics import YOLO

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://rabbitmq")
params = pika.URLParameters(RABBITMQ_URL)
params.heartbeat = 3600

print("Connecting to RabbitMQ...")
connection = pika.BlockingConnection(params)
channel = connection.channel()

# We assume YOLO is installed
model = YOLO("yolov8n.pt")  # or the path you want
model.to("cpu")  # or "cuda" if needed and available

# Declare the same exchanges
channel.exchange_declare(exchange="video_frames_exchange", exchange_type="fanout", durable=True)
channel.exchange_declare(exchange="processed_frames_exchange", exchange_type="fanout", durable=True)

# Create ephemeral queue for consuming from video_frames_exchange
q = channel.queue_declare(queue='', exclusive=True)
queue_name = q.method.queue
channel.queue_bind(exchange='video_frames_exchange', queue=queue_name)

print("frame_processor: Listening for raw frames...")

def callback(ch, method, properties, body):
    """
    body is the raw frame bytes. We'll do YOLO and publish annotated frame.
    """
    try:
        # read headers for timestamp
        timestamp_ns = properties.headers.get("timestamp_ns") if properties.headers else None

        # decode raw image
        np_arr = np.frombuffer(body, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            print("Invalid frame data.")
            return
        
        # YOLO inference
        results = model(frame, conf=0.5)

        # draw bounding boxes
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = result.names[class_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # re-encode as jpg
        _, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        annotated_bytes = encoded.tobytes()

        # publish to processed_frames_exchange as binary
        channel.basic_publish(
            exchange="processed_frames_exchange",
            routing_key='',
            body=annotated_bytes,  # raw binary
            properties=pika.BasicProperties(
                content_type="application/octet-stream",
                headers={"timestamp_ns": timestamp_ns or "0"}
            )
        )
    except Exception as e:
        print(f"Error in frame_processor callback: {e}")

channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
channel.start_consuming()
