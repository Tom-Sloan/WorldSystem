#!/usr/bin/env python3
"""Test SLAM3R by sending a frame directly to RabbitMQ."""

import asyncio
import aio_pika
import json
import numpy as np
import time
from PIL import Image
import io
import cv2

async def test_slam3r():
    # Connect to RabbitMQ
    connection = await aio_pika.connect_robust("amqp://localhost")
    channel = await connection.channel()
    
    # Get the exchange
    exchange = await channel.get_exchange("video_frames_exchange")
    
    # Create a test image (640x480 RGB)
    width, height = 640, 480
    # Create a gradient pattern for better visibility
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            img[y, x] = [x % 256, y % 256, (x + y) % 256]
    
    # Encode as JPEG bytes (not base64)
    success, encoded = cv2.imencode('.jpg', img)
    if not success:
        print("Failed to encode image")
        return
    
    img_bytes = encoded.tobytes()
    
    print(f"Sending test frame to SLAM3R...")
    print(f"Frame size: {width}x{height}")
    print(f"Encoded size: {len(img_bytes)} bytes")
    print(f"Timestamp: {time.time_ns()}")
    
    # Publish raw image bytes
    await exchange.publish(
        aio_pika.Message(body=img_bytes),
        routing_key=""
    )
    
    print("Message sent! Check SLAM3R logs for processing.")
    
    await connection.close()

if __name__ == "__main__":
    asyncio.run(test_slam3r())