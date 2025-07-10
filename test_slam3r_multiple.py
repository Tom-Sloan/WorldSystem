#!/usr/bin/env python3
"""Test SLAM3R by sending multiple frames to trigger initialization."""

import asyncio
import aio_pika
import numpy as np
import time
import cv2

async def test_slam3r():
    # Connect to RabbitMQ
    connection = await aio_pika.connect_robust("amqp://localhost")
    channel = await connection.channel()
    
    # Get the exchange
    exchange = await channel.get_exchange("video_frames_exchange")
    
    # Send 10 frames to trigger initialization
    width, height = 640, 480
    
    for i in range(10):
        # Create a slightly different image each time (moving pattern)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                # Add some movement to the pattern
                img[y, x] = [(x + i*10) % 256, (y + i*10) % 256, (x + y + i*10) % 256]
        
        # Add some features for SLAM to track
        cv2.rectangle(img, (100 + i*5, 100), (200 + i*5, 200), (255, 255, 255), -1)
        cv2.circle(img, (300 + i*3, 300), 50, (255, 0, 0), -1)
        
        # Encode as JPEG
        success, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            print(f"Failed to encode image {i}")
            continue
        
        img_bytes = encoded.tobytes()
        
        print(f"Sending frame {i+1}/10 ({len(img_bytes)} bytes)")
        
        # Publish raw image bytes
        await exchange.publish(
            aio_pika.Message(body=img_bytes),
            routing_key=""
        )
        
        # Small delay between frames
        await asyncio.sleep(0.1)
    
    print("\nAll frames sent! Check SLAM3R logs for processing.")
    print("SLAM3R needs multiple frames to initialize the scene.")
    
    await connection.close()

if __name__ == "__main__":
    asyncio.run(test_slam3r())