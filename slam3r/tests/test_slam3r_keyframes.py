#!/usr/bin/env python3
"""Test SLAM3R keyframe generation with enough frames to trigger keyframes."""

import asyncio
import aio_pika
import numpy as np
import time
import cv2

async def test_slam3r_keyframes():
    # Connect to RabbitMQ
    connection = await aio_pika.connect_robust("amqp://localhost")
    channel = await connection.channel()
    
    # Get the exchange
    exchange = await channel.get_exchange("video_frames_exchange")
    
    # Send 30 frames to ensure keyframe generation
    width, height = 640, 480
    
    print("Sending 30 frames to trigger keyframe generation...")
    print("SLAM3R typically creates keyframes every 5-15 frames based on motion")
    
    for i in range(30):
        # Create a more realistic image with features
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add background gradient
        for y in range(height):
            for x in range(width):
                img[y, x] = [(x//3 + i*5) % 256, (y//3 + i*5) % 256, 100]
        
        # Add moving features that SLAM can track
        # Moving rectangles
        cv2.rectangle(img, (100 + i*10, 100), (200 + i*10, 200), (255, 255, 255), -1)
        cv2.rectangle(img, (300 - i*5, 250), (400 - i*5, 350), (255, 0, 0), -1)
        
        # Moving circles  
        cv2.circle(img, (320 + i*3, 240 - i*2), 50, (0, 255, 0), -1)
        cv2.circle(img, (150 + i*7, 350 + i*3), 30, (255, 255, 0), -1)
        
        # Add some texture
        for _ in range(50):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            cv2.circle(img, (x, y), 2, (255, 255, 255), -1)
        
        # Encode as JPEG
        success, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            print(f"Failed to encode image {i}")
            continue
        
        img_bytes = encoded.tobytes()
        
        print(f"  Frame {i+1}/30 ({len(img_bytes)} bytes)", end='\r')
        
        # Publish raw image bytes
        await exchange.publish(
            aio_pika.Message(body=img_bytes),
            routing_key=""
        )
        
        # Simulate camera frame rate (30fps)
        await asyncio.sleep(0.033)
    
    print("\n\nAll frames sent! SLAM3R should generate several keyframes.")
    print("Check mesh service logs for processing.")
    
    # Give time for processing
    await asyncio.sleep(2)
    
    await connection.close()

if __name__ == "__main__":
    asyncio.run(test_slam3r_keyframes())