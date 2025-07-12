#!/usr/bin/env python3
"""
Simple test script for SLAM3R v2 processor
"""

import asyncio
import numpy as np
import cv2
import aio_pika
import time

async def test_slam3r():
    """Send test frames to SLAM3R v2"""
    print("Connecting to RabbitMQ...")
    
    # Connect to RabbitMQ
    connection = await aio_pika.connect_robust("amqp://guest:guest@localhost/")
    channel = await connection.channel()
    
    # Declare the exchange
    exchange = await channel.declare_exchange(
        "video_frames_exchange", 
        aio_pika.ExchangeType.FANOUT, 
        durable=True
    )
    
    print("Sending test frames...")
    
    # Create synthetic test frames
    for i in range(10):
        # Create a simple test image
        img = np.ones((480, 640, 3), dtype=np.uint8) * 100
        
        # Add some variation
        cv2.rectangle(img, (100 + i*10, 100), (200 + i*10, 200), (0, 255, 0), -1)
        cv2.putText(img, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Encode as JPEG
        _, encoded = cv2.imencode('.jpg', img)
        
        # Create timestamp
        timestamp_ns = int(time.time() * 1e9) + i * int(1e8)  # 100ms apart
        
        # Send message
        message = aio_pika.Message(
            body=encoded.tobytes(),
            headers={
                "timestamp_ns": str(timestamp_ns),
                "video_segment": "test_segment",
                "frame_id": f"frame_{i:06d}"
            }
        )
        
        await exchange.publish(message, routing_key="")
        print(f"Sent frame {i}")
        
        # Wait a bit between frames
        await asyncio.sleep(0.2)
    
    await connection.close()
    print("Done! Check SLAM3R logs for processing output")

if __name__ == "__main__":
    asyncio.run(test_slam3r())