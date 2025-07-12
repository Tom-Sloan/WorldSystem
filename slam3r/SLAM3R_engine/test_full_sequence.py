#!/usr/bin/env python3
"""
Test SLAM3R v2 with a longer sequence to see full processing
"""

import asyncio
import numpy as np
import cv2
import aio_pika
import time

async def test_full_sequence():
    """Send more frames to test full pipeline"""
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
    
    print("Sending test sequence...")
    
    # Create more varied test frames
    for i in range(30):  # Send 30 frames
        # Create a test image with moving pattern
        img = np.ones((480, 640, 3), dtype=np.uint8) * 50
        
        # Add moving rectangles to create visual features
        x = 100 + i * 5
        y = 100 + int(20 * np.sin(i * 0.3))
        cv2.rectangle(img, (x, y), (x + 100, y + 100), (0, 255, 0), -1)
        cv2.rectangle(img, (300, 200), (400, 300), (255, 0, 0), -1)
        cv2.circle(img, (320 + i * 2, 250), 20, (0, 0, 255), -1)
        
        # Add text
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
                "video_segment": "test_sequence",
                "frame_id": f"frame_{i:06d}"
            }
        )
        
        await exchange.publish(message, routing_key="")
        
        if i % 5 == 0:
            print(f"Sent frame {i}")
        
        # Send at moderate rate
        await asyncio.sleep(0.1)  # 10 fps
    
    await connection.close()
    print("\nDone! Sent 30 frames")
    print("\nExpected behavior:")
    print("1. Frames 0-4: Initialization")
    print("2. Frame 5+: Should see 'SLAM initialization successful'")
    print("3. Frames 5-9: First batch processing")
    print("4. Frame 10+: Continued processing with results")

if __name__ == "__main__":
    asyncio.run(test_full_sequence())