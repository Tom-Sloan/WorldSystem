#!/usr/bin/env python3
"""Test the SLAM3R fixes for image dimensions and keyframe publishing."""

import asyncio
import aio_pika
import numpy as np
import cv2
import time
import msgpack


async def main():
    # Connect to RabbitMQ
    connection = await aio_pika.connect_robust("amqp://guest:guest@localhost:5672/")
    channel = await connection.channel()
    
    # Declare the video frames exchange
    exchange = await channel.declare_exchange(
        "video_frames_exchange",
        aio_pika.ExchangeType.FANOUT,
        durable=True
    )
    
    print("Sending test frames to SLAM3R...")
    
    # Create test frames with problematic dimensions (854x480)
    for i in range(30):
        # Create a test frame with 854x480 dimensions (not divisible by 16)
        frame = np.random.randint(0, 255, (480, 854, 3), dtype=np.uint8)
        
        # Add some pattern to make it more realistic
        cv2.rectangle(frame, (50 + i*5, 50), (200 + i*5, 200), (255, 0, 0), 3)
        cv2.putText(frame, f"Frame {i}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Encode as JPEG
        _, encoded = cv2.imencode('.jpg', frame)
        
        # Create message with headers
        message = aio_pika.Message(
            body=encoded.tobytes(),
            headers={
                "timestamp_ns": str(time.time_ns()),
                "video_id": "test_854x480"
            }
        )
        
        # Publish
        await exchange.publish(message, routing_key="")
        print(f"Sent frame {i} (854x480)")
        
        # Small delay between frames
        await asyncio.sleep(0.033)  # ~30 fps
    
    print("\nTest complete! Check SLAM3R logs for:")
    print("1. Image resizing messages (854x480 -> 864x480)")
    print("2. Patch embed type logging")
    print("3. Keyframe publishing without errors")
    print("4. No 'width not multiple of 16' errors")
    
    await connection.close()


if __name__ == "__main__":
    asyncio.run(main())