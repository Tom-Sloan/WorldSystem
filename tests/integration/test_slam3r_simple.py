#!/usr/bin/env python3
"""
Simple test to send frames to SLAM3R via RabbitMQ.
Assumes RabbitMQ and SLAM3R are already running.
"""

import asyncio
import numpy as np
import cv2
import aio_pika
import time

# Test video path
VIDEO_PATH = "/home/sam3/Desktop/Toms_Workspace/WorldSystem/data/20250617_211214/mav0/video_segments/20250617_211214_segment_1.mp4"

async def send_test_frames():
    """Send a few video frames to SLAM3R."""
    print("Connecting to RabbitMQ...")
    
    # Connect to RabbitMQ
    connection = await aio_pika.connect_robust("amqp://guest@localhost/")
    channel = await connection.channel()
    
    # Use the exchange that SLAM3R expects
    exchange = await channel.declare_exchange("video_frames", aio_pika.ExchangeType.FANOUT, durable=True)
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {fps:.2f} fps")
    print("Sending frames to SLAM3R...")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        # Send 20 frames to test
        while frame_count < 20:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Create timestamp
            timestamp_ns = int((frame_count / fps) * 1e9)
            
            # Send raw image bytes as body with metadata in headers
            await exchange.publish(
                aio_pika.Message(
                    body=cv2.imencode('.jpg', frame)[1].tobytes(),  # JPEG encode for smaller size
                    headers={
                        "timestamp_ns": str(timestamp_ns),
                        "video_segment": "test_segment",
                        "frame_id": f"test_frame_{frame_count:06d}"
                    }
                ),
                routing_key=""  # Fanout exchange ignores routing key
            )
            
            frame_count += 1
            print(f"Sent frame {frame_count}")
            
            # Small delay between frames
            await asyncio.sleep(0.1)
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        cap.release()
        await connection.close()
        
    print(f"\nSent {frame_count} frames in {time.time() - start_time:.1f}s")
    print("Check docker logs slam3r to see processing results")

if __name__ == "__main__":
    asyncio.run(send_test_frames())