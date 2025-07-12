#!/usr/bin/env python3
"""
Test SLAM3R with real video to verify the I2P window fix resolves the tensor reshape error.
"""

import asyncio
import json
import numpy as np
import cv2
import aio_pika
from pathlib import Path
import time

# Test video path
VIDEO_PATH = "/home/sam3/Desktop/Toms_Workspace/WorldSystem/data/20250617_211214/mav0/video_segments/20250617_211214_segment_1.mp4"

async def send_video_frames():
    """Send video frames to SLAM3R via RabbitMQ."""
    # Connect to RabbitMQ
    connection = await aio_pika.connect_robust("amqp://guest@localhost/")
    channel = await connection.channel()
    
    # Declare exchange and queue
    exchange = await channel.declare_exchange("frames", aio_pika.ExchangeType.TOPIC)
    queue = await channel.declare_queue("slam3r_frames", durable=True)
    await queue.bind(exchange, "frame.slam3r")
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {fps:.2f} fps, {total_frames} frames total")
    print("Sending frames to SLAM3R...")
    
    frame_count = 0
    start_time = time.time()
    
    # Send frames with timing to avoid overwhelming the system
    try:
        while frame_count < 100:  # Send first 100 frames to test
            ret, frame = cap.read()
            if not ret:
                break
            
            # Prepare message
            timestamp_ns = int((frame_count / fps) * 1e9)
            
            message = {
                "frame_id": f"test_frame_{frame_count:06d}",
                "timestamp": timestamp_ns,
                "video_segment": "test_segment",
                "width": frame.shape[1],
                "height": frame.shape[0],
                "image_data": frame.tobytes(),
                "encoding": "bgr8",
                "detector_data": {
                    "num_detections": 0,
                    "detections": []
                }
            }
            
            # Publish to RabbitMQ - send raw image bytes as body
            # with metadata in headers
            await exchange.publish(
                aio_pika.Message(
                    body=message["image_data"],
                    headers={
                        "timestamp_ns": str(timestamp_ns),
                        "video_segment": message["video_segment"],
                        "frame_id": message["frame_id"]
                    }
                ),
                routing_key="frame.slam3r"
            )
            
            frame_count += 1
            
            # Log progress
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                actual_fps = frame_count / elapsed
                print(f"Sent {frame_count} frames, {actual_fps:.1f} fps")
            
            # Control frame rate (send at 15 fps)
            await asyncio.sleep(1.0 / 15.0)
    
    except Exception as e:
        print(f"Error sending frames: {e}")
    
    finally:
        cap.release()
        await connection.close()
        
    print(f"\nCompleted: sent {frame_count} frames in {time.time() - start_time:.1f}s")
    print("Check SLAM3R logs for processing results")

if __name__ == "__main__":
    asyncio.run(send_video_frames())