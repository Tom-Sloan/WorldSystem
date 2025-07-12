#!/usr/bin/env python3
"""
Test SLAM3R with frames after it has processed a restart message.
"""

import asyncio
import numpy as np
import cv2
import aio_pika
import time

# Test video path
VIDEO_PATH = "/home/sam3/Desktop/Toms_Workspace/WorldSystem/data/20250617_211214/mav0/video_segments/20250617_211214_segment_1.mp4"

async def send_test_frames():
    """Send video frames to SLAM3R after restart."""
    print("Waiting for SLAM3R to process restart message...")
    await asyncio.sleep(3)  # Give SLAM3R time to process the restart
    
    print("Connecting to RabbitMQ...")
    
    # Connect to RabbitMQ
    connection = await aio_pika.connect_robust("amqp://guest@localhost/")
    channel = await connection.channel()
    
    # Declare the exchange
    exchange = await channel.declare_exchange("video_frames_exchange", aio_pika.ExchangeType.FANOUT, durable=True)
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps:.2f} fps")
    print("Sending frames to SLAM3R...")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        # Send enough frames to trigger bootstrap and incremental processing
        while frame_count < 40:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Create timestamp
            timestamp_ns = int((frame_count / fps) * 1e9)
            
            # Encode frame as JPEG (SLAM3R expects encoded image)
            _, encoded = cv2.imencode('.jpg', frame)
            
            # Send encoded image bytes
            message = aio_pika.Message(
                body=encoded.tobytes(),
                headers={
                    "timestamp_ns": str(timestamp_ns),
                    "video_segment": "test_after_restart",
                    "frame_id": f"frame_{frame_count:06d}"
                }
            )
            
            await exchange.publish(message, routing_key="")
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Sent {frame_count} frames...")
            
            # Send at moderate rate
            await asyncio.sleep(0.15)  # ~6-7 fps
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        await connection.close()
        
    print(f"\nSent {frame_count} frames in {time.time() - start_time:.1f}s")
    print("\nCheck SLAM3R logs for:")
    print("1. 'Processing frame X for initialization'")
    print("2. 'Bootstrap complete with X keyframes'") 
    print("3. 'I2P window: X views' (should show 7 views with win_r=3)")
    print("4. Any tensor reshape errors")

if __name__ == "__main__":
    asyncio.run(send_test_frames())