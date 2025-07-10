#!/usr/bin/env python3
"""Test SLAM3R with real drone video to debug keyframe generation and tensor reshape error."""

import asyncio
import aio_pika
import cv2
import time

async def test_slam3r_real_video():
    # Use real drone footage
    video_path = "/home/sam3/Desktop/Toms_Workspace/WorldSystem/data/20250617_211214/mav0/video_segments/20250617_211214_segment_1.mp4"
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info:")
    print(f"  Path: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    
    # Connect to RabbitMQ
    connection = await aio_pika.connect_robust("amqp://localhost")
    channel = await connection.channel()
    
    # Get the exchange
    exchange = await channel.get_exchange("video_frames_exchange")
    
    print("\nSending real drone video frames to SLAM3R...")
    print("This should trigger keyframe generation and potentially reproduce the tensor reshape error")
    
    frame_count = 0
    start_time = time.time()
    
    # Send frames at realistic rate
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Encode as JPEG
        success, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            print(f"Failed to encode frame {frame_count}")
            continue
        
        img_bytes = encoded.tobytes()
        
        # Publish raw image bytes
        await exchange.publish(
            aio_pika.Message(body=img_bytes),
            routing_key=""
        )
        
        frame_count += 1
        
        # Progress update
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            actual_fps = frame_count / elapsed
            print(f"  Sent {frame_count}/{total_frames} frames ({actual_fps:.1f} fps)", end='\r')
        
        # Send first 300 frames (10 seconds at 30fps) to test
        if frame_count >= 300:
            break
        
        # Simulate realistic frame rate
        await asyncio.sleep(1.0 / fps)
    
    print(f"\n\nSent {frame_count} frames from real drone video!")
    print("Check SLAM3R logs for:")
    print("  1. Keyframe generation messages")
    print("  2. Tensor reshape errors")
    print("  3. Shared memory writes")
    print("\nAlso check mesh service logs for keyframe processing")
    
    # Give time for processing
    await asyncio.sleep(5)
    
    cap.release()
    await connection.close()

if __name__ == "__main__":
    asyncio.run(test_slam3r_real_video())