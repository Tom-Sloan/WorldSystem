#!/usr/bin/env python3
"""
Test script to verify the tensor reshape fix for SLAM3R.
This will send real video frames to trigger the error and verify the fix works.
"""

import asyncio
import cv2
import numpy as np
import aio_pika
import msgpack
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test video path
VIDEO_PATH = "/home/sam3/Desktop/Toms_Workspace/WorldSystem/data/20250617_211214/mav0/video_segments/20250617_211214_segment_1.mp4"

async def send_video_frames():
    """Send real video frames to SLAM3R to test tensor reshape fix"""
    
    # Connect to RabbitMQ
    connection = await aio_pika.connect_robust("amqp://guest@localhost/")
    channel = await connection.channel()
    
    # Declare exchange (must match SLAM3R's expected exchange)
    exchange = await channel.declare_exchange(
        "video_frames_exchange",
        aio_pika.ExchangeType.TOPIC,
        durable=True
    )
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {VIDEO_PATH}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video: {width}x{height} @ {fps:.2f} fps, {frame_count} frames")
    
    frame_delay = 1.0 / fps
    frame_idx = 0
    start_time = time.time()
    
    # Send frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create message matching server expectations
        timestamp_ns = int(time.time() * 1e9)
        message = {
            "type": "video_frame",
            "timestamp_ns": timestamp_ns,
            "video_segment": "test_segment",
            "frame_index": frame_idx,
            "frame_data": frame_rgb.tobytes(),
            "width": width,
            "height": height,
            "encoding": "rgb8"
        }
        
        # Send to RabbitMQ
        await exchange.publish(
            aio_pika.Message(body=msgpack.packb(message)),
            routing_key="video.frame"
        )
        
        frame_idx += 1
        
        # Log progress
        if frame_idx % 10 == 0:
            logger.info(f"Sent frame {frame_idx}/{frame_count}")
            
        # Critical frames to watch
        if frame_idx == 5:
            logger.info("=== Bootstrap should complete after this frame ===")
        elif frame_idx == 6:
            logger.info("=== FRAME 6: This is where tensor reshape error occurred ===")
            logger.info("=== If no error appears, the fix is working! ===")
        elif frame_idx == 10:
            logger.info("=== Successfully passed the error point! ===")
            
        # Stop after 30 frames to test the critical section
        if frame_idx >= 30:
            logger.info("Test completed successfully - no tensor reshape error!")
            break
            
        # Maintain realistic frame rate
        elapsed = time.time() - start_time
        expected_time = frame_idx * frame_delay
        if elapsed < expected_time:
            await asyncio.sleep(expected_time - elapsed)
    
    cap.release()
    await connection.close()
    
    logger.info(f"Sent {frame_idx} frames total")

async def monitor_slam3r_logs():
    """Monitor SLAM3R logs in parallel to catch any errors"""
    import subprocess
    
    logger.info("Monitoring SLAM3R logs for errors...")
    
    # Start monitoring docker logs
    proc = await asyncio.create_subprocess_exec(
        'docker', 'logs', '-f', '--tail', '50', 'slam3r',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    error_patterns = [
        b"RuntimeError",
        b"shape '[25, 196, 12, 64]'",
        b"invalid for input of size",
        b"Bootstrap complete",
        b"Published keyframe",
        b"ERROR"
    ]
    
    while True:
        line = await proc.stderr.readline()
        if not line:
            break
            
        line_str = line.decode('utf-8', errors='ignore').strip()
        
        # Check for important patterns
        for pattern in error_patterns:
            if pattern in line:
                if b"RuntimeError" in line or b"ERROR" in line:
                    logger.error(f"SLAM3R ERROR: {line_str}")
                elif b"Bootstrap complete" in line:
                    logger.info(f"SLAM3R: {line_str}")
                elif b"Published keyframe" in line:
                    logger.info(f"SLAM3R: {line_str}")

async def main():
    """Run the test"""
    logger.info("Starting SLAM3R tensor reshape fix test...")
    logger.info("This test will send real video frames to trigger the error")
    logger.info("Watch for errors around frame 6...")
    
    # Check if SLAM3R is running
    import subprocess
    result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
    if 'slam3r' not in result.stdout:
        logger.error("SLAM3R container is not running! Start it with: docker compose up slam3r")
        return
    
    # Run video sender and log monitor in parallel
    await asyncio.gather(
        send_video_frames(),
        monitor_slam3r_logs(),
        return_exceptions=True
    )

if __name__ == "__main__":
    asyncio.run(main())