#!/usr/bin/env python3
"""
Simple test to check if SLAM3R tensor reshape error is fixed.
Sends just enough frames to trigger the error.
"""

import asyncio
import cv2
import numpy as np
import aio_pika
import msgpack
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VIDEO_PATH = "/home/sam3/Desktop/Toms_Workspace/WorldSystem/data/20250617_211214/mav0/video_segments/20250617_211214_segment_1.mp4"

async def send_test_frames():
    """Send exactly 10 frames to test bootstrap and first incremental frame"""
    
    # Connect to RabbitMQ
    connection = await aio_pika.connect_robust("amqp://guest@localhost/")
    channel = await connection.channel()
    
    # Declare exchange
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
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video: {width}x{height}")
    logger.info("Sending 10 frames to test tensor reshape fix...")
    
    for frame_idx in range(10):
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame")
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create message
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
        
        if frame_idx == 4:
            logger.info(f"Frame {frame_idx}: Bootstrap should complete after this")
        elif frame_idx == 5:
            logger.info(f"Frame {frame_idx}: CRITICAL - Tensor reshape error occurred here before")
        elif frame_idx == 6:
            logger.info(f"Frame {frame_idx}: If we get here, the fix is working!")
            
        # Small delay between frames
        await asyncio.sleep(0.1)
    
    cap.release()
    await connection.close()
    
    logger.info("Test complete - check SLAM3R logs for results")

async def main():
    logger.info("Starting simple SLAM3R test...")
    await send_test_frames()
    
    # Wait a bit for processing
    logger.info("Waiting 5 seconds for SLAM3R to process...")
    await asyncio.sleep(5)
    
    # Check logs
    import subprocess
    logger.info("\n=== SLAM3R Recent Logs ===")
    result = subprocess.run(
        ['docker', 'logs', 'slam3r', '--tail', '50'],
        capture_output=True, text=True
    )
    
    # Look for key indicators
    if "RuntimeError" in result.stderr or "shape '[25, 196, 12, 64]'" in result.stderr:
        logger.error("❌ TENSOR RESHAPE ERROR STILL PRESENT!")
        logger.error("Error details:")
        for line in result.stderr.split('\n'):
            if "RuntimeError" in line or "shape" in line:
                logger.error(f"  {line}")
    elif "Bootstrap complete" in result.stderr:
        logger.info("✅ Bootstrap completed successfully!")
        # Check if we processed beyond frame 5
        if "Processing frame 6" in result.stderr or "frame_index=6" in result.stderr:
            logger.info("✅ TENSOR RESHAPE FIX SUCCESSFUL! Frame 6 processed!")
        else:
            logger.warning("⚠️  Did not see frame 6 processing confirmation")
    else:
        logger.warning("⚠️  Could not determine test result - check logs manually")

if __name__ == "__main__":
    asyncio.run(main())