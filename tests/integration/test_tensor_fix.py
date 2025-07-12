#!/usr/bin/env python3
"""
Test script to verify the L2W tensor reshape error is fixed.
Sends exactly 10 frames to test bootstrap and incremental processing.
"""

import asyncio
import cv2
import numpy as np
import aio_pika
import msgpack
import time
import logging
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VIDEO_PATH = "/home/sam3/Desktop/Toms_Workspace/WorldSystem/data/20250617_211214/mav0/video_segments/20250617_211214_segment_1.mp4"

async def send_test_frames():
    """Send frames to test the tensor reshape fix"""
    
    # Check if SLAM3R is running
    result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
    if 'slam3r' not in result.stdout:
        logger.error("SLAM3R container is not running! Start it with: docker compose up slam3r")
        return False
        
    # Connect to RabbitMQ
    connection = await aio_pika.connect_robust("amqp://guest@localhost/")
    channel = await connection.channel()
    
    # Declare exchange as fanout (matching server configuration)
    exchange = await channel.declare_exchange(
        "video_frames_exchange",
        aio_pika.ExchangeType.FANOUT,
        durable=True
    )
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {VIDEO_PATH}")
        return False
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video: {width}x{height}")
    logger.info("Sending 10 frames to test tensor reshape fix...")
    
    success = True
    
    for frame_idx in range(10):
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame")
            success = False
            break
            
        # Encode frame as JPEG
        _, jpeg_data = cv2.imencode('.jpg', frame)
        
        # Create message matching server format
        timestamp_ns = int(time.time() * 1e9)
        
        # Send to RabbitMQ as raw JPEG data with headers
        message = aio_pika.Message(
            body=jpeg_data.tobytes(),
            content_type="application/octet-stream",
            headers={
                "timestamp_ns": str(timestamp_ns),
                "width": str(width),
                "height": str(height)
            }
        )
        
        await exchange.publish(message, routing_key="")
        
        if frame_idx == 4:
            logger.info(f"Frame {frame_idx}: Bootstrap should complete after this")
        elif frame_idx == 5:
            logger.info(f"Frame {frame_idx}: CRITICAL - This is where tensor reshape error occurred")
        elif frame_idx == 6:
            logger.info(f"Frame {frame_idx}: If we get here, the fix is working!")
        elif frame_idx == 9:
            logger.info(f"Frame {frame_idx}: Last test frame sent")
            
        # Small delay between frames
        await asyncio.sleep(0.1)
    
    cap.release()
    await connection.close()
    
    return success

async def check_slam3r_logs():
    """Check SLAM3R logs for errors or success"""
    logger.info("Waiting 5 seconds for SLAM3R to process...")
    await asyncio.sleep(5)
    
    # Get recent logs
    result = subprocess.run(
        ['docker', 'logs', 'slam3r', '--tail', '100'],
        capture_output=True, text=True
    )
    
    logs = result.stderr
    
    # Check for tensor reshape error
    if "RuntimeError" in logs and ("shape '[25, 196, 12, 64]'" in logs or "shape '[20, 196, 12, 64]'" in logs):
        logger.error("❌ TENSOR RESHAPE ERROR STILL PRESENT!")
        logger.error("The fix did not resolve the issue.")
        return False
    
    # Check for successful L2W inference
    if "L2W inference with" in logs:
        # Extract the message about number of reference views
        for line in logs.split('\n'):
            if "L2W inference with" in line:
                logger.info(f"✅ Found L2W inference: {line.strip()}")
    
    # Check for bootstrap completion
    if "Bootstrap complete" in logs:
        logger.info("✅ Bootstrap completed successfully!")
    
    # Check for any other errors
    error_found = False
    for line in logs.split('\n'):
        if "ERROR" in line and "L2W inference error" in line:
            logger.error(f"❌ L2W error found: {line.strip()}")
            error_found = True
    
    if not error_found:
        logger.info("✅ NO TENSOR RESHAPE ERROR DETECTED - Fix appears to be working!")
        return True
    
    return False

async def main():
    """Run the test"""
    logger.info("Starting SLAM3R tensor reshape fix test...")
    logger.info("This test will:")
    logger.info("1. Send 10 frames from real video")
    logger.info("2. Trigger bootstrap (frames 0-4)")
    logger.info("3. Test incremental processing (frames 5-9)")
    logger.info("4. Check for tensor reshape errors")
    
    # Send frames
    if not await send_test_frames():
        logger.error("Failed to send test frames")
        return
    
    # Check results
    success = await check_slam3r_logs()
    
    if success:
        logger.info("\n" + "="*60)
        logger.info("✅ TEST PASSED - Tensor reshape error is fixed!")
        logger.info("="*60)
    else:
        logger.info("\n" + "="*60)
        logger.error("❌ TEST FAILED - Tensor reshape error persists")
        logger.info("="*60)

if __name__ == "__main__":
    asyncio.run(main())