#!/usr/bin/env python3
"""
Test script to verify SLAM3R streaming keyframes to mesh_service works.
This tests the full pipeline including shared memory publishing.
"""

import sys
import os
import numpy as np
import cv2
import asyncio
import logging
import time
import posix_ipc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add SLAM3R engine to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_shared_memory(keyframe_id):
    """Check if shared memory segment was created."""
    shm_name = f"/slam3r_keyframe_{keyframe_id}"
    try:
        # Try to open the shared memory segment
        shm = posix_ipc.SharedMemory(shm_name, posix_ipc.O_RDONLY)
        logger.info(f"✓ Shared memory segment {shm_name} exists!")
        
        # Map and read header
        import mmap
        import struct
        
        # Map just the header first
        header_format = "QII" + "f" * 16 + "f" * 6  # timestamp, count, channels, pose, bbox
        header_size = struct.calcsize(header_format)
        
        mapfile = mmap.mmap(shm.fd, header_size, access=mmap.ACCESS_READ)
        header_data = struct.unpack(header_format, mapfile[:header_size])
        
        timestamp_ns = header_data[0]
        point_count = header_data[1]
        color_channels = header_data[2]
        
        logger.info(f"  Timestamp: {timestamp_ns}")
        logger.info(f"  Point count: {point_count}")
        logger.info(f"  Color channels: {color_channels}")
        
        mapfile.close()
        shm.close_fd()
        
        # Try to unlink (cleanup)
        try:
            shm.unlink()
            logger.info(f"  Cleaned up shared memory segment")
        except:
            pass
            
        return True
        
    except posix_ipc.ExistentialError:
        logger.error(f"✗ Shared memory segment {shm_name} does not exist!")
        return False
    except Exception as e:
        logger.error(f"✗ Error checking shared memory: {e}")
        return False

async def test_streaming_pipeline():
    """Test the full SLAM3R streaming pipeline."""
    
    # Import here to ensure environment is set up
    from slam3r_processor import SLAM3RProcessor
    
    # Create processor with correct config path
    processor = SLAM3RProcessor(config_path="slam3r/SLAM3R_engine/configs/wild.yaml")
    
    # Check if models are initialized
    if not processor.slam3r:
        logger.error("SLAM3R models not initialized!")
        return
    
    logger.info("=" * 80)
    logger.info("Testing SLAM3R Streaming Pipeline")
    logger.info("=" * 80)
    
    # Create a simple test video segment
    num_frames = 20
    keyframes_published = []
    
    for i in range(num_frames):
        # Create test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add moving pattern
        x = int(640 * 0.2 + i * 15)
        y = int(480 * 0.3)
        cv2.rectangle(img, (x, y), (x + 150, y + 150), (255, 128, 0), -1)
        cv2.circle(img, (x + 75, y + 75), 40, (0, 255, 0), -1)
        cv2.putText(img, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        logger.info(f"\nProcessing frame {i}")
        
        # Process frame
        result = processor.slam3r.process_frame(
            image=img_rgb,
            timestamp=i * 1000000000  # Nanoseconds
        )
        
        if result and result.get('is_keyframe', False):
            logger.info(f"  Keyframe detected!")
            
            # Add RGB image to result
            result['rgb_image'] = img_rgb
            
            # Increment keyframe counter
            processor.keyframe_count += 1
            
            # Try to publish
            try:
                await processor._publish_keyframe(result, i * 1000000000)
                keyframes_published.append(str(processor.keyframe_count))
                logger.info(f"  Published keyframe {processor.keyframe_count}")
            except Exception as e:
                logger.error(f"  Failed to publish keyframe: {e}")
        
        # Small delay to simulate real processing
        await asyncio.sleep(0.05)
    
    logger.info("\n" + "=" * 80)
    logger.info("Checking Shared Memory Segments")
    logger.info("=" * 80)
    
    # Check if shared memory segments were created
    success_count = 0
    for kf_id in keyframes_published:
        if check_shared_memory(kf_id):
            success_count += 1
    
    logger.info("\n" + "=" * 80)
    logger.info(f"Test Results: {success_count}/{len(keyframes_published)} keyframes successfully streamed")
    logger.info("=" * 80)
    
    if success_count == len(keyframes_published):
        logger.info("✓ SUCCESS: All keyframes were successfully published to shared memory!")
    elif success_count > 0:
        logger.warning(f"⚠ PARTIAL SUCCESS: {success_count} keyframes published, {len(keyframes_published) - success_count} failed")
    else:
        logger.error("✗ FAIL: No keyframes were successfully published!")

if __name__ == "__main__":
    # Set environment
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["SLAM3R_ENABLE_KEYFRAME_STREAMING"] = "true"
    
    # Run test
    asyncio.run(test_streaming_pipeline())