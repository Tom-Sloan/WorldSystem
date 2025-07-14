#!/usr/bin/env python3
"""
Test script to verify SLAM3R pipeline is producing valid keyframes.
Run this to check if the fixes are working properly.
"""

import sys
import os
import numpy as np
import cv2
import asyncio
import logging
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add SLAM3R engine to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the processor
from slam3r_processor import SLAM3RProcessor

def create_test_images(num_images=10, size=(640, 480)):
    """Create synthetic test images with some motion."""
    images = []
    for i in range(num_images):
        # Create a simple pattern that moves
        img = np.zeros((*size, 3), dtype=np.uint8)
        
        # Add moving rectangle
        x = int(size[1] * 0.3 + i * 20)
        y = int(size[0] * 0.3)
        cv2.rectangle(img, (x, y), (x + 100, y + 100), (255, 128, 0), -1)
        
        # Add some texture
        cv2.circle(img, (x + 50, y + 50), 30, (0, 255, 0), -1)
        
        # Add frame number
        cv2.putText(img, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        images.append(img)
    
    return images

async def test_slam3r_pipeline():
    """Test the SLAM3R pipeline with synthetic images."""
    
    # Create processor
    processor = SLAM3RProcessor()
    
    # Initialize models
    await processor._initialize_models()
    
    # Create test images
    test_images = create_test_images(15)  # Need at least 5 for initialization
    
    logger.info("=" * 80)
    logger.info("Testing SLAM3R Pipeline")
    logger.info("=" * 80)
    
    # Process each image
    for i, img_bgr in enumerate(test_images):
        logger.info(f"\nProcessing frame {i}")
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Process frame
        result = processor.slam3r.process_frame(
            image=img_rgb,
            timestamp=i * 1000000000  # Nanoseconds
        )
        
        if result and result.get('is_keyframe', False):
            logger.info(f"KEYFRAME detected at frame {i}")
            
            # Check the keyframe data
            pts3d = result.get('pts3d_world')
            conf = result.get('conf_world')
            
            if pts3d is not None and conf is not None:
                if torch.is_tensor(pts3d):
                    pts3d_np = pts3d.cpu().numpy()
                    conf_np = conf.cpu().numpy()
                else:
                    pts3d_np = pts3d
                    conf_np = conf
                
                logger.info(f"  pts3d shape: {pts3d_np.shape}")
                logger.info(f"  conf shape: {conf_np.shape}")
                logger.info(f"  conf range: [{conf_np.min():.3f}, {conf_np.max():.3f}]")
                
                # Try to extract points like the publisher would
                if pts3d_np.ndim == 4 and pts3d_np.shape[0] == 1:
                    pts3d_np = pts3d_np.squeeze(0)
                if conf_np.ndim == 3 and conf_np.shape[0] == 1:
                    conf_np = conf_np.squeeze(0)
                
                pts_flat = pts3d_np.reshape(-1, 3)
                conf_flat = conf_np.squeeze().reshape(-1)
                
                # Check confidence filtering
                conf_thresh = 12.0  # L2W threshold
                mask = conf_flat > conf_thresh
                if mask.sum() < 3:
                    conf_thresh = 0.5 * conf_thresh
                    mask = conf_flat > conf_thresh
                
                valid_points = pts_flat[mask]
                logger.info(f"  Valid points after filtering: {len(valid_points)}")
                
                if len(valid_points) > 0:
                    logger.info(f"  ✓ SUCCESS: Keyframe has {len(valid_points)} valid 3D points!")
                    logger.info(f"  Point cloud range: X[{valid_points[:,0].min():.3f}, {valid_points[:,0].max():.3f}], "
                               f"Y[{valid_points[:,1].min():.3f}, {valid_points[:,1].max():.3f}], "
                               f"Z[{valid_points[:,2].min():.3f}, {valid_points[:,2].max():.3f}]")
                else:
                    logger.warning(f"  ✗ FAIL: Keyframe has NO valid points after filtering!")
            else:
                logger.error(f"  ✗ FAIL: Keyframe missing pts3d_world or conf_world!")
        
    logger.info("\n" + "=" * 80)
    logger.info("Test completed")
    logger.info("=" * 80)

if __name__ == "__main__":
    # Set environment to avoid CUDA issues
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Run test
    asyncio.run(test_slam3r_pipeline())