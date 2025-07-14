#!/usr/bin/env python3
"""
Test script for SLAM3R pipeline to debug empty keyframes issue.

This script:
1. Creates synthetic test images
2. Processes them through StreamingSLAM3R
3. Logs detailed information about the outputs
"""

import numpy as np
import torch
import logging
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from streaming_slam3r import StreamingSLAM3R
from slam3r.models import Image2PointsModel, Local2WorldModel

# Configure logging to show all debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('slam3r_test.log')
    ]
)
logger = logging.getLogger(__name__)


def create_test_image(frame_idx: int, size=(224, 224)) -> np.ndarray:
    """Create a synthetic test image with some features."""
    # Create a gradient image with some patterns
    img = np.zeros((*size, 3), dtype=np.uint8)
    
    # Add gradient
    for i in range(size[0]):
        for j in range(size[1]):
            img[i, j] = [
                int(255 * i / size[0]),
                int(255 * j / size[1]),
                int(255 * (frame_idx % 10) / 10)
            ]
    
    # Add some squares for features
    square_size = 20
    for i in range(0, size[0], 50):
        for j in range(0, size[1], 50):
            img[i:i+square_size, j:j+square_size] = [255, 255, 255]
    
    return img


def test_slam3r_pipeline():
    """Test the SLAM3R pipeline with synthetic data."""
    logger.info("Starting SLAM3R pipeline test")
    
    # Load configuration
    config = {
        'recon_pipeline': {
            'initial_winsize': 5,
            'conf_thres_i2p': 1.5,
            'conf_thres_l2w': 12,
            'num_scene_frame': 5,
            'norm_input_l2w': True
        },
        'window_size': 20,
        'initial_keyframe_stride': 5,
        'batch_size': 5,
        'use_adaptive_stride': False  # Disable for testing
    }
    
    try:
        # Initialize models
        logger.info("Loading models...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models from HuggingFace
        i2p_model = Image2PointsModel.from_pretrained("siyan824/slam3r_i2p").to(device).eval()
        l2w_model = Local2WorldModel.from_pretrained("siyan824/slam3r_l2w").to(device).eval()
        
        logger.info(f"Models loaded successfully on {device}")
        
        # Create StreamingSLAM3R instance
        slam3r = StreamingSLAM3R(
            i2p_model=i2p_model,
            l2w_model=l2w_model,
            config=config,
            device=str(device)
        )
        
        # Process test frames
        num_frames = 20
        keyframe_results = []
        
        for i in range(num_frames):
            # Create test image
            img = create_test_image(i)
            timestamp = i * 1000000000  # nanoseconds
            
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing frame {i}")
            logger.info(f"{'='*50}")
            
            # Process frame
            result = slam3r.process_frame(img, timestamp)
            
            if result:
                logger.info(f"✓ Got keyframe result for frame {i}:")
                logger.info(f"  - Frame ID: {result['frame_id']}")
                logger.info(f"  - Timestamp: {result['timestamp']}")
                logger.info(f"  - Is keyframe: {result['is_keyframe']}")
                
                pts3d = result['pts3d_world']
                conf = result['conf_world']
                
                logger.info(f"  - Points shape: {pts3d.shape}")
                logger.info(f"  - Confidence shape: {conf.shape}")
                logger.info(f"  - Confidence range: [{conf.min().item():.3f}, {conf.max().item():.3f}]")
                
                # Check if points would pass filtering
                conf_thresh = config['recon_pipeline']['conf_thres_i2p']
                num_valid = (conf > conf_thresh).sum().item()
                logger.info(f"  - Points passing threshold {conf_thresh}: {num_valid}")
                
                keyframe_results.append(result)
            else:
                logger.info(f"✗ No keyframe result for frame {i}")
        
        # Summary
        logger.info(f"\n{'='*50}")
        logger.info("Test Summary:")
        logger.info(f"{'='*50}")
        logger.info(f"Total frames processed: {num_frames}")
        logger.info(f"Keyframes produced: {len(keyframe_results)}")
        
        if len(keyframe_results) == 0:
            logger.error("❌ No keyframes were produced! Check the logs above for errors.")
        else:
            logger.info("✅ Pipeline produced keyframes successfully!")
            
            # Analyze keyframe quality
            total_points = 0
            for kf in keyframe_results:
                pts3d = kf['pts3d_world']
                conf = kf['conf_world']
                conf_thresh = config['recon_pipeline']['conf_thres_i2p']
                num_valid = (conf > conf_thresh).sum().item()
                total_points += num_valid
            
            logger.info(f"Average points per keyframe: {total_points / len(keyframe_results):.1f}")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return False
    
    return len(keyframe_results) > 0


if __name__ == "__main__":
    success = test_slam3r_pipeline()
    sys.exit(0 if success else 1)