#!/usr/bin/env python3
"""
Test script to verify SLAM3R resolution fix.
Creates a test image and processes it through the pipeline.
"""

import numpy as np
import torch
import logging
from streaming_slam3r import StreamingSLAM3R
from slam3r.models import Image2PointsModel, Local2WorldModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_image(size=(640, 480)):
    """Create a test image with known pattern."""
    img = np.zeros((*size, 3), dtype=np.uint8)
    # Add some pattern
    img[::20, ::20] = [255, 0, 0]  # Red dots
    img[10::20, 10::20] = [0, 255, 0]  # Green dots
    return img

def test_resolution_handling():
    """Test that the pipeline correctly handles different input resolutions."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create dummy models (would need real models for actual test)
    logger.info("Note: This test requires actual SLAM3R models to run properly")
    logger.info("Testing resolution handling logic...")
    
    # Test config
    config = {
        'recon_pipeline': {
            'conf_thres_i2p': 1.5,
            'conf_thres_l2w': 12.0,
            'initial_winsize': 5,
        },
        'batch_size': 1,
        'window_size': 10,
        'initial_keyframe_stride': 5,
    }
    
    # Test different input sizes
    test_sizes = [
        (640, 480),   # Standard VGA
        (1920, 1080), # Full HD
        (224, 224),   # Target size
        (256, 256),   # Multiple of 16
        (300, 400),   # Arbitrary size
    ]
    
    for size in test_sizes:
        logger.info(f"\nTesting input size: {size}")
        img = create_test_image(size)
        
        # The key test: verify image gets resized to 224x224
        from streaming_slam3r import StreamingSLAM3R
        slam = StreamingSLAM3R.__new__(StreamingSLAM3R)
        slam.device = device
        
        # Test _create_frame method
        frame = slam._create_frame(img, timestamp=1000)
        
        # Verify output
        assert frame.image_tensor.shape == (1, 3, 224, 224), \
            f"Expected shape (1, 3, 224, 224), got {frame.image_tensor.shape}"
        assert frame.true_shape.tolist() == [224, 224], \
            f"Expected true_shape [224, 224], got {frame.true_shape.tolist()}"
        
        logger.info(f"✓ Input {size} correctly resized to 224x224")
        logger.info(f"  Tensor shape: {frame.image_tensor.shape}")
        logger.info(f"  True shape: {frame.true_shape.tolist()}")

if __name__ == "__main__":
    test_resolution_handling()
    logger.info("\nResolution handling test completed successfully!")
    logger.info("All input sizes are correctly resized to 224x224 as required by SLAM3R.")