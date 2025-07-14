#!/usr/bin/env python3
"""
Test SLAM3R with adjusted confidence thresholds based on observed values.
"""

import sys
import os
import numpy as np
import cv2
import torch
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add SLAM3R engine to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slam3r_processor import SLAM3RProcessor
from streaming_slam3r import StreamingSLAM3R

def analyze_keyframe_with_thresholds(keyframe_data, test_thresholds=[12.0, 6.0, 3.0, 1.5, 1.0]):
    """Analyze keyframe data with different confidence thresholds."""
    pts3d = keyframe_data.get('pts3d_world')
    conf = keyframe_data.get('conf_world')
    
    if pts3d is None or conf is None:
        logger.error("Missing pts3d_world or conf_world!")
        return
    
    # Convert to numpy
    if torch.is_tensor(pts3d):
        pts3d_np = pts3d.cpu().numpy()
        conf_np = conf.cpu().numpy()
    else:
        pts3d_np = pts3d
        conf_np = conf
    
    # Handle dimensions
    if pts3d_np.ndim == 4 and pts3d_np.shape[0] == 1:
        pts3d_np = pts3d_np.squeeze(0)
    if conf_np.ndim == 3 and conf_np.shape[0] == 1:
        conf_np = conf_np.squeeze(0)
    
    pts_flat = pts3d_np.reshape(-1, 3)
    conf_flat = conf_np.squeeze().reshape(-1)
    
    logger.info(f"Confidence statistics:")
    logger.info(f"  Min: {conf_flat.min():.3f}")
    logger.info(f"  Max: {conf_flat.max():.3f}")
    logger.info(f"  Mean: {conf_flat.mean():.3f}")
    logger.info(f"  Std: {conf_flat.std():.3f}")
    logger.info(f"  Percentiles: 25%={np.percentile(conf_flat, 25):.3f}, "
                f"50%={np.percentile(conf_flat, 50):.3f}, "
                f"75%={np.percentile(conf_flat, 75):.3f}, "
                f"95%={np.percentile(conf_flat, 95):.3f}")
    
    logger.info(f"\nPoints passing different thresholds:")
    for thresh in test_thresholds:
        mask = conf_flat > thresh
        count = mask.sum()
        percentage = 100.0 * count / len(conf_flat)
        logger.info(f"  Threshold {thresh:4.1f}: {count:6d} points ({percentage:5.1f}%)")
        
        # With fallback
        if count < 3:
            fallback_thresh = 0.5 * thresh
            mask_fallback = conf_flat > fallback_thresh
            count_fallback = mask_fallback.sum()
            percentage_fallback = 100.0 * count_fallback / len(conf_flat)
            logger.info(f"    Fallback {fallback_thresh:4.1f}: {count_fallback:6d} points ({percentage_fallback:5.1f}%)")

def test_adjusted_thresholds():
    """Test with adjusted thresholds."""
    # Create processor
    processor = SLAM3RProcessor(config_path="slam3r/SLAM3R_engine/configs/wild.yaml")
    
    # Override confidence thresholds based on observed values
    # Original: conf_thres_l2w=12.0, but observed values are 1-5
    processor.config['recon_pipeline']['conf_thres_l2w'] = 1.0  # Much lower threshold
    processor.config['recon_pipeline']['conf_thres_i2p'] = 1.5  # Keep this
    
    # Reinitialize StreamingSLAM3R with new config
    from slam3r.models import Image2PointsModel, Local2WorldModel
    i2p_model = Image2PointsModel.from_pretrained("siyan824/slam3r_i2p").to(processor.device).eval()
    l2w_model = Local2WorldModel.from_pretrained("siyan824/slam3r_l2w").to(processor.device).eval()
    
    processor.slam3r = StreamingSLAM3R(
        i2p_model=i2p_model,
        l2w_model=l2w_model,
        config=processor.config,
        device=str(processor.device)
    )
    
    logger.info("=" * 80)
    logger.info("Testing with adjusted confidence thresholds")
    logger.info(f"conf_thres_l2w: {processor.config['recon_pipeline']['conf_thres_l2w']}")
    logger.info(f"conf_thres_i2p: {processor.config['recon_pipeline']['conf_thres_i2p']}")
    logger.info("=" * 80)
    
    # Create test images
    num_frames = 15
    keyframe_count = 0
    
    for i in range(num_frames):
        # Create test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        x = int(640 * 0.2 + i * 20)
        y = int(480 * 0.3)
        cv2.rectangle(img, (x, y), (x + 150, y + 150), (255, 128, 0), -1)
        cv2.circle(img, (x + 75, y + 75), 40, (0, 255, 0), -1)
        cv2.putText(img, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process frame
        result = processor.slam3r.process_frame(
            image=img_rgb,
            timestamp=i * 1000000000
        )
        
        if result and result.get('is_keyframe', False):
            keyframe_count += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"KEYFRAME {keyframe_count} at frame {i}")
            logger.info(f"{'='*60}")
            
            # Analyze with different thresholds
            analyze_keyframe_with_thresholds(result)
            
            # Try to publish with adjusted threshold
            result['rgb_image'] = img_rgb
            processor.keyframe_count += 1
            
            # Temporarily override threshold for publishing
            original_thresh = processor.config.get('recon_pipeline', {}).get('conf_thres_l2w', 12.0)
            processor.config['recon_pipeline']['conf_thres_l2w'] = 1.0
            
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(processor._publish_keyframe(result, i * 1000000000))
                logger.info("✓ Successfully published keyframe!")
            except Exception as e:
                logger.error(f"Failed to publish: {e}")
            finally:
                processor.config['recon_pipeline']['conf_thres_l2w'] = original_thresh
    
    logger.info("\n" + "=" * 80)
    logger.info(f"Summary: {keyframe_count} keyframes detected")
    logger.info("=" * 80)

if __name__ == "__main__":
    # Set environment
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["SLAM3R_ENABLE_KEYFRAME_STREAMING"] = "true"
    
    # Run test
    test_adjusted_thresholds()