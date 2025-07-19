"""
Segment Anything Model 2 (SAM2) detector implementation.

This provides class-agnostic object detection using Meta's SAM2,
which can detect any object without being limited to specific classes.
SAM2 offers improved performance and quality over SAM1.
"""

import torch
import numpy as np
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import asyncio
import cv2

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2, build_sam2_video_predictor

from .base import Detector, Detection
from core.utils import get_logger, PerformanceTimer


logger = get_logger(__name__)


def mask_to_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Convert binary mask to bounding box.
    
    Args:
        mask: Binary mask array
        
    Returns:
        Tuple of (x1, y1, x2, y2)
    """
    # Find mask coordinates
    coords = np.column_stack(np.where(mask))
    if len(coords) == 0:
        return (0, 0, 0, 0)
    
    # Get bounding box - note: coords are (y, x) format
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return (int(x_min), int(y_min), int(x_max), int(y_max))


class SAMDetector(Detector):
    """
    SAM2 detector for class-agnostic object detection.
    
    This detector can find any object in an image without being limited
    to predefined classes, making it perfect for discovering objects
    that YOLO would miss (outlets, thermostats, paper, etc.).
    SAM2 offers improved performance over SAM1.
    """
    
    def __init__(self, 
                 model_cfg: str = "sam2_hiera_l.yaml",
                 model_path: str = "sam2_hiera_large.pt",
                 device: str = "cuda",
                 points_per_side: int = 24,
                 pred_iou_thresh: float = 0.86,
                 stability_score_thresh: float = 0.92,
                 min_mask_region_area: int = 500,
                 **kwargs):
        """
        Initialize SAM2 detector.
        
        Args:
            model_cfg: SAM2 model configuration file
            model_path: Path to SAM2 checkpoint file
            device: Device to run on (cuda/cpu)
            points_per_side: Number of points sampled per side for mask generation
            pred_iou_thresh: Threshold for predicted IoU quality
            stability_score_thresh: Threshold for mask stability
            min_mask_region_area: Minimum area for valid masks
        """
        self.model_cfg = model_cfg
        self.model_path = model_path
        self.device = device
        
        # Check model file exists
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"SAM2 model not found at {model_path}. "
                f"Download from: https://github.com/facebookresearch/sam2#model-checkpoints"
            )
        
        logger.info(f"Loading SAM2 model {model_cfg} from {model_path}")
        
        try:
            # Build SAM2 model
            self.sam2 = build_sam2(model_cfg, model_path, device=device)
            
            # Create automatic mask generator with tuned parameters
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=self.sam2,
                points_per_side=points_per_side,
                points_per_batch=64,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                stability_score_offset=1.0,
                box_nms_thresh=0.7,
                crop_n_layers=1,
                crop_nms_thresh=0.7,
                crop_overlap_ratio=0.34,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=min_mask_region_area,
                output_mode="binary_mask"
            )
            
            logger.info(
                f"SAM2 initialized successfully - "
                f"config: {model_cfg}, "
                f"points_per_side: {points_per_side}, "
                f"iou_thresh: {pred_iou_thresh}, "
                f"min_area: {min_mask_region_area}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize SAM2: {e}")
            raise
        
        # Store configuration
        self.config = {
            'model_cfg': model_cfg,
            'points_per_side': points_per_side,
            'pred_iou_thresh': pred_iou_thresh,
            'stability_score_thresh': stability_score_thresh,
            'min_mask_region_area': min_mask_region_area
        }
    
    async def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect all objects in frame using SAM2.
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            List of Detection objects
        """
        with PerformanceTimer("sam2_inference", logger):
            try:
                # Convert BGR to RGB for SAM2
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run SAM2 inference in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                masks = await loop.run_in_executor(
                    None,
                    self.mask_generator.generate,
                    image_rgb
                )
                
                # Convert masks to detections
                detections = []
                for idx, mask_data in enumerate(masks):
                    # Extract mask and convert to bbox
                    segmentation = mask_data['segmentation']
                    bbox = mask_to_bbox(segmentation)
                    
                    # Skip invalid bboxes
                    x1, y1, x2, y2 = bbox
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Create detection
                    detection = Detection(
                        bbox=bbox,
                        confidence=float(mask_data['predicted_iou']),
                        class_id=0,  # All objects are class 0
                        class_name="object",  # Generic object class
                        mask=segmentation  # Include mask for downstream use
                    )
                    
                    # Add additional SAM2 metadata
                    setattr(detection, 'sam2_metadata', {
                        'stability_score': mask_data['stability_score'],
                        'area': int(mask_data['area']),
                        'bbox_area': (x2 - x1) * (y2 - y1),
                        'point_coords': mask_data.get('point_coords', [])
                    })
                    
                    detections.append(detection)
                
                # Sort by confidence (predicted IoU)
                detections.sort(key=lambda d: d.confidence, reverse=True)
                
                logger.debug(f"SAM2 detected {len(detections)} objects")
                return detections
                
            except Exception as e:
                logger.error(f"SAM2 inference failed: {e}", exc_info=True)
                return []
    
    async def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Batch detection for multiple frames.
        
        Note: SAM doesn't have native batch support, so we process sequentially.
        """
        results = []
        for frame in frames:
            detections = await self.detect(frame)
            results.append(detections)
        return results
    
    @property
    def name(self) -> str:
        """Detector name for logging/metrics."""
        return f"SAM2-{self.model_cfg.replace('.yaml', '')}"
    
    @property
    def supported_classes(self) -> List[str]:
        """SAM is class-agnostic, returns generic 'object'."""
        return ["object"]
    
    def warmup(self) -> None:
        """Warmup the model with dummy inference."""
        try:
            logger.info("Warming up SAM2 model...")
            # Create dummy image
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            dummy_image_rgb = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2RGB)
            
            # Run inference
            _ = self.mask_generator.generate(dummy_image_rgb)
            
            logger.info("SAM2 warmup complete")
        except Exception as e:
            logger.warning(f"SAM2 warmup failed: {e}")
    
    def update_parameters(self, **kwargs):
        """
        Update SAM parameters at runtime.
        
        Args:
            points_per_side: Number of sampling points
            pred_iou_thresh: IoU threshold
            stability_score_thresh: Stability threshold
            min_mask_region_area: Minimum mask area
        """
        # Update generator parameters
        for key, value in kwargs.items():
            if hasattr(self.mask_generator, key):
                setattr(self.mask_generator, key, value)
                self.config[key] = value
                logger.info(f"Updated SAM parameter {key} to {value}")