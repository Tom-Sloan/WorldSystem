"""
FastSAM detector implementation.

FastSAM provides fast class-agnostic segmentation using a YOLO-based architecture,
offering 50x speedup over SAM with comparable quality.
"""

import torch
import numpy as np
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import asyncio
import cv2

from ultralytics import YOLO
from ultralytics.engine.results import Results

from .base import Detector, Detection
from core.utils import get_logger, PerformanceTimer


logger = get_logger(__name__)


class FastSAMDetector(Detector):
    """
    FastSAM detector for high-speed class-agnostic object detection.
    
    FastSAM achieves similar results to SAM but runs much faster,
    making it ideal for real-time applications.
    """
    
    def __init__(self,
                 model_path: str = "FastSAM-x.pt",
                 device: str = "cuda",
                 conf_threshold: float = 0.4,
                 iou_threshold: float = 0.9,
                 retina_masks: bool = True,
                 max_det: int = 300,
                 **kwargs):
        """
        Initialize FastSAM detector.
        
        Args:
            model_path: Path to FastSAM model weights
            device: Device to run on (cuda/cpu)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            retina_masks: Use high-resolution masks
            max_det: Maximum detections per image
        """
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Check model file
        if not Path(model_path).exists():
            # Try to download
            logger.info(f"FastSAM model not found at {model_path}, attempting download...")
            try:
                # FastSAM uses YOLO format, so it can auto-download
                self.model = YOLO(model_path)
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to load/download FastSAM model: {e}\n"
                    f"Download manually from: https://github.com/CASIA-IVA-Lab/FastSAM"
                )
        else:
            logger.info(f"Loading FastSAM model from {model_path}")
            self.model = YOLO(model_path)
        
        # Move model to device
        self.model.to(device)
        
        # Set model parameters
        self.model.overrides['conf'] = conf_threshold
        self.model.overrides['iou'] = iou_threshold
        self.model.overrides['retina_masks'] = retina_masks
        self.model.overrides['max_det'] = max_det
        
        logger.info(
            f"FastSAM initialized - "
            f"conf: {conf_threshold}, iou: {iou_threshold}, "
            f"max_det: {max_det}"
        )
    
    async def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects using FastSAM.
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            List of Detection objects
        """
        with PerformanceTimer("fastsam_inference", logger):
            try:
                # Run inference
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    lambda: self.model(
                        frame,
                        task='segment',
                        boxes=False,  # We want masks, not boxes
                        verbose=False
                    )
                )
                
                detections = []
                
                # Process results
                for result in results:
                    if result.masks is None:
                        continue
                    
                    masks = result.masks.data.cpu().numpy()
                    
                    # Process each mask
                    for idx, mask in enumerate(masks):
                        # Convert mask to binary
                        binary_mask = (mask > 0.5).astype(np.uint8)
                        
                        # Get bounding box from mask
                        contours, _ = cv2.findContours(
                            binary_mask, 
                            cv2.RETR_EXTERNAL, 
                            cv2.CHAIN_APPROX_SIMPLE
                        )
                        
                        if not contours:
                            continue
                        
                        # Get largest contour
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        
                        # Skip tiny objects
                        if w * h < 100:  # Min area threshold
                            continue
                        
                        # Calculate confidence from mask area ratio
                        mask_area = np.sum(binary_mask)
                        bbox_area = w * h
                        area_ratio = mask_area / bbox_area if bbox_area > 0 else 0
                        
                        # Create detection
                        detection = Detection(
                            bbox=(x, y, x + w, y + h),
                            confidence=min(0.99, area_ratio),  # Use area ratio as confidence
                            class_id=0,
                            class_name="object",
                            mask=binary_mask
                        )
                        
                        # Add metadata
                        setattr(detection, 'fastsam_metadata', {
                            'mask_area': int(mask_area),
                            'bbox_area': bbox_area,
                            'area_ratio': area_ratio
                        })
                        
                        detections.append(detection)
                
                # Apply NMS to remove duplicates
                detections = self._apply_nms(detections, self.iou_threshold)
                
                logger.debug(f"FastSAM detected {len(detections)} objects")
                return detections
                
            except Exception as e:
                logger.error(f"FastSAM inference failed: {e}", exc_info=True)
                return []
    
    async def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Batch detection using FastSAM.
        
        FastSAM supports batch inference natively.
        """
        with PerformanceTimer(f"fastsam_batch_{len(frames)}", logger):
            try:
                # Run batch inference
                results = self.model(
                    frames,
                    task='segment',
                    boxes=False,
                    verbose=False,
                    stream=False
                )
                
                batch_detections = []
                
                # Process each frame's results
                for frame_idx, result in enumerate(results):
                    frame_detections = []
                    
                    if result.masks is not None:
                        masks = result.masks.data.cpu().numpy()
                        
                        for mask in masks:
                            binary_mask = (mask > 0.5).astype(np.uint8)
                            
                            # Get bbox from mask
                            contours, _ = cv2.findContours(
                                binary_mask,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE
                            )
                            
                            if contours:
                                largest = max(contours, key=cv2.contourArea)
                                x, y, w, h = cv2.boundingRect(largest)
                                
                                if w * h >= 100:
                                    frame_detections.append(Detection(
                                        bbox=(x, y, x + w, y + h),
                                        confidence=0.9,
                                        class_id=0,
                                        class_name="object",
                                        mask=binary_mask
                                    ))
                    
                    batch_detections.append(frame_detections)
                
                return batch_detections
                
            except Exception as e:
                logger.error(f"FastSAM batch inference failed: {e}")
                return [[] for _ in frames]
    
    def _apply_nms(self, detections: List[Detection], 
                   iou_threshold: float) -> List[Detection]:
        """Apply Non-Maximum Suppression to detections."""
        if not detections:
            return detections
        
        # Convert to tensors for NMS
        boxes = torch.tensor([d.bbox for d in detections], dtype=torch.float32)
        scores = torch.tensor([d.confidence for d in detections], dtype=torch.float32)
        
        # Convert to x1,y1,x2,y2 format if needed
        keep_indices = torch.ops.torchvision.nms(boxes, scores, iou_threshold)
        
        # Filter detections
        return [detections[i] for i in keep_indices.tolist()]
    
    @property
    def name(self) -> str:
        """Detector name for logging/metrics."""
        return "FastSAM"
    
    @property
    def supported_classes(self) -> List[str]:
        """FastSAM is class-agnostic."""
        return ["object"]
    
    def warmup(self) -> None:
        """Warmup the model."""
        try:
            logger.info("Warming up FastSAM model...")
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy, task='segment', verbose=False)
            logger.info("FastSAM warmup complete")
        except Exception as e:
            logger.warning(f"FastSAM warmup failed: {e}")