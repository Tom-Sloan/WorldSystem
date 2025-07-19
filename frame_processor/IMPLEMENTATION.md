# Detailed Implementation Plan: SAM-L and FastSAM Integration

## Overview

This plan details the complete implementation of both SAM-L and FastSAM detectors into your existing frame processor architecture, allowing runtime selection between them.

## GitHub Repositories & Documentation

### Segment Anything Model (SAM)
- **GitHub**: https://github.com/facebookresearch/segment-anything
- **Paper**: https://arxiv.org/abs/2304.02643
- **Model Weights**: https://github.com/facebookresearch/segment-anything#model-checkpoints
- **Documentation**: https://segment-anything.com/

### FastSAM
- **GitHub**: https://github.com/CASIA-IVA-Lab/FastSAM
- **Paper**: https://arxiv.org/abs/2306.12156
- **Documentation**: https://github.com/CASIA-IVA-Lab/FastSAM/blob/main/README.md

## Files to Modify/Create

```
frame_processor/
├── requirements.txt                    # UPDATE: Add new dependencies
├── Dockerfile                         # UPDATE: Add model downloads
├── core/
│   └── config.py                      # UPDATE: Add SAM/FastSAM config options
├── detection/
│   ├── __init__.py                    # UPDATE: Export new detectors
│   ├── sam.py                         # CREATE: SAM detector implementation
│   └── fastsam.py                     # CREATE: FastSAM detector implementation
├── pipeline/
│   └── processor.py                   # UPDATE: Add to detector registry
├── models/                            # CREATE: Directory for model weights
│   ├── sam_vit_l_0b3195.pth         # DOWNLOAD: SAM-L weights
│   └── FastSAM-x.pt                  # DOWNLOAD: FastSAM weights
└── docker-compose.yml                 # UPDATE: Mount model directories
```

## 1. Update requirements.txt

```python
# frame_processor/requirements.txt
# Add these dependencies:

# Segment Anything Model
segment-anything @ git+https://github.com/facebookresearch/segment-anything.git

# FastSAM dependencies
ultralytics>=8.0.230
fastsam @ git+https://github.com/CASIA-IVA-Lab/FastSAM.git

# Additional dependencies for SAM/FastSAM
matplotlib>=3.7.1
pycocotools>=2.0.7
onnx>=1.14.1
onnxruntime-gpu>=1.16.0
```

## 2. Create SAM Detector Implementation

```python
# frame_processor/detection/sam.py
"""
Segment Anything Model (SAM) detector implementation.

This provides class-agnostic object detection using Meta's SAM,
which can detect any object without being limited to specific classes.
"""

import torch
import numpy as np
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import asyncio
import cv2

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

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
    SAM detector for class-agnostic object detection.
    
    This detector can find any object in an image without being limited
    to predefined classes, making it perfect for discovering objects
    that YOLO would miss (outlets, thermostats, paper, etc.).
    """
    
    def __init__(self, 
                 model_path: str = "sam_vit_l_0b3195.pth",
                 model_type: str = "vit_l",
                 device: str = "cuda",
                 points_per_side: int = 24,
                 pred_iou_thresh: float = 0.86,
                 stability_score_thresh: float = 0.92,
                 min_mask_region_area: int = 500,
                 **kwargs):
        """
        Initialize SAM detector.
        
        Args:
            model_path: Path to SAM checkpoint file
            model_type: Model architecture (vit_h, vit_l, vit_b)
            device: Device to run on (cuda/cpu)
            points_per_side: Number of points sampled per side for mask generation
            pred_iou_thresh: Threshold for predicted IoU quality
            stability_score_thresh: Threshold for mask stability
            min_mask_region_area: Minimum area for valid masks
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        
        # Check model file exists
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"SAM model not found at {model_path}. "
                f"Download from: https://dl.fbaipublicfiles.com/segment_anything/{Path(model_path).name}"
            )
        
        logger.info(f"Loading SAM model {model_type} from {model_path}")
        
        try:
            # Load SAM model
            self.sam = sam_model_registry[model_type](checkpoint=model_path)
            self.sam.to(device=device)
            self.sam.eval()  # Set to evaluation mode
            
            # Create automatic mask generator with tuned parameters
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
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
                f"SAM initialized successfully - "
                f"points_per_side: {points_per_side}, "
                f"iou_thresh: {pred_iou_thresh}, "
                f"min_area: {min_mask_region_area}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize SAM: {e}")
            raise
        
        # Store configuration
        self.config = {
            'points_per_side': points_per_side,
            'pred_iou_thresh': pred_iou_thresh,
            'stability_score_thresh': stability_score_thresh,
            'min_mask_region_area': min_mask_region_area
        }
    
    async def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect all objects in frame using SAM.
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            List of Detection objects
        """
        with PerformanceTimer("sam_inference", logger):
            try:
                # Convert BGR to RGB for SAM
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run SAM inference in thread pool to avoid blocking
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
                    
                    # Add additional SAM metadata
                    detection.sam_metadata = {
                        'stability_score': mask_data['stability_score'],
                        'area': int(mask_data['area']),
                        'bbox_area': (x2 - x1) * (y2 - y1),
                        'point_coords': mask_data.get('point_coords', [])
                    }
                    
                    detections.append(detection)
                
                # Sort by confidence (predicted IoU)
                detections.sort(key=lambda d: d.confidence, reverse=True)
                
                logger.debug(f"SAM detected {len(detections)} objects")
                return detections
                
            except Exception as e:
                logger.error(f"SAM inference failed: {e}", exc_info=True)
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
        return f"SAM-{self.model_type}"
    
    @property
    def supported_classes(self) -> List[str]:
        """SAM is class-agnostic, returns generic 'object'."""
        return ["object"]
    
    def warmup(self) -> None:
        """Warmup the model with dummy inference."""
        try:
            logger.info("Warming up SAM model...")
            # Create dummy image
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            dummy_image_rgb = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2RGB)
            
            # Run inference
            _ = self.mask_generator.generate(dummy_image_rgb)
            
            logger.info("SAM warmup complete")
        except Exception as e:
            logger.warning(f"SAM warmup failed: {e}")
    
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
```

## 3. Create FastSAM Detector Implementation

```python
# frame_processor/detection/fastsam.py
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
                        detection.fastsam_metadata = {
                            'mask_area': int(mask_area),
                            'bbox_area': bbox_area,
                            'area_ratio': area_ratio
                        }
                        
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
```

## 4. Update Configuration

```python
# frame_processor/core/config.py
# Add these to the Config class:

class Config(BaseSettings):
    # ... existing config ...
    
    # ========== Component Selection ==========
    detector_type: Literal["yolo", "sam", "fastsam", "detectron2", "grounding_dino"] = Field(
        default="yolo",
        description="Detection algorithm to use"
    )
    
    # ========== SAM Configuration ==========
    sam_model_type: Literal["vit_h", "vit_l", "vit_b"] = Field(
        default="vit_l",
        description="SAM model architecture"
    )
    sam_checkpoint_path: str = Field(
        default="sam_vit_l_0b3195.pth",
        description="Path to SAM checkpoint file"
    )
    sam_points_per_side: int = Field(
        default=24,
        ge=1,
        le=64,
        description="Number of points sampled per side"
    )
    sam_pred_iou_thresh: float = Field(
        default=0.86,
        ge=0.0,
        le=1.0,
        description="Threshold for predicted IoU"
    )
    sam_stability_score_thresh: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description="Threshold for mask stability"
    )
    sam_min_mask_region_area: int = Field(
        default=500,
        ge=0,
        description="Minimum area for valid masks"
    )
    
    # ========== FastSAM Configuration ==========
    fastsam_model_path: str = Field(
        default="FastSAM-x.pt",
        description="Path to FastSAM model"
    )
    fastsam_conf_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for FastSAM"
    )
    fastsam_iou_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="IoU threshold for NMS"
    )
    fastsam_max_det: int = Field(
        default=300,
        ge=1,
        le=1000,
        description="Maximum detections per image"
    )
    
    @field_validator('detector_type')
    def validate_detector(cls, v):
        """Ensure detector type is supported."""
        supported = ["yolo", "sam", "fastsam", "detectron2", "grounding_dino"]
        if v not in supported:
            raise ValueError(f"Detector must be one of {supported}, got {v}")
        return v
```

## 5. Update Pipeline Processor

```python
# frame_processor/pipeline/processor.py
# Update the ComponentFactory class:

from detection.sam import SAMDetector
from detection.fastsam import FastSAMDetector

class ComponentFactory:
    """
    Factory for creating processing components based on configuration.
    """
    
    # Registry of available detectors
    DETECTORS: Dict[str, Type[Detector]] = {
        "yolo": YOLODetector,
        "sam": SAMDetector,
        "fastsam": FastSAMDetector,
        # Future: "detectron2": Detectron2Detector,
        # Future: "grounding_dino": GroundingDINODetector,
    }
    
    @classmethod
    def create_detector(cls, config: Config) -> Detector:
        """
        Create detector instance based on configuration.
        """
        detector_type = config.detector_type.lower()
        
        if detector_type not in cls.DETECTORS:
            raise ValueError(
                f"Unknown detector type: {detector_type}. "
                f"Available: {list(cls.DETECTORS.keys())}"
            )
        
        detector_class = cls.DETECTORS[detector_type]
        
        # Create detector with appropriate parameters
        if detector_type == "yolo":
            return detector_class(
                model_path=config.detector_model,
                confidence=config.detector_confidence,
                device=config.detector_device
            )
        elif detector_type == "sam":
            return detector_class(
                model_path=config.sam_checkpoint_path,
                model_type=config.sam_model_type,
                device=config.detector_device,
                points_per_side=config.sam_points_per_side,
                pred_iou_thresh=config.sam_pred_iou_thresh,
                stability_score_thresh=config.sam_stability_score_thresh,
                min_mask_region_area=config.sam_min_mask_region_area
            )
        elif detector_type == "fastsam":
            return detector_class(
                model_path=config.fastsam_model_path,
                device=config.detector_device,
                conf_threshold=config.fastsam_conf_threshold,
                iou_threshold=config.fastsam_iou_threshold,
                max_det=config.fastsam_max_det
            )
        else:
            raise NotImplementedError(f"Factory not implemented for {detector_type}")
```

## 6. Update Detection Module Init

```python
# frame_processor/detection/__init__.py
"""
Object detection modules.

This package provides:
- Base detection interfaces
- YOLO detector implementation
- SAM detector for class-agnostic detection
- FastSAM for high-speed class-agnostic detection
- Support for additional detectors (detectron2, grounding_dino)
"""

from .base import Detection, Detector
from .yolo import YOLODetector
from .sam import SAMDetector
from .fastsam import FastSAMDetector

__all__ = [
    'Detection',
    'Detector',
    'YOLODetector',
    'SAMDetector',
    'FastSAMDetector',
]
```

## 7. Update Dockerfile

```dockerfile
# frame_processor/Dockerfile
# Add these sections:

# Download model weights during build
RUN mkdir -p /app/models && \
    cd /app/models && \
    # Download SAM-L weights
    wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth && \
    # Download FastSAM weights
    wget -q https://github.com/CASIA-IVA-Lab/FastSAM/releases/download/v0.0.1/FastSAM-x.pt

# Ensure models are accessible
COPY models/*.pth /app/models/
COPY models/*.pt /app/models/
```

## 8. Update docker-compose.yml

```yaml
# docker-compose.yml
# Update frame_processor service:

frame_processor:
    # ... existing config ...
    environment:
      # Add SAM/FastSAM configuration
      - DETECTOR_TYPE=${DETECTOR_TYPE:-sam}  # Change default to SAM
      
      # SAM Configuration
      - SAM_MODEL_TYPE=${SAM_MODEL_TYPE:-vit_l}
      - SAM_CHECKPOINT_PATH=${SAM_CHECKPOINT_PATH:-/app/models/sam_vit_l_0b3195.pth}
      - SAM_POINTS_PER_SIDE=${SAM_POINTS_PER_SIDE:-24}
      - SAM_PRED_IOU_THRESH=${SAM_PRED_IOU_THRESH:-0.86}
      - SAM_STABILITY_SCORE_THRESH=${SAM_STABILITY_SCORE_THRESH:-0.92}
      - SAM_MIN_MASK_REGION_AREA=${SAM_MIN_MASK_REGION_AREA:-500}
      
      # FastSAM Configuration
      - FASTSAM_MODEL_PATH=${FASTSAM_MODEL_PATH:-/app/models/FastSAM-x.pt}
      - FASTSAM_CONF_THRESHOLD=${FASTSAM_CONF_THRESHOLD:-0.4}
      - FASTSAM_IOU_THRESHOLD=${FASTSAM_IOU_THRESHOLD:-0.9}
      - FASTSAM_MAX_DET=${FASTSAM_MAX_DET:-300}
      
    volumes:
      # Mount model directory
      - ${WORKSPACE}/frame_processor/models:/app/models:ro
```

## 9. Testing Script

```python
# frame_processor/test_detectors.py
"""
Test script to compare SAM and FastSAM detectors.
"""

import asyncio
import cv2
import time
import numpy as np
from pathlib import Path

from core.config import Config
from detection.sam import SAMDetector
from detection.fastsam import FastSAMDetector
from detection.yolo import YOLODetector


async def test_detector(detector, image_path: str, name: str):
    """Test a detector on an image."""
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print(f"{'='*60}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    print(f"Image shape: {image.shape}")
    
    # Warmup
    print("Warming up...")
    detector.warmup()
    
    # Time detection
    print("Running detection...")
    start = time.time()
    detections = await detector.detect(image)
    inference_time = (time.time() - start) * 1000
    
    print(f"\nResults:")
    print(f"- Inference time: {inference_time:.1f}ms")
    print(f"- FPS: {1000/inference_time:.1f}")
    print(f"- Detections: {len(detections)}")
    
    # Analyze detections
    if detections:
        confidences = [d.confidence for d in detections]
        areas = [(d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]) for d in detections]
        
        print(f"\nDetection statistics:")
        print(f"- Avg confidence: {np.mean(confidences):.3f}")
        print(f"- Min confidence: {np.min(confidences):.3f}")
        print(f"- Max confidence: {np.max(confidences):.3f}")
        print(f"- Avg area: {np.mean(areas):.0f} pixels")
        print(f"- Total coverage: {sum(areas)/(image.shape[0]*image.shape[1])*100:.1f}%")
    
    # Draw and save results
    result_image = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result_image, f"{det.confidence:.2f}", 
                   (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    output_path = f"test_results_{name.lower().replace(' ', '_')}.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"\nSaved visualization to: {output_path}")
    
    return detections


async def compare_detectors():
    """Compare all detectors on the same image."""
    # Test image - replace with your test image
    test_image = "test_office.jpg"
    
    if not Path(test_image).exists():
        print(f"Test image not found: {test_image}")
        print("Creating a test image with various objects...")
        # Create synthetic test image
        img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
        # Add some objects
        cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 0), -1)  # Black square
        cv2.circle(img, (500, 200), 80, (128, 128, 128), -1)  # Gray circle
        cv2.rectangle(img, (700, 150), (900, 250), (200, 200, 200), -1)  # Light rect
        cv2.imwrite(test_image, img)
    
    # Initialize detectors
    config = Config()
    
    # YOLO baseline
    yolo = YOLODetector(
        model_path="yolov11l.pt",
        confidence=0.25,  # Lower threshold
        device="cuda"
    )
    
    # SAM-L
    sam = SAMDetector(
        model_path="/app/models/sam_vit_l_0b3195.pth",
        model_type="vit_l",
        device="cuda",
        points_per_side=24
    )
    
    # FastSAM
    fastsam = FastSAMDetector(
        model_path="/app/models/FastSAM-x.pt",
        device="cuda",
        conf_threshold=0.4
    )
    
    # Test each detector
    yolo_results = await test_detector(yolo, test_image, "YOLO v11")
    sam_results = await test_detector(sam, test_image, "SAM-L")
    fastsam_results = await test_detector(fastsam, test_image, "FastSAM")
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Detector':<15} {'Detections':<15} {'Unique Objects':<15}")
    print(f"{'-'*45}")
    print(f"{'YOLO v11':<15} {len(yolo_results):<15} {len(set(d.class_name for d in yolo_results)):<15}")
    print(f"{'SAM-L':<15} {len(sam_results):<15} {'All objects':<15}")
    print(f"{'FastSAM':<15} {len(fastsam_results):<15} {'All objects':<15}")


if __name__ == "__main__":
    asyncio.run(compare_detectors())
```

## 10. Installation Script

```bash
#!/bin/bash
# frame_processor/install_models.sh

echo "Installing SAM and FastSAM models..."

# Create models directory
mkdir -p models

# Download SAM-L checkpoint
echo "Downloading SAM-L model (1.2GB)..."
wget -O models/sam_vit_l_0b3195.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# Download FastSAM model
echo "Downloading FastSAM-x model (138MB)..."
wget -O models/FastSAM-x.pt \
  https://github.com/CASIA-IVA-Lab/FastSAM/releases/download/v0.0.1/FastSAM-x.pt

# Verify downloads
echo "Verifying downloads..."
ls -lh models/

echo "Model installation complete!"
```

## Usage Instructions

### 1. Install Models
```bash
cd frame_processor
chmod +x install_models.sh
./install_models.sh
```

### 2. Switch Between Detectors

**Use SAM-L (better quality, slower):**
```bash
export DETECTOR_TYPE=sam
docker-compose --profile frame_processor up frame_processor
```

**Use FastSAM (faster, good quality):**
```bash
export DETECTOR_TYPE=fastsam
docker-compose --profile frame_processor up frame_processor
```

### 3. Runtime Configuration

You can also update `.env` file:
```env
# Detector selection
DETECTOR_TYPE=sam  # or fastsam

# SAM tuning
SAM_POINTS_PER_SIDE=32  # Increase for better coverage
SAM_MIN_MASK_REGION_AREA=300  # Decrease to catch smaller objects

# FastSAM tuning
FASTSAM_CONF_THRESHOLD=0.3  # Lower to detect more objects
```

### 4. Monitor Performance

The logs will show:
```
SAM-L: ~5-8 FPS, excellent coverage
FastSAM: ~15-20 FPS, very good coverage
```

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size or points_per_side:
```env
SAM_POINTS_PER_SIDE=16
FASTSAM_MAX_DET=200
```

### Issue: Too Many False Positives
**Solution**: Increase confidence thresholds:
```env
SAM_PRED_IOU_THRESH=0.90
FASTSAM_CONF_THRESHOLD=0.5
```

### Issue: Missing Small Objects
**Solution**: Decrease minimum area:
```env
SAM_MIN_MASK_REGION_AREA=100
```

## Performance Optimization Tips

1. **For Maximum Speed**: Use FastSAM with reduced max_det
2. **For Maximum Quality**: Use SAM-L with higher points_per_side
3. **For Balance**: Use SAM-L with default settings
4. **For Memory Efficiency**: Process every 3rd frame instead of every frame

## Links and References

- **SAM Paper**: https://arxiv.org/abs/2304.02643
- **SAM GitHub**: https://github.com/facebookresearch/segment-anything
- **SAM Demo**: https://segment-anything.com/demo
- **FastSAM Paper**: https://arxiv.org/abs/2306.12156
- **FastSAM GitHub**: https://github.com/CASIA-IVA-Lab/FastSAM
- **Ultralytics Docs**: https://docs.ultralytics.com/
- **Comparison Blog**: https://blog.roboflow.com/sam-fast-sam-mobile-sam-edge-sam/

This implementation gives you the flexibility to test both detectors and choose the best one for your specific use case!