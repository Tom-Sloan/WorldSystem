"""
YOLO detector implementation.

This wraps the existing YOLO functionality in the modular detector interface,
preserving all current behavior while enabling easy swapping of detectors.
"""

from ultralytics import YOLO
import torch
from typing import List, Optional
import numpy as np
from pathlib import Path
import asyncio

from .base import Detector, Detection
from core.utils import get_logger, PerformanceTimer


logger = get_logger(__name__)


class YOLODetector(Detector):
    """
    YOLO v8/v11 detector implementation.
    
    This preserves the exact same YOLO functionality from the original
    frame_processor.py, just wrapped in our modular interface.
    """
    
    def __init__(self, model_path: str, confidence: float, device: str, **kwargs):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model weights (e.g., "yolov11l.pt")
            confidence: Minimum confidence threshold
            device: Device to run on ("cuda" or "cpu")
            **kwargs: Additional YOLO parameters
        """
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        
        logger.info(f"Initializing YOLO detector with model: {model_path}")
        
        # Load model - this matches the original implementation
        try:
            self.model = YOLO(model_path)
            
            # Check CUDA availability
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
            
            self.model.to(self.device)
            logger.info(f"YOLO model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            # Try to download if not found (matching original behavior)
            if not Path(model_path).exists():
                logger.info(f"Model file not found, attempting to download {model_path}")
                self.model = YOLO(model_path)  # This triggers download
                self.model.to(self.device)
                logger.info(f"Model downloaded and loaded on {self.device}")
            else:
                raise
        
        # Store model info
        self._model_name = Path(model_path).stem
        self._class_names = self.model.names if hasattr(self.model, 'names') else {}
    
    async def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLO detection on single frame.
        
        This matches the detection logic from the original frame_processor.py
        """
        with PerformanceTimer("yolo_inference", logger):
            try:
                # Run inference in executor to avoid blocking event loop
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,  # Use default executor
                    lambda: self.model(frame, conf=self.confidence)
                )
                detections = []
                
                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        logger.debug("No boxes found in YOLO result")
                        continue
                    
                    # Process each detection
                    for box in boxes:
                        try:
                            # Extract bbox coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Validate coordinates (from original)
                            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                                logger.warning(
                                    f"Bounding box out of frame bounds: {[x1, y1, x2, y2]}, "
                                    f"frame shape: {frame.shape[:2]}"
                                )
                                # Clip to frame bounds
                                x1 = max(0, x1)
                                y1 = max(0, y1)
                                x2 = min(frame.shape[1], x2)
                                y2 = min(frame.shape[0], y2)
                            
                            # Extract class and confidence
                            class_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = self._class_names.get(class_id, f"class_{class_id}")
                            
                            # Create detection object
                            detections.append(Detection(
                                bbox=(x1, y1, x2, y2),
                                confidence=conf,
                                class_id=class_id,
                                class_name=class_name
                            ))
                            
                        except Exception as e:
                            logger.error(f"Failed to parse YOLO box: {e}")
                            continue
                
                logger.debug(f"YOLO detected {len(detections)} objects")
                return detections
                
            except Exception as e:
                logger.error(f"YOLO inference failed: {e}", exc_info=True)
                return []
    
    async def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Batch detection for efficiency.
        
        YOLO supports batch inference natively.
        """
        if not frames:
            return []
        
        with PerformanceTimer(f"yolo_batch_inference_{len(frames)}_frames", logger):
            try:
                # Run batch inference
                results = self.model(frames, conf=self.confidence)
                batch_detections = []
                
                # Process results for each frame
                for frame_idx, result in enumerate(results):
                    frame_detections = []
                    boxes = result.boxes
                    
                    if boxes is not None:
                        for box in boxes:
                            try:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                
                                # Clip to frame bounds
                                h, w = frames[frame_idx].shape[:2]
                                x1 = max(0, min(x1, w))
                                y1 = max(0, min(y1, h))
                                x2 = max(0, min(x2, w))
                                y2 = max(0, min(y2, h))
                                
                                class_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                class_name = self._class_names.get(class_id, f"class_{class_id}")
                                
                                frame_detections.append(Detection(
                                    bbox=(x1, y1, x2, y2),
                                    confidence=conf,
                                    class_id=class_id,
                                    class_name=class_name
                                ))
                            except Exception as e:
                                logger.error(f"Failed to parse box in batch: {e}")
                                continue
                    
                    batch_detections.append(frame_detections)
                
                return batch_detections
                
            except Exception as e:
                logger.error(f"YOLO batch inference failed: {e}", exc_info=True)
                # Return empty lists for each frame
                return [[] for _ in frames]
    
    @property
    def name(self) -> str:
        """Detector name for logging/metrics."""
        return f"YOLO-{self._model_name}"
    
    @property
    def supported_classes(self) -> List[str]:
        """List of classes this detector can identify."""
        return list(self._class_names.values())
    
    def warmup(self) -> None:
        """
        Warmup the model with a dummy inference.
        
        This can improve performance for the first real inference.
        """
        try:
            logger.info("Warming up YOLO model...")
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_frame, conf=self.confidence)
            logger.info("YOLO warmup complete")
        except Exception as e:
            logger.warning(f"YOLO warmup failed: {e}")