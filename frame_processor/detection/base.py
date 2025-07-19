"""
Abstract base classes for detection modules.

This module defines the interface that all detection implementations must follow,
enabling easy swapping of detection algorithms (YOLO, Detectron2, etc.).
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class Detection:
    """
    Universal detection format regardless of detector.
    
    Attributes:
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        confidence: Detection confidence score [0.0-1.0]
        class_id: Numeric class identifier
        class_name: Human-readable class name
        embedding: Optional feature embedding for ReID trackers
        mask: Optional segmentation mask
    """
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    embedding: Optional[np.ndarray] = None  # For ReID trackers
    mask: Optional[np.ndarray] = None      # For segmentation
    
    def __post_init__(self):
        """Validate detection data."""
        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox: {self.bbox}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0,1], got {self.confidence}")


class Detector(ABC):
    """
    Abstract base class for all detectors.
    
    All detection implementations (YOLO, Detectron2, etc.) must inherit from
    this class and implement the required methods.
    """
    
    @abstractmethod
    async def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a single frame.
        
        Args:
            frame: Input image as numpy array (H, W, C) in BGR format
            
        Returns:
            List of Detection objects found in the frame
        """
        pass
    
    @abstractmethod
    async def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Detect objects in multiple frames (batch processing).
        
        Args:
            frames: List of images as numpy arrays
            
        Returns:
            List of detection lists, one per input frame
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Detector name for logging/metrics.
        
        Returns:
            String identifier for this detector
        """
        pass
    
    @property
    @abstractmethod
    def supported_classes(self) -> List[str]:
        """
        List of classes this detector can identify.
        
        Returns:
            List of class names supported by this detector
        """
        pass
    
    def warmup(self) -> None:
        """
        Optional warmup method to initialize detector.
        
        Can be used to pre-load models, run dummy inference, etc.
        Default implementation does nothing.
        """
        pass