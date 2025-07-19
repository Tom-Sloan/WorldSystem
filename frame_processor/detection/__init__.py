"""
Object detection modules.

This package provides:
- Base detection interfaces
- YOLO detector implementation
- Support for additional detectors (detectron2, grounding_dino)
"""

from .base import Detection, Detector
from .yolo import YOLODetector

__all__ = [
    'Detection',
    'Detector',
    'YOLODetector',
]