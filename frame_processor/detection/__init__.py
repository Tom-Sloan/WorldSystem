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