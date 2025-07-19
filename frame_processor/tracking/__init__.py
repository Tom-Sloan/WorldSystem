"""
Object tracking modules.

This package provides:
- Base tracking interfaces
- IOU-based tracker
- Support for additional trackers (SORT, DeepSORT, ByteTrack)
"""

from .base import TrackedObject, Tracker
from .iou_tracker import IOUTracker

__all__ = [
    'TrackedObject',
    'Tracker',
    'IOUTracker',
]