# Frame Processor Modules
"""
This package contains the modules for the enhanced frame processor:

- tracker: Object tracking with IOU-based matching
- frame_scorer: Quality scoring for frame selection
- enhancement: Image enhancement for better API recognition
- api_client: Google Lens and Perplexity API integration
- scene_scaler: Scene scale calculation from object dimensions
"""

from .tracker import ObjectTracker, TrackedObject
from .frame_scorer import FrameQualityScorer
from .enhancement import ImageEnhancer
from .api_client import APIClient
from .scene_scaler import SceneScaler

__all__ = [
    'ObjectTracker',
    'TrackedObject',
    'FrameQualityScorer',
    'ImageEnhancer',
    'APIClient',
    'SceneScaler'
]