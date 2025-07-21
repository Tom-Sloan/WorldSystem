"""
Video processing modules.

This package provides:
- Video processing orchestration
- SAM2 real-time video tracking
- Prompt generation strategies
- Base classes for video tracking
"""

from .base import VideoTracker, VideoTrackingResult, StreamState
from .tracker import SAM2RealtimeTracker
from .processor import VideoProcessor, VideoProcessingResult
from .prompt_strategies import (
    PromptStrategy,
    GridPromptStrategy,
    MotionPromptStrategy,
    SaliencyPromptStrategy,
    HybridPromptStrategy,
    create_prompt_strategy
)

__all__ = [
    # Base classes
    'VideoTracker',
    'VideoTrackingResult', 
    'StreamState',
    
    # Implementations
    'SAM2RealtimeTracker',
    'VideoProcessor',
    'VideoProcessingResult',
    
    # Strategies
    'PromptStrategy',
    'GridPromptStrategy',
    'MotionPromptStrategy',
    'SaliencyPromptStrategy',
    'HybridPromptStrategy',
    'create_prompt_strategy',
]