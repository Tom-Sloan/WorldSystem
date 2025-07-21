"""
Abstract base classes for video-aware tracking.

This module defines the interface that all video tracking implementations must follow,
enabling easy swapping of tracking algorithms while maintaining video context.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass, field
import time


@dataclass
class VideoTrackingResult:
    """Result from video tracking."""
    masks: List[Dict[str, Any]]  # List of mask dictionaries with segmentation data
    tracks: List[Dict[str, Any]]  # Track information with IDs and bboxes
    object_count: int
    processing_time_ms: float
    frame_number: int
    confidence_scores: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "masks": self.masks,
            "tracks": self.tracks,
            "object_count": self.object_count,
            "processing_time_ms": self.processing_time_ms,
            "frame_number": self.frame_number,
            "confidence_scores": self.confidence_scores
        }


class VideoTracker(ABC):
    """
    Abstract base class for video-aware trackers.
    Provides interface for both real-time and batch processing.
    """
    
    @abstractmethod
    async def initialize_stream(self, stream_id: str, first_frame: np.ndarray, 
                                prompts: Optional[Dict] = None) -> VideoTrackingResult:
        """
        Initialize tracking for a new stream.
        
        Args:
            stream_id: Unique identifier for the stream
            first_frame: First frame of the video stream
            prompts: Optional prompting information (points, boxes, etc.)
            
        Returns:
            VideoTrackingResult with initial detections
        """
        pass
    
    @abstractmethod
    async def process_frame(self, stream_id: str, frame: np.ndarray, 
                           timestamp: int) -> VideoTrackingResult:
        """
        Process a single frame in the video stream.
        
        Args:
            stream_id: Stream identifier
            frame: Current frame to process
            timestamp: Frame timestamp in milliseconds
            
        Returns:
            VideoTrackingResult with tracked objects
        """
        pass
    
    @abstractmethod
    async def cleanup_stream(self, stream_id: str):
        """
        Clean up resources for a stream.
        
        Args:
            stream_id: Stream to clean up
        """
        pass
    
    @abstractmethod
    def get_stream_status(self, stream_id: str) -> Dict[str, Any]:
        """
        Get current status of a stream.
        
        Args:
            stream_id: Stream to query
            
        Returns:
            Dictionary with status information
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tracker name for logging."""
        pass
    
    @property
    @abstractmethod
    def supports_batching(self) -> bool:
        """Whether this tracker supports batch processing."""
        pass
    
    @property
    def max_concurrent_streams(self) -> int:
        """Maximum number of concurrent streams supported."""
        return 10
    
    @property
    def requires_gpu(self) -> bool:
        """Whether this tracker requires GPU."""
        return True
    
    async def process_batch(self, stream_id: str, frames: List[np.ndarray], 
                           timestamps: List[int]) -> List[VideoTrackingResult]:
        """
        Process multiple frames at once (if supported).
        
        Default implementation processes frames sequentially.
        Subclasses can override for true batch processing.
        
        Args:
            stream_id: Stream identifier
            frames: List of frames to process
            timestamps: Corresponding timestamps
            
        Returns:
            List of tracking results
        """
        if not self.supports_batching:
            # Process sequentially
            results = []
            for frame, timestamp in zip(frames, timestamps):
                result = await self.process_frame(stream_id, frame, timestamp)
                results.append(result)
            return results
        
        # Subclasses should implement actual batch processing
        raise NotImplementedError("Batch processing not implemented")
    
    def validate_frame(self, frame: np.ndarray) -> bool:
        """
        Validate that frame is suitable for processing.
        
        Args:
            frame: Frame to validate
            
        Returns:
            True if frame is valid
        """
        if frame is None or frame.size == 0:
            return False
        
        # Check dimensions
        if len(frame.shape) != 3:
            return False
        
        # Check channels (should be RGB or BGR)
        if frame.shape[2] not in [3, 4]:
            return False
        
        return True
    
    def get_memory_usage(self, stream_id: str) -> Dict[str, int]:
        """
        Get memory usage for a stream.
        
        Args:
            stream_id: Stream to query
            
        Returns:
            Dictionary with memory usage in bytes
        """
        return {
            "gpu_memory": 0,
            "cpu_memory": 0,
            "total": 0
        }


@dataclass
class StreamState:
    """Base class for maintaining stream state."""
    stream_id: str
    frame_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    is_active: bool = True
    
    def update(self):
        """Update timestamp when state changes."""
        self.last_update = time.time()
        self.frame_count += 1
    
    @property
    def age(self) -> float:
        """Age of the stream in seconds."""
        return time.time() - self.created_at
    
    @property
    def time_since_update(self) -> float:
        """Time since last update in seconds."""
        return time.time() - self.last_update