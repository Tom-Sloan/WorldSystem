"""
Abstract base classes for tracking modules.

This module defines the interface that all tracking implementations must follow,
enabling easy swapping of tracking algorithms (IOU, SORT, DeepSORT, etc.).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import time

from detection.base import Detection


@dataclass
class TrackedObject:
    """
    Universal tracked object format.
    
    Attributes:
        id: Unique identifier for this track
        class_name: Object class name
        bbox: Current bounding box (x1, y1, x2, y2)
        confidence: Current detection confidence
        age: Number of frames since first detection
        time_since_update: Frames since last detection match
        hits: Total number of matched detections
        best_frame: Best quality ROI for API processing (memory efficient)
        best_score: Quality score of best frame
        created_at: Timestamp when track was created
        api_result: Results from external API processing
    """
    id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    
    # Tracking state
    age: int = 0  # Frames since first seen
    time_since_update: int = 0
    hits: int = 1
    
    # For API processing - store only ROI, not full frame
    best_frame: Optional[np.ndarray] = None  # Just the ROI
    best_score: float = 0.0
    created_at: float = field(default_factory=time.time)
    first_seen_time: float = field(default_factory=time.time)  # Same as created_at for compatibility
    
    # Results from external processing
    api_result: Optional[Dict[str, Any]] = None
    api_processed: bool = False  # Flag to track if API processing has been attempted
    estimated_dimensions: Optional[Dict[str, float]] = None  # Estimated dimensions from API
    
    # Additional attributes for enhanced visualization
    is_being_processed: bool = False  # Currently being processed by API
    frame_history: List[Dict] = field(default_factory=list)  # History of frames for visualization
    identified_products: List[Dict] = field(default_factory=list)  # Products identified by API
    score_components: Optional[Dict[str, float]] = None  # Quality score breakdown
    processing_time: float = 0.0  # Time taken for API processing
    
    def should_process(self, current_time: float, 
                      process_after: float = 1.5,
                      reprocess_interval: float = 3.0) -> bool:
        """
        Check if object should be sent for API processing.
        
        Args:
            current_time: Current timestamp
            process_after: Seconds to wait before first processing
            reprocess_interval: Seconds between reprocessing attempts
            
        Returns:
            True if object should be processed
        """
        time_since_creation = current_time - self.created_at
        
        if self.api_result is None:
            # First processing after initial delay
            return time_since_creation >= process_after
        else:
            # Reprocessing logic could be added here if needed
            return False
    
    def update_bbox(self, bbox: Tuple[int, int, int, int], confidence: float):
        """Update bounding box and confidence."""
        self.bbox = bbox
        self.confidence = confidence
        self.time_since_update = 0
        self.hits += 1
        self.age += 1
    
    def mark_missed(self):
        """Mark that this track was not matched in current frame."""
        self.time_since_update += 1
        self.age += 1
    
    def get_memory_usage(self) -> int:
        """Calculate approximate memory usage in bytes."""
        memory = 0
        if self.best_frame is not None:
            memory += self.best_frame.nbytes
        # Add some overhead for other attributes
        memory += 1024  # Rough estimate for other fields
        return memory


class Tracker(ABC):
    """
    Abstract base class for all trackers.
    
    All tracking implementations (IOU, SORT, DeepSORT, etc.) must inherit from
    this class and implement the required methods.
    """
    
    @abstractmethod
    def update(self, detections: List[Detection], frame: np.ndarray, 
               frame_number: int) -> List[TrackedObject]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections from current frame
            frame: Current frame for extracting ROIs
            frame_number: Sequential frame number
            
        Returns:
            List of TrackedObjects ready for API processing
        """
        pass
    
    @abstractmethod
    def get_active_tracks(self) -> List[TrackedObject]:
        """
        Get all currently active tracks.
        
        Returns:
            List of all TrackedObjects being tracked
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Tracker name for logging/metrics.
        
        Returns:
            String identifier for this tracker
        """
        pass
    
    def get_track_by_id(self, track_id: int) -> Optional[TrackedObject]:
        """
        Get a specific track by ID.
        
        Args:
            track_id: Track identifier
            
        Returns:
            TrackedObject if found, None otherwise
        """
        for track in self.get_active_tracks():
            if track.id == track_id:
                return track
        return None
    
    def get_memory_usage(self) -> int:
        """
        Calculate total memory usage of all tracks.
        
        Returns:
            Total memory usage in bytes
        """
        return sum(track.get_memory_usage() for track in self.get_active_tracks())