"""
IOU-based object tracker implementation.

This implements the same IOU tracking logic from the original frame_processor.py
but with memory optimization (storing only ROIs instead of full frames) and
wrapped in the modular tracker interface.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import time
from collections import defaultdict

from .base import Tracker, TrackedObject
from detection.base import Detection
from core.utils import get_logger, calculate_iou, extract_roi_with_padding
from core.config import Config


logger = get_logger(__name__)


class IOUTracker(Tracker):
    """
    Simple IOU-based tracker with optimized memory usage.
    
    This preserves the tracking logic from the original implementation
    but stores only ROIs instead of full frames for memory efficiency.
    """
    
    def __init__(self, 
                 iou_threshold: float = 0.3,
                 max_lost: int = 10,
                 process_after_seconds: float = 1.5,
                 max_tracks: int = 100,
                 **kwargs):
        """
        Initialize IOU tracker.
        
        Args:
            iou_threshold: Minimum IOU for matching detection to track
            max_lost: Maximum frames before removing lost track
            process_after_seconds: Time before sending track for API processing
            max_tracks: Maximum number of simultaneous tracks
            **kwargs: Additional parameters
        """
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.process_after_seconds = process_after_seconds
        self.max_tracks = max_tracks
        
        # Track storage
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_id = 1
        
        # Frame quality scorer (will be set by processor)
        self.frame_scorer = None
        
        # Statistics
        self.total_tracks_created = 0
        self.total_tracks_lost = 0
        
        logger.info(
            f"IOUTracker initialized with iou_threshold={iou_threshold}, "
            f"max_lost={max_lost}, process_after={process_after_seconds}s"
        )
    
    def update(self, detections: List[Detection], frame: np.ndarray, 
               frame_number: int) -> List[TrackedObject]:
        """
        Update tracker with new detections.
        
        This implements the same matching logic as the original tracker.
        """
        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_trks = self._match_detections(detections)
        
        # Update matched tracks
        for det_idx, trk_id in matched:
            detection = detections[det_idx]
            track = self.tracks[trk_id]
            
            # Update track state
            track.update_bbox(detection.bbox, detection.confidence)
            
            # Update best frame if we have a scorer
            if self.frame_scorer:
                self._update_best_frame(track, frame, detection.bbox)
            else:
                # Without scorer, just use first frame
                if track.best_frame is None:
                    roi, _ = extract_roi_with_padding(frame, detection.bbox)
                    track.best_frame = roi
                    track.best_score = track.confidence
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            if len(self.tracks) >= self.max_tracks:
                logger.warning(f"Maximum tracks ({self.max_tracks}) reached, skipping new detection")
                continue
                
            detection = detections[det_idx]
            track = TrackedObject(
                id=self.next_id,
                class_name=detection.class_name,
                bbox=detection.bbox,
                confidence=detection.confidence
            )
            
            # Extract initial ROI
            roi, _ = extract_roi_with_padding(frame, detection.bbox)
            track.best_frame = roi
            track.best_score = detection.confidence
            
            self.tracks[self.next_id] = track
            self.next_id += 1
            self.total_tracks_created += 1
            
            logger.debug(f"Created new track #{track.id} for {track.class_name}")
        
        # Update lost tracks
        tracks_to_remove = []
        for trk_id in unmatched_trks:
            track = self.tracks[trk_id]
            track.mark_missed()
            
            if track.time_since_update > self.max_lost:
                tracks_to_remove.append(trk_id)
        
        # Remove lost tracks
        for trk_id in tracks_to_remove:
            track = self.tracks.pop(trk_id)
            self.total_tracks_lost += 1
            logger.debug(f"Removed lost track #{trk_id} ({track.class_name})")
        
        # Find tracks ready for API processing
        ready_tracks = []
        current_time = time.time()
        
        for track in self.tracks.values():
            if track.should_process(current_time, self.process_after_seconds):
                ready_tracks.append(track)
                logger.debug(
                    f"Track #{track.id} ready for processing after "
                    f"{current_time - track.created_at:.1f}s"
                )
        
        # Log statistics periodically
        if frame_number % 100 == 0:
            logger.info(
                f"Tracking stats: {len(self.tracks)} active, "
                f"{self.total_tracks_created} created, {self.total_tracks_lost} lost"
            )
        
        return ready_tracks
    
    def get_active_tracks(self) -> List[TrackedObject]:
        """Get all currently active tracks."""
        return list(self.tracks.values())
    
    @property
    def name(self) -> str:
        """Tracker name for logging/metrics."""
        return "IOU-Tracker"
    
    def set_frame_scorer(self, scorer):
        """
        Set frame quality scorer for best frame selection.
        
        Args:
            scorer: Frame scorer instance
        """
        self.frame_scorer = scorer
        logger.debug("Frame scorer attached to tracker")
    
    def _match_detections(self, detections: List[Detection]) -> Tuple:
        """
        Match detections to existing tracks using IOU.
        
        Returns:
            Tuple of (matched, unmatched_detections, unmatched_tracks)
        """
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(self.tracks.keys())
        
        # Build IOU matrix
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(detections), len(track_ids)))
        
        for d_idx, det in enumerate(detections):
            for t_idx, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                iou = calculate_iou(det.bbox, track.bbox)
                iou_matrix[d_idx, t_idx] = iou
        
        # Greedy matching (same as original)
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = track_ids.copy()
        
        # Continue matching until no good matches remain
        while iou_matrix.size > 0:
            # Find best match
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[max_idx]
            
            if max_iou < self.iou_threshold:
                break
            
            # Record match
            d_idx, t_idx = max_idx
            matched.append((unmatched_dets[d_idx], unmatched_trks[t_idx]))
            
            # Remove matched items
            unmatched_dets.pop(d_idx)
            unmatched_trks.pop(t_idx)
            
            # Remove from IOU matrix
            iou_matrix = np.delete(iou_matrix, d_idx, axis=0)
            iou_matrix = np.delete(iou_matrix, t_idx, axis=1)
        
        return matched, unmatched_dets, unmatched_trks
    
    def _update_best_frame(self, track: TrackedObject, frame: np.ndarray, bbox: Tuple):
        """
        Update best frame for track if current frame has better quality.
        
        Args:
            track: Track to update
            frame: Current frame
            bbox: Current bounding box
        """
        if not self.frame_scorer:
            return
        
        # Calculate quality score
        score, components = self.frame_scorer.score_frame(frame, bbox, frame.shape[:2])
        
        # Update if better
        if score > track.best_score:
            # Extract ROI with padding
            roi, _ = extract_roi_with_padding(frame, bbox)
            
            # Store only ROI for memory efficiency
            track.best_frame = roi
            track.best_score = score
            track.score_components = components
            
            logger.debug(
                f"Updated best frame for track #{track.id}: "
                f"score={score:.3f}, components={components}"
            )
    
    def get_memory_usage(self) -> int:
        """Calculate total memory usage of all tracks."""
        total_memory = super().get_memory_usage()
        
        # Add tracker overhead
        total_memory += len(self.tracks) * 1024  # Rough estimate for dict overhead
        
        return total_memory
    
    def cleanup_old_tracks(self, max_age_seconds: float = 300):
        """
        Remove tracks older than specified age.
        
        Args:
            max_age_seconds: Maximum age in seconds before removal
        """
        current_time = time.time()
        tracks_to_remove = []
        
        for track_id, track in self.tracks.items():
            age = current_time - track.created_at
            if age > max_age_seconds:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            track = self.tracks.pop(track_id)
            logger.info(f"Removed old track #{track_id} (age: {age:.1f}s)")
        
        return len(tracks_to_remove)