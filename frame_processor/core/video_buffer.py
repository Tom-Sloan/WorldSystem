"""
Video buffer implementation with SAM2Long memory tree for error prevention.

This module provides buffering for video streams with multiple hypothesis tracking
to prevent error accumulation in long videos.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import asyncio
from dataclasses import dataclass
import time

from .utils import get_logger

logger = get_logger(__name__)


class VideoBufferBase(ABC):
    """Abstract base class for video buffers."""
    
    @abstractmethod
    async def add_frame(self, stream_id: str, frame: np.ndarray, timestamp: int):
        """Add a frame to the buffer."""
        pass
    
    @abstractmethod
    async def get_frames(self, stream_id: str, count: int) -> List[Tuple[np.ndarray, int]]:
        """Get recent frames with timestamps."""
        pass
    
    @abstractmethod
    async def cleanup_stream(self, stream_id: str):
        """Clean up resources for a stream."""
        pass


class SAM2LongVideoBuffer(VideoBufferBase):
    """
    Enhanced video buffer with SAM2Long memory tree for error prevention.
    Maintains multiple hypothesis branches to prevent error accumulation.
    """
    
    def __init__(self, buffer_size: int = 30, tree_branches: int = 3):
        self.buffers: Dict[str, deque] = {}
        self.memory_trees: Dict[str, MemoryTree] = {}
        self.tree_branches = tree_branches
        self.buffer_size = buffer_size
        self._locks: Dict[str, asyncio.Lock] = {}
        self.stream_stats: Dict[str, StreamStats] = {}
        
        logger.info(f"Initialized SAM2LongVideoBuffer with buffer_size={buffer_size}, "
                   f"tree_branches={tree_branches}")
        
    async def add_frame(self, stream_id: str, frame: np.ndarray, timestamp: int):
        """Add frame and maintain memory tree branches."""
        if stream_id not in self._locks:
            self._locks[stream_id] = asyncio.Lock()
            
        async with self._locks[stream_id]:
            # Initialize stream if new
            if stream_id not in self.buffers:
                self.buffers[stream_id] = deque(maxlen=self.buffer_size)
                self.memory_trees[stream_id] = MemoryTree(self.tree_branches)
                self.stream_stats[stream_id] = StreamStats(stream_id)
                logger.info(f"Initialized new stream buffer for {stream_id}")
            
            # Add frame to buffer
            self.buffers[stream_id].append((frame, timestamp))
            
            # Update stats
            stats = self.stream_stats[stream_id]
            stats.frame_count += 1
            stats.last_frame_time = time.time()
            
            # Note: Memory tree update happens when masks are available
            logger.debug(f"Added frame to stream {stream_id}, buffer size: {len(self.buffers[stream_id])}")
    
    async def get_frames(self, stream_id: str, count: int) -> List[Tuple[np.ndarray, int]]:
        """Get recent frames for processing."""
        if stream_id not in self.buffers:
            return []
            
        async with self._locks[stream_id]:
            frames = list(self.buffers[stream_id])[-count:]
            return frames
    
    def get_optimal_memory_path(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Select best segmentation path from memory tree."""
        if stream_id not in self.memory_trees:
            return None
        return self.memory_trees[stream_id].get_best_path()
    
    def update_memory_tree(self, stream_id: str, masks: List[Dict], timestamp: int):
        """Update memory tree with new segmentation masks."""
        if stream_id in self.memory_trees:
            self.memory_trees[stream_id].update(masks, timestamp)
    
    async def cleanup_stream(self, stream_id: str):
        """Clean up resources for a stream."""
        if stream_id in self.buffers:
            async with self._locks[stream_id]:
                # Log final stats
                if stream_id in self.stream_stats:
                    stats = self.stream_stats[stream_id]
                    logger.info(f"Stream {stream_id} cleanup - frames: {stats.frame_count}, "
                               f"duration: {time.time() - stats.start_time:.1f}s")
                    del self.stream_stats[stream_id]
                
                # Clean up resources
                del self.buffers[stream_id]
                del self.memory_trees[stream_id]
                
            del self._locks[stream_id]
            logger.info(f"Cleaned up stream {stream_id}")
    
    def get_stream_stats(self, stream_id: str) -> Optional['StreamStats']:
        """Get statistics for a stream."""
        return self.stream_stats.get(stream_id)


@dataclass
class StreamStats:
    """Statistics for a video stream."""
    stream_id: str
    frame_count: int = 0
    start_time: float = None
    last_frame_time: float = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.time()
        if self.last_frame_time is None:
            self.last_frame_time = time.time()


class MemoryTree:
    """SAM2Long-style memory tree for maintaining multiple segmentation hypotheses."""
    
    def __init__(self, max_branches: int = 3, consistency_threshold: float = 0.8):
        self.max_branches = max_branches
        self.consistency_threshold = consistency_threshold
        self.branches: List[MemoryBranch] = []
        self.active_branch_idx = 0
        
        logger.debug(f"Initialized MemoryTree with max_branches={max_branches}, "
                    f"consistency_threshold={consistency_threshold}")
        
    def update(self, masks: List[Dict], timestamp: int):
        """Update memory tree with new segmentation results."""
        if not self.branches:
            # Initialize first branch
            self.branches.append(MemoryBranch())
            logger.debug("Created initial memory branch")
        
        # Update existing branches with new masks
        branch_scores = []
        for i, branch in enumerate(self.branches):
            score = branch.update(masks, timestamp)
            branch_scores.append(score)
            logger.debug(f"Branch {i} consistency score: {score:.3f}")
        
        # Create new branches if consistency drops
        if branch_scores[self.active_branch_idx] < self.consistency_threshold:
            self._spawn_new_branch(masks, timestamp)
        
        # Prune poorly performing branches
        self._prune_branches()
        
        # Select best branch as active
        if self.branches:
            scores = [b.overall_score for b in self.branches]
            self.active_branch_idx = np.argmax(scores)
            logger.debug(f"Active branch: {self.active_branch_idx} (score: {scores[self.active_branch_idx]:.3f})")
    
    def _spawn_new_branch(self, masks: List[Dict], timestamp: int):
        """Create a new hypothesis branch when confidence drops."""
        if len(self.branches) < self.max_branches:
            new_branch = MemoryBranch()
            # Initialize with current masks as high confidence
            new_branch.update(masks, timestamp, is_keyframe=True)
            self.branches.append(new_branch)
            logger.info(f"Spawned new memory branch (total: {len(self.branches)})")
    
    def _prune_branches(self):
        """Remove poorly performing branches."""
        if len(self.branches) <= 1:
            return
        
        # Keep at least one branch
        scores = [b.overall_score for b in self.branches]
        min_score_idx = np.argmin(scores)
        
        # Remove if significantly worse than best branch
        if scores[min_score_idx] < 0.5 * max(scores):
            self.branches.pop(min_score_idx)
            if self.active_branch_idx >= len(self.branches):
                self.active_branch_idx = 0
            logger.info(f"Pruned low-scoring branch (remaining: {len(self.branches)})")
    
    def get_best_path(self) -> Dict[str, Any]:
        """Return the highest scoring segmentation path."""
        if not self.branches:
            return {"masks": [], "confidence": 0.0}
        
        best_branch = self.branches[self.active_branch_idx]
        return {
            "masks": best_branch.current_masks,
            "confidence": best_branch.overall_score,
            "history_length": len(best_branch.history),
            "branch_index": self.active_branch_idx,
            "total_branches": len(self.branches)
        }


class MemoryBranch:
    """A single hypothesis branch in the memory tree."""
    
    def __init__(self, history_limit: int = 30):
        self.history: deque = deque(maxlen=history_limit)
        self.current_masks: List[Dict] = []
        self.overall_score: float = 1.0
        self.consistency_scores: deque = deque(maxlen=10)
        self.object_tracks: Dict[int, ObjectTrack] = {}
        
    def update(self, masks: List[Dict], timestamp: int, is_keyframe: bool = False) -> float:
        """Update branch with new masks and compute consistency score."""
        if not self.current_masks:
            # First update
            self.current_masks = masks
            self.consistency_scores.append(1.0)
            self.overall_score = 1.0
            self._initialize_tracks(masks)
            return 1.0
        
        # Compute consistency with previous masks
        consistency = self._compute_consistency(self.current_masks, masks)
        self.consistency_scores.append(consistency)
        
        # Update history
        self.history.append({
            "masks": self.current_masks,
            "timestamp": timestamp,
            "score": consistency
        })
        
        # Update object tracks
        self._update_tracks(masks, timestamp)
        
        # Update current masks
        self.current_masks = masks
        
        # Update overall score (weighted average of recent consistency)
        if len(self.consistency_scores) > 0:
            weights = np.exp(np.linspace(-1, 0, len(self.consistency_scores)))
            self.overall_score = np.average(list(self.consistency_scores), weights=weights)
        
        # Boost score for keyframes
        if is_keyframe:
            self.overall_score = min(1.0, self.overall_score * 1.2)
        
        return consistency
    
    def _compute_consistency(self, prev_masks: List[Dict], curr_masks: List[Dict]) -> float:
        """Compute consistency score between two sets of masks."""
        if len(prev_masks) != len(curr_masks):
            # Penalize changes in object count
            return 0.7
        
        if not prev_masks:
            return 1.0
        
        # Compute IoU-based consistency for matched masks
        total_iou = 0.0
        matched = 0
        
        for prev in prev_masks:
            best_iou = 0.0
            for curr in curr_masks:
                if 'segmentation' in prev and 'segmentation' in curr:
                    intersection = np.logical_and(prev['segmentation'], curr['segmentation']).sum()
                    union = np.logical_or(prev['segmentation'], curr['segmentation']).sum()
                    iou = intersection / max(union, 1)
                    best_iou = max(best_iou, iou)
            
            total_iou += best_iou
            if best_iou > 0.5:  # Consider it a match
                matched += 1
        
        # Combine IoU score with match ratio
        iou_score = total_iou / max(len(prev_masks), 1)
        match_ratio = matched / max(len(prev_masks), 1)
        
        return 0.7 * iou_score + 0.3 * match_ratio
    
    def _initialize_tracks(self, masks: List[Dict]):
        """Initialize object tracks from first set of masks."""
        for i, mask in enumerate(masks):
            if 'object_id' in mask:
                obj_id = mask['object_id']
            else:
                obj_id = i
            
            self.object_tracks[obj_id] = ObjectTrack(obj_id)
    
    def _update_tracks(self, masks: List[Dict], timestamp: int):
        """Update object tracking information."""
        # Mark all tracks as potentially lost
        for track in self.object_tracks.values():
            track.frames_since_seen += 1
        
        # Update matched tracks
        for mask in masks:
            obj_id = mask.get('object_id', -1)
            if obj_id in self.object_tracks:
                track = self.object_tracks[obj_id]
                track.frames_since_seen = 0
                track.last_seen = timestamp
                track.total_frames += 1
                
                # Update confidence history
                if 'confidence' in mask:
                    track.confidence_history.append(mask['confidence'])


@dataclass
class ObjectTrack:
    """Track information for a single object."""
    object_id: int
    first_seen: float = None
    last_seen: float = None
    total_frames: int = 0
    frames_since_seen: int = 0
    confidence_history: deque = None
    
    def __post_init__(self):
        if self.first_seen is None:
            self.first_seen = time.time()
        if self.last_seen is None:
            self.last_seen = time.time()
        if self.confidence_history is None:
            self.confidence_history = deque(maxlen=30)
    
    @property
    def average_confidence(self) -> float:
        """Get average confidence score."""
        if not self.confidence_history:
            return 0.0
        return np.mean(list(self.confidence_history))
    
    @property
    def is_stable(self) -> bool:
        """Check if track is stable (consistently detected)."""
        return self.frames_since_seen < 5 and self.total_frames > 10