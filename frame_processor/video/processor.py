"""
Video processing pipeline orchestrator.

This module coordinates the video processing pipeline with modular components,
resolution normalization, and real-time performance monitoring.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass
import cv2
import time
from collections import deque

from core.config import Config
from core.utils import get_logger, PerformanceTimer
from core.video_buffer import SAM2LongVideoBuffer
from .base import VideoTracker, VideoTrackingResult
from .tracker import SAM2RealtimeTracker
from .prompt_strategies import create_prompt_strategy, PromptStrategy
from pipeline.enhancer import ImageEnhancer

logger = get_logger(__name__)


@dataclass
class VideoProcessingResult:
    """Result from video processing pipeline."""
    frame_id: str
    tracking_result: VideoTrackingResult
    enhanced_crops: List[Dict[str, Any]]
    processing_time_ms: float
    original_resolution: Tuple[int, int]
    processed_resolution: Tuple[int, int]
    fps: float = 0.0


class VideoProcessor:
    """
    Orchestrates video processing pipeline with modular components.
    Supports different trackers and maintains real-time performance.
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize components
        logger.info("Initializing VideoProcessor components...")
        
        # Video buffer
        self.video_buffer = SAM2LongVideoBuffer(
            buffer_size=config.video_buffer_size,
            tree_branches=config.memory_tree_branches
        )
        
        # Create tracker based on configuration
        self.video_tracker = self._create_tracker(config)
        
        # Prompting strategy
        self.prompt_strategy = self._create_prompt_strategy(config)
        
        # Enhancement pipeline
        self.enhancer = None
        if hasattr(config, 'enhancement_enabled') and config.enhancement_enabled:
            try:
                self.enhancer = ImageEnhancer(config)
                logger.info("Image enhancement enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize enhancer: {e}")
        
        # Object tracking for API
        self.pending_identifications: Dict[str, Dict] = {}
        self.processed_objects: Dict[str, Any] = {}
        
        # Performance monitoring
        self.fps_monitor = FPSMonitor(target_fps=config.target_fps)
        self.last_quality_adjustment_time = 0
        self.quality_adjustment_cooldown = 30.0  # seconds
        
        # Stream management
        self.active_streams: Dict[str, StreamInfo] = {}
        self._stream_locks: Dict[str, asyncio.Lock] = {}
        
        logger.info(f"VideoProcessor initialized with {self.video_tracker.name}")
    
    def _create_tracker(self, config: Config) -> VideoTracker:
        """Factory method to create video tracker."""
        tracker_types = {
            "sam2_realtime": SAM2RealtimeTracker,
            # Future: "yolo_track": YOLOVideoTracker,
            # Future: "grounded_sam2": GroundedSAM2Tracker,
        }
        
        tracker_type = config.video_tracker_type
        if tracker_type not in tracker_types:
            raise ValueError(f"Unknown tracker type: {tracker_type}")
        
        logger.info(f"Creating video tracker: {tracker_type}")
        return tracker_types[tracker_type](config)
    
    def _create_prompt_strategy(self, config: Config) -> PromptStrategy:
        """Factory method to create prompting strategy."""
        strategy_name = config.sam2_prompt_strategy
        
        # Strategy-specific parameters
        strategy_params = {
            "grid": {
                "points_per_side": config.grid_prompt_density,
                "margin": 50
            },
            "motion": {
                "threshold": 25.0,
                "min_area": 100
            },
            "saliency": {
                "num_prompts": 20
            }
        }
        
        params = strategy_params.get(strategy_name, {})
        logger.info(f"Creating prompt strategy: {strategy_name}")
        
        return create_prompt_strategy(strategy_name, **params)
    
    async def process_stream_frame(self, stream_id: str, frame: np.ndarray, 
                                   timestamp: int) -> VideoProcessingResult:
        """
        Process a frame from a video stream.
        Maintains real-time performance with quality adjustments.
        """
        # Ensure we have a lock for this stream
        if stream_id not in self._stream_locks:
            self._stream_locks[stream_id] = asyncio.Lock()
        
        async with self._stream_locks[stream_id]:
            with PerformanceTimer("video_process_frame", logger) as timer:
                # Track stream info
                if stream_id not in self.active_streams:
                    self.active_streams[stream_id] = StreamInfo(stream_id)
                
                stream_info = self.active_streams[stream_id]
                stream_info.frame_count += 1
                
                # Normalize resolution if needed
                original_shape = frame.shape
                normalized_frame = self._normalize_resolution(frame)
                
                # Add to buffer (original resolution for quality)
                await self.video_buffer.add_frame(stream_id, frame, timestamp)
                
                # Check if we need to initialize stream
                stream_status = self.video_tracker.get_stream_status(stream_id)
                logger.debug(f"Stream {stream_id} status: {stream_status}")
                
                if stream_status["status"] == "not_initialized":
                    logger.info(f"Initializing stream {stream_id} with SAM2 tracker")
                    # Generate initial prompts
                    prompts = await self.prompt_strategy.generate_prompts(normalized_frame)
                    logger.debug(f"Generated {len(prompts.get('points', []))} prompts")
                    tracking_result = await self.video_tracker.initialize_stream(
                        stream_id, normalized_frame, prompts
                    )
                    logger.info(f"Stream initialized with {tracking_result.object_count} objects")
                else:
                    # Continue tracking
                    logger.debug(f"Processing frame {stream_info.frame_count} for stream {stream_id}")
                    tracking_result = await self.video_tracker.process_frame(
                        stream_id, normalized_frame, timestamp
                    )
                
                logger.debug(f"Tracking result: {tracking_result.object_count} objects, "
                           f"{len(tracking_result.masks)} masks, "
                           f"processing time: {tracking_result.processing_time_ms}ms")
                
                # Update memory tree in buffer
                if tracking_result.masks:
                    self.video_buffer.update_memory_tree(
                        stream_id, tracking_result.masks, timestamp
                    )
                
                # Scale tracking results back to original resolution if needed
                if normalized_frame.shape != original_shape:
                    tracking_result = self._scale_tracking_result(
                        tracking_result, normalized_frame.shape, original_shape
                    )
                
                # Process tracked objects for enhancement
                enhanced_crops = await self._process_tracked_objects(
                    frame, tracking_result, stream_id, timestamp
                )
                
                # Update FPS monitor
                self.fps_monitor.update()
                current_fps = self.fps_monitor.current_fps
                
                # Adjust quality if needed
                if self.fps_monitor.is_below_target():
                    await self._adjust_quality_settings()
                
                # Create result
                result = VideoProcessingResult(
                    frame_id=f"{stream_id}_{timestamp}",
                    tracking_result=tracking_result,
                    enhanced_crops=enhanced_crops,
                    processing_time_ms=timer.elapsed_ms,
                    original_resolution=(original_shape[1], original_shape[0]),
                    processed_resolution=(normalized_frame.shape[1], normalized_frame.shape[0]),
                    fps=current_fps
                )
                
                # Update stream info
                stream_info.last_fps = current_fps
                stream_info.total_processing_time += timer.elapsed_ms
                
                # Log performance periodically
                if stream_info.frame_count % 30 == 0:
                    avg_time = stream_info.total_processing_time / stream_info.frame_count
                    logger.info(f"Stream {stream_id} - Frame: {stream_info.frame_count}, "
                               f"FPS: {current_fps:.1f}, Avg time: {avg_time:.1f}ms")
                
                return result
    
    def _normalize_resolution(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame resolution for processing."""
        h, w = frame.shape[:2]
        max_dim = max(h, w)
        
        # Check if resizing is needed
        if max_dim <= self.config.processing_resolution:
            return frame
        
        # Calculate scale factor
        scale = self.config.processing_resolution / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Ensure dimensions are even (required for some codecs)
        new_w = new_w if new_w % 2 == 0 else new_w - 1
        new_h = new_h if new_h % 2 == 0 else new_h - 1
        
        # Resize frame
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        logger.debug(f"Normalized resolution from {w}x{h} to {new_w}x{new_h}")
        return resized
    
    def _scale_tracking_result(self, result: VideoTrackingResult, 
                              normalized_shape: tuple, original_shape: tuple) -> VideoTrackingResult:
        """Scale tracking results back to original resolution."""
        if normalized_shape == original_shape:
            return result
        
        # Calculate scale factors
        scale_y = original_shape[0] / normalized_shape[0]
        scale_x = original_shape[1] / normalized_shape[1]
        
        # Scale tracks
        scaled_tracks = []
        for track in result.tracks:
            scaled_track = track.copy()
            if 'bbox' in scaled_track:
                bbox = scaled_track['bbox']
                scaled_track['bbox'] = [
                    int(bbox[0] * scale_x),
                    int(bbox[1] * scale_y),
                    int(bbox[2] * scale_x),
                    int(bbox[3] * scale_y)
                ]
            scaled_tracks.append(scaled_track)
        
        # Scale masks
        scaled_masks = []
        for mask in result.masks:
            scaled_mask = mask.copy()
            if 'segmentation' in scaled_mask:
                # Resize mask back to original resolution
                seg = mask['segmentation'].astype(np.uint8)
                scaled_seg = cv2.resize(
                    seg, 
                    (original_shape[1], original_shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                )
                scaled_mask['segmentation'] = scaled_seg.astype(bool)
                scaled_mask['area'] = int(scaled_seg.sum())
            scaled_masks.append(scaled_mask)
        
        # Return scaled result
        return VideoTrackingResult(
            masks=scaled_masks,
            tracks=scaled_tracks,
            object_count=result.object_count,
            processing_time_ms=result.processing_time_ms,
            frame_number=result.frame_number,
            confidence_scores=result.confidence_scores
        )
    
    async def _process_tracked_objects(self, frame: np.ndarray, 
                                       tracking_result: VideoTrackingResult,
                                       stream_id: str, timestamp: int) -> List[Dict]:
        """Extract and enhance tracked objects."""
        enhanced_crops = []
        total_enhancement_time = 0.0
        
        # Skip entirely if enhancement is disabled and no API processing
        if not self.enhancer and not (hasattr(self.config, 'use_serpapi') and self.config.use_serpapi):
            return enhanced_crops
        
        for track in tracking_result.tracks:
            track_id = track["track_id"]
            
            # Check if we've already processed this object recently
            if track_id in self.processed_objects:
                last_processed = self.processed_objects[track_id]["timestamp"]
                if timestamp - last_processed < self.config.reprocess_interval_ms:
                    continue
            
            # Extract object crop
            bbox = track["bbox"]
            crop = self._extract_crop(frame, bbox)
            
            # Skip if crop is too small
            if crop.shape[0] < 32 or crop.shape[1] < 32:
                continue
            
            # Enhance if enabled
            enhanced = crop
            enhancement_start = time.time()
            if self.enhancer:
                try:
                    enhanced = self.enhancer.enhance_roi(crop)
                except Exception as e:
                    logger.error(f"Enhancement failed for object {track_id}: {e}")
            enhancement_time = (time.time() - enhancement_start) * 1000
            total_enhancement_time += enhancement_time
            
            # Prepare for identification
            crop_data = {
                "object_id": track_id,
                "track_data": track,
                "original_crop": crop,
                "enhanced_crop": enhanced,
                "timestamp": timestamp,
                "bbox": bbox,
                "confidence": track.get("confidence", 0.0)
            }
            
            enhanced_crops.append(crop_data)
            
            # Mark as pending identification
            self.pending_identifications[track_id] = crop_data
            
        # Record total enhancement time
        if total_enhancement_time > 0:
            from core.performance_monitor import get_performance_monitor
            monitor = get_performance_monitor()
            monitor.record_timing('object_enhancement_total', total_enhancement_time)
            
        return enhanced_crops
    
    def _extract_crop(self, frame: np.ndarray, bbox: List[int], 
                      padding: float = 0.1) -> np.ndarray:
        """Extract object crop with padding."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # Add padding
        pad_x = int((x2 - x1) * padding)
        pad_y = int((y2 - y1) * padding)
        
        # Clip to frame bounds
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        return frame[y1:y2, x1:x2]
    
    async def _adjust_quality_settings(self):
        """Dynamically adjust settings to maintain target FPS."""
        current_fps = self.fps_monitor.current_fps
        target_fps = self.config.target_fps
        current_time = time.time()
        
        # Check cooldown to avoid spam
        if current_time - self.last_quality_adjustment_time < self.quality_adjustment_cooldown:
            return
        
        if current_fps < target_fps * 0.5:  # Below 50% of target (more reasonable threshold)
            self.last_quality_adjustment_time = current_time
            logger.warning(f"FPS dropped to {current_fps:.1f}, adjusting quality...")
            
            # Reduce prompt density
            if hasattr(self.prompt_strategy, 'points_per_side'):
                old_density = self.prompt_strategy.points_per_side
                self.prompt_strategy.points_per_side = max(8, old_density - 2)
                logger.info(f"Reduced prompt density from {old_density} to {self.prompt_strategy.points_per_side}")
            
            # Increase reprompt interval
            if hasattr(self.video_tracker, 'reprompt_interval'):
                old_interval = self.video_tracker.reprompt_interval
                self.video_tracker.reprompt_interval = min(120, old_interval + 10)
                logger.info(f"Increased reprompt interval from {old_interval} to {self.video_tracker.reprompt_interval}")
    
    async def get_pending_identifications(self) -> List[Dict]:
        """Get objects ready for Google Lens identification."""
        ready = []
        current_time = time.time()
        
        for track_id, crop_data in list(self.pending_identifications.items()):
            # Check if enough time has passed
            time_since_detection = (current_time * 1000 - crop_data["timestamp"]) / 1000.0
            if time_since_detection >= self.config.process_after_seconds:
                ready.append(crop_data)
                del self.pending_identifications[track_id]
                self.processed_objects[track_id] = {
                    "timestamp": crop_data["timestamp"],
                    "processed_at": current_time * 1000
                }
        
        return ready
    
    async def cleanup_stream(self, stream_id: str):
        """Clean up resources for a stream."""
        logger.info(f"Cleaning up stream {stream_id}")
        
        # Clean up video buffer
        await self.video_buffer.cleanup_stream(stream_id)
        
        # Clean up tracker
        await self.video_tracker.cleanup_stream(stream_id)
        
        # Clean up pending identifications for this stream
        to_remove = [k for k in self.pending_identifications.keys() 
                     if k.startswith(stream_id)]
        for key in to_remove:
            del self.pending_identifications[key]
        
        # Clean up processed objects
        to_remove = [k for k in self.processed_objects.keys() 
                     if k.startswith(stream_id)]
        for key in to_remove:
            del self.processed_objects[key]
        
        # Clean up stream info
        if stream_id in self.active_streams:
            stream_info = self.active_streams[stream_id]
            logger.info(f"Stream {stream_id} stats - Frames: {stream_info.frame_count}, "
                       f"Avg FPS: {stream_info.last_fps:.1f}")
            del self.active_streams[stream_id]
        
        # Clean up lock
        if stream_id in self._stream_locks:
            del self._stream_locks[stream_id]
    
    def get_stream_info(self, stream_id: str) -> Optional['StreamInfo']:
        """Get information about a stream."""
        return self.active_streams.get(stream_id)
    
    def get_all_streams(self) -> Dict[str, 'StreamInfo']:
        """Get information about all active streams."""
        return self.active_streams.copy()


@dataclass
class StreamInfo:
    """Information about an active stream."""
    stream_id: str
    frame_count: int = 0
    start_time: float = None
    last_fps: float = 0.0
    total_processing_time: float = 0.0
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.time()
    
    @property
    def duration(self) -> float:
        """Stream duration in seconds."""
        return time.time() - self.start_time
    
    @property
    def average_fps(self) -> float:
        """Average FPS over stream lifetime."""
        if self.duration > 0:
            return self.frame_count / self.duration
        return 0.0


class FPSMonitor:
    """Monitor and track FPS performance."""
    
    def __init__(self, target_fps: int = 15, window_size: int = 30):
        self.target_fps = target_fps
        self.window_size = window_size
        self.frame_times: deque = deque(maxlen=window_size)
        self.last_time = None
        
    def update(self):
        """Update with new frame timestamp."""
        current_time = time.time()
        if self.last_time is not None:
            self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
    
    @property
    def current_fps(self) -> float:
        """Calculate current FPS."""
        if len(self.frame_times) < 2:
            return 0.0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def is_below_target(self) -> bool:
        """Check if FPS is below target."""
        return self.current_fps < self.target_fps * 0.9  # 90% threshold