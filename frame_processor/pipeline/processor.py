"""
Main frame processing pipeline with factory pattern.

This module orchestrates the entire frame processing workflow,
using factories to create swappable components based on configuration.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Type
from dataclasses import dataclass
import numpy as np

from core.config import Config
from core.utils import get_logger, PerformanceTimer, get_ntp_time_ns
from detection.base import Detector, Detection
from detection.yolo import YOLODetector
from tracking.base import Tracker, TrackedObject
from tracking.iou_tracker import IOUTracker
from external.api_client import APIClient
from visualization.rerun_client import RerunClient
from .scorer import FrameScorer
from .enhancer import ImageEnhancer


logger = get_logger(__name__)


@dataclass
class ProcessingResult:
    """Result of frame processing."""
    frame_number: int
    timestamp_ns: int
    detections: List[Detection]
    tracks_for_api: List[TrackedObject]
    processing_time_ms: float
    detection_count: int
    active_track_count: int


class ComponentFactory:
    """
    Factory for creating processing components based on configuration.
    
    This enables easy swapping of detection/tracking algorithms.
    """
    
    # Registry of available detectors
    DETECTORS: Dict[str, Type[Detector]] = {
        "yolo": YOLODetector,
        # Future: "detectron2": Detectron2Detector,
        # Future: "mmdetection": MMDetector,
    }
    
    # Registry of available trackers
    TRACKERS: Dict[str, Type[Tracker]] = {
        "iou": IOUTracker,
        # Future: "sort": SORTTracker,
        # Future: "deepsort": DeepSORTTracker,
        # Future: "bytetrack": ByteTracker,
    }
    
    @classmethod
    def create_detector(cls, config: Config) -> Detector:
        """
        Create detector instance based on configuration.
        
        Args:
            config: Application configuration
            
        Returns:
            Detector instance
            
        Raises:
            ValueError: If detector type not found
        """
        detector_type = config.detector_type.lower()
        
        if detector_type not in cls.DETECTORS:
            raise ValueError(
                f"Unknown detector type: {detector_type}. "
                f"Available: {list(cls.DETECTORS.keys())}"
            )
        
        detector_class = cls.DETECTORS[detector_type]
        
        # Create detector with appropriate parameters
        if detector_type == "yolo":
            return detector_class(
                model_path=config.detector_model,
                confidence=config.detector_confidence,
                device=config.detector_device
            )
        else:
            # Future detectors would have their own parameter mappings
            raise NotImplementedError(f"Factory not implemented for {detector_type}")
    
    @classmethod
    def create_tracker(cls, config: Config) -> Tracker:
        """
        Create tracker instance based on configuration.
        
        Args:
            config: Application configuration
            
        Returns:
            Tracker instance
            
        Raises:
            ValueError: If tracker type not found
        """
        tracker_type = config.tracker_type.lower()
        
        if tracker_type not in cls.TRACKERS:
            raise ValueError(
                f"Unknown tracker type: {tracker_type}. "
                f"Available: {list(cls.TRACKERS.keys())}"
            )
        
        tracker_class = cls.TRACKERS[tracker_type]
        
        # Create tracker with appropriate parameters
        if tracker_type == "iou":
            return tracker_class(
                iou_threshold=config.tracker_iou_threshold,
                max_lost=config.tracker_max_lost,
                process_after_seconds=config.process_after_seconds,
                max_tracks=config.tracker_max_tracks
            )
        else:
            # Future trackers would have their own parameter mappings
            raise NotImplementedError(f"Factory not implemented for {tracker_type}")


class FrameProcessor:
    """
    Main frame processing pipeline.
    
    This orchestrates the entire workflow from detection through API processing,
    maintaining the same functionality as the original but with better modularity.
    """
    
    def __init__(self, config: Config):
        """
        Initialize frame processor with all components.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.frame_number = 0
        
        logger.info("Initializing FrameProcessor components...")
        
        # Create components using factory
        self.detector = ComponentFactory.create_detector(config)
        self.tracker = ComponentFactory.create_tracker(config)
        
        # Create other components
        self.scorer = FrameScorer()
        self.enhancer = ImageEnhancer(config)
        self.api_client = APIClient(config)
        self.rerun_client = RerunClient(config) if config.rerun_enabled else None
        
        # Connect scorer to tracker
        if hasattr(self.tracker, 'set_frame_scorer'):
            self.tracker.set_frame_scorer(self.scorer)
        
        # Load API caches
        self.api_client.load_caches()
        
        # Warmup detector if supported
        if hasattr(self.detector, 'warmup'):
            self.detector.warmup()
        
        # Processing statistics
        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'tracks_created': 0,
            'api_calls_made': 0,
            'api_calls_cached': 0,
            'total_processing_time_ms': 0
        }
        
        logger.info(
            f"FrameProcessor initialized - "
            f"Detector: {self.detector.name}, Tracker: {self.tracker.name}"
        )
    
    async def process_frame(self, frame: np.ndarray, 
                          timestamp_ns: Optional[int] = None) -> ProcessingResult:
        """
        Process a single frame through the entire pipeline.
        
        Args:
            frame: Input video frame
            timestamp_ns: Optional timestamp in nanoseconds
            
        Returns:
            ProcessingResult with all processing outputs
        """
        start_time = time.time()
        self.frame_number += 1
        
        if timestamp_ns is None:
            timestamp_ns = get_ntp_time_ns()
        
        with PerformanceTimer(f"frame_{self.frame_number}_total", logger):
            # Step 1: Run detection
            detections = await self.detector.detect(frame)
            self.stats['total_detections'] += len(detections)
            
            # Step 2: Update tracking
            tracks_ready = self.tracker.update(detections, frame, self.frame_number)
            
            # Step 3: Process tracks ready for API
            api_tasks = []
            for track in tracks_ready:
                if track.best_frame is not None and not track.api_processed:
                    api_tasks.append(self._process_track_for_api(track))
            
            # Run API processing concurrently
            if api_tasks:
                await asyncio.gather(*api_tasks)
            
            # Step 4: Update visualization if enabled
            if self.rerun_client:
                active_tracks = self.tracker.get_active_tracks()
                self.rerun_client.log_frame(
                    frame, detections, active_tracks, 
                    self.frame_number, timestamp_ns
                )
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            self.stats['frames_processed'] += 1
            self.stats['total_processing_time_ms'] += processing_time_ms
            
            # Log performance periodically
            if self.frame_number % 100 == 0:
                self._log_performance_stats()
            
            return ProcessingResult(
                frame_number=self.frame_number,
                timestamp_ns=timestamp_ns,
                detections=detections,
                tracks_for_api=tracks_ready,
                processing_time_ms=processing_time_ms,
                detection_count=len(detections),
                active_track_count=len(self.tracker.get_active_tracks())
            )
    
    async def _process_track_for_api(self, track: TrackedObject):
        """
        Process a track through the API pipeline.
        
        Args:
            track: Track to process
        """
        try:
            logger.info(f"Processing track #{track.id} ({track.class_name}) for API")
            
            # Mark as being processed
            track.is_being_processed = True
            start_time = time.time()
            
            # Enhance image if enabled
            enhanced_image = self.enhancer.enhance_roi(track.best_frame)
            
            # Process through API client
            api_result = await self.api_client.process_object_for_dimensions(
                enhanced_image, track.class_name
            )
            
            # Store result in track
            track.api_result = api_result
            track.api_processed = True
            track.is_being_processed = False
            track.processing_time = time.time() - start_time
            
            # Extract dimensions and products if available
            if api_result.get('dimensions'):
                track.estimated_dimensions = api_result['dimensions']
            if api_result.get('all_products'):
                track.identified_products = api_result['all_products']
            
            # Update stats
            self.stats['api_calls_made'] += 1
            
            # Log to Rerun if enabled
            if self.rerun_client and api_result.get('dimensions'):
                self.rerun_client.log_processed_object(track)
            
            logger.info(
                f"API processing complete for track #{track.id}: "
                f"Product: {api_result.get('product_name', 'Unknown')}, "
                f"Dimensions: {api_result.get('dimensions', 'None')}"
            )
            
        except Exception as e:
            logger.error(f"API processing failed for track #{track.id}: {e}")
            track.api_processed = True  # Mark as processed to avoid retry
            track.api_result = {"error": str(e)}
            track.is_being_processed = False
    
    def update_analysis_mode(self, mode: str):
        """
        Update the analysis mode (e.g., "none", "yolo", etc.).
        
        Args:
            mode: New analysis mode
        """
        logger.info(f"Updating analysis mode to: {mode}")
        
        if mode.lower() == "none":
            # Disable detection
            self.config.detection_enabled = False
        else:
            # Enable detection
            self.config.detection_enabled = True
            
            # If mode specifies a different detector, could switch here
            # For now, just log the mode change
    
    def _log_performance_stats(self):
        """Log performance statistics."""
        if self.stats['frames_processed'] == 0:
            return
        
        avg_time = self.stats['total_processing_time_ms'] / self.stats['frames_processed']
        avg_detections = self.stats['total_detections'] / self.stats['frames_processed']
        
        logger.info(
            f"Performance stats - "
            f"Frames: {self.stats['frames_processed']}, "
            f"Avg time: {avg_time:.1f}ms, "
            f"Avg detections: {avg_detections:.1f}, "
            f"API calls: {self.stats['api_calls_made']}, "
            f"Active tracks: {len(self.tracker.get_active_tracks())}"
        )
        
        # Log to Rerun if enabled
        if self.rerun_client:
            self.rerun_client.log_metric("/metrics/avg_processing_time_ms", avg_time)
            self.rerun_client.log_metric("/metrics/avg_detections", avg_detections)
            self.rerun_client.log_metric("/metrics/active_tracks", 
                                       len(self.tracker.get_active_tracks()))
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with current stats
        """
        stats = self.stats.copy()
        
        # Add component-specific stats
        stats['detector_name'] = self.detector.name
        stats['tracker_name'] = self.tracker.name
        stats['tracker_memory_mb'] = self.tracker.get_memory_usage() / (1024 * 1024)
        stats['api_cache_stats'] = self.api_client.get_cache_stats()
        stats['enhancement_stats'] = self.enhancer.get_enhancement_stats()
        
        return stats
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up FrameProcessor resources...")
        
        # Cleanup old tracks
        if hasattr(self.tracker, 'cleanup_old_tracks'):
            removed = self.tracker.cleanup_old_tracks()
            logger.info(f"Removed {removed} old tracks")
        
        # Cleanup API client session
        if hasattr(self.api_client, 'close'):
            await self.api_client.close()
        
        # Log final stats
        self._log_performance_stats()
        
        logger.info("FrameProcessor cleanup complete")