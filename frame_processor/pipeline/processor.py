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
import psutil
import torch

from core.config import Config
from core.utils import get_logger, PerformanceTimer, get_ntp_time_ns
from core.performance_monitor import DetailedTimer, get_performance_monitor
from detection.base import Detector, Detection
from detection.yolo import YOLODetector
from detection.sam import SAMDetector
from detection.fastsam import FastSAMDetector
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
        "sam": SAMDetector,
        "fastsam": FastSAMDetector,
        # Future: "detectron2": Detectron2Detector,
        # Future: "grounding_dino": GroundingDINODetector,
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
        elif detector_type == "sam":
            return detector_class(
                model_cfg=config.sam_model_cfg,
                model_path=config.sam_checkpoint_path,
                device=config.detector_device,
                points_per_side=config.sam_points_per_side,
                pred_iou_thresh=config.sam_pred_iou_thresh,
                stability_score_thresh=config.sam_stability_score_thresh,
                min_mask_region_area=config.sam_min_mask_region_area
            )
        elif detector_type == "fastsam":
            return detector_class(
                model_path=config.fastsam_model_path,
                device=config.detector_device,
                conf_threshold=config.fastsam_conf_threshold,
                iou_threshold=config.fastsam_iou_threshold,
                max_det=config.fastsam_max_det
            )
        else:
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
        Process a single frame through the entire pipeline with detailed timing.
        """
        monitor = get_performance_monitor()
        frame_breakdown = {}
        
        with DetailedTimer("total_frame_processing"):
            start_time = time.time()
            self.frame_number += 1
            
            if timestamp_ns is None:
                timestamp_ns = get_ntp_time_ns()
            
            # Step 1: Run detection with timing
            if self.config.detection_enabled:
                with DetailedTimer("detection") as timer:
                    detections = await self.detector.detect(frame)
                    self.stats['total_detections'] += len(detections)
                frame_breakdown['detection'] = timer.elapsed_ms
                
                # Log detection details
                monitor.add_event(
                    f"Detected {len(detections)} objects ({timer.elapsed_ms:.1f}ms)",
                    "info" if timer.elapsed_ms < 50 else "warning"
                )
            else:
                detections = []
                frame_breakdown['detection'] = 0.0
            
            # Step 2: Update tracking with timing
            with DetailedTimer("tracking") as timer:
                tracks_ready = self.tracker.update(detections, frame, self.frame_number)
            frame_breakdown['tracking'] = timer.elapsed_ms
            
            # Step 3: Process tracks ready for API
            api_tasks = []
            api_start = time.perf_counter()
            
            for track in tracks_ready:
                if track.best_frame is not None and not track.api_processed:
                    api_tasks.append(self._process_track_for_api(track))
            
            # Run API processing concurrently
            if api_tasks:
                with DetailedTimer(f"api_processing_{len(api_tasks)}_tracks"):
                    await asyncio.gather(*api_tasks)
                    monitor.add_event(
                        f"Processed {len(api_tasks)} tracks through API",
                        "success"
                    )
            
            frame_breakdown['api_processing'] = (time.perf_counter() - api_start) * 1000
            
            # Step 4: Update visualization with timing
            if self.rerun_client:
                with DetailedTimer("visualization") as timer:
                    active_tracks = self.tracker.get_active_tracks()
                    self.rerun_client.log_frame(
                        frame, detections, active_tracks, 
                        self.frame_number, timestamp_ns
                    )
                frame_breakdown['visualization'] = timer.elapsed_ms
            else:
                frame_breakdown['visualization'] = 0.0
            
            # Calculate total processing time
            processing_time_ms = (time.time() - start_time) * 1000
            frame_breakdown['total'] = processing_time_ms
            
            # Update monitor metrics
            monitor.record_frame_breakdown(frame_breakdown)
            monitor.update_metric('frames_processed', self.stats['frames_processed'])
            monitor.update_metric('active_tracks', len(self.tracker.get_active_tracks()))
            monitor.update_metric('detections_per_frame', 
                                self.stats['total_detections'] / max(1, self.stats['frames_processed']))
            
            # Update memory usage
            process = psutil.Process()
            monitor.update_metric('memory_mb', process.memory_info().rss / 1024 / 1024)
            
            if torch.cuda.is_available():
                monitor.update_metric('gpu_memory_mb', 
                                    torch.cuda.memory_allocated() / 1024 / 1024)
            
            # Calculate FPS
            self.stats['frames_processed'] += 1
            self.stats['total_processing_time_ms'] += processing_time_ms
            
            if self.frame_number % 30 == 0:  # Update FPS every 30 frames
                avg_time = self.stats['total_processing_time_ms'] / self.stats['frames_processed']
                fps = 1000.0 / avg_time if avg_time > 0 else 0
                monitor.update_metric('fps', fps)
            
            # Log performance periodically with better formatting
            if self.frame_number % 100 == 0:
                monitor.add_event(
                    f"Milestone: {self.frame_number} frames processed",
                    "success"
                )
            
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
        """Process a track through the API pipeline with detailed timing."""
        monitor = get_performance_monitor()
        
        try:
            track.is_being_processed = True
            start_time = time.time()
            
            # Enhance image with timing
            with DetailedTimer(f"enhancement_track_{track.id}"):
                enhanced_image = self.enhancer.enhance_roi(track.best_frame)
            
            # Log the enhanced object to Rerun IMMEDIATELY after enhancement
            if self.rerun_client:
                # Create a temporary track object with enhanced image
                enhanced_track = TrackedObject(
                    id=track.id,
                    class_name=track.class_name,
                    bbox=track.bbox,
                    confidence=track.confidence
                )
                enhanced_track.best_frame = enhanced_image
                
                # Log to grid
                self.rerun_client.log_enhanced_object(enhanced_track)
            
            # Process through API client with timing
            with DetailedTimer(f"api_call_track_{track.id}") as timer:
                api_result = await self.api_client.process_object_for_dimensions(
                    enhanced_image, track.class_name
                )
            
            # Check if it was a cache hit
            if timer.elapsed_ms < 10:  # Likely a cache hit if very fast
                monitor.update_metric('api_cache_hits', 
                                    self.stats.get('api_cache_hits', 0) + 1)
            
            # Store result
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
            monitor.update_metric('api_calls', self.stats['api_calls_made'])
            
            # Log success/failure
            if api_result.get('dimensions'):
                monitor.add_event(
                    f"✅ Track #{track.id}: {api_result.get('product_name', 'Unknown')}",
                    "success"
                )
            else:
                monitor.add_event(
                    f"❌ Track #{track.id}: Failed to get dimensions",
                    "warning"
                )
            
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
            monitor.add_event(f"API error for track #{track.id}: {str(e)}", "error")
            track.api_processed = True
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