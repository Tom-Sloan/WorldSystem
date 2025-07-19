"""
Rerun visualization client.

This module provides a clean interface to the enhanced Rerun visualizer,
preserving all existing visualization functionality while fitting into
the modular architecture.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path

from .enhanced_visualizer import (
    EnhancedRerunVisualizer, 
    ViewMode,
    ProcessedObjectInfo,
    show_both_pages,
    show_live_page_only,
    show_process_page_only
)
from detection.base import Detection
from tracking.base import TrackedObject
from core.utils import get_logger
from core.config import Config

import rerun as rr


logger = get_logger(__name__)


class RerunClient:
    """
    Wrapper for enhanced Rerun visualization.
    
    This provides a clean interface to the existing enhanced visualizer
    while fitting into the new modular architecture.
    """
    
    def __init__(self, config: Config):
        """
        Initialize Rerun client.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.enabled = config.rerun_enabled
        
        if not self.enabled:
            logger.info("Rerun visualization disabled")
            return
        
        # Initialize Rerun
        logger.info("Initializing Rerun visualization")
        
        # Initialize Rerun SDK (matching original setup)
        rr.init("frame_processor", spawn=False)
        
        # Connect to viewer
        try:
            logger.info(f"Connecting to Rerun viewer at {config.rerun_connect_url}")
            rr.connect_grpc(config.rerun_connect_url)
            logger.info("Connected to Rerun viewer successfully")
        except Exception as e:
            logger.warning(f"Failed to connect to Rerun viewer: {e}")
            logger.info("Will continue logging data - viewer can connect later")
        
        # Create enhanced visualizer instance
        self.visualizer = EnhancedRerunVisualizer()
        
        # Set default view mode
        self.set_view_mode(ViewMode.BOTH)
        
        logger.info("Rerun visualization initialized with enhanced visualizer")
    
    def log_frame(self, frame: np.ndarray, detections: List[Detection], 
                  active_tracks: List[TrackedObject], frame_number: int,
                  timestamp_ns: Optional[int] = None):
        """
        Log frame with detections and tracking info.
        
        Args:
            frame: Current video frame
            detections: List of detections from detector
            active_tracks: List of active tracked objects
            frame_number: Sequential frame number
            timestamp_ns: Optional timestamp in nanoseconds
        """
        if not self.enabled:
            return
        
        # Convert our Detection objects to format expected by visualizer
        detection_info = []
        for det in detections:
            detection_info.append({
                'bbox': det.bbox,
                'class_name': det.class_name,
                'confidence': det.confidence
            })
        
        # Convert active_tracks list to dictionary for enhanced visualizer
        tracked_objects_dict = {}
        if active_tracks:
            for track in active_tracks:
                tracked_objects_dict[track.id] = track
        
        # Log frame with overlays using enhanced visualizer
        self.visualizer.log_frame_with_overlays(
            frame=frame,
            tracked_objects=tracked_objects_dict,
            frame_number=frame_number,
            timestamp_ns=timestamp_ns
        )
        
        # Also log raw detections for analysis
        if detection_info:
            self._log_yolo_detections(detection_info, frame.shape[:2])
    
    def _log_yolo_detections(self, detections: List[Dict], frame_shape: tuple):
        """Log YOLO detections in Rerun format."""
        if not detections:
            return
        
        boxes_2d = []
        labels = []
        colors = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Convert to center + half-size format
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            half_width = (x2 - x1) / 2
            half_height = (y2 - y1) / 2
            
            boxes_2d.append([center_x, center_y, half_width, half_height])
            labels.append(f"{det['class_name']} ({det['confidence']:.2f})")
            
            # Color based on class name
            color_hash = hash(det['class_name']) % 360
            colors.append([color_hash, 255, 255])  # HSV
        
        # Log to Rerun
        try:
            boxes_array = np.array(boxes_2d, dtype=np.float32)
            centers = boxes_array[:, :2]
            half_sizes = boxes_array[:, 2:]
            
            rr.log(
                "detections/yolo",
                rr.Boxes2D(
                    centers=centers,
                    half_sizes=half_sizes,
                    labels=labels,
                    colors=colors
                )
            )
        except Exception as e:
            logger.error(f"Failed to log YOLO detections: {e}")
    
    def log_processed_object(self, track: TrackedObject):
        """
        Log a processed object with API results.
        
        Args:
            track: TrackedObject with api_result populated
        """
        if not self.enabled or not track.api_result:
            return
        
        # Create ProcessedObjectInfo for gallery
        if track.best_frame is not None:
            info = ProcessedObjectInfo(
                track_id=track.id,
                class_name=track.class_name,
                enhanced_image=track.best_frame,
                dimensions=track.api_result.get('dimensions'),
                confidence=track.confidence,
                product_name=track.api_result.get('product_name'),
                processing_time=0.0,  # Could track this if needed
                timestamp=track.created_at
            )
            
            # Add to gallery
            self.visualizer.add_to_gallery(info)
    
    def log_scene_scale(self, scale_info: Dict[str, Any]):
        """
        Log scene scale information.
        
        Args:
            scale_info: Dictionary with scale_factor, confidence, etc.
        """
        if not self.enabled:
            return
        
        # Log scale info as text document
        scale_text = f"""### Scene Scale Estimation

**Scale Factor:** {scale_info.get('scale_factor', 0):.4f} m/unit
**Confidence:** {scale_info.get('confidence', 0):.2%}
**Based on:** {scale_info.get('num_estimates', 0)} objects

**Average Dimensions:**
- Width: {scale_info.get('avg_dimensions_m', {}).get('width', 0):.3f} m
- Height: {scale_info.get('avg_dimensions_m', {}).get('height', 0):.3f} m
- Depth: {scale_info.get('avg_dimensions_m', {}).get('depth', 0):.3f} m
"""
        
        try:
            rr.log("/scale", rr.TextDocument(scale_text, media_type=rr.MediaType.MARKDOWN))
        except Exception as e:
            logger.error(f"Failed to log scene scale: {e}")
    
    def set_view_mode(self, mode: ViewMode):
        """
        Change visualization view mode.
        
        Args:
            mode: ViewMode.BOTH, ViewMode.LIVE_ONLY, or ViewMode.PROCESS_ONLY
        """
        if not self.enabled:
            return
        
        self.visualizer.current_view_mode = mode
        
        if mode == ViewMode.BOTH:
            show_both_pages(self.visualizer)
        elif mode == ViewMode.LIVE_ONLY:
            show_live_page_only(self.visualizer)
        elif mode == ViewMode.PROCESS_ONLY:
            show_process_page_only(self.visualizer)
        
        logger.info(f"Rerun view mode changed to: {mode.value}")
    
    def log_text(self, path: str, text: str, level: str = "INFO"):
        """
        Log text message to Rerun.
        
        Args:
            path: Rerun path for the log
            text: Message text
            level: Log level (INFO, WARNING, ERROR, etc.)
        """
        if not self.enabled:
            return
        
        try:
            rr.log(path, rr.TextLog(text, level=level))
        except Exception as e:
            logger.error(f"Failed to log text to Rerun: {e}")
    
    def log_metric(self, path: str, value: float):
        """
        Log scalar metric to Rerun.
        
        Args:
            path: Rerun path for the metric
            value: Scalar value
        """
        if not self.enabled:
            return
        
        try:
            rr.log(path, rr.Scalar(value))
        except Exception as e:
            logger.error(f"Failed to log metric to Rerun: {e}")