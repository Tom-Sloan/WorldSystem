"""
Rerun visualization client.

This module provides a clean interface to the enhanced Rerun visualizer,
preserving all existing visualization functionality while fitting into
the modular architecture.
"""

import numpy as np
import cv2
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
        
        # Frame buffer for grid view
        self.frame_buffer = []
        self.max_buffer_size = 12  # 3x4 grid
        self.grid_update_interval = 30  # Update grid every 30 frames (1 second at 30fps)
        self.frames_since_grid_update = 0
        self.last_grid_update_time = 0
        
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
        
        # Add frame to buffer for grid view
        frame_copy = frame.copy()
        self.frame_buffer.append((frame_copy, frame_number, timestamp_ns))
        
        # Keep buffer size limited
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
        
        # Update grid view periodically (time-based for smoother updates)
        import time
        current_time = time.time()
        self.frames_since_grid_update += 1
        
        # Update grid based on time (1 second) OR frame count
        if (current_time - self.last_grid_update_time >= 1.0 or 
            self.frames_since_grid_update >= self.grid_update_interval):
            self._log_grid_view()
            self.frames_since_grid_update = 0
            self.last_grid_update_time = current_time
        
        # Clear previous frame's detections to prevent persistence
        rr.log("/page1/live/overlays", rr.Clear(recursive=False))
        rr.log("detections/yolo", rr.Clear(recursive=False))
        rr.log("detections/sam_edges", rr.Clear(recursive=False))
        
        # For grid view, we'll minimize overlays on individual frames
        # Just log the current frame without heavy overlays
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rr.log("/page1/live/camera", rr.Image(frame_rgb))
        
        # Check if we're using SAM/FastSAM (many detections) or YOLO (few detections)
        is_segmentation = self.config.detector_type in ['sam', 'fastsam']
        
        # Only show minimal detection info for cleaner view
        if active_tracks and len(active_tracks) < 5:  # Only show if few tracks
            # Convert active_tracks list to dictionary
            tracked_objects_dict = {}
            for track in active_tracks:
                tracked_objects_dict[track.id] = track
            
            # Log minimal overlays
            self._log_minimal_overlays(tracked_objects_dict, frame_number, timestamp_ns)
    
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
    
    def _filter_sam_detections(self, detections: List[Detection], 
                               frame_shape: tuple) -> List[Detection]:
        """
        Filter SAM detections to reduce clutter.
        
        Args:
            detections: List of all SAM detections
            frame_shape: (height, width) of frame
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        filtered = []
        frame_area = frame_shape[0] * frame_shape[1]
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            bbox_area = (x2 - x1) * (y2 - y1)
            
            # Filter criteria:
            # 1. Minimum size: at least 2% of frame area
            # 2. Maximum size: no more than 50% of frame area
            # 3. Confidence threshold higher for SAM
            size_ratio = bbox_area / frame_area
            
            if (size_ratio >= 0.02 and 
                size_ratio <= 0.5 and 
                det.confidence >= 0.85):  # Higher threshold for SAM
                filtered.append(det)
        
        # Sort by area (largest first) and take top N
        filtered.sort(key=lambda d: (d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]), 
                     reverse=True)
        
        # Limit to top 10 segments
        return filtered[:10]
    
    def _log_sam_detections(self, detections: List[Dict], frame_shape: tuple):
        """
        Log SAM detections with edge-only visualization.
        
        Uses contours instead of filled boxes for cleaner visualization.
        """
        if not detections:
            return
        
        import cv2
        
        # Create an edge map for visualization
        edge_map = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            
            # Draw rectangle edges
            color = self._get_segment_color(i, det['confidence'])
            cv2.rectangle(edge_map, (x1, y1), (x2, y2), color, 2)
            
            # Add subtle label
            label = f"{det['class_name']} {det['confidence']:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            
            # Background for text (semi-transparent effect)
            cv2.rectangle(edge_map, 
                         (x1, y1 - label_size[1] - 4),
                         (x1 + label_size[0], y1),
                         color, -1)
            cv2.putText(edge_map, label,
                       (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                       (255, 255, 255), 1)
        
        # Log edge map as separate image
        try:
            rr.log(
                "detections/sam_edges",
                rr.Image(edge_map)
            )
        except Exception as e:
            logger.error(f"Failed to log SAM detections: {e}")
    
    def _get_segment_color(self, index: int, confidence: float) -> tuple:
        """Get color for segment based on index and confidence."""
        # Use HSV color space for better distribution
        hue = (index * 30) % 180  # Distribute hues
        saturation = int(255 * confidence)  # Higher confidence = more saturated
        value = 255
        
        # Convert HSV to BGR for OpenCV
        hsv = np.array([[[hue, saturation, value]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return tuple(int(x) for x in bgr[0, 0])
    
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
    
    def _log_grid_view(self):
        """Create and log a grid view of recent frames."""
        if not self.frame_buffer:
            return
        
        # Create a 3x4 grid
        grid_rows = 3
        grid_cols = 4
        
        # Get dimensions from first frame
        sample_frame = self.frame_buffer[0][0]
        h, w = sample_frame.shape[:2]
        
        # Scale down frames for grid
        scale_factor = 0.33  # Make each frame 1/3 size
        cell_h = int(h * scale_factor)
        cell_w = int(w * scale_factor)
        
        # Create grid canvas
        grid_h = cell_h * grid_rows
        grid_w = cell_w * grid_cols
        grid_canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        # Fill grid with frames
        for idx, (frame, frame_num, _) in enumerate(self.frame_buffer[-12:]):
            if idx >= grid_rows * grid_cols:
                break
            
            row = idx // grid_cols
            col = idx % grid_cols
            
            # Resize frame
            resized = cv2.resize(frame, (cell_w, cell_h))
            
            # Place in grid
            y1 = row * cell_h
            y2 = (row + 1) * cell_h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w
            
            grid_canvas[y1:y2, x1:x2] = resized
            
            # Add subtle frame number
            cv2.putText(grid_canvas, f"#{frame_num}", 
                       (x1 + 5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 1)
        
        # Convert to RGB and log
        grid_rgb = cv2.cvtColor(grid_canvas, cv2.COLOR_BGR2RGB)
        rr.log("/grid_view", rr.Image(grid_rgb))
    
    def _log_minimal_overlays(self, tracked_objects: Dict, frame_number: int, 
                              timestamp_ns: Optional[int] = None):
        """Log minimal overlays for cleaner visualization."""
        # Set time context
        rr.set_time("frame", sequence=frame_number)
        if timestamp_ns:
            rr.set_time("sensor_time", timestamp=timestamp_ns / 1e9)
        
        # Create minimal overlays - just small labels
        for track_id, track in tracked_objects.items():
            x1, y1, x2, y2 = track.bbox
            
            # Log just a small label at top-left of bbox
            label_text = f"#{track.id}"
            if hasattr(track, 'estimated_dimensions') and track.estimated_dimensions:
                label_text += " ✓"  # Checkmark for identified objects
            
            # Create a small text annotation
            rr.log(
                f"/page1/live/labels/track_{track_id}",
                rr.TextLog(label_text, position=[x1, y1])
            )