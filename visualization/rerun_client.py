"""
Rerun visualization client.

This module provides a clean interface to the enhanced Rerun visualizer,
preserving all existing visualization functionality while fitting into
the modular architecture.
"""

import numpy as np
import cv2
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from .enhanced_visualizer import (
    EnhancedRerunVisualizer, 
    ViewMode,
    ProcessedObjectInfo,
    show_both_pages,
    show_live_page_only,
    show_process_page_only
)
# from detection.base import Detection
# from tracking.base import TrackedObject
# Temporary placeholder classes until we update visualization for video-only
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Detection:
    bbox: List[float]
    class_name: str
    confidence: float
    mask: Optional[Any] = None

@dataclass  
class TrackedObject:
    id: int
    bbox: List[float]
    class_name: str
    best_frame: Optional[np.ndarray] = None
    created_at: Optional[int] = None
    confidence: float = 0.0
    api_result: Optional[Dict] = None
import logging

logger = logging.getLogger(__name__)

import rerun as rr
import rerun.blueprint as rrb
import time
import av


# logger is already defined above


class RerunClient:
    """
    Wrapper for enhanced Rerun visualization.
    
    This provides a clean interface to the existing enhanced visualizer
    while fitting into the new modular architecture.
    """
    
    def __init__(self, rerun_enabled: bool = True, rerun_connect_url: str = "127.0.0.1:9876"):
        """
        Initialize Rerun client.
        
        Args:
            rerun_enabled: Whether Rerun visualization is enabled
            rerun_connect_url: URL to connect to Rerun viewer
        """
        self.enabled = rerun_enabled
        
        if not self.enabled:
            logger.info("Rerun visualization disabled")
            return
        
        # Initialize Rerun
        logger.info("Initializing Rerun visualization")
        
        # Initialize Rerun SDK (matching original setup)
        rr.init("frame_processor", spawn=False)
        
        # Connect to viewer
        try:
            logger.info(f"Connecting to Rerun viewer at {rerun_connect_url}")
            rr.connect_grpc(rerun_connect_url)
            logger.info("Connected to Rerun viewer successfully")
        except Exception as e:
            logger.warning(f"Failed to connect to Rerun viewer: {e}")
            logger.info("Will continue logging data - viewer can connect later")
        
        # Create enhanced visualizer instance
        self.visualizer = EnhancedRerunVisualizer()
        
        # Set default view mode
        self.set_view_mode(ViewMode.BOTH)
        
        # Setup blueprint for enhanced visualization
        self._setup_blueprint()
        
        # Frame buffer for grid view
        self.frame_buffer = []
        self.max_buffer_size = 12  # 3x4 grid
        self.grid_update_interval = 10  # Update grid every N frames
        self.frames_since_grid_update = 0
        
        # Enhanced objects buffer for grid view
        self.enhanced_objects_buffer = []  # List of (enhanced_image, track_id, timestamp)
        self.max_enhanced_objects = 12  # 4x3 grid
        self.grid_rows = 3
        self.grid_cols = 4
        self.grid_cell_size = (200, 200)  # Fixed size for grid cells
        
        logger.info("Rerun visualization initialized with enhanced visualizer")
        
        # Initialize video streams and encoders
        self.video_streams = {}  # websocket_id -> av.CodecContext
        self.video_containers = {}  # websocket_id -> av.container.OutputContainer
        self.video_frame_counts = {}  # websocket_id -> frame count
    
    def log_frame(self, frame: np.ndarray, detections: List[Detection], 
                  active_tracks: List[TrackedObject], frame_number: int,
                  timestamp_ns: Optional[int] = None, websocket_id: str = "default"):
        """
        Log frame with SAM-style segmentation visualization using video streaming.
        """
        if not self.enabled:
            return
        
        # Set time context for all subsequent logs
        rr.set_time_sequence("frame", frame_number)
        if timestamp_ns:
            rr.set_time_nanos("sensor_time", timestamp_ns)
        
        # Store frame with detections for gallery
        self.frame_buffer.append((frame.copy(), frame_number, timestamp_ns, detections))
        if len(self.frame_buffer) > 20:
            self.frame_buffer.pop(0)
        
        # Create composite image with segmentation overlay
        if detections:
            composite_frame = self._create_composite_with_detections(detections, frame)
        else:
            composite_frame = frame.copy()
        
        # Convert to RGB for video streaming
        composite_rgb = cv2.cvtColor(composite_frame, cv2.COLOR_BGR2RGB)
        
        # Initialize video stream if needed
        self._init_video_stream_if_needed(websocket_id, composite_rgb.shape[:2])
        
        # Stream the composite frame
        self._stream_video_frame(websocket_id, composite_rgb, frame_number, timestamp_ns)
        
        # Update gallery periodically
        if frame_number % 30 == 0:  # Every 30 frames
            self._log_segmentation_gallery()
        
        # Log statistics - always update
        self._log_segmentation_statistics(detections, active_tracks, frame_number)
        
        # No need for explicit flush - Rerun handles this
    
    # YOLO detection logging removed - using SAM2 exclusively
    
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
    
    def _create_composite_with_detections(self, detections: List[Detection], frame: np.ndarray) -> np.ndarray:
        """
        Create a composite frame with colored segmentation masks overlaid.
        Returns BGR frame with overlays.
        """
        if not detections:
            return frame.copy()
        
        # Create segmentation mask and color overlay
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA for composite
        
        # Color palette similar to SAM2 demo - vibrant, distinct colors
        colors = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (255, 128, 0),    # Orange
            (128, 255, 0),    # Lime
            (255, 0, 128),    # Pink
            (0, 128, 255),    # Sky Blue
            (128, 0, 255),    # Purple
            (255, 128, 128),  # Light Pink
            (128, 255, 128),  # Light Green
            (128, 128, 255),  # Light Blue
        ]
        
        # Process each detection
        for idx, det in enumerate(detections):
            # Get color for this detection
            color = colors[idx % len(colors)]
            
            # If detection has a mask, use it; otherwise create from bbox
            if hasattr(det, 'mask') and det.mask is not None:
                mask = det.mask
            else:
                # Create mask from bbox
                mask = np.zeros((h, w), dtype=np.uint8)
                x1, y1, x2, y2 = det.bbox
                mask[y1:y2, x1:x2] = 1
            
            # Apply color to overlay for composite view
            mask_indices = mask > 0
            overlay[mask_indices] = (*color, int(255 * 0.5))  # 50% opacity
        
        # Create a composite image (original + colored overlay)
        composite = frame.copy()
        mask_rgb = overlay[:, :, :3]
        mask_alpha = overlay[:, :, 3:4] / 255.0
        
        # Blend the overlay with the original image
        composite = (composite * (1 - mask_alpha) + mask_rgb * mask_alpha).astype(np.uint8)
        
        return composite
    
    def _log_sam_detections(self, detections: List[Detection], frame: np.ndarray):
        """
        Log SAM detections with colorful segmentation masks like SAM2 demo.
        Note: This is kept for backward compatibility but video streaming is preferred.
        """
        if not detections:
            return
        
        # Create composite with detections
        composite = self._create_composite_with_detections(detections, frame)
        
        # Log the composite image (for non-video mode)
        composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
        rr.log("/page1/live/composite", rr.Image(composite_rgb))
    
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
    
    async def log_video_tracking(self, result):
        """
        Log video tracking results to Rerun.
        
        Args:
            result: VideoTrackingResult or similar object with tracking data
        """
        if not self.enabled:
            return
        
        try:
            # Log masks if available
            if hasattr(result, 'masks') and result.masks:
                for i, mask_data in enumerate(result.masks):
                    if 'segmentation' in mask_data:
                        # Convert segmentation to image
                        seg = mask_data['segmentation']
                        if isinstance(seg, np.ndarray):
                            # Create colored mask for visualization
                            colored_mask = np.zeros((*seg.shape, 3), dtype=np.uint8)
                            colored_mask[seg > 0] = [255, 0, 0]  # Red color for masks
                            
                            rr.log(
                                f"video/masks/mask_{i}",
                                rr.Image(colored_mask)
                            )
            
            # Log tracks
            if hasattr(result, 'tracks') and result.tracks:
                for track in result.tracks:
                    track_id = track.get('id', 0)
                    bbox = track.get('bbox', [])
                    
                    if bbox and len(bbox) == 4:
                        # Log bounding box
                        x, y, w, h = bbox
                        rr.log(
                            f"video/tracks/track_{track_id}",
                            rr.Boxes2D(
                                array=[[x, y, w, h]],
                                array_format=rr.Box2DFormat.XYWH,
                                labels=[f"Track {track_id}"],
                                class_ids=[track_id]
                            )
                        )
            
            # Log metrics
            if hasattr(result, 'object_count'):
                rr.log("metrics/video/object_count", rr.Scalar(result.object_count))
            
            if hasattr(result, 'processing_time_ms'):
                rr.log("metrics/video/processing_time_ms", rr.Scalar(result.processing_time_ms))
                
        except Exception as e:
            logger.error(f"Error logging video tracking results: {e}")

    def _generate_distinct_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate n visually distinct colors using HSV color space."""
        colors = []
        for i in range(n):
            # Use golden ratio to distribute hues evenly
            hue = int((i * 137.5) % 360)  # Golden angle approximation
            
            # Vary saturation and value for better distinction
            saturation = 200 + (i % 3) * 25  # High saturation
            value = 200 + (i % 2) * 55  # Bright colors
            
            # Convert HSV to RGB
            hsv = np.array([[[hue/2, saturation, value]]], dtype=np.uint8)  # OpenCV uses 0-179 for hue
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
            colors.append(tuple(int(c) for c in rgb))
        
        return colors
    
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
    
    def log_enhanced_object(self, track: TrackedObject):
        """
        Log an enhanced object for the grid view.
        
        This is called after enhancement is complete but before API processing.
        """
        if not self.enabled or track.best_frame is None:
            return
        
        # The best_frame is already the cropped ROI from the tracker
        enhanced_crop = track.best_frame
        
        # Add to enhanced objects buffer
        self.enhanced_objects_buffer.append({
            'image': enhanced_crop.copy(),
            'track_id': track.id,
            'timestamp': time.time(),
            'bbox': track.bbox  # Original bbox for reference
        })
        
        # Maintain FIFO buffer size
        if len(self.enhanced_objects_buffer) > self.max_enhanced_objects:
            self.enhanced_objects_buffer.pop(0)  # Remove oldest
        
        # Update the grid view
        self._update_enhanced_objects_grid()
    
    def _update_enhanced_objects_grid(self):
        """Create and log a grid view of enhanced object crops."""
        if not self.enhanced_objects_buffer:
            return
        
        cell_w, cell_h = self.grid_cell_size
        grid_w = cell_w * self.grid_cols
        grid_h = cell_h * self.grid_rows
        
        # Create blank grid canvas (dark background)
        grid_canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 40  # Dark gray
        
        # Fill grid with enhanced objects
        for idx, obj_data in enumerate(self.enhanced_objects_buffer):
            if idx >= self.max_enhanced_objects:
                break
            
            row = idx // self.grid_cols
            col = idx % self.grid_cols
            
            # Get the enhanced crop
            enhanced_crop = obj_data['image']
            
            # Resize to fit grid cell while maintaining aspect ratio
            h, w = enhanced_crop.shape[:2]
            scale = min(cell_w / w, cell_h / h) * 0.9  # 90% to leave some padding
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized = cv2.resize(enhanced_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Calculate position to center in cell
            x_offset = (cell_w - new_w) // 2
            y_offset = (cell_h - new_h) // 2
            
            # Place in grid
            x1 = col * cell_w + x_offset
            y1 = row * cell_h + y_offset
            x2 = x1 + new_w
            y2 = y1 + new_h
            
            # Add a subtle border around the image
            cv2.rectangle(grid_canvas, (x1-2, y1-2), (x2+2, y2+2), (80, 80, 80), 2)
            
            # Place the enhanced image
            grid_canvas[y1:y2, x1:x2] = resized
        
        # Convert to RGB and log
        grid_rgb = cv2.cvtColor(grid_canvas, cv2.COLOR_BGR2RGB)
        rr.log("/page2/enhanced_objects/grid", rr.Image(grid_rgb))
    
    def _log_segmentation_gallery(self):
        """Create a gallery view of recent segmentation results."""
        if not self.frame_buffer:
            return
        
        # Get the most recent frames with good segmentations
        gallery_frames = []
        for frame, frame_num, timestamp, detections in self.frame_buffer[-6:]:
            if detections and len(detections) > 3:  # Only frames with multiple detections
                gallery_frames.append((frame, detections))
        
        if not gallery_frames:
            return
        
        # Create a horizontal gallery
        gallery_height = 200
        gallery_images = []
        
        for frame, detections in gallery_frames:
            # Resize frame
            h, w = frame.shape[:2]
            scale = gallery_height / h
            new_w = int(w * scale)
            resized = cv2.resize(frame, (new_w, gallery_height))
            
            # Apply segmentation overlay
            overlay = self._create_segmentation_overlay(resized, detections, scale)
            gallery_images.append(overlay)
        
        # Concatenate horizontally
        if gallery_images:
            gallery = np.hstack(gallery_images)
            gallery_rgb = cv2.cvtColor(gallery, cv2.COLOR_BGR2RGB)
            rr.log("/segmentation/gallery", rr.Image(gallery_rgb))
    
    def _create_segmentation_overlay(self, frame: np.ndarray, detections: List[Detection], scale: float) -> np.ndarray:
        """Create a segmentation overlay for a scaled frame."""
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        
        colors = self._generate_distinct_colors(len(detections))
        
        for idx, det in enumerate(detections[:10]):  # Limit to 10 detections
            color = colors[idx % len(colors)]
            
            # Scale bbox
            x1, y1, x2, y2 = det.bbox
            x1 = int(x1 * scale)
            y1 = int(y1 * scale)
            x2 = int(x2 * scale)
            y2 = int(y2 * scale)
            
            # Create mask
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1
            
            # Apply color
            mask_indices = mask > 0
            overlay[mask_indices] = (*color, 128)  # 50% opacity
        
        # Blend with frame
        composite = frame.copy()
        mask_rgb = overlay[:, :, :3]
        mask_alpha = overlay[:, :, 3:4] / 255.0
        composite = (composite * (1 - mask_alpha) + mask_rgb * mask_alpha).astype(np.uint8)
        
        return composite
    
    def _log_segmentation_statistics(self, detections: List[Detection], 
                                    active_tracks: List[TrackedObject], frame_number: int):
        """Log statistics about segmentation results."""
        num_segments = len(detections)
        num_enhanced = len(self.enhanced_objects_buffer)
        
        # Calculate coverage
        if detections:
            # Get frame dimensions from config or use default
            frame_area = 640 * 480  # Default, should get from actual frame
            total_area = sum((d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]) for d in detections)
            coverage = total_area / frame_area if frame_area > 0 else 0
        else:
            coverage = 0
        
        stats_text = f"""# 📊 Processing Statistics

**Frame:** {frame_number}
**Segments Detected:** {num_segments}
**Active Tracks:** {len(active_tracks)}
**Enhanced Objects:** {num_enhanced}/12
**Scene Coverage:** {coverage:.1%}

## Detection Distribution
"""
        
        # Add confidence distribution
        if detections:
            confidences = [d.confidence for d in detections]
            stats_text += f"""
- **Avg Confidence:** {np.mean(confidences):.3f}
- **Min Confidence:** {np.min(confidences):.3f}  
- **Max Confidence:** {np.max(confidences):.3f}

The grid shows the 12 most recent enhanced object crops.
Objects are added after enhancement completes.
"""
        
        rr.log("/page1/stats", rr.TextDocument(stats_text, media_type=rr.MediaType.MARKDOWN))
    
    def clear_enhanced_objects_older_than(self, seconds: float):
        """Remove enhanced objects older than specified seconds."""
        current_time = time.time()
        self.enhanced_objects_buffer = [
            obj for obj in self.enhanced_objects_buffer 
            if current_time - obj['timestamp'] < seconds
        ]
        self._update_enhanced_objects_grid()
    
    def _setup_blueprint(self):
        """Setup the blueprint for both live view and enhanced objects grid."""
        blueprint = self._create_both_pages_blueprint()
        rr.send_blueprint(blueprint)
    
    def _create_both_pages_blueprint(self) -> rrb.Blueprint:
        """Create blueprint showing both live view and enhanced objects grid."""
        return rrb.Blueprint(
            rrb.Vertical(
                # Top row: Live feed and enhanced objects grid
                rrb.Horizontal(
                    rrb.Spatial2DView(
                        name="📹 Live Video Stream",
                        origin="/page1/live/video",
                        contents=[
                            "/page1/live/video/**"
                        ]
                    ),
                    rrb.Spatial2DView(
                        name="🔍 Enhanced Objects",
                        origin="/page2/enhanced_objects",
                        contents=["/page2/enhanced_objects/grid/**"]
                    ),
                    column_shares=[1, 1]
                ),
                # Bottom row: Stats and timeline
                rrb.Horizontal(
                    rrb.TextDocumentView(
                        name="📊 Processing Stats",
                        origin="/page1/stats",
                        contents=["/page1/stats/**"]
                    ),
                    rrb.TimeSeriesView(
                        name="⏱️ Timeline",
                        origin="/timeline",
                        contents=["/timeline/**"]
                    ),
                    column_shares=[1, 1]
                ),
                row_shares=[3, 1]
            )
        )
    
    def _init_video_stream_if_needed(self, websocket_id: str, frame_shape: Tuple[int, int]):
        """Initialize video stream for a websocket connection if not already initialized."""
        if websocket_id not in self.video_streams:
            height, width = frame_shape
            
            # Create container for H.264 encoding
            # Using null output as we're only interested in the encoded packets
            container = av.open("/dev/null", mode='w', format='h264')
            stream = container.add_stream('libx264', rate=30)  # 30 fps
            stream.width = width
            stream.height = height
            stream.pix_fmt = 'yuv420p'
            
            # Configure encoder for real-time streaming
            stream.options = {
                'preset': 'ultrafast',  # Fastest encoding
                'tune': 'zerolatency',  # Low latency
                'crf': '23',  # Quality (lower is better, 23 is default)
            }
            # Disable B-frames as Rerun doesn't support them yet
            stream.max_b_frames = 0
            
            self.video_containers[websocket_id] = container
            self.video_streams[websocket_id] = stream
            self.video_frame_counts[websocket_id] = 0
            
            # Initialize the video stream in Rerun with codec metadata
            rr.log(f"/page1/live/video/{websocket_id}", rr.VideoStream(codec=rr.VideoCodec.H264), static=True)
            
            logger.info(f"Initialized video stream for websocket {websocket_id} with resolution {width}x{height}")
    
    def _stream_video_frame(self, websocket_id: str, frame_rgb: np.ndarray, frame_number: int, timestamp_ns: Optional[int]):
        """Encode and stream a video frame to Rerun."""
        if websocket_id not in self.video_streams:
            logger.warning(f"Video stream not initialized for websocket {websocket_id}")
            return
        
        try:
            stream = self.video_streams[websocket_id]
            
            # Create video frame from numpy array
            av_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
            av_frame.pts = self.video_frame_counts[websocket_id]
            self.video_frame_counts[websocket_id] += 1
            
            # Encode the frame
            for packet in stream.encode(av_frame):
                if packet.pts is None:
                    continue
                
                # Set time context based on packet timing
                if timestamp_ns:
                    # Use actual timestamp if available
                    time_seconds = timestamp_ns / 1e9
                    rr.set_time_seconds("video_time", time_seconds)
                else:
                    # Use packet timing
                    rr.set_time("time", duration=float(packet.pts * packet.time_base))
                
                # Log the video packet using the correct method
                rr.log(
                    f"/page1/live/video/{websocket_id}", 
                    rr.VideoStream.from_fields(sample=bytes(packet))
                )
                
        except Exception as e:
            logger.error(f"Error streaming video frame: {e}")
    
    def cleanup_video_stream(self, websocket_id: str):
        """Clean up video stream resources for a websocket connection."""
        if websocket_id in self.video_streams:
            try:
                # Flush any remaining frames
                stream = self.video_streams[websocket_id]
                for packet in stream.encode():
                    if packet.pts is not None:
                        rr.log(
                            f"/page1/live/video/{websocket_id}",
                            rr.VideoStream.from_fields(sample=bytes(packet))
                        )
                
                # Close the container
                if websocket_id in self.video_containers:
                    self.video_containers[websocket_id].close()
                    del self.video_containers[websocket_id]
                
                del self.video_streams[websocket_id]
                del self.video_frame_counts[websocket_id]
                
                logger.info(f"Cleaned up video stream for websocket {websocket_id}")
            except Exception as e:
                logger.error(f"Error cleaning up video stream: {e}")
    
    def _create_segmentation_blueprint(self) -> rrb.Blueprint:
        """Create blueprint optimized for SAM segmentation visualization."""
        return rrb.Blueprint(
            rrb.Vertical(
                # Main segmentation view
                rrb.Horizontal(
                    rrb.Spatial2DView(
                        name="🎯 Segmentation Results",
                        origin="/segmentation/composite",
                        contents=["/segmentation/composite/**"]
                    ),
                    rrb.Spatial2DView(
                        name="🎨 Segmentation Masks",
                        origin="/segmentation/masks", 
                        contents=["/segmentation/masks/**"]
                    ),
                    column_shares=[1, 1]
                ),
                # Bottom panels
                rrb.Horizontal(
                    rrb.Spatial2DView(
                        name="📸 Recent Segmentations",
                        origin="/segmentation/gallery",
                        contents=["/segmentation/gallery/**"]
                    ),
                    rrb.TextDocumentView(
                        name="📊 Segmentation Stats",
                        origin="/segmentation/stats",
                        contents=["/segmentation/stats/**"]
                    ),
                    column_shares=[2, 1]
                ),
                row_shares=[3, 1]
            )
        )