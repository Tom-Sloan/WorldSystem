"""
Minimal enhanced Rerun visualizer for refactored architecture.

This provides the essential visualization functionality without the complexity
of the experimental enhanced visualizer.
"""

import rerun as rr
import rerun.blueprint as rrb
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import time

import logging

logger = logging.getLogger(__name__)


class ViewMode(Enum):
    """Visualization view modes."""
    BOTH = "both"
    LIVE_ONLY = "live"
    PROCESS_ONLY = "process"


@dataclass
class ProcessedObjectInfo:
    """Information about a processed object for gallery display."""
    track_id: int
    class_name: str
    enhanced_image: np.ndarray
    dimensions: Optional[Dict[str, float]]
    confidence: float
    product_name: Optional[str]
    processing_time: float
    timestamp: float


class EnhancedRerunVisualizer:
    """
    Minimal Rerun visualizer for the frame processor.
    
    This provides basic functionality for logging frames and detections
    without the complexity of the experimental enhanced version.
    """
    
    def __init__(self, process_after_seconds: float = 1.5):
        self.process_after_seconds = process_after_seconds
        self.current_view_mode = ViewMode.BOTH
        self.selected_object_id = None
        self.gallery_objects = {}
        self.max_gallery_items = 20
        self.successful_identifications = 0
        self.blueprint_update_needed = True
        
        # Initialize with default blueprint
        self.setup_blueprint()
        
    def set_view_mode(self, mode: ViewMode):
        """Change the current view mode and update blueprint."""
        if self.current_view_mode != mode:
            self.current_view_mode = mode
            self.blueprint_update_needed = True
            self.setup_blueprint()
            logger.info(f"View mode changed to: {mode.value}")
    
    def setup_blueprint(self):
        """Create the enhanced blueprint layout based on current view mode."""
        if self.current_view_mode == ViewMode.BOTH:
            blueprint = self._create_both_pages_blueprint()
        elif self.current_view_mode == ViewMode.LIVE_ONLY:
            blueprint = self._create_live_page_blueprint()
        else:
            blueprint = self._create_process_page_blueprint()
        
        try:
            rr.send_blueprint(blueprint)
            self.blueprint_update_needed = False
        except Exception as e:
            logger.error(f"Failed to send blueprint: {e}")
    
    def _create_both_pages_blueprint(self) -> rrb.Blueprint:
        """Create blueprint showing both pages side by side."""
        return rrb.Blueprint(
            rrb.Horizontal(
                # Page 1: Live Monitoring with Grid
                rrb.Vertical(
                    rrb.Horizontal(
                        rrb.Spatial2DView(
                            name="📹 Live Camera View",
                            origin="/page1/live",
                            contents=[
                                "/page1/live/camera/**",
                                "/page1/live/overlays/**",
                                "/page1/live/labels/**",
                            ]
                        ),
                        rrb.Spatial2DView(
                            name="🎬 Frame Grid",
                            origin="/grid_view",
                            contents=["/grid_view/**"]
                        ),
                        column_shares=[1, 1]
                    ),
                    rrb.Horizontal(
                        rrb.TextDocumentView(
                            name="🎯 Selected Object",
                            origin="/page1/selected",
                            contents=["/page1/selected/**"]
                        ),
                        rrb.TimeSeriesView(
                            name="⏱️ Processing Timeline", 
                            origin="/page1/timeline",
                            contents=["/page1/timeline/**"]
                        ),
                        column_shares=[1, 1]
                    ),
                    row_shares=[2, 1]
                ),
                # Page 2: Process Gallery
                rrb.Vertical(
                    rrb.Spatial2DView(
                        name="🖼️ Processed Gallery",
                        origin="/page2/gallery",
                        contents=["/page2/gallery/**"]
                    ),
                    rrb.Horizontal(
                        rrb.TextDocumentView(
                            name="📋 Identification Results",
                            origin="/page2/results", 
                            contents=["/page2/results/**"]
                        ),
                        rrb.TextDocumentView(
                            name="📊 Statistics",
                            origin="/page2/stats",
                            contents=["/page2/stats/**"]
                        ),
                        column_shares=[2, 1]
                    ),
                    row_shares=[2, 1]
                ),
                column_shares=[1, 1]
            )
        )
    
    def _create_live_page_blueprint(self) -> rrb.Blueprint:
        """Create blueprint showing only the live monitoring page."""
        return rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(
                        name="📹 Live Camera View",
                        origin="/page1/live",
                        contents=[
                            "/page1/live/camera/**",
                            "/page1/live/overlays/**",
                            "/page1/live/labels/**",
                        ]
                    ),
                    rrb.Spatial2DView(
                        name="🎬 Frame Grid",
                        origin="/grid_view",
                        contents=["/grid_view/**"]
                    ),
                    column_shares=[1, 1]
                ),
                rrb.Horizontal(
                    rrb.TextDocumentView(
                        name="🎯 Selected Object",
                        origin="/page1/selected",
                        contents=["/page1/selected/**"]
                    ),
                    rrb.TextDocumentView(
                        name="📊 Live Statistics",
                        origin="/page1/stats",
                        contents=["/page1/stats/**"]
                    ),
                    rrb.TimeSeriesView(
                        name="⏱️ Processing Timeline",
                        origin="/page1/timeline", 
                        contents=["/page1/timeline/**"]
                    ),
                    column_shares=[1, 1, 1]
                ),
                row_shares=[2, 1]
            )
        )
    
    def _create_process_page_blueprint(self) -> rrb.Blueprint:
        """Create blueprint showing only the process/gallery page."""
        return rrb.Blueprint(
            rrb.Vertical(
                rrb.Spatial2DView(
                    name="🖼️ Processed Gallery",
                    origin="/page2/gallery",
                    contents=["/page2/gallery/**"]
                ),
                rrb.Horizontal(
                    rrb.TextDocumentView(
                        name="📋 Identification Results",
                        origin="/page2/results",
                        contents=["/page2/results/**"]
                    ),
                    rrb.TextDocumentView(
                        name="📊 Statistics & Scale",
                        origin="/page2/stats",
                        contents=["/page2/stats/**", "/page2/scale/**"]
                    ),
                    column_shares=[2, 1]
                ),
                row_shares=[2, 1]
            )
        )
        
    def log_frame_with_overlays(self, frame: np.ndarray, tracked_objects: Dict,
                               frame_number: int, timestamp_ns: Optional[int] = None):
        """
        Log camera frame with minimal overlays for cleaner visualization.
        
        Args:
            frame: Input frame (BGR)
            tracked_objects: Dictionary of tracked objects
            frame_number: Frame sequence number
            timestamp_ns: Optional timestamp in nanoseconds
        """
        # Set time context
        rr.set_time("frame", sequence=frame_number)
        if timestamp_ns:
            rr.set_time("sensor_time", timestamp=timestamp_ns / 1e9)
        
        # Log base frame to Page 1 live view (convert BGR to RGB for Rerun)
        frame_rgb = frame[..., ::-1].copy()  # BGR to RGB
        try:
            rr.log("/page1/live/camera", rr.Image(frame_rgb).compress(jpeg_quality=85))
        except AttributeError:
            rr.log("/page1/live/camera", rr.Image(frame_rgb))
        
        # Only log minimal overlays if there are few tracked objects
        if tracked_objects and len(tracked_objects) < 5:
            # Log timeline data for tracked objects
            self._log_timeline_data(tracked_objects, frame_number)
        
        # Log statistics
        self._log_live_statistics(frame_number, tracked_objects)
    
    def _log_timeline_data(self, tracked_objects: Dict, frame_number: int):
        """Log timeline data for all tracked objects."""
        for track_id, track in tracked_objects.items():
            # Log presence
            rr.log(f"/page1/timeline/object_{track_id}/present", rr.Scalar(1.0))
            
            # Log processing stage
            if hasattr(track, 'estimated_dimensions') and track.estimated_dimensions:
                rr.log(f"/page1/timeline/object_{track_id}/stage", rr.Scalar(3.0))  # Identified
            elif hasattr(track, 'is_being_processed') and track.is_being_processed:
                rr.log(f"/page1/timeline/object_{track_id}/stage", rr.Scalar(2.0))  # Processing
            else:
                rr.log(f"/page1/timeline/object_{track_id}/stage", rr.Scalar(1.0))  # Tracking
            
            # Log quality score if available
            if hasattr(track, 'best_score') and track.best_score > 0:
                rr.log(f"/page1/timeline/object_{track_id}/quality", rr.Scalar(track.best_score))
    
    def _log_live_statistics(self, frame_number: int, tracked_objects: Dict):
        """Log live statistics to Page 1."""
        num_objects = len(tracked_objects)
        
        # Create statistics text
        stats_text = f"""# 📊 Live Statistics

**Frame:** {frame_number}
**Active Objects:** {num_objects}
**Successful IDs:** {self.successful_identifications}

Tracking {num_objects} object{'s' if num_objects != 1 else ''}
"""
        
        rr.log("/page1/stats", rr.TextDocument(stats_text, media_type=rr.MediaType.MARKDOWN))
        
        # Log scalar values for timeline
        rr.log("/page1/timeline/objects", rr.Scalar(num_objects))
    
    def log_selected_object_details(self, track: Any, enhanced_image: Optional[np.ndarray] = None):
        """
        Log details for a selected object.
        
        Args:
            track: TrackedObject instance
            enhanced_image: Optional enhanced image
        """
        self.selected_object_id = track.id
        
        # Log enhanced image if available
        if enhanced_image is not None:
            enhanced_rgb = enhanced_image[..., ::-1].copy()  # BGR to RGB
            rr.log(f"/selected/object_{track.id}", rr.Image(enhanced_rgb))
        
        # Format detailed information as rich markdown
        details = f"""# 🎯 Object #{track.id}

## Identification
**Class:** {track.class_name}  
**Confidence:** {track.confidence:.1%}  
"""
        
        if hasattr(track, 'estimated_dimensions') and track.estimated_dimensions:
            dims = track.estimated_dimensions
            details += "\n### ✅ Status: Identified\n\n"
            details += """### Real-World Dimensions
| Dimension | Measurement |
|-----------|-------------|
| Width | {:.1f} cm |
| Height | {:.1f} cm |
| Depth | {:.1f} cm |
""".format(
                dims.get('width_m', 0) * 100,
                dims.get('height_m', 0) * 100,
                dims.get('depth_m', 0) * 100
            )
            self.successful_identifications += 1
        elif hasattr(track, 'is_being_processed') and track.is_being_processed:
            details += "\n### 🔄 Status: Processing...\n"
            details += "Enhancing image and identifying dimensions\n"
        else:
            time_until = max(0, (self.process_after_seconds or 0) - (time.time() - track.created_at))
            details += f"\n### ⏳ Status: Tracking\n"
            details += f"Processing will begin in {time_until:.1f} seconds\n"
        
        # Log to Page 1
        rr.log("/page1/selected", rr.TextDocument(details, media_type=rr.MediaType.MARKDOWN))
    
    def add_to_gallery(self, processed_object: ProcessedObjectInfo):
        """Add processed object to gallery."""
        self.gallery_objects[processed_object.track_id] = processed_object
        
        # Keep gallery size limited
        if len(self.gallery_objects) > self.max_gallery_items:
            oldest_id = min(self.gallery_objects.keys())
            del self.gallery_objects[oldest_id]
        
        # Log to Page 2 gallery
        if processed_object.enhanced_image is not None:
            enhanced_rgb = processed_object.enhanced_image[..., ::-1].copy()
            try:
                rr.log(f"/page2/gallery/object_{processed_object.track_id}", 
                      rr.Image(enhanced_rgb).compress(jpeg_quality=90))
            except AttributeError:
                rr.log(f"/page2/gallery/object_{processed_object.track_id}", rr.Image(enhanced_rgb))
        
        # Update gallery results
        self._update_gallery_results()
        self._update_process_statistics()
    
    def _update_gallery_results(self):
        """Update the identification results panel."""
        results_text = "# 📋 Identification Results\n\n"
        
        for obj_id, obj in sorted(self.gallery_objects.items(), 
                                 key=lambda x: x[1].timestamp, reverse=True):
            results_text += f"## Object #{obj_id}\n"
            results_text += f"**Type:** {obj.class_name}\n"
            
            if obj.product_name:
                results_text += f"**Product:** {obj.product_name}\n"
            
            if obj.dimensions:
                results_text += f"**Dimensions:** "
                results_text += f"{obj.dimensions.get('width_m', 0)*100:.1f} × "
                results_text += f"{obj.dimensions.get('height_m', 0)*100:.1f} × "
                results_text += f"{obj.dimensions.get('depth_m', 0)*100:.1f} cm\n"
            
            results_text += f"**Processing Time:** {obj.processing_time:.2f}s\n"
            results_text += "\n---\n\n"
        
        rr.log("/page2/results", rr.TextDocument(results_text, media_type=rr.MediaType.MARKDOWN))
    
    def _update_process_statistics(self):
        """Update the process statistics panel."""
        total_processed = len(self.gallery_objects)
        successful = sum(1 for obj in self.gallery_objects.values() if obj.dimensions)
        
        stats_text = f"""# 📊 Process Statistics

## Summary
- **Total Processed:** {total_processed}
- **Successfully Identified:** {successful}
- **Success Rate:** {(successful/total_processed*100) if total_processed > 0 else 0:.1f}%

## Recent Processing
"""
        
        # Show last 5 processed objects
        recent = sorted(self.gallery_objects.items(), 
                       key=lambda x: x[1].timestamp, reverse=True)[:5]
        
        for obj_id, obj in recent:
            status = "✅" if obj.dimensions else "❌"
            stats_text += f"- {status} Object #{obj_id} ({obj.class_name})\n"
        
        rr.log("/page2/stats", rr.TextDocument(stats_text, media_type=rr.MediaType.MARKDOWN))


# Convenience functions for view mode changes
def show_both_pages(visualizer: EnhancedRerunVisualizer):
    """Show both live and process pages."""
    visualizer.set_view_mode(ViewMode.BOTH)


def show_live_page_only(visualizer: EnhancedRerunVisualizer):
    """Show only the live page."""
    visualizer.set_view_mode(ViewMode.LIVE_ONLY)


def show_process_page_only(visualizer: EnhancedRerunVisualizer):
    """Show only the process page."""
    visualizer.set_view_mode(ViewMode.PROCESS_ONLY)