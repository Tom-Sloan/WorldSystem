"""
Enhanced Rerun visualization for frame processor with dual-page system.
Page 1: Live monitoring with integrated overlays
Page 2: Process view with gallery and enhanced images

Compatible with Rerun 0.23.2
"""

import rerun as rr
import rerun.blueprint as rrb
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
import time
import threading
import queue
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class ViewMode(Enum):
    """Enum for different view modes."""
    BOTH = "both"
    LIVE_ONLY = "live_only" 
    PROCESS_ONLY = "process_only"


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
    """Enhanced visualization with dual-page system and integrated overlays."""
    
    def __init__(self):
        # View mode control
        self.current_view_mode = ViewMode.BOTH
        self.blueprint_update_needed = True
        
        # Performance tracking
        self.fps_counter = []
        self.last_fps_update = time.time()
        
        # Gallery management  
        self.gallery_objects: Dict[int, ProcessedObjectInfo] = {}
        self.max_gallery_items = 20
        
        # Statistics tracking
        self.total_objects_processed = 0
        self.successful_identifications = 0
        self.current_scene_scale = None
        self.last_processing_time = 0
        
        # Selected object tracking
        self.selected_object_id = None
        
        # Tracker configuration (will be set by integration)
        self.process_after_seconds = None
        self.reprocess_interval_seconds = None
        
        # Initialize with default blueprint
        self.setup_blueprint()
    
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
            print(f"[EnhancedRerunVisualizer] Failed to send blueprint: {e}")
    
    def _create_both_pages_blueprint(self) -> rrb.Blueprint:
        """Create blueprint showing both pages side by side."""
        return rrb.Blueprint(
            rrb.Horizontal(
                # Page 1: Live Monitoring
                rrb.Vertical(
                    rrb.Spatial2DView(
                        name="📹 Live Camera View",
                        origin="/page1/live",
                        contents=[
                            "/page1/live/camera/**",
                            "/page1/live/overlays/**",
                        ]
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
                    row_shares=[3, 1]
                ),
                # Page 2: Process View
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
                column_shares=[3, 2]
            )
        )
    
    def _create_live_page_blueprint(self) -> rrb.Blueprint:
        """Create blueprint showing only the live monitoring page."""
        return rrb.Blueprint(
            rrb.Vertical(
                rrb.Spatial2DView(
                    name="📹 Live Camera View",
                    origin="/page1/live",
                    contents=[
                        "/page1/live/camera/**",
                        "/page1/live/overlays/**",
                    ]
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
                row_shares=[3, 1]
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
    
    def set_view_mode(self, mode: ViewMode):
        """Change the current view mode and update blueprint."""
        if self.current_view_mode != mode:
            self.current_view_mode = mode
            self.blueprint_update_needed = True
            self.setup_blueprint()
    
    def log_frame_with_overlays(self, frame: np.ndarray, tracked_objects: Dict,
                               frame_number: int, timestamp_ns: Optional[int] = None):
        """Log camera frame with integrated detection overlays for Page 1."""
        # Set time context
        rr.set_time("frame", sequence=frame_number)
        if timestamp_ns:
            rr.set_time("sensor_time", timestamp=timestamp_ns / 1e9)
        
        # Log base frame first (convert BGR to RGB for Rerun)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            rr.log("/page1/live/camera", rr.Image(frame_rgb).compress(jpeg_quality=85))
        except AttributeError:
            rr.log("/page1/live/camera", rr.Image(frame_rgb))
        
        # Log bounding boxes as overlay
        if tracked_objects:
            boxes_2d = []
            labels = []
            class_ids = []
            colors = []
            
            for track_id, track in tracked_objects.items():
                x1, y1, x2, y2 = track.bbox
                
                # Convert to center + half-size format
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                half_width = (x2 - x1) / 2
                half_height = (y2 - y1) / 2
                
                boxes_2d.append([center_x, center_y, half_width, half_height])
                
                # Create informative label
                label_parts = [f"{track.class_name} #{track.id}"]
                
                # Add dimensions if available
                if track.estimated_dimensions:
                    dims = track.estimated_dimensions
                    size_str = f"{dims.get('width_m', 0)*100:.1f}cm"
                    label_parts.append(size_str)
                
                # Add confidence
                label_parts.append(f"{track.confidence:.0%}")
                
                # Add stage indicator
                if track.estimated_dimensions:
                    label_parts.append("✓")
                elif track.is_being_processed:
                    label_parts.append("⚡")
                else:
                    time_until = max(0, (self.process_after_seconds or 0) - (time.time() - track.first_seen_time))
                    label_parts.append(f"⏳{time_until:.1f}s")
                
                labels.append(" • ".join(label_parts))
                # Use only the track ID as class_id for Rerun (must be integer)
                class_ids.append(track.id)
                
                # Color based on stage (RGB format for Rerun)
                if track.estimated_dimensions:
                    colors.append([0, 255, 0])  # Green
                elif track.is_being_processed:
                    colors.append([255, 165, 0])  # Orange
                else:
                    colors.append([247, 195, 79])  # Light blue (was BGR, now RGB)
            
            # Log boxes overlay
            if boxes_2d:
                try:
                    boxes_array = np.array(boxes_2d, dtype=np.float32)
                    centers = boxes_array[:, :2]
                    half_sizes = boxes_array[:, 2:]
                    
                    rr.log(
                        "/page1/live/overlays",
                        rr.Boxes2D(
                            centers=centers,
                            half_sizes=half_sizes,
                            labels=labels,
                            class_ids=class_ids,
                            colors=colors
                        )
                    )
                except Exception as e:
                    print(f"[EnhancedRerunVisualizer] Failed to log boxes: {e}")
        
        # Update FPS tracking
        self._update_fps()
        
        # Log tracking data for timeline
        self._log_timeline_data(tracked_objects, frame_number)
        
        # Update live statistics
        self._log_live_statistics(frame_number, tracked_objects)
    
    def log_selected_object(self, track):
        """Log detailed information about the selected object to Page 1."""
        if not track:
            return
        
        self.selected_object_id = track.id
        
        # Format detailed information as rich markdown
        details = f"""# 🎯 Object #{track.id}

## Identification
**Class:** {track.class_name}  
**Confidence:** {track.confidence:.1%}  
**Tracking Duration:** {len(track.frame_history)} frames  
"""
        
        if track.estimated_dimensions:
            details += "\n### ✅ Status: Identified\n\n"
            
            # Add product information
            if hasattr(track, 'identified_products') and track.identified_products:
                product = track.identified_products[0]
                details += f"**Product:** {product.get('title', 'Unknown')}\n\n"
            
            # Add dimensions table
            dims = track.estimated_dimensions
            details += """### Real-World Dimensions
| Dimension | Measurement |
|-----------|-------------|
"""
            details += f"| Width | {dims.get('width', 0):.1f} {dims.get('unit', '')} ({dims.get('width_m', 0)*100:.1f} cm) |\n"
            details += f"| Height | {dims.get('height', 0):.1f} {dims.get('unit', '')} ({dims.get('height_m', 0)*100:.1f} cm) |\n"
            details += f"| Depth | {dims.get('depth', 0):.1f} {dims.get('unit', '')} ({dims.get('depth_m', 0)*100:.1f} cm) |\n"
            
        elif track.is_being_processed:
            details += "\n### 🔄 Status: Processing...\n"
            details += "Enhancing image and identifying dimensions\n"
        else:
            time_until = max(0, (self.process_after_seconds or 0) - (time.time() - track.first_seen_time))
            details += f"\n### ⏳ Status: Tracking\n"
            details += f"Processing will begin in {time_until:.1f} seconds\n"
        
        # Add quality analysis if available
        if hasattr(track, 'score_components') and track.score_components:
            details += "\n### Frame Quality Analysis\n"
            details += "| Component | Score | Status |\n"
            details += "|-----------|-------|--------|\n"
            
            for component, score in track.score_components.items():
                emoji = self._get_quality_emoji(score)
                details += f"| {component.title()} | {score:.2f} | {emoji} |\n"
            
            if hasattr(track, 'best_score'):
                overall_emoji = self._get_quality_emoji(track.best_score)
                details += f"\n**Overall Score:** {track.best_score:.2f} {overall_emoji}\n"
        
        # Log to Page 1
        rr.log("/page1/selected", rr.TextDocument(details, media_type=rr.MediaType.MARKDOWN))
    
    def log_processed_object_to_gallery(self, track, enhanced_image: np.ndarray):
        """Log a processed object to the gallery on Page 2."""
        # Create gallery entry
        gallery_info = ProcessedObjectInfo(
            track_id=track.id,
            class_name=track.class_name,
            enhanced_image=enhanced_image,
            dimensions=track.estimated_dimensions,
            confidence=track.confidence,
            product_name=track.identified_products[0].get('title') if track.identified_products else None,
            processing_time=getattr(track, 'processing_time', 0),
            timestamp=time.time()
        )
        
        # Add to gallery (limit size)
        self.gallery_objects[track.id] = gallery_info
        if len(self.gallery_objects) > self.max_gallery_items:
            # Remove oldest item
            oldest_id = min(self.gallery_objects.keys(), 
                          key=lambda k: self.gallery_objects[k].timestamp)
            del self.gallery_objects[oldest_id]
        
        # Update statistics
        self.total_objects_processed += 1
        if track.estimated_dimensions:
            self.successful_identifications += 1
        
        # Create gallery grid layout
        self._update_gallery_grid()
        
        # Update gallery results display
        self._update_gallery_results()
        
        # Update statistics
        self._update_process_statistics()
    
    def _update_gallery_grid(self):
        """Update the gallery grid display with all processed objects."""
        # Create a grid of processed objects
        grid_cols = 4
        cell_size = 200  # pixels per cell
        padding = 10
        
        # Sort by most recent
        sorted_objects = sorted(self.gallery_objects.values(), 
                              key=lambda x: x.timestamp, reverse=True)
        
        for idx, obj in enumerate(sorted_objects):
            # Calculate grid position
            row = idx // grid_cols
            col = idx % grid_cols
            
            # Resize image to fit cell
            resized = cv2.resize(obj.enhanced_image, (cell_size - 2*padding, cell_size - 2*padding))
            
            # Create cell with label
            cell = np.ones((cell_size, cell_size, 3), dtype=np.uint8) * 40  # Dark gray background
            y_offset = padding
            x_offset = padding
            cell[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized
            
            # Add text label at bottom
            label = f"#{obj.track_id} - {obj.class_name[:15]}"
            cv2.putText(cell, label, (padding, cell_size - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Log to gallery (convert BGR to RGB for Rerun)
            cell_rgb = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
            try:
                rr.log(f"/page2/gallery/item_{idx}", 
                      rr.Image(cell_rgb).compress(jpeg_quality=90))
            except AttributeError:
                rr.log(f"/page2/gallery/item_{idx}", rr.Image(cell_rgb))
    
    def _update_gallery_results(self):
        """Update the identification results display on Page 2."""
        results_text = "# 📋 Identification Results\n\n"
        
        # Sort by most recent
        sorted_objects = sorted(self.gallery_objects.values(), 
                              key=lambda x: x.timestamp, reverse=True)
        
        for obj in sorted_objects[:10]:  # Show last 10
            icon = "📷" if "camera" in obj.class_name.lower() else "📱" if "phone" in obj.class_name.lower() else "🖥️"
            
            results_text += f"### {icon} {obj.class_name} #{obj.track_id}\n"
            
            if obj.product_name:
                results_text += f"**Product:** {obj.product_name}\n"
            
            if obj.dimensions:
                dims = obj.dimensions
                results_text += f"**Size:** {dims.get('width', 0):.1f}×{dims.get('height', 0):.1f}×{dims.get('depth', 0):.1f} {dims.get('unit', '')}\n"
            
            results_text += f"**Confidence:** {obj.confidence:.0%}\n"
            results_text += f"**Processing Time:** {obj.processing_time:.1f}s\n"
            results_text += "\n---\n\n"
        
        rr.log("/page2/results", rr.TextDocument(results_text, media_type=rr.MediaType.MARKDOWN))
    
    def _update_process_statistics(self):
        """Update the statistics display on Page 2."""
        success_rate = (self.successful_identifications / self.total_objects_processed * 100 
                       if self.total_objects_processed > 0 else 0)
        
        stats_text = f"""# 📊 Processing Statistics

## Overall Performance
- **Total Processed:** {self.total_objects_processed}
- **Successfully Identified:** {self.successful_identifications}
- **Success Rate:** {success_rate:.1f}%
- **Gallery Items:** {len(self.gallery_objects)}

## Recent Performance
- **Last Processing Time:** {self.last_processing_time:.1f}s
- **Average Confidence:** {self._calculate_average_confidence():.1f}%
"""
        
        if self.current_scene_scale:
            stats_text += f"""
## Scene Scale
- **Scale Factor:** {self.current_scene_scale.get('scale_factor', 0):.4f} m/unit
- **Confidence:** {self.current_scene_scale.get('confidence', 0):.0%}
- **Based on:** {self.current_scene_scale.get('num_estimates', 0)} objects
"""
        
        rr.log("/page2/stats", rr.TextDocument(stats_text, media_type=rr.MediaType.MARKDOWN))
    
    def log_scene_scale(self, scale_info: Dict):
        """Log scene scale information."""
        self.current_scene_scale = scale_info
        
        # Create visual scale representation
        scale_doc = f"""# 📏 Scene Understanding

## Real-World Scale Factor
**{scale_info.get('scale_factor', 1.0):.4f} m/unit**  
or {1/scale_info.get('scale_factor', 1.0):.1f} units per meter

### Confidence: {scale_info.get('confidence', 0):.0%}
Based on {scale_info.get('num_estimates', 0)} identified objects
"""
        
        # Add contributing objects if available
        if 'estimates' in scale_info:
            scale_doc += "\n## Contributing Objects\n"
            scale_doc += "| Object | Dimensions | Confidence |\n"
            scale_doc += "|--------|------------|------------|\n"
            
            for est in scale_info['estimates'][:5]:  # Show top 5
                dims = est['dimensions_m']
                scale_doc += f"| {est['product']} | "
                scale_doc += f"{dims['width']:.2f}×{dims['height']:.2f}×{dims['depth']:.2f}m | "
                scale_doc += f"{est['confidence']:.0%} |\n"
        
        # Log to both pages
        rr.log("/page1/stats/scale", rr.TextDocument(scale_doc, media_type=rr.MediaType.MARKDOWN))
        rr.log("/page2/scale", rr.TextDocument(scale_doc, media_type=rr.MediaType.MARKDOWN))
        
        # Update statistics display
        self._update_process_statistics()
    
    def _log_timeline_data(self, tracked_objects: Dict, frame_number: int):
        """Log timeline data for Page 1."""
        for track_id, track in tracked_objects.items():
            # Log presence
            rr.log(f"/page1/timeline/object_{track_id}/present", rr.Scalar(1.0))
            
            # Log processing stage
            if track.estimated_dimensions:
                rr.log(f"/page1/timeline/object_{track_id}/stage", rr.Scalar(3.0))  # Identified
            elif track.is_being_processed:
                rr.log(f"/page1/timeline/object_{track_id}/stage", rr.Scalar(2.0))  # Processing
            else:
                rr.log(f"/page1/timeline/object_{track_id}/stage", rr.Scalar(1.0))  # Tracking
            
            # Log quality score if available
            if hasattr(track, 'best_score') and track.best_score > 0:
                rr.log(f"/page1/timeline/object_{track_id}/quality", rr.Scalar(track.best_score))
    
    def _log_live_statistics(self, frame_number: int, tracked_objects: Dict):
        """Log live statistics for Page 1."""
        fps = len(self.fps_counter)
        num_objects = len(tracked_objects)
        
        # Count objects by stage
        tracking = sum(1 for t in tracked_objects.values() 
                      if not t.is_being_processed and not t.estimated_dimensions)
        processing = sum(1 for t in tracked_objects.values() if t.is_being_processed)
        identified = sum(1 for t in tracked_objects.values() if t.estimated_dimensions)
        
        stats_text = f"""# 📊 Live Statistics

## Performance
- **FPS:** {fps}
- **Frame:** {frame_number}

## Objects
- **Total Active:** {num_objects}
- **Tracking:** {tracking}
- **Processing:** {processing}
- **Identified:** {identified}

## Processing
- **Queue Size:** {getattr(self, 'queue_size', 0)}
- **Total Processed:** {self.total_objects_processed}
- **Success Rate:** {(self.successful_identifications / self.total_objects_processed * 100) if self.total_objects_processed > 0 else 0:.1f}%
"""
        
        rr.log("/page1/stats", rr.TextDocument(stats_text, media_type=rr.MediaType.MARKDOWN))
        
        # Log scalar values for timeline
        rr.log("/page1/timeline/fps", rr.Scalar(fps))
        rr.log("/page1/timeline/objects", rr.Scalar(num_objects))
    
    def _update_fps(self):
        """Update FPS counter."""
        current_time = time.time()
        self.fps_counter.append(current_time)
        self.fps_counter = [t for t in self.fps_counter if current_time - t < 1.0]
    
    def _calculate_average_confidence(self) -> float:
        """Calculate average confidence of gallery objects."""
        if not self.gallery_objects:
            return 0.0
        return sum(obj.confidence for obj in self.gallery_objects.values()) / len(self.gallery_objects) * 100
    
    def _get_quality_color(self, score: float) -> Tuple[int, int, int]:
        """Get color based on quality score (BGR format)."""
        if score < 0.3:
            return (0, 0, 255)  # Red
        elif score < 0.7:
            return (0, 165, 255)  # Orange
        else:
            return (0, 255, 0)  # Green
    
    def _get_quality_emoji(self, score: float) -> str:
        """Get emoji representation of quality score."""
        if score < 0.3:
            return "❌"
        elif score < 0.7:
            return "⚠️"
        else:
            return "✅"
    
    def set_queue_size(self, size: int):
        """Update queue size for statistics."""
        self.queue_size = size
    
    def set_last_processing_time(self, time_ms: float):
        """Update last processing time."""
        self.last_processing_time = time_ms / 1000.0  # Convert to seconds
    
    def set_tracker_config(self, process_after_seconds: float, reprocess_interval_seconds: float):
        """Set tracker configuration values."""
        self.process_after_seconds = process_after_seconds
        self.reprocess_interval_seconds = reprocess_interval_seconds


# Convenience functions for view mode switching
def show_both_pages(visualizer: EnhancedRerunVisualizer):
    """Show both pages side by side."""
    visualizer.set_view_mode(ViewMode.BOTH)


def show_live_page_only(visualizer: EnhancedRerunVisualizer):
    """Show only the live monitoring page."""
    visualizer.set_view_mode(ViewMode.LIVE_ONLY)


def show_process_page_only(visualizer: EnhancedRerunVisualizer):
    """Show only the process/gallery page."""
    visualizer.set_view_mode(ViewMode.PROCESS_ONLY)


# Integration helper
def integrate_with_frame_processor(processor):
    """
    Integrate the enhanced visualizer with an existing frame processor.
    
    Usage:
        visualizer = integrate_with_frame_processor(processor)
    """
    visualizer = EnhancedRerunVisualizer()
    
    # Store reference in processor
    processor.rerun_visualizer = visualizer
    
    print("[EnhancedRerunVisualizer] Integration complete. Use visualizer methods to log data.")
    print("[EnhancedRerunVisualizer] Switch views with: show_live_page_only(), show_process_page_only(), show_both_pages()")
    
    return visualizer