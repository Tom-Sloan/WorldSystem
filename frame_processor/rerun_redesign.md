# enhanced_rerun_visualizer.py
"""
Enhanced Rerun visualization for frame processor with integrated overlays
and intuitive information display.
"""

import rerun as rr
import rerun.blueprint as bp
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
import time
import threading
import queue
import torch


class EnhancedRerunVisualizer:
    """Enhanced visualization with integrated overlays and meaningful displays."""
    
    def __init__(self):
        # Initialize blueprint
        self.setup_blueprint()
        
        # Track selected object for detailed view
        self.selected_object_id = None
        
        # Performance tracking
        self.fps_counter = []
        self.last_fps_update = time.time()
        
        # Gallery tracking
        self.processed_objects = {}  # Store processed objects for gallery
        self.gallery_update_counter = 0
        
    def setup_blueprint(self):
        """Create the enhanced blueprint layout with both views visible."""
        blueprint = bp.Blueprint(
            bp.Horizontal(
                # Left side: Live Monitoring Components
                bp.Vertical(
                    # Main camera view with integrated overlays
                    bp.SpaceView(
                        name="📹 Live Camera View",
                        origin="/live/camera",
                        contents=[
                            "/live/camera/**",
                            "/live/detections/**",
                            "/live/tracking/**",
                            "/live/annotations/**"
                        ]
                    ),
                    bp.Horizontal(
                        bp.TextDocumentView(
                            name="🎯 Selected Object",
                            origin="/live/selected_object",
                            contents=["/live/selected_object/**"]
                        ),
                        bp.TimeSeriesView(
                            name="⏱️ Processing Timeline",
                            origin="/live/timeline",
                            contents=["/live/timeline/**"]
                        ),
                        column_shares=[1, 1]
                    ),
                    row_shares=[3, 1]
                ),
                # Right side: Gallery and Statistics
                bp.Vertical(
                    # Gallery grid showing processed objects
                    bp.SpaceView(
                        name="🖼️ Processed Objects Gallery",
                        origin="/gallery",
                        contents=[
                            "/gallery/**",
                            "/enhancement/**"
                        ]
                    ),
                    bp.Horizontal(
                        bp.TextDocumentView(
                            name="📋 Identification Results",
                            origin="/gallery/results",
                            contents=["/gallery/results/**"]
                        ),
                        bp.TextDocumentView(
                            name="📊 Statistics",
                            origin="/stats",
                            contents=["/stats/**", "/scene/**"]
                        ),
                        column_shares=[2, 1]
                    ),
                    row_shares=[2, 1]
                ),
                column_shares=[3, 2]
            )
        )
        rr.send_blueprint(blueprint)
    
    def log_frame_with_overlays(self, frame: np.ndarray, detections: List[Tuple],
                               tracked_objects: Dict, frame_number: int):
        """Log camera frame with integrated bounding box overlays."""
        # Log the base frame
        rr.set_time("frame", sequence=frame_number)
        try:
            rr.log("/live/camera/image", rr.Image(frame).compress(jpeg_quality=85))
        except AttributeError:
            # Fallback if compress is not available
            rr.log("/live/camera/image", rr.Image(frame))
        
        # Create annotated frame for display
        annotated = frame.copy()
        
        # Draw all tracked objects with enhanced labels
        for track_id, track in tracked_objects.items():
            x1, y1, x2, y2 = track.bbox
            
            # Determine color based on status
            if track.estimated_dimensions:
                color = (0, 255, 0)  # Green for identified objects
                thickness = 3
            elif track.is_being_processed:
                color = (255, 165, 0)  # Orange for processing
                thickness = 2
            else:
                color = (79, 195, 247)  # Light blue for tracking
                thickness = 2
            
            # Draw enhanced bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Create informative label
            label_parts = [f"{track.class_name} #{track.id}"]
            
            # Add dimensions if available
            if track.estimated_dimensions:
                dims = track.estimated_dimensions
                size_m = dims.get('width_m', 0)
                label_parts.append(f"{size_m:.2f}m")
            
            # Add confidence
            label_parts.append(f"{track.confidence:.0%}")
            
            label = " • ".join(label_parts)
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - 25), (x1 + label_w + 10, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Add quality indicator if best frame exists
            if track.best_score > 0:
                quality_bar_width = 60
                quality_bar_height = 4
                bar_x = x1
                bar_y = y2 + 5
                
                # Background
                cv2.rectangle(annotated, (bar_x, bar_y), 
                            (bar_x + quality_bar_width, bar_y + quality_bar_height),
                            (50, 50, 50), -1)
                
                # Quality fill
                fill_width = int(quality_bar_width * track.best_score)
                quality_color = self._get_quality_color(track.best_score)
                cv2.rectangle(annotated, (bar_x, bar_y),
                            (bar_x + fill_width, bar_y + quality_bar_height),
                            quality_color, -1)
        
        # Log the annotated frame
        try:
            rr.log("/live/camera/annotated", rr.Image(annotated).compress(jpeg_quality=85))
        except AttributeError:
            rr.log("/live/camera/annotated", rr.Image(annotated))
        
        # Log tracking information for timeline
        self._log_timeline_data(tracked_objects, frame_number)
    
    def log_selected_object(self, track):
        """Log detailed information about the selected object."""
        if not track:
            return
        
        # Format object details as rich markdown
        details = f"""# 🎯 Object #{track.id}

## Identification
**Class:** {track.class_name}  
**Confidence:** {track.confidence:.1%}  
**Tracking Duration:** {len(track.frame_history)} frames  
**Processing Count:** {track.processing_count}

## Status
"""
        
        if track.estimated_dimensions:
            details += "✅ **Identified**\n\n"
            
            # Add product information if available
            if track.identified_products:
                product = track.identified_products[0]
                details += f"### Product Details\n"
                details += f"**Name:** {product.get('title', 'Unknown')}\n\n"
            
            # Add dimensions
            dims = track.estimated_dimensions
            details += f"""### Real-World Dimensions
| Dimension | Imperial | Metric |
|-----------|----------|---------|
| Width     | {dims.get('width', 0):.1f} {dims.get('unit', '')} | {dims.get('width_m', 0)*100:.1f} cm |
| Height    | {dims.get('height', 0):.1f} {dims.get('unit', '')} | {dims.get('height_m', 0)*100:.1f} cm |
| Depth     | {dims.get('depth', 0):.1f} {dims.get('unit', '')} | {dims.get('depth_m', 0)*100:.1f} cm |
"""
        elif track.is_being_processed:
            details += "🔄 **Processing...**\n"
        else:
            time_until_process = max(0, 1.5 - (time.time() - track.first_seen_time))
            details += f"⏳ **Tracking** (processing in {time_until_process:.1f}s)\n"
        
        # Add quality scores if available
        if track.score_components:
            details += f"""
### Frame Quality Analysis
| Component | Score | Status |
|-----------|-------|---------|
| Sharpness | {track.score_components['sharpness']:.2f} | {self._get_quality_emoji(track.score_components['sharpness'])} |
| Exposure  | {track.score_components['exposure']:.2f} | {self._get_quality_emoji(track.score_components['exposure'])} |
| Size      | {track.score_components['size']:.2f} | {self._get_quality_emoji(track.score_components['size'])} |
| Centering | {track.score_components['centering']:.2f} | {self._get_quality_emoji(track.score_components['centering'])} |

**Overall Score:** {track.best_score:.2f} {self._get_quality_emoji(track.best_score)}
"""
        
        rr.log("/live/selected_object", rr.TextDocument(details, media_type=rr.MediaType.MARKDOWN))
    
    def log_live_statistics(self, frame_count: int, processing_time_ms: float,
                          num_objects: int, gpu_memory_mb: float):
        """Log live statistics in a user-friendly format."""
        # Calculate FPS
        current_time = time.time()
        self.fps_counter.append(current_time)
        self.fps_counter = [t for t in self.fps_counter if current_time - t < 1.0]
        fps = len(self.fps_counter)
        
        stats = f"""# 📊 Live Statistics

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">

<div style="background: #2a2a2a; padding: 15px; border-radius: 8px; text-align: center;">
<h2 style="color: #4fc3f7; margin: 0;">{fps}</h2>
<p style="margin: 0; color: #888; font-size: 12px;">FPS</p>
</div>

<div style="background: #2a2a2a; padding: 15px; border-radius: 8px; text-align: center;">
<h2 style="color: #4fc3f7; margin: 0;">{num_objects}</h2>
<p style="margin: 0; color: #888; font-size: 12px;">Objects</p>
</div>

<div style="background: #2a2a2a; padding: 15px; border-radius: 8px; text-align: center;">
<h2 style="color: #4fc3f7; margin: 0;">{processing_time_ms:.0f}ms</h2>
<p style="margin: 0; color: #888; font-size: 12px;">Latency</p>
</div>

<div style="background: #2a2a2a; padding: 15px; border-radius: 8px; text-align: center;">
<h2 style="color: #4fc3f7; margin: 0;">{gpu_memory_mb:.1f}GB</h2>
<p style="margin: 0; color: #888; font-size: 12px;">GPU Memory</p>
</div>

</div>

### Processing Status
- **Frames Processed:** {frame_count:,}
- **Average Processing:** {processing_time_ms:.1f}ms
- **GPU Utilization:** {min(100, (gpu_memory_mb / 24) * 100):.0f}%
"""
        
        # Log to the statistics panel
        rr.log("/stats/live", rr.TextDocument(stats, media_type=rr.MediaType.MARKDOWN))
        
        # Also log time series data
        rr.log("/live/timeline/fps", rr.Scalar(fps))
        rr.log("/live/timeline/latency", rr.Scalar(processing_time_ms))
        rr.log("/live/timeline/objects", rr.Scalar(num_objects))
    
    def log_scene_scale(self, scale_info: Dict):
        """Log scene scale information in an intuitive format."""
        scale_factor = scale_info.get('scale_factor', 1.0)
        confidence = scale_info.get('confidence', 0.0)
        num_estimates = scale_info.get('num_estimates', 0)
        
        # Create visual representation
        scale_doc = f"""# 📏 Scene Understanding

<div style="background: linear-gradient(135deg, #1a237e, #3949ab); padding: 20px; border-radius: 12px; text-align: center; color: white;">

## Real-World Scale Factor
<h1 style="font-size: 36px; margin: 10px 0;">{scale_factor:.4f} m/unit</h1>
<p style="font-size: 14px; opacity: 0.9;">or {1/scale_factor:.1f} units per meter</p>

### Confidence: {confidence:.0%}
<div style="background: rgba(255,255,255,0.2); height: 8px; border-radius: 4px; margin: 10px 0;">
<div style="background: #4fc3f7; height: 100%; width: {confidence*100}%; border-radius: 4px;"></div>
</div>

<p style="font-size: 12px; opacity: 0.8;">Based on {num_estimates} identified objects</p>

</div>

## Contributing Objects
"""
        
        # Add list of objects used for scale
        if 'estimates' in scale_info:
            scale_doc += "\n| Object | Dimensions | Confidence |\n|--------|------------|------------|\n"
            for est in scale_info['estimates']:
                dims = est['dimensions_m']
                scale_doc += f"| {est['product']} | {dims['width']:.2f}×{dims['height']:.2f}×{dims['depth']:.2f}m | {est['confidence']:.0%} |\n"
        
        # Log to both statistics panel and scene section
        rr.log("/scene/scale", rr.TextDocument(scale_doc, media_type=rr.MediaType.MARKDOWN))
        rr.log("/stats/scene_scale", rr.TextDocument(scale_doc, media_type=rr.MediaType.MARKDOWN))
    
    def _log_timeline_data(self, tracked_objects: Dict, frame_number: int):
        """Log timeline data for each tracked object."""
        for track_id, track in tracked_objects.items():
            # Log object presence
            rr.log(f"/live/timeline/object_{track_id}/present", rr.Scalar(1.0))
            
            # Log processing status
            if track.is_being_processed:
                rr.log(f"/live/timeline/object_{track_id}/processing", rr.Scalar(1.0))
            elif track.estimated_dimensions:
                rr.log(f"/live/timeline/object_{track_id}/identified", rr.Scalar(1.0))
            
            # Log quality score
            if track.best_score > 0:
                rr.log(f"/live/timeline/object_{track_id}/quality", rr.Scalar(track.best_score))
    
    def _get_quality_color(self, score: float) -> Tuple[int, int, int]:
        """Get color based on quality score."""
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


# Integration with existing frame processor
def integrate_enhanced_visualization(processor):
    """Integrate the enhanced visualizer with the existing frame processor."""
    
    # Create visualizer instance
    visualizer = EnhancedRerunVisualizer()
    
    # Monkey patch the process_frame method to use enhanced visualization
    original_process_frame = processor.process_frame
    
    def enhanced_process_frame(frame, properties, frame_number):
        # Call original processing
        result = original_process_frame(frame, properties, frame_number)
        
        # Get current tracking data
        tracked_objects = processor.tracker.tracked_objects
        detections = []  # Would need to capture from YOLO results
        
        # Log enhanced visualization
        visualizer.log_frame_with_overlays(frame, detections, tracked_objects, frame_number)
        
        # Update statistics
        gpu_memory_mb = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        visualizer.log_live_statistics(
            processor.frame_count,
            processor.last_frame_time * 1000,
            len(tracked_objects),
            gpu_memory_mb
        )
        
        # Update selected object (could be based on mouse click in future)
        if tracked_objects:
            # For now, select the first object with dimensions
            for track in tracked_objects.values():
                if track.estimated_dimensions:
                    visualizer.log_selected_object(track)
                    break
        
        # Update scene scale
        scale_info = processor.scene_scaler.calculate_weighted_scale()
        if scale_info['confidence'] > 0:
            visualizer.log_scene_scale(scale_info)
        
        return result
    
    # Monkey patch the enhancement worker to capture enhanced images
    original_enhancement_worker = processor.enhancement_worker
    
    def enhanced_enhancement_worker():
        """Enhanced worker that logs to gallery."""
        while True:
            try:
                track = processor.enhancement_queue.get(timeout=1)
                
                # Log processing start
                rr.log("/logs", rr.TextLog(
                    f"Processing object {track.id} ({track.class_name})",
                    level="INFO"
                ))
                
                # Extract best frame region
                x1, y1, x2, y2 = track.best_bbox
                roi = track.best_frame[y1:y2, x1:x2]
                
                # Enhance if enabled
                if processor.enhancement_enabled:
                    enhanced_roi = processor.enhancer.enhance_frame(roi)
                else:
                    enhanced_roi = roi
                
                # Process with API for dimensions
                start_time = time.time()
                dimension_result = processor.api_client.process_object_for_dimensions(
                    enhanced_roi, track.id, track.class_name
                )
                processing_time = time.time() - start_time
                
                if dimension_result:
                    # Update track with dimension info
                    track.identified_products = dimension_result.get('all_products', [])
                    track.estimated_dimensions = dimension_result.get('dimensions')
                    track.processing_time = processing_time
                    
                    # Add to scene scaler
                    processor.scene_scaler.add_dimension_estimate(
                        track.id, track.class_name, dimension_result, track.confidence
                    )
                    
                    # Log to gallery
                    visualizer.log_processed_object_to_gallery(track, enhanced_roi)
                    
                    # Update scene scale and publish
                    processor.update_and_publish_scene_scale()
                
                # Mark processing complete
                track.is_being_processed = False
                
            except queue.Empty:
                continue
            except Exception as e:
                rr.log("/logs", rr.TextLog(f"Processing error: {e}", level="ERROR"))
    
    # Replace methods
    processor.process_frame = enhanced_process_frame
    processor.enhancement_worker = enhanced_enhancement_worker
    
    # Start the enhancement worker thread
    if hasattr(processor, 'enhancement_thread'):
        processor.enhancement_thread = threading.Thread(target=enhanced_enhancement_worker)
        processor.enhancement_thread.daemon = True
        processor.enhancement_thread.start()
    
    return visualizer


# Usage Instructions:
"""
To use this enhanced visualization with your existing frame processor:

1. Import the visualizer:
   from enhanced_rerun_visualizer import integrate_enhanced_visualization

2. After creating your processor:
   processor = EnhancedFrameProcessor()
   visualizer = integrate_enhanced_visualization(processor)

3. The visualizer will automatically:
   - Show live camera feed with integrated bounding boxes on the left
   - Display processed objects gallery on the right
   - Update statistics and timeline in real-time
   
4. In the Rerun viewer, you can:
   - Collapse/expand the left panel to focus on live monitoring
   - Collapse/expand the right panel to focus on gallery
   - Use the entity tree to hide/show specific components
   - Adjust panel sizes by dragging dividers

Note: This implementation is compatible with Rerun 0.23.2 and uses:
- SpaceView for 2D camera and gallery views
- TextDocumentView for markdown-formatted information
- TimeSeriesView for temporal data
- Horizontal/Vertical layouts for organization
"""