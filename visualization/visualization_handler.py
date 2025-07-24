"""
Handler for processing visualization messages and logging to Rerun.
Decodes base64 images and creates structured visualizations.
"""

import base64
import numpy as np
import cv2
import rerun as rr
from typing import Dict, Any, List, Optional
import json
import time
from collections import deque

from blueprint_manager import BlueprintManager, ViewMode


class VisualizationHandler:
    """Handles incoming visualization messages and logs to Rerun."""
    
    def __init__(self):
        """Initialize the visualization handler."""
        self.blueprint_manager = BlueprintManager()
        self.frame_count = 0
        self.last_update_time = time.time()
        self.fps_tracker = deque(maxlen=30)
        
        # Gallery management
        self.enhanced_gallery = deque(maxlen=20)  # FIFO gallery
        self.object_timelines = {}  # object_id -> timeline data
        
        # Statistics tracking
        self.stats = {
            'total_frames': 0,
            'total_objects': 0,
            'active_tracks': 0,
            'processing_fps': 0.0,
            'last_update': time.time()
        }
        
        # Apply initial blueprint
        self.apply_blueprint(ViewMode.BOTH)
    
    def apply_blueprint(self, view_mode: ViewMode):
        """Apply the visualization blueprint."""
        blueprint = self.blueprint_manager.create_blueprint(view_mode)
        rr.send_blueprint(blueprint)
    
    def decode_base64_image(self, base64_str: str) -> Optional[np.ndarray]:
        """Decode base64 string to numpy array."""
        try:
            # Decode base64
            img_bytes = base64.b64decode(base64_str)
            
            # Convert to numpy array
            nparr = np.frombuffer(img_bytes, np.uint8)
            
            # Decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert BGR to RGB
            if img is not None and len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            return img
        except Exception as e:
            print(f"Error decoding base64 image: {e}")
            return None
    
    def decode_masks(self, base64_str: str, mask_info: List[Dict]) -> Dict[int, np.ndarray]:
        """Decode base64 encoded masks."""
        if not base64_str:
            return {}
        
        try:
            # Decode base64
            img_bytes = base64.b64decode(base64_str)
            nparr = np.frombuffer(img_bytes, np.uint8)
            
            # Decode multi-channel mask image
            mask_array = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            
            if mask_array is None:
                return {}
            
            # Extract individual masks
            masks = {}
            for i, info in enumerate(mask_info):
                if i < mask_array.shape[2]:
                    mask = mask_array[:, :, i] > 0
                    masks[info['instance_id']] = mask
            
            return masks
        except Exception as e:
            print(f"Error decoding masks: {e}")
            return {}
    
    def process_message(self, message_data: Dict[str, Any]):
        """Process incoming visualization message."""
        msg_type = message_data.get('type')
        
        if msg_type == 'visualization_update':
            self.handle_visualization_update(message_data)
        elif msg_type == 'processed_frame':
            self.handle_processed_frame(message_data)
        elif msg_type == 'session_complete':
            self.handle_session_complete(message_data)
    
    def handle_visualization_update(self, message: Dict[str, Any]):
        """Handle visualization update message."""
        frame_number = message.get('frame_number', 0)
        timestamp = message.get('timestamp', time.time())
        data = message.get('data', {})
        
        # Set time context
        rr.set_time_seconds("time", timestamp)
        rr.set_time_sequence("frame", frame_number)
        
        # Process each data type
        frame_data = data.get('frame', {})
        vis_data = data.get('visualization', {})
        mask_data = data.get('masks', {})
        enhanced_data = data.get('enhanced', {})
        stats_data = data.get('statistics', {})
        
        # 1. Log original frame (Page 1: Live Camera View)
        if frame_data.get('content'):
            frame = self.decode_base64_image(frame_data['content'])
            if frame is not None:
                rr.log("page1/live_camera/frame", rr.Image(frame))
        
        # 2. Log visualization with overlays (Page 1: Frame Grid)
        if vis_data.get('content'):
            vis_frame = self.decode_base64_image(vis_data['content'])
            if vis_frame is not None:
                # Log to frame grid (updates every 10 frames)
                if frame_number % 10 == 0:
                    grid_idx = (frame_number // 10) % 12  # 3x4 grid
                    rr.log(f"page1/frame_grid/frame_{grid_idx}", rr.Image(vis_frame))
                
                # Also log as main visualization
                rr.log("page1/live_camera/visualization", rr.Image(vis_frame))
        
        # 3. Process masks if available
        if mask_data.get('content'):
            masks = self.decode_masks(mask_data['content'], mask_data['metadata']['mask_info'])
            self.visualize_masks(masks, mask_data['metadata']['mask_info'])
        
        # 4. Update enhanced gallery (Page 2)
        if enhanced_data.get('content'):
            self.update_enhanced_gallery(enhanced_data['content'])
        
        # 5. Update statistics
        if stats_data.get('content'):
            self.update_statistics(stats_data['content'])
        
        # Track FPS
        current_time = time.time()
        self.fps_tracker.append(current_time)
        if len(self.fps_tracker) > 1:
            fps = len(self.fps_tracker) / (self.fps_tracker[-1] - self.fps_tracker[0])
            self.stats['processing_fps'] = fps
        
        self.frame_count += 1
    
    def visualize_masks(self, masks: Dict[int, np.ndarray], mask_info: List[Dict]):
        """Visualize segmentation masks with colors."""
        if not masks:
            return
        
        # Create composite mask visualization
        first_mask = next(iter(masks.values()))
        h, w = first_mask.shape
        composite = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Color palette
        colors = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255],
            [255, 255, 0], [255, 0, 255], [0, 255, 255],
            [255, 128, 0], [128, 255, 0], [255, 0, 128],
            [0, 128, 255], [128, 0, 255], [255, 128, 128]
        ]
        
        # Apply colors to masks
        for i, (obj_id, mask) in enumerate(masks.items()):
            color = colors[i % len(colors)]
            composite[mask] = color
        
        # Log composite mask
        rr.log("page1/live_camera/masks", rr.Image(composite))
        
        # Update object timeline data
        for info in mask_info:
            obj_id = info['instance_id']
            if obj_id not in self.object_timelines:
                self.object_timelines[obj_id] = []
            
            self.object_timelines[obj_id].append({
                'frame': self.frame_count,
                'confidence': info.get('confidence', 1.0),
                'area': info.get('area', 0)
            })
    
    def update_enhanced_gallery(self, enhanced_objects: List[Dict]):
        """Update the enhanced objects gallery."""
        for obj_data in enhanced_objects:
            if 'data' in obj_data:
                img = self.decode_base64_image(obj_data['data'])
                if img is not None:
                    # Add to gallery with metadata
                    self.enhanced_gallery.append({
                        'image': img,
                        'metadata': obj_data.get('metadata', {}),
                        'timestamp': time.time()
                    })
        
        # Update gallery view (Page 2)
        for i, gallery_item in enumerate(list(self.enhanced_gallery)[-12:]):  # Show last 12
            rr.log(f"page2/gallery/object_{i}", rr.Image(gallery_item['image']))
    
    def update_statistics(self, stats: Dict[str, Any]):
        """Update and display statistics."""
        self.stats.update(stats)
        
        # Format statistics for Page 1
        live_stats = f"""# 📊 Live Statistics

**Frame**: {self.stats.get('frame_count', 0)}
**Active Objects**: {self.stats.get('active_tracks', 0)}
**Unique Objects**: {self.stats.get('unique_objects', 0)}
**Processing FPS**: {self.stats.get('processing_fps', 0):.1f}
"""
        rr.log("page1/stats/live", rr.TextDocument(live_stats, media_type="text/markdown"))
        
        # Format process statistics for Page 2
        process_stats = f"""# 📈 Process Statistics

**Total Frames**: {self.stats.get('frame_count', 0)}
**Objects Tracked**: {self.stats.get('unique_objects', 0)}
**Gallery Items**: {len(self.enhanced_gallery)}
**Avg Processing**: {self.stats.get('processing_time_ms', 0):.1f}ms
"""
        rr.log("page2/stats/process", rr.TextDocument(process_stats, media_type="text/markdown"))
        
        # Update timeline for active objects
        if self.frame_count % 30 == 0:  # Update every 30 frames
            self.update_object_timelines()
    
    def update_object_timelines(self):
        """Update object tracking timelines."""
        # Create timeline data for each tracked object
        for obj_id, timeline in self.object_timelines.items():
            if timeline:
                frames = [t['frame'] for t in timeline[-100:]]  # Last 100 points
                confidences = [t['confidence'] for t in timeline[-100:]]
                
                # Log as time series
                for frame, conf in zip(frames, confidences):
                    rr.set_time_sequence("frame", frame)
                    rr.log(f"page1/timeline/object_{obj_id}/confidence", rr.TimeSeriesScalar(conf))
    
    def handle_processed_frame(self, message: Dict[str, Any]):
        """Handle processed frame message from frame processor."""
        frame = message.get('frame')
        if frame is None:
            return
            
        frame_number = message.get('frame_number', 0)
        timestamp_ns = message.get('timestamp_ns', 0)
        timestamp = timestamp_ns / 1e9 if timestamp_ns > 0 else time.time()
        
        # Set time context
        rr.set_time_seconds("time", timestamp)
        rr.set_time_sequence("frame", frame_number)
        
        # Update stats
        self.frame_count += 1
        self.stats['total_frames'] += 1
        
        # Convert BGR to RGB if needed (OpenCV uses BGR by default)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
            
        # Log the processed frame to multiple views
        # 1. Main live camera view
        rr.log("page1/live_camera/frame", rr.Image(frame_rgb))
        
        # 2. Frame grid (updates every 10 frames)
        if frame_number % 10 == 0:
            grid_idx = (frame_number // 10) % 12  # 3x4 grid
            rr.log(f"page1/frame_grid/frame_{grid_idx}", rr.Image(frame_rgb))
        
        # 3. Log detection summary if available
        class_summary = message.get('class_summary', {})
        if class_summary:
            detection_text = "Detections:\n"
            for class_name, count in class_summary.items():
                detection_text += f"- {class_name}: {count}\n"
            rr.log("page1/detections", rr.TextDocument(detection_text))
        
        # 4. Update statistics
        processing_time = message.get('processing_time_ms', 0)
        detection_count = message.get('detection_count', 0)
        
        stats_text = f"""# 📊 Processing Statistics

**Frame**: {frame_number}
**Processing Time**: {processing_time:.1f}ms
**Detections**: {detection_count}
**Total Frames**: {self.stats['total_frames']}
**Active Tracks**: {self.stats.get('active_tracks', 0)}
"""
        rr.log("page1/statistics", rr.TextDocument(stats_text, media_type="text/markdown"))
        
        # Track FPS
        current_time = time.time()
        self.fps_tracker.append(current_time)
        if len(self.fps_tracker) > 1:
            fps = len(self.fps_tracker) / (self.fps_tracker[-1] - self.fps_tracker[0])
            self.stats['processing_fps'] = fps
            rr.log("page1/metrics/fps", rr.TimeSeriesScalar(fps))
    
    def handle_session_complete(self, message: Dict[str, Any]):
        """Handle session completion message."""
        summary = f"""# 🎬 Session Complete

**Session ID**: {message.get('session_id', 'unknown')}
**Total Frames**: {message.get('total_frames', 0)}
**Unique Objects**: {message.get('unique_objects', 0)}
"""
        rr.log("session_summary", rr.TextDocument(summary, media_type="text/markdown"))