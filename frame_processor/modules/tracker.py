import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time


@dataclass
class TrackedObject:
    """Represents a tracked object with its history and metadata."""
    id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_name: str
    confidence: float
    frame_history: List[Dict] = field(default_factory=list)
    best_frame: Optional[np.ndarray] = None
    best_score: float = 0.0
    last_seen_frame: int = 0
    first_seen_time: float = field(default_factory=time.time)
    last_processed_time: float = 0.0
    processing_count: int = 0
    is_being_processed: bool = False
    best_bbox: Optional[Tuple[int, int, int, int]] = None
    best_frame_number: int = 0
    score_components: Optional[Dict] = None
    # For dimension tracking
    identified_products: List[Dict] = field(default_factory=list)
    estimated_dimensions: Optional[Dict] = None  # {width, height, depth, unit, confidence}


class ObjectTracker:
    """Tracks objects across frames with IOU-based matching."""
    
    def __init__(self, 
                 iou_threshold=0.3,
                 max_lost_frames=10,
                 process_after_seconds=1.5,
                 reprocess_interval_seconds=3.0,
                 max_frame_history=50):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.process_after_seconds = process_after_seconds
        self.reprocess_interval_seconds = reprocess_interval_seconds
        self.max_frame_history = max_frame_history  # Limit frame history to prevent memory issues
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_id = 0
        
    def calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                      bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
            
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def update(self, detections: List[Tuple], frame: np.ndarray, frame_number: int) -> List[TrackedObject]:
        """Update tracking with new detections and return objects ready for processing."""
        current_time = time.time()
        objects_to_process = []
        
        # Match detections to existing tracks
        matched_detection_indices = set()
        
        for detection_idx, (bbox, class_name, confidence) in enumerate(detections):
            best_iou = 0
            best_track_id = None
            
            # Find best matching track
            for track_id, track in self.tracked_objects.items():
                if track.class_name == class_name:
                    iou = self.calculate_iou(bbox, track.bbox)
                    if iou > best_iou and iou > self.iou_threshold:
                        best_iou = iou
                        best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                track = self.tracked_objects[best_track_id]
                track.bbox = bbox
                track.confidence = confidence
                track.last_seen_frame = frame_number
                track.frame_history.append({
                    'frame': frame.copy(),
                    'bbox': bbox,
                    'frame_number': frame_number,
                    'confidence': confidence,
                    'timestamp': current_time
                })
                # Limit frame history to prevent memory issues
                if len(track.frame_history) > self.max_frame_history:
                    track.frame_history.pop(0)  # Remove oldest frame
                matched_detection_indices.add(detection_idx)
            else:
                # Create new track
                new_track = TrackedObject(
                    id=self.next_id,
                    bbox=bbox,
                    class_name=class_name,
                    confidence=confidence,
                    frame_history=[{
                        'frame': frame.copy(),
                        'bbox': bbox,
                        'frame_number': frame_number,
                        'confidence': confidence,
                        'timestamp': current_time
                    }],
                    last_seen_frame=frame_number,
                    first_seen_time=current_time
                )
                self.tracked_objects[self.next_id] = new_track
                self.next_id += 1
        
        # Check for objects ready for processing
        for track_id, track in self.tracked_objects.items():
            time_since_first_seen = current_time - track.first_seen_time
            time_since_last_processed = current_time - track.last_processed_time
            
            if (time_since_first_seen >= self.process_after_seconds and 
                not track.is_being_processed and
                (track.processing_count == 0 or 
                 time_since_last_processed >= self.reprocess_interval_seconds)):
                
                objects_to_process.append(track)
                track.is_being_processed = True
                track.last_processed_time = current_time
                track.processing_count += 1
        
        # Remove lost tracks
        lost_track_ids = []
        for track_id, track in self.tracked_objects.items():
            if frame_number - track.last_seen_frame > self.max_lost_frames:
                lost_track_ids.append(track_id)
        
        for track_id in lost_track_ids:
            del self.tracked_objects[track_id]
        
        return objects_to_process
    
    def get_active_tracks(self) -> List[TrackedObject]:
        """Get all currently active tracks."""
        return list(self.tracked_objects.values())
    
    def get_tracks_with_dimensions(self) -> List[TrackedObject]:
        """Get tracks that have estimated dimensions."""
        return [track for track in self.tracked_objects.values() 
                if track.estimated_dimensions is not None]