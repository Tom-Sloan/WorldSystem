"""
Grounded SAM2 video tracker with continuous object detection.

This module implements a video tracker that combines GroundingDINO for open-vocabulary
detection with SAM2 for segmentation and tracking. It performs periodic re-detection
to handle new objects entering the scene.
"""

import sys
import torch
import numpy as np
import time
import cv2
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio

# Add Grounded-SAM-2 to path
sys.path.append('/app/Grounded-SAM-2')
sys.path.append('/app/Grounded-SAM-2/grounding_dino')
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, predict

from core.config import Config
from core.utils import get_logger
from .base import VideoTracker, VideoTrackingResult, StreamState

logger = get_logger(__name__)


@dataclass
class TrackedObject:
    """Represents a tracked object with consistent ID."""
    instance_id: int  # Unique ID for continuous tracking
    mask: Optional[np.ndarray] = None
    mask_dict: Dict[str, Any] = field(default_factory=dict)  # Full mask dictionary
    class_name: str = ""
    bbox: List[int] = field(default_factory=lambda: [0, 0, 0, 0])  # [x1, y1, x2, y2]
    confidence: float = 0.0
    last_seen_frame: int = 0
    frames_since_seen: int = 0
    iou_history: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'instance_id': self.instance_id,
            'class_name': self.class_name,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'area': int(self.mask.sum()) if self.mask is not None else 0
        }


@dataclass
class GroundedStreamState(StreamState):
    """Extended stream state for Grounded SAM2 tracking."""
    tracked_objects: Dict[int, TrackedObject] = field(default_factory=dict)
    next_instance_id: int = 0  # For continuous ID tracking
    frames_since_detection: int = 0
    sam2_initialized: bool = False
    current_prompt: str = ""
    reset_tracker: bool = False
    tracks_initialized: bool = False


class GroundedSAM2Tracker(VideoTracker):
    """
    Video tracker combining GroundingDINO detection with SAM2 tracking.
    Performs continuous detection to handle new objects entering the scene.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Stream states
        self.stream_states: Dict[str, GroundedStreamState] = {}
        self.stream_locks: Dict[str, asyncio.Lock] = {}
        
        # Detection settings
        self.detection_interval = int(config.grounded_detection_interval) if hasattr(config, 'grounded_detection_interval') else 30
        self.text_prompt = config.grounded_text_prompt if hasattr(config, 'grounded_text_prompt') else "all objects. item. thing. stuff."
        self.box_threshold = config.grounded_box_threshold if hasattr(config, 'grounded_box_threshold') else 0.25
        self.text_threshold = config.grounded_text_threshold if hasattr(config, 'grounded_text_threshold') else 0.2
        self.iou_threshold = config.grounded_iou_threshold if hasattr(config, 'grounded_iou_threshold') else 0.5
        
        # Initialize models
        self._initialize_models()
        
        logger.info(f"GroundedSAM2Tracker initialized with detection interval: {self.detection_interval}")
    
    def _initialize_models(self):
        """Initialize GroundingDINO and SAM2 models."""
        logger.info("Loading GroundingDINO model...")
        
        # GroundingDINO setup
        grounding_dino_config = "/app/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        grounding_dino_checkpoint = "/app/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
        self.grounding_model = load_model(grounding_dino_config, grounding_dino_checkpoint)
        
        logger.info("Loading SAM2 model...")
        
        # SAM2 setup - use configuration from model_configs
        if self.config.sam_checkpoint_path and self.config.sam_model_cfg:
            sam2_checkpoint = self.config.sam_checkpoint_path
            model_cfg = self.config.sam_model_cfg
        else:
            # Default to small model
            sam2_checkpoint = "/app/models/sam2_hiera_small.pt"
            model_cfg = "sam2_hiera_s.yaml"
        
        # Image predictor for segmentation
        self.sam2_image_predictor = SAM2ImagePredictor(
            build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        )
        
        # Video predictor for tracking
        self.sam2_video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        
        logger.info("Models loaded successfully")
    
    async def initialize_stream(self, stream_id: str, first_frame: np.ndarray,
                                prompts: Optional[Dict] = None) -> VideoTrackingResult:
        """Initialize tracking for a new stream with first frame detection."""
        if stream_id not in self.stream_locks:
            self.stream_locks[stream_id] = asyncio.Lock()
        
        async with self.stream_locks[stream_id]:
            # Create stream state
            state = GroundedStreamState(stream_id=stream_id)
            state.current_prompt = prompts.get('text_prompt', self.text_prompt) if prompts else self.text_prompt
            self.stream_states[stream_id] = state
            
            # Initialize SAM2 video predictor
            self.sam2_video_predictor.load_first_frame(first_frame)
            state.sam2_initialized = True
            
            # Perform initial detection
            result = await self._detect_and_track(stream_id, first_frame, 0)
            
            state.update()
            return result
    
    async def process_frame(self, stream_id: str, frame: np.ndarray,
                           timestamp: int) -> VideoTrackingResult:
        """Process a frame with periodic re-detection."""
        if stream_id not in self.stream_states:
            # Auto-initialize if needed
            return await self.initialize_stream(stream_id, frame)
        
        async with self.stream_locks[stream_id]:
            state = self.stream_states[stream_id]
            
            # Decide whether to detect or just track
            if state.frames_since_detection >= self.detection_interval:
                # Perform full detection
                result = await self._detect_and_track(stream_id, frame, state.frame_count)
                state.frames_since_detection = 0
            else:
                # Just track existing objects
                result = await self._track_only(stream_id, frame, state.frame_count)
                state.frames_since_detection += 1
            
            state.update()
            return result
    
    async def _detect_and_track(self, stream_id: str, frame: np.ndarray,
                               frame_number: int) -> VideoTrackingResult:
        """Perform full detection with GroundingDINO and initialize/update tracking."""
        state = self.stream_states[stream_id]
        start_time = time.time()
        
        # Detect objects with GroundingDINO
        detections = await self._detect_objects(frame, state.current_prompt)
        
        if not detections:
            # No detections, just track existing
            return await self._track_only(stream_id, frame, frame_number)
        
        # Match detections to existing tracks
        matched_tracks, new_detections = self._match_detections_to_tracks(
            detections, state.tracked_objects
        )
        
        # Initialize new tracks for unmatched detections
        new_track_ids = []
        for det in new_detections:
            # Generate new object ID
            obj_id = state.next_object_id
            state.next_object_id += 1
            
            # Add to SAM2 tracker
            _, out_obj_ids, _ = self.sam2_video_predictor.add_new_prompt(
                frame_idx=frame_number,
                obj_id=obj_id,
                bbox=np.array([det['bbox']])
            )
            
            if out_obj_ids is not None and len(out_obj_ids) > 0:
                track_id = out_obj_ids[0]
                new_track_ids.append(track_id)
                
                # Create tracked object
                tracked = TrackedObject(
                    object_id=obj_id,
                    track_id=track_id,
                    class_name=det['class_name'],
                    bbox=det['bbox'],
                    confidence=det['confidence'],
                    last_seen_frame=frame_number
                )
                state.tracked_objects[obj_id] = tracked
        
        # Track all objects (existing + new)
        out_obj_ids, out_mask_logits = self.sam2_video_predictor.track(frame)
        
        # Process results
        masks = []
        tracks = []
        
        if out_obj_ids is not None and len(out_obj_ids) > 0:
            binary_masks = (out_mask_logits > 0.0).cpu().numpy()
            
            for i, (track_id, mask) in enumerate(zip(out_obj_ids, binary_masks)):
                # Find object by track ID
                obj = None
                for obj_id, tracked in state.tracked_objects.items():
                    if tracked.track_id == track_id:
                        obj = tracked
                        break
                
                if obj:
                    # Update object
                    obj.mask = mask
                    obj.last_seen_frame = frame_number
                    obj.frames_since_seen = 0
                    
                    # Update bbox from mask
                    bbox = self._mask_to_bbox(mask)
                    if bbox:
                        obj.bbox = bbox
                    
                    # Add to results
                    masks.append({
                        'object_id': obj.object_id,
                        'segmentation': mask,
                        'confidence': obj.confidence
                    })
                    
                    tracks.append({
                        'object_id': obj.object_id,
                        'track_id': obj.track_id,
                        'bbox': obj.bbox,
                        'class_name': obj.class_name,
                        'confidence': obj.confidence
                    })
        
        # Clean up lost tracks
        self._cleanup_lost_tracks(state, frame_number)
        
        processing_time = (time.time() - start_time) * 1000
        
        return VideoTrackingResult(
            masks=masks,
            tracks=tracks,
            object_count=len(tracks),
            processing_time_ms=processing_time,
            frame_number=frame_number,
            confidence_scores=[t['confidence'] for t in tracks]
        )
    
    async def _track_only(self, stream_id: str, frame: np.ndarray,
                         frame_number: int) -> VideoTrackingResult:
        """Track existing objects without new detection."""
        state = self.stream_states[stream_id]
        start_time = time.time()
        
        # Track with SAM2
        out_obj_ids, out_mask_logits = self.sam2_video_predictor.track(frame)
        
        masks = []
        tracks = []
        
        if out_obj_ids is not None and len(out_obj_ids) > 0:
            binary_masks = (out_mask_logits > 0.0).cpu().numpy()
            
            for track_id, mask in zip(out_obj_ids, binary_masks):
                # Find object by track ID
                obj = None
                for obj_id, tracked in state.tracked_objects.items():
                    if tracked.track_id == track_id:
                        obj = tracked
                        break
                
                if obj:
                    # Update tracking info
                    obj.mask = mask
                    obj.last_seen_frame = frame_number
                    obj.frames_since_seen = 0
                    
                    # Update bbox from mask
                    bbox = self._mask_to_bbox(mask)
                    if bbox:
                        obj.bbox = bbox
                    
                    masks.append({
                        'object_id': obj.object_id,
                        'segmentation': mask,
                        'confidence': obj.confidence
                    })
                    
                    tracks.append({
                        'object_id': obj.object_id,
                        'track_id': obj.track_id,
                        'bbox': obj.bbox,
                        'class_name': obj.class_name,
                        'confidence': obj.confidence
                    })
        
        # Update frames since seen for missing objects
        for obj_id, obj in state.tracked_objects.items():
            if obj.last_seen_frame < frame_number:
                obj.frames_since_seen += 1
        
        # Clean up lost tracks
        self._cleanup_lost_tracks(state, frame_number)
        
        processing_time = (time.time() - start_time) * 1000
        
        return VideoTrackingResult(
            masks=masks,
            tracks=tracks,
            object_count=len(tracks),
            processing_time_ms=processing_time,
            frame_number=frame_number,
            confidence_scores=[t['confidence'] for t in tracks]
        )
    
    async def _detect_objects(self, frame: np.ndarray, text_prompt: str) -> List[Dict]:
        """Detect objects using GroundingDINO."""
        from PIL import Image
        
        # Convert to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Run detection
        boxes, logits, phrases = predict(
            model=self.grounding_model,
            image=pil_image,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )
        
        # Convert to list of detections
        detections = []
        h, w = frame.shape[:2]
        
        for box, logit, phrase in zip(boxes, logits, phrases):
            # Convert normalized coords to pixels
            x1, y1, x2, y2 = box * torch.tensor([w, h, w, h])
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            
            detections.append({
                'bbox': bbox,
                'confidence': float(logit),
                'class_name': phrase
            })
        
        return detections
    
    def _update_masks_with_continuous_id(self, masks: List[Dict], state: GroundedStreamState) -> List[Dict]:
        """
        Update masks with continuous instance IDs using IOU matching.
        This is the key method for continuous ID tracking across frames.
        """
        if not masks:
            return []
        
        updated_masks = []
        used_ids = set()
        
        # For each new mask, find the best matching existing object
        for mask_dict in masks:
            mask = mask_dict.get('segmentation', np.zeros((1, 1)))
            bbox = self._mask_to_bbox(mask)
            
            if bbox is None:
                continue
            
            best_iou = 0.0
            best_match_id = None
            
            # Compare with all tracked objects
            for obj_id, tracked_obj in state.tracked_objects.items():
                if obj_id in used_ids:
                    continue
                    
                # Calculate IOU between current mask and tracked object
                if tracked_obj.bbox and len(tracked_obj.bbox) == 4:
                    iou = self._calculate_iou(bbox, tracked_obj.bbox)
                    if iou > best_iou and iou > self.iou_threshold:
                        best_iou = iou
                        best_match_id = obj_id
            
            # Assign ID
            if best_match_id is not None:
                # Update existing object
                instance_id = best_match_id
                used_ids.add(instance_id)
                tracked_obj = state.tracked_objects[instance_id]
                tracked_obj.mask = mask
                tracked_obj.bbox = bbox
                tracked_obj.confidence = mask_dict.get('confidence', 0.9)
                tracked_obj.last_seen_frame = state.frame_count
                tracked_obj.frames_since_seen = 0
            else:
                # Create new object with new ID
                instance_id = state.next_instance_id
                state.next_instance_id += 1
                
                tracked_obj = TrackedObject(
                    instance_id=instance_id,
                    mask=mask,
                    mask_dict=mask_dict,
                    class_name=mask_dict.get('class_name', f'object_{instance_id}'),
                    bbox=bbox,
                    confidence=mask_dict.get('confidence', 0.9),
                    last_seen_frame=state.frame_count
                )
                state.tracked_objects[instance_id] = tracked_obj
            
            # Update mask dict with instance ID
            mask_dict['instance_id'] = instance_id
            updated_masks.append(mask_dict)
        
        # Update frames_since_seen for unmatched objects
        for obj_id, obj in state.tracked_objects.items():
            if obj_id not in used_ids and obj.last_seen_frame < state.frame_count:
                obj.frames_since_seen += 1
        
        return updated_masks
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union between two bboxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max < x_min or y_max < y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _mask_to_bbox(self, mask: np.ndarray) -> Optional[List[int]]:
        """Convert binary mask to bounding box."""
        if mask.sum() == 0:
            return None
        
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None
        
        return [
            int(x_indices.min()),
            int(y_indices.min()),
            int(x_indices.max()),
            int(y_indices.max())
        ]
    
    def _cleanup_lost_tracks(self, state: GroundedStreamState, current_frame: int):
        """Remove tracks that have been lost for too long."""
        to_remove = []
        
        for obj_id, obj in state.tracked_objects.items():
            if obj.frames_since_seen > self.config.tracker_max_lost:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del state.tracked_objects[obj_id]
            logger.debug(f"Removed lost track {obj_id}")
    
    async def cleanup_stream(self, stream_id: str):
        """Clean up resources for a stream."""
        if stream_id in self.stream_states:
            del self.stream_states[stream_id]
        if stream_id in self.stream_locks:
            del self.stream_locks[stream_id]
        
        logger.info(f"Cleaned up stream {stream_id}")
    
    def get_stream_status(self, stream_id: str) -> Dict[str, Any]:
        """Get current status of a stream."""
        if stream_id not in self.stream_states:
            return {"status": "not_found"}
        
        state = self.stream_states[stream_id]
        return {
            "status": "active" if state.is_active else "inactive",
            "frame_count": state.frame_count,
            "active_tracks": len([obj for obj in state.tracked_objects.values() 
                                if obj.frames_since_seen < self.config.tracker_max_lost]),
            "total_objects_seen": state.next_object_id,
            "frames_since_detection": state.frames_since_detection,
            "age_seconds": state.age
        }
    
    @property
    def name(self) -> str:
        """Tracker name for logging."""
        return "GroundedSAM2Tracker"
    
    @property
    def supports_batching(self) -> bool:
        """Whether this tracker supports batch processing."""
        return False