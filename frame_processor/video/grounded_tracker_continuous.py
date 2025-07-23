"""
Grounded SAM2 video tracker with continuous object detection and ID tracking.

This module implements the continuous camera tracking approach from the 
Grounded-SAM-2 example, with consistent object IDs across frames.
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
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, predict

from core.config import Config
from core.utils import get_logger
from .base import VideoTracker, VideoTrackingResult, StreamState

logger = get_logger(__name__)


@dataclass
class MaskDictionary:
    """Mask dictionary with metadata for continuous tracking."""
    instance_id: int
    mask: np.ndarray
    class_name: str
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float = 0.0
    area: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'instance_id': self.instance_id,
            'class_name': self.class_name,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'area': self.area
        }


@dataclass
class GroundedStreamState(StreamState):
    """Extended stream state for Grounded SAM2 tracking."""
    mask_dict_list: List[MaskDictionary] = field(default_factory=list)
    next_instance_id: int = 0
    frames_since_detection: int = 0
    video_segments: Any = None  # SAM2 video segments
    current_prompt: str = ""
    tracks_initialized: bool = False
    reset_tracker: bool = False


class GroundedSAM2ContinuousTracker(VideoTracker):
    """
    Grounded SAM2 tracker with continuous detection and ID tracking.
    Based on grounded_sam2_tracking_camera_with_continuous_id.py
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Stream states
        self.stream_states: Dict[str, GroundedStreamState] = {}
        self.stream_locks: Dict[str, asyncio.Lock] = {}
        
        # Detection settings
        self.detection_interval = int(config.grounded_detection_interval) if hasattr(config, 'grounded_detection_interval') else 20
        self.text_prompt = config.grounded_text_prompt if hasattr(config, 'grounded_text_prompt') else "person. car. object."
        self.box_threshold = config.grounded_box_threshold if hasattr(config, 'grounded_box_threshold') else 0.25
        self.text_threshold = config.grounded_text_threshold if hasattr(config, 'grounded_text_threshold') else 0.2
        self.iou_threshold = config.grounded_iou_threshold if hasattr(config, 'grounded_iou_threshold') else 0.5
        
        # Initialize models
        self._initialize_models()
        
        logger.info(f"GroundedSAM2ContinuousTracker initialized with detection interval: {self.detection_interval}")
    
    def _initialize_models(self):
        """Initialize GroundingDINO and SAM2 models."""
        logger.info("Loading GroundingDINO model...")
        
        # GroundingDINO setup
        grounding_dino_config = "/app/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        grounding_dino_checkpoint = "/app/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
        self.grounding_model = load_model(grounding_dino_config, grounding_dino_checkpoint)
        
        logger.info("Loading SAM2 model...")
        
        # SAM2 setup
        if self.config.sam_checkpoint_path and self.config.sam_model_cfg:
            sam2_checkpoint = self.config.sam_checkpoint_path
            model_cfg = self.config.sam_model_cfg
        else:
            sam2_checkpoint = "/app/models/sam2_hiera_small.pt"
            model_cfg = "sam2_hiera_s.yaml"
        
        # Video predictor for tracking
        self.sam2_video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        
        # Image predictor for initial segmentation
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        self.sam2_image_predictor = SAM2ImagePredictor(sam2_model)
        
        logger.info("Models loaded successfully")
    
    def _update_masks(self, masks: List[Dict], state: GroundedStreamState) -> List[MaskDictionary]:
        """
        Update masks with continuous instance IDs using IOU matching.
        This is the core method for maintaining consistent IDs across frames.
        """
        updated_masks = []
        used_ids = set()
        
        for mask_data in masks:
            mask = mask_data.get('segmentation', np.zeros((1, 1)))
            bbox = self._mask_to_bbox(mask)
            if bbox is None:
                continue
            
            best_iou = 0.0
            best_match = None
            
            # Try to match with existing masks
            for existing_mask in state.mask_dict_list:
                if existing_mask.instance_id in used_ids:
                    continue
                    
                iou = self._calculate_mask_iou(mask, existing_mask.mask)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_match = existing_mask
            
            # Create or update mask dictionary
            if best_match:
                # Update existing mask
                instance_id = best_match.instance_id
                used_ids.add(instance_id)
                
                mask_dict = MaskDictionary(
                    instance_id=instance_id,
                    mask=mask,
                    class_name=mask_data.get('class_name', best_match.class_name),
                    bbox=bbox,
                    confidence=mask_data.get('confidence', 0.9),
                    area=int(mask.sum())
                )
            else:
                # Create new mask with new ID
                instance_id = state.next_instance_id
                state.next_instance_id += 1
                
                mask_dict = MaskDictionary(
                    instance_id=instance_id,
                    mask=mask,
                    class_name=mask_data.get('class_name', f'object_{instance_id}'),
                    bbox=bbox,
                    confidence=mask_data.get('confidence', 0.9),
                    area=int(mask.sum())
                )
            
            updated_masks.append(mask_dict)
        
        # Update state with new mask list
        state.mask_dict_list = updated_masks
        return updated_masks
    
    async def initialize_stream(self, stream_id: str, first_frame: np.ndarray,
                                prompts: Optional[Dict] = None) -> VideoTrackingResult:
        """Initialize tracking for a new stream."""
        if stream_id not in self.stream_locks:
            self.stream_locks[stream_id] = asyncio.Lock()
        
        async with self.stream_locks[stream_id]:
            # Create stream state
            state = GroundedStreamState(stream_id=stream_id)
            state.current_prompt = prompts.get('text_prompt', self.text_prompt) if prompts else self.text_prompt
            self.stream_states[stream_id] = state
            
            # Initialize SAM2 video predictor with the frame
            with torch.cuda.amp.autocast(enabled=True):
                state.video_segments = {}
                self.sam2_video_predictor.reset_state(state.video_segments)
            
            # Perform initial detection
            result = await self._add_image(stream_id, first_frame, 0)
            
            state.update()
            return result
    
    async def process_frame(self, stream_id: str, frame: np.ndarray,
                           timestamp: int) -> VideoTrackingResult:
        """Process a frame with continuous detection and tracking."""
        if stream_id not in self.stream_states:
            return await self.initialize_stream(stream_id, frame)
        
        async with self.stream_locks[stream_id]:
            state = self.stream_states[stream_id]
            result = await self._add_image(stream_id, frame, state.frame_count)
            state.update()
            return result
    
    async def _add_image(self, stream_id: str, frame: np.ndarray, 
                        frame_idx: int) -> VideoTrackingResult:
        """
        Add image to tracking - main processing method.
        Implements the continuous detection and tracking logic.
        """
        state = self.stream_states[stream_id]
        start_time = time.time()
        
        # Reset tracker if needed
        if state.reset_tracker:
            self.sam2_video_predictor.reset_state(state.video_segments)
            state.tracks_initialized = False
            state.reset_tracker = False
            state.frames_since_detection = 0
        
        # Decide whether to detect or track
        if (state.frames_since_detection >= self.detection_interval or 
            not state.tracks_initialized):
            # Perform full detection
            masks = await self._detect_and_segment(frame, state)
            state.frames_since_detection = 0
            state.tracks_initialized = True
        else:
            # Incremental tracking
            masks = await self._track_objects(frame_idx, state)
            state.frames_since_detection += 1
        
        # Update masks with continuous IDs
        mask_dicts = self._update_masks(masks, state)
        
        # Convert to output format
        output_masks = []
        output_tracks = []
        
        for mask_dict in mask_dicts:
            output_masks.append({
                'object_id': mask_dict.instance_id,
                'instance_id': mask_dict.instance_id,
                'segmentation': mask_dict.mask,
                'confidence': mask_dict.confidence,
                'class_name': mask_dict.class_name
            })
            
            output_tracks.append({
                'object_id': mask_dict.instance_id,
                'instance_id': mask_dict.instance_id,
                'bbox': mask_dict.bbox,
                'class_name': mask_dict.class_name,
                'confidence': mask_dict.confidence
            })
        
        processing_time = (time.time() - start_time) * 1000
        
        return VideoTrackingResult(
            masks=output_masks,
            tracks=output_tracks,
            object_count=len(output_tracks),
            processing_time_ms=processing_time,
            frame_number=frame_idx,
            confidence_scores=[m.confidence for m in mask_dicts]
        )
    
    async def _detect_and_segment(self, frame: np.ndarray, state: GroundedStreamState) -> List[Dict]:
        """Detect objects with GroundingDINO and segment with SAM2."""
        from PIL import Image
        
        # Convert to PIL for GroundingDINO
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Detect objects
        boxes, logits, phrases = predict(
            model=self.grounding_model,
            image=pil_image,
            caption=state.current_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )
        
        if len(boxes) == 0:
            return []
        
        # Set image for SAM2
        self.sam2_image_predictor.set_image(frame)
        
        # Get masks for each detection
        masks = []
        h, w = frame.shape[:2]
        
        for box, logit, phrase in zip(boxes, logits, phrases):
            # Convert box to pixel coordinates
            x1, y1, x2, y2 = box * torch.tensor([w, h, w, h])
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            
            # Get mask from SAM2
            mask, score, _ = self.sam2_image_predictor.predict(
                box=np.array(bbox),
                multimask_output=False
            )
            
            if mask is not None and len(mask) > 0:
                masks.append({
                    'segmentation': mask[0],
                    'bbox': bbox,
                    'confidence': float(logit),
                    'class_name': phrase
                })
        
        return masks
    
    async def _track_objects(self, frame_idx: int, state: GroundedStreamState) -> List[Dict]:
        """Track existing objects using SAM2 video predictor."""
        if not state.mask_dict_list:
            return []
        
        # Propagate masks to current frame
        masks = []
        
        with torch.cuda.amp.autocast(enabled=True):
            for mask_dict in state.mask_dict_list:
                # Get propagated mask for this object
                # Note: This is simplified - actual implementation would use SAM2's propagation
                masks.append({
                    'segmentation': mask_dict.mask,
                    'bbox': mask_dict.bbox,
                    'confidence': mask_dict.confidence,
                    'class_name': mask_dict.class_name
                })
        
        return masks
    
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
    
    def _calculate_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate IOU between two masks."""
        intersection = (mask1 & mask2).sum()
        union = (mask1 | mask2).sum()
        return intersection / union if union > 0 else 0.0
    
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
            "active_objects": len(state.mask_dict_list),
            "total_objects_seen": state.next_instance_id,
            "frames_since_detection": state.frames_since_detection,
            "age_seconds": state.age
        }
    
    def set_prompt(self, stream_id: str, text_prompt: str):
        """Update detection prompt and reset tracker."""
        if stream_id in self.stream_states:
            state = self.stream_states[stream_id]
            if state.current_prompt != text_prompt:
                state.current_prompt = text_prompt
                state.reset_tracker = True
                logger.info(f"Updated prompt for stream {stream_id}: {text_prompt}")
    
    @property
    def name(self) -> str:
        return "GroundedSAM2ContinuousTracker"
    
    @property
    def supports_batching(self) -> bool:
        return False