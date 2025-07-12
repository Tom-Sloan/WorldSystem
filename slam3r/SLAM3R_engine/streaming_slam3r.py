#!/usr/bin/env python3
"""
StreamingSLAM3R: A clean architecture for real-time SLAM3R processing.

This module provides a proper abstraction layer between streaming input
and the batch-oriented SLAM3R models, handling all dimension management
and state tracking in a clean, maintainable way.
"""

import torch
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, field
import asyncio
from threading import Lock

# SLAM3R imports
from recon import (
    get_img_tokens,
    initialize_scene,
    adapt_keyframe_stride,
    i2p_inference_batch,
    l2w_inference,
    scene_frame_retrieve,
)
from slam3r.models import Image2PointsModel, Local2WorldModel
from slam3r.utils.device import to_device, collate_with_cat
from slam3r.utils.recon_utils import unsqueeze_view

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Clean data structure for frame information"""
    timestamp: int
    image_tensor: torch.Tensor
    true_shape: torch.Tensor
    frame_id: int
    
    # Computed features (filled later)
    img_tokens: Optional[torch.Tensor] = None
    img_pos: Optional[torch.Tensor] = None
    pts3d_cam: Optional[torch.Tensor] = None
    pts3d_world: Optional[torch.Tensor] = None
    conf_cam: Optional[torch.Tensor] = None
    conf_world: Optional[torch.Tensor] = None
    is_keyframe: bool = False


class TokenCache:
    """Efficient token storage and retrieval with proper dimension management"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.cache = {}
        self.lock = Lock()
    
    def add(self, frame_id: int, img_tokens: torch.Tensor, img_pos: torch.Tensor):
        """Store tokens with consistent dimensions"""
        with self.lock:
            # Store tokens as-is, maintaining their dimensions
            self.cache[frame_id] = {
                'img_tokens': img_tokens.to(self.device),
                'img_pos': img_pos.to(self.device)
            }
    
    def get(self, frame_id: int) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieve tokens for a frame"""
        with self.lock:
            return self.cache.get(frame_id)
    
    def cleanup(self, keep_recent: int = 100):
        """Remove old tokens to manage memory"""
        with self.lock:
            if len(self.cache) > keep_recent:
                # Keep only the most recent frames
                sorted_ids = sorted(self.cache.keys())
                for fid in sorted_ids[:-keep_recent]:
                    del self.cache[fid]


class BatchAccumulator:
    """Accumulate frames for efficient batch processing"""
    
    def __init__(self, target_batch_size: int = 5, timeout_ms: int = 100):
        self.target_size = target_batch_size
        self.timeout_ms = timeout_ms
        self.frames = []
        self.last_add_time = None
        self.lock = Lock()
    
    def add_frame(self, frame: FrameData) -> Optional[List[FrameData]]:
        """Add frame and return batch if ready"""
        import time
        
        with self.lock:
            self.frames.append(frame)
            current_time = time.time() * 1000
            
            if self.last_add_time is None:
                self.last_add_time = current_time
            
            # Return batch if we have enough frames or timeout
            if (len(self.frames) >= self.target_size or 
                (current_time - self.last_add_time) > self.timeout_ms):
                batch = self.frames
                self.frames = []
                self.last_add_time = None
                return batch
            
            return None


class ViewFactory:
    """Factory for creating consistent view structures"""
    
    @staticmethod
    def create_view(frame: FrameData, include_image: bool = False) -> Dict[str, Any]:
        """Create a view with consistent structure"""
        view = {
            'true_shape': frame.true_shape,
            'frame_id': frame.frame_id,
            'timestamp': frame.timestamp
        }
        
        if include_image and frame.image_tensor is not None:
            view['img'] = frame.image_tensor
        
        if frame.img_tokens is not None:
            view['img_tokens'] = frame.img_tokens
            
        if frame.img_pos is not None:
            view['img_pos'] = frame.img_pos
            
        if frame.pts3d_cam is not None:
            view['pts3d_cam'] = frame.pts3d_cam
            
        if frame.pts3d_world is not None:
            view['pts3d_world'] = frame.pts3d_world
            
        return view
    
    @staticmethod
    def prepare_for_model(views: List[Dict[str, Any]], 
                         add_batch_dim: bool = True) -> List[Dict[str, Any]]:
        """Prepare views for model input with proper dimensions"""
        prepared = []
        
        for view in views:
            prep_view = {}
            
            for key, value in view.items():
                if isinstance(value, torch.Tensor) and add_batch_dim:
                    # Add batch dimension if needed
                    if key in ['img', 'img_tokens', 'img_pos', 'pts3d_cam', 'pts3d_world']:
                        if key == 'true_shape' and value.dim() == 1:
                            prep_view[key] = value.unsqueeze(0)
                        elif key != 'true_shape' and value.dim() == 2:
                            prep_view[key] = value.unsqueeze(0)
                        else:
                            prep_view[key] = value
                    else:
                        prep_view[key] = value
                else:
                    prep_view[key] = value
                    
            prepared.append(prep_view)
            
        return prepared


class SlidingWindowProcessor:
    """Manage a sliding window of frames for reference selection"""
    
    def __init__(self, window_size: int = 20, keyframe_stride: int = 5):
        self.window = deque(maxlen=window_size)
        self.keyframe_stride = keyframe_stride
        self.keyframes = []
        
    def add_frame(self, frame: FrameData):
        """Add frame to window and update keyframes"""
        self.window.append(frame)
        
        # Check if this should be a keyframe
        if frame.frame_id % self.keyframe_stride == 0:
            frame.is_keyframe = True
            self.keyframes.append(frame)
            
            # Limit keyframes
            if len(self.keyframes) > 10:
                self.keyframes = self.keyframes[-10:]
    
    def get_reference_frames(self, num_refs: int = 5) -> List[FrameData]:
        """Select best reference frames from window"""
        # For now, return the most recent keyframes
        return self.keyframes[-num_refs:] if len(self.keyframes) >= num_refs else list(self.keyframes)


class StreamingSLAM3R:
    """Main class for streaming SLAM3R processing"""
    
    def __init__(self, 
                 i2p_model: Image2PointsModel,
                 l2w_model: Local2WorldModel,
                 config: Dict[str, Any],
                 device: str = 'cuda'):
        
        self.i2p_model = i2p_model
        self.l2w_model = l2w_model
        self.config = config
        self.device = device
        
        # Core components
        self.token_cache = TokenCache(device)
        self.batch_accumulator = BatchAccumulator(
            target_batch_size=config.get('batch_size', 5)
        )
        self.window_processor = SlidingWindowProcessor(
            window_size=config.get('window_size', 20),
            keyframe_stride=config.get('initial_keyframe_stride', 5)
        )
        
        # State tracking
        self.frame_counter = 0
        self.is_initialized = False
        self.initialization_frames = []
        
        logger.info("StreamingSLAM3R initialized with config: %s", config)
    
    def process_frame(self, image: np.ndarray, timestamp: int) -> Optional[Dict[str, Any]]:
        """
        Process a single frame through the SLAM3R pipeline.
        
        Args:
            image: Input image as numpy array
            timestamp: Timestamp in nanoseconds
            
        Returns:
            Dict with pose, points, and confidence if successful
        """
        logger.info(f"Processing frame {self.frame_counter} with timestamp {timestamp}")
        
        # Create frame data
        frame = self._create_frame(image, timestamp)
        
        # Generate tokens immediately
        self._generate_tokens(frame)
        
        # Add to window
        self.window_processor.add_frame(frame)
        
        # Handle initialization
        if not self.is_initialized:
            logger.info(f"Handling initialization, have {len(self.initialization_frames)} frames")
            return self._handle_initialization(frame)
        
        # Try to accumulate for batch processing
        batch = self.batch_accumulator.add_frame(frame)
        if batch is not None:
            logger.info(f"Processing batch of {len(batch)} frames")
            return self._process_batch(batch)
        
        logger.debug("Frame added to accumulator, waiting for batch")
        return None
    
    def _create_frame(self, image: np.ndarray, timestamp: int) -> FrameData:
        """Create a FrameData object from raw input"""
        # Convert image to tensor and normalize
        img_tensor = torch.from_numpy(image).float() / 255.0
        if img_tensor.dim() == 2:
            img_tensor = img_tensor.unsqueeze(-1).repeat(1, 1, 3)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Get true shape
        true_shape = torch.tensor([image.shape[0], image.shape[1]], 
                                 dtype=torch.float32, device=self.device)
        
        frame = FrameData(
            timestamp=timestamp,
            image_tensor=img_tensor,
            true_shape=true_shape,
            frame_id=self.frame_counter
        )
        
        self.frame_counter += 1
        return frame
    
    def _generate_tokens(self, frame: FrameData):
        """Generate and cache tokens for a frame"""
        # Prepare input for get_img_tokens
        view = {'img': frame.image_tensor, 'true_shape': frame.true_shape.unsqueeze(0)}
        
        # Generate tokens
        _, img_tokens, img_pos = get_img_tokens([view], self.i2p_model)
        
        # Store in frame (with batch dim for initialize_scene compatibility)
        # img_tokens and img_pos should have shape [1, 196, C]
        frame.img_tokens = img_tokens[0] if img_tokens[0].dim() == 3 else img_tokens[0].unsqueeze(0)
        frame.img_pos = img_pos[0] if img_pos[0].dim() == 3 else img_pos[0].unsqueeze(0)
        
        # Cache tokens
        self.token_cache.add(frame.frame_id, frame.img_tokens, frame.img_pos)
    
    def _handle_initialization(self, frame: FrameData) -> Optional[Dict[str, Any]]:
        """Handle initialization phase"""
        self.initialization_frames.append(frame)
        
        if len(self.initialization_frames) >= self.config.get('init_frames', 5):
            # Run initialization
            success = self._initialize_slam()
            if success:
                self.is_initialized = True
                # Process the initialization frames
                return self._process_batch(self.initialization_frames)
            else:
                # Reset and try again
                self.initialization_frames = []
                
        return None
    
    def _initialize_slam(self) -> bool:
        """Initialize SLAM with first frames"""
        try:
            # Create views for initialization
            # The views need img_tokens, img_pos, and true_shape for initialize_scene to work
            init_views = []
            for frame in self.initialization_frames:
                # Make sure frame has tokens generated
                if frame.img_tokens is None:
                    self._generate_tokens(frame)
                
                # Create view with required fields
                view = {
                    'img_tokens': frame.img_tokens,
                    'img_pos': frame.img_pos,
                    'true_shape': frame.true_shape,
                    'label': f'frame_{frame.frame_id}'
                }
                init_views.append(view)
            
            # Run initialization
            # initialize_scene returns (pcs, confs) or (pcs, confs, ref_id)
            pcs, confs, ref_id = initialize_scene(
                init_views, 
                self.i2p_model,
                winsize=self.config.get('initial_winsize', 5),
                conf_thres=self.config.get('conf_thres_i2p', 5.0),
                return_ref_id=True
            )
            
            # Check if initialization was successful
            if pcs is not None and len(pcs) > 0:
                # Store results in frames
                for i, frame_data in enumerate(zip(self.initialization_frames, pcs, confs)):
                    frame, pc, conf = frame_data
                    frame.pts3d_world = pc
                    frame.conf_world = conf
                    frame.is_keyframe = True
                
                logger.info("SLAM initialization successful with %d frames", 
                          len(self.initialization_frames))
                return True
            
        except Exception as e:
            logger.error("SLAM initialization failed: %s", e, exc_info=True)
            
        return False
    
    def _process_batch(self, frames: List[FrameData]) -> Optional[Dict[str, Any]]:
        """Process a batch of frames"""
        try:
            results = []
            
            for frame in frames:
                # Get reference frames
                ref_frames = self.window_processor.get_reference_frames()
                
                if len(ref_frames) < 2:
                    logger.warning("Not enough reference frames for processing")
                    continue
                
                # Process frame
                result = self._process_single_frame(frame, ref_frames)
                if result is not None:
                    results.append(result)
            
            # Return last result (most recent frame)
            return results[-1] if results else None
            
        except Exception as e:
            logger.error("Batch processing failed: %s", e)
            return None
    
    def _process_single_frame(self, 
                            frame: FrameData, 
                            ref_frames: List[FrameData]) -> Optional[Dict[str, Any]]:
        """Process a single frame with reference frames"""
        
        # Run I2P inference
        self._run_i2p_inference(frame, ref_frames[-1] if ref_frames else None)
        
        # Run L2W inference if we have reference frames
        if ref_frames:
            self._run_l2w_inference(frame, ref_frames)
        
        # Extract pose from pts3d transformation
        pose = self._extract_pose(frame)
        
        # Prepare output
        if frame.pts3d_world is not None and frame.conf_world is not None:
            # Convert to numpy for output
            points = frame.pts3d_world.cpu().numpy().reshape(-1, 3)
            confidence = frame.conf_world.cpu().numpy().reshape(-1)
            
            # Filter by confidence
            mask = confidence > self.config.get('conf_thres_i2p', 5.0)
            
            return {
                'pose': pose,
                'points': points[mask],
                'confidence': confidence[mask],
                'frame_id': frame.frame_id,
                'timestamp': frame.timestamp,
                'is_keyframe': frame.is_keyframe
            }
        
        return None
    
    def _run_i2p_inference(self, frame: FrameData, ref_frame: Optional[FrameData]):
        """Run Image-to-Points inference"""
        if ref_frame is None:
            # No reference, can't compute relative points
            return
        
        # Make sure both frames have tokens
        if ref_frame.img_tokens is None:
            self._generate_tokens(ref_frame)
        if frame.img_tokens is None:
            self._generate_tokens(frame)
        
        # Create views with the required fields
        views = [
            {
                'img_tokens': ref_frame.img_tokens,
                'img_pos': ref_frame.img_pos,
                'true_shape': ref_frame.true_shape,
                'label': f'frame_{ref_frame.frame_id}'
            },
            {
                'img_tokens': frame.img_tokens,
                'img_pos': frame.img_pos,
                'true_shape': frame.true_shape,
                'label': f'frame_{frame.frame_id}'
            }
        ]
        
        # Run inference (unsqueeze=False since views don't have batch dims)
        output = i2p_inference_batch(
            [views], 
            self.i2p_model,
            ref_id=0,
            tocpu=False,
            unsqueeze=False
        )
        
        # Extract results
        preds = output['preds']
        if len(preds) > 1:
            # Frame is source view (index 1)
            frame.pts3d_cam = preds[1]['pts3d_in_other_view']
            frame.conf_cam = preds[1]['conf']
    
    def _run_l2w_inference(self, frame: FrameData, ref_frames: List[FrameData]):
        """Run Local-to-World inference"""
        # Create candidate views for scene_frame_retrieve
        cand_views = []
        for ref in ref_frames:
            if ref.pts3d_world is not None and ref.img_tokens is not None:
                view = {
                    'img_tokens': ref.img_tokens,
                    'img_pos': ref.img_pos,
                    'true_shape': ref.true_shape,
                    'pts3d_world': ref.pts3d_world,
                    'label': f'frame_{ref.frame_id}'
                }
                cand_views.append(view)
        
        if not cand_views:
            return
        
        # Make sure frame has tokens
        if frame.img_tokens is None:
            self._generate_tokens(frame)
        
        # Create source view
        src_view = {
            'img_tokens': frame.img_tokens,
            'img_pos': frame.img_pos,
            'true_shape': frame.true_shape,
            'label': f'frame_{frame.frame_id}'
        }
        
        if frame.pts3d_cam is not None:
            src_view['pts3d_cam'] = frame.pts3d_cam
        
        # Run scene frame retrieval
        selected_refs, _ = scene_frame_retrieve(
            cand_views,
            [src_view],
            self.i2p_model,
            sel_num=min(self.config.get('num_scene_frame', 5), len(cand_views))
        )
        
        if not selected_refs:
            return
        
        # Prepare views for L2W
        l2w_views = selected_refs + [src_view]
        
        # Run L2W inference
        output = l2w_inference(
            l2w_views,
            self.l2w_model,
            ref_ids=list(range(len(selected_refs))),
            device=self.device,
            normalize=self.config.get('norm_input_l2w', False)
        )
        
        # Extract results (last output is for source frame)
        if output and len(output) > len(selected_refs):
            result = output[-1]
            frame.pts3d_world = result.get('pts3d_in_other_view')
            frame.conf_world = result.get('conf')
    
    def _extract_pose(self, frame: FrameData) -> Optional[np.ndarray]:
        """Extract camera pose from frame data"""
        # This is a simplified version - you'd implement proper pose extraction
        # based on the point cloud alignment between camera and world coordinates
        
        if frame.pts3d_cam is not None and frame.pts3d_world is not None:
            # For now, return identity matrix
            # In practice, you'd compute the transformation
            return np.eye(4, dtype=np.float32)
        
        return None


# Async wrapper for RabbitMQ integration
class AsyncStreamingSLAM3R:
    """Async wrapper for StreamingSLAM3R to work with RabbitMQ"""
    
    def __init__(self, slam3r: StreamingSLAM3R):
        self.slam3r = slam3r
        self.processing_lock = asyncio.Lock()
    
    async def process_frame(self, image: np.ndarray, timestamp: int) -> Optional[Dict[str, Any]]:
        """Async frame processing"""
        async with self.processing_lock:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self.slam3r.process_frame, 
                image, 
                timestamp
            )
            return result