"""
SAM2 real-time video tracker implementation.

Optimized for 15+ FPS performance on RTX 3090 with SAM2Long memory tree
for robust long-term tracking.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import gc
from dataclasses import dataclass, field
import time

try:
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    # Placeholder for development
    build_sam2_video_predictor = None
    SAM2ImagePredictor = None

from tracking.video_base import VideoTracker, VideoTrackingResult, StreamState
from core.config import Config
from core.utils import get_logger, PerformanceTimer
from core.video_buffer import MemoryTree

logger = get_logger(__name__)


@dataclass
class SAM2StreamState(StreamState):
    """Extended stream state for SAM2 tracking."""
    inference_state: Any = None
    object_ids: List[int] = field(default_factory=list)
    memory_tree: MemoryTree = None
    last_good_result: Optional[VideoTrackingResult] = None
    current_model_size: str = "small"
    error_count: int = 0
    last_prompt_frame: int = 0
    
    def __post_init__(self):
        super().__init__(self.stream_id)
        if self.memory_tree is None:
            self.memory_tree = MemoryTree()


class SAM2RealtimeTracker(VideoTracker):
    """
    SAM2 tracker optimized for real-time performance (15+ FPS on RTX 3090).
    Uses SAM2Long memory tree approach for robust tracking.
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Model configuration
        self.model_configs = {
            "tiny": "sam2_hiera_t.yaml",    # 25+ FPS
            "small": "sam2_hiera_s.yaml",   # 15-20 FPS
            "base": "sam2_hiera_b+.yaml",   # 10-15 FPS
            "large": "sam2_hiera_l.yaml"    # 5-10 FPS
        }
        
        # Initialize model
        self.current_model_size = config.sam2_model_size
        self.predictor = None
        self.image_predictor = None
        self._initialize_model()
        
        # Per-stream state management
        self.stream_states: Dict[str, SAM2StreamState] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        
        # Performance settings
        self.points_per_side = config.grid_prompt_density
        self.reprompt_interval = config.reprompt_interval
        self.min_object_area = config.min_object_area
        
        # Error recovery settings
        self.max_retries = getattr(config, 'max_retry_attempts', 2)
        self.retry_delay_ms = getattr(config, 'retry_delay_ms', 100)
        
        # GPU memory management
        self.enable_dynamic_switching = getattr(config, 'enable_dynamic_model_switching', True)
        self.model_switch_threshold_mb = getattr(config, 'model_switch_threshold_mb', 3000)
        
        logger.info(f"Initialized SAM2RealtimeTracker with {self.current_model_size} model")
    
    def _initialize_model(self):
        """Initialize SAM2 model with error handling."""
        if build_sam2_video_predictor is None:
            logger.warning("SAM2 not installed, using mock predictor")
            return
        
        try:
            model_cfg = self.model_configs.get(self.current_model_size, "sam2_hiera_s.yaml")
            logger.info(f"Loading SAM2 {self.current_model_size} model...")
            
            # Build video predictor
            self.predictor = build_sam2_video_predictor(
                model_cfg,
                self.config.sam_checkpoint_path,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                vos_optimized=True  # Critical for speed
            )
            
            # Compile model for additional speedup (PyTorch 2.0+)
            if self.config.enable_model_compilation and torch.__version__ >= "2.0":
                logger.info("Compiling SAM2 model for improved performance...")
                self.predictor = torch.compile(self.predictor)
            
            # Initialize image predictor for initial prompting
            if hasattr(self.predictor, 'model'):
                self.image_predictor = SAM2ImagePredictor(self.predictor.model)
            
            logger.info(f"SAM2 {self.current_model_size} model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SAM2 model: {e}")
            raise
    
    async def initialize_stream(self, stream_id: str, first_frame: np.ndarray, 
                                prompts: Optional[Dict] = None) -> VideoTrackingResult:
        """Initialize tracking for a new stream with automatic object discovery."""
        if stream_id not in self._locks:
            self._locks[stream_id] = asyncio.Lock()
        
        try:
            async with self._locks[stream_id]:
                with PerformanceTimer("sam2_stream_init", logger) as timer:
                    # Create stream state
                    state = SAM2StreamState(
                        stream_id=stream_id,
                        memory_tree=MemoryTree(self.config.memory_tree_branches)
                    )
                    
                    # Initialize SAM2 state with error handling
                    try:
                        if self.predictor:
                            state.inference_state = self.predictor.init_state(video_path=None)
                    except torch.cuda.OutOfMemoryError:
                        logger.error(f"GPU OOM initializing stream {stream_id}")
                        # Try cleanup and retry once
                        await self._handle_gpu_oom()
                        if self.predictor:
                            state.inference_state = self.predictor.init_state(video_path=None)
                    
                    # Generate prompts if not provided
                    if prompts is None:
                        prompts = await self._generate_initial_prompts(first_frame)
                    
                    # Add prompts and get initial masks
                    masks = []
                    if state.inference_state:
                        masks = await self._add_prompts_to_state(
                            state.inference_state, 
                            prompts, 
                            first_frame,
                            frame_idx=0
                        )
                    
                    # Store state
                    self.stream_states[stream_id] = state
                    
                    # Create tracking result
                    result = VideoTrackingResult(
                        masks=masks,
                        tracks=self._masks_to_tracks(masks, stream_id),
                        object_count=len(masks),
                        processing_time_ms=timer.elapsed_ms,
                        frame_number=0
                    )
                    
                    # Store as last good result
                    state.last_good_result = result
                    
                    return result
                    
        except Exception as e:
            logger.error(f"Failed to initialize stream {stream_id}: {e}", exc_info=True)
            
            # Clean up on failure
            if stream_id in self.stream_states:
                del self.stream_states[stream_id]
            if stream_id in self._locks:
                del self._locks[stream_id]
            
            # Return empty result
            return VideoTrackingResult(
                masks=[],
                tracks=[],
                object_count=0,
                processing_time_ms=0,
                frame_number=0
            )
    
    async def process_frame(self, stream_id: str, frame: np.ndarray, 
                           timestamp: int) -> VideoTrackingResult:
        """Process frame with memory-aware tracking and comprehensive error handling."""
        if not self.validate_frame(frame):
            logger.error(f"Invalid frame for stream {stream_id}")
            return self._empty_result(stream_id)
        
        if stream_id not in self.stream_states:
            # Auto-initialize if needed
            return await self.initialize_stream(stream_id, frame)
        
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                async with self._locks[stream_id]:
                    with PerformanceTimer("sam2_frame_process", logger) as timer:
                        state = self.stream_states[stream_id]
                        state.update()  # Update frame count and timestamp
                        
                        # Check if we need to re-prompt for new objects
                        if state.frame_count % self.reprompt_interval == 0:
                            await self._reprompt_for_new_objects(state, frame)
                        
                        # Process with SAM2
                        masks = []
                        if state.inference_state and self.predictor:
                            masks = await self._propagate_masks(
                                state.inference_state,
                                frame,
                                state.frame_count
                            )
                        
                        # Update memory tree
                        state.memory_tree.update(masks, timestamp)
                        
                        # Get best path from memory tree
                        best_path = state.memory_tree.get_best_path()
                        if best_path['confidence'] > 0.9:
                            masks = best_path.get('masks', masks)
                        
                        # Filter small objects for performance
                        masks = [m for m in masks if m.get('area', 0) >= self.min_object_area]
                        
                        # Create result
                        result = VideoTrackingResult(
                            masks=masks,
                            tracks=self._masks_to_tracks(masks, stream_id),
                            object_count=len(masks),
                            processing_time_ms=timer.elapsed_ms,
                            frame_number=state.frame_count,
                            confidence_scores=[m.get('confidence', 0.0) for m in masks]
                        )
                        
                        # Store as last good result
                        state.last_good_result = result
                        state.error_count = 0  # Reset error count on success
                        
                        return result
                        
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"GPU OOM in stream {stream_id}, attempt {retry_count + 1}/{self.max_retries + 1}")
                
                # Try to recover
                await self._handle_gpu_oom()
                
                # Switch to smaller model if available
                if retry_count == 0 and self.enable_dynamic_switching:
                    success = await self._try_smaller_model()
                    if not success:
                        break
                
                retry_count += 1
                if retry_count > self.max_retries:
                    return self._fallback_result(stream_id)
                
                # Wait before retry
                await asyncio.sleep(self.retry_delay_ms / 1000.0 * retry_count)
                
            except Exception as e:
                logger.error(f"Unexpected error processing frame for stream {stream_id}: {e}", exc_info=True)
                
                # Update error count
                if stream_id in self.stream_states:
                    state = self.stream_states[stream_id]
                    state.error_count += 1
                    
                    # Return last good result if available
                    if state.last_good_result and state.error_count < 5:
                        logger.info(f"Returning last good result for stream {stream_id}")
                        return state.last_good_result
                
                return self._empty_result(stream_id)
        
        return self._empty_result(stream_id)
    
    async def cleanup_stream(self, stream_id: str):
        """Clean up resources for a stream."""
        if stream_id in self.stream_states:
            try:
                async with self._locks[stream_id]:
                    state = self.stream_states[stream_id]
                    
                    # Log stream statistics
                    logger.info(f"Cleaning up stream {stream_id} - "
                               f"frames: {state.frame_count}, "
                               f"age: {state.age:.1f}s, "
                               f"errors: {state.error_count}")
                    
                    # Clear SAM2 state
                    if state.inference_state is not None:
                        del state.inference_state
                    
                    del self.stream_states[stream_id]
                    
            except Exception as e:
                logger.error(f"Error cleaning up stream {stream_id}: {e}")
            finally:
                if stream_id in self._locks:
                    del self._locks[stream_id]
    
    def get_stream_status(self, stream_id: str) -> Dict[str, Any]:
        """Get current status of a stream."""
        if stream_id not in self.stream_states:
            return {"status": "not_initialized"}
        
        state = self.stream_states[stream_id]
        memory_info = state.memory_tree.get_best_path()
        
        return {
            "status": "active" if state.is_active else "inactive",
            "frame_count": state.frame_count,
            "object_count": len(state.object_ids),
            "memory_tree_branches": memory_info.get('total_branches', 0),
            "memory_confidence": memory_info.get('confidence', 0.0),
            "age_seconds": state.age,
            "error_count": state.error_count,
            "current_model": state.current_model_size
        }
    
    @property
    def name(self) -> str:
        return f"SAM2-Realtime-{self.current_model_size}"
    
    @property
    def supports_batching(self) -> bool:
        return False  # Real-time mode processes frame-by-frame
    
    async def _generate_initial_prompts(self, frame: np.ndarray) -> Dict:
        """Generate grid prompts for automatic object discovery."""
        h, w = frame.shape[:2]
        
        # Adjust density based on resolution
        density = self.points_per_side
        if h > 1080 or w > 1920:
            density = max(8, density // 2)  # Reduce for high res
        
        # Generate grid points with margin
        margin = 50
        y_coords = np.linspace(margin, h - margin, density)
        x_coords = np.linspace(margin, w - margin, density)
        
        points = []
        for y in y_coords:
            for x in x_coords:
                points.append([int(x), int(y)])
        
        logger.debug(f"Generated {len(points)} grid prompts for {w}x{h} frame")
        
        return {
            "points": np.array(points),
            "labels": np.ones(len(points), dtype=np.int32)  # All positive
        }
    
    async def _add_prompts_to_state(self, inference_state, prompts, frame, frame_idx):
        """Add prompts to SAM2 state and get masks."""
        if not self.predictor:
            return []
        
        try:
            # Convert to SAM2 format
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
                inference_state,
                frame_idx=frame_idx,
                obj_id=0,  # Will auto-increment
                points=prompts["points"],
                labels=prompts["labels"],
                clear_old_points=True
            )
            
            # Convert logits to masks
            masks = []
            for i, obj_id in enumerate(out_obj_ids):
                mask_data = {
                    "object_id": int(obj_id),
                    "segmentation": (out_mask_logits[i] > 0.0).cpu().numpy(),
                    "area": int((out_mask_logits[i] > 0.0).sum()),
                    "confidence": float(torch.sigmoid(out_mask_logits[i]).mean())
                }
                masks.append(mask_data)
            
            return masks
            
        except Exception as e:
            logger.error(f"Error adding prompts: {e}")
            return []
    
    async def _propagate_masks(self, inference_state, frame, frame_idx):
        """Propagate masks to current frame."""
        if not self.predictor:
            return []
        
        try:
            # Get masks for current frame
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                inference_state,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=1
            ):
                if out_frame_idx == frame_idx:
                    # Convert to mask format
                    masks = []
                    for i, obj_id in enumerate(out_obj_ids):
                        mask_data = {
                            "object_id": int(obj_id),
                            "segmentation": (out_mask_logits[i] > 0.0).cpu().numpy(),
                            "area": int((out_mask_logits[i] > 0.0).sum()),
                            "confidence": float(torch.sigmoid(out_mask_logits[i]).mean())
                        }
                        masks.append(mask_data)
                    return masks
            
            return []
            
        except Exception as e:
            logger.error(f"Error propagating masks: {e}")
            return []
    
    async def _reprompt_for_new_objects(self, state: SAM2StreamState, frame: np.ndarray):
        """Periodically check for new objects not being tracked."""
        # Generate sparse grid for efficiency
        prompts = await self._generate_initial_prompts(frame)
        prompts["points"] = prompts["points"][::2]  # Use every other point
        prompts["labels"] = prompts["labels"][::2]
        
        logger.debug(f"Re-prompting with {len(prompts['points'])} points")
        state.last_prompt_frame = state.frame_count
        
        # Add new prompts without clearing existing tracks
        if state.inference_state:
            await self._add_prompts_to_state(
                state.inference_state,
                prompts,
                frame,
                state.frame_count
            )
    
    def _masks_to_tracks(self, masks: List[Dict], stream_id: str) -> List[Dict]:
        """Convert masks to track format for compatibility."""
        tracks = []
        for mask in masks:
            # Get bounding box from mask
            segmentation = mask.get("segmentation")
            if segmentation is None or segmentation.sum() == 0:
                continue
            
            y_coords, x_coords = np.where(segmentation)
            if len(y_coords) == 0:
                continue
            
            bbox = [
                int(x_coords.min()),
                int(y_coords.min()),
                int(x_coords.max()),
                int(y_coords.max())
            ]
            
            track = {
                "track_id": f"{stream_id}_{mask['object_id']}",
                "object_id": mask["object_id"],
                "bbox": bbox,
                "confidence": mask.get("confidence", 0.0),
                "area": mask.get("area", 0),
                "mask": mask["segmentation"]
            }
            tracks.append(track)
        
        return tracks
    
    async def _handle_gpu_oom(self):
        """Handle GPU out of memory error."""
        logger.warning("Handling GPU OOM - clearing cache")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        # Log memory status
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_info = torch.cuda.mem_get_info(i)
                free_gb = mem_info[0] / (1024**3)
                total_gb = mem_info[1] / (1024**3)
                logger.info(f"GPU {i}: {free_gb:.1f}/{total_gb:.1f} GB free")
    
    async def _try_smaller_model(self) -> bool:
        """Try to switch to a smaller model."""
        model_hierarchy = ["large", "base", "small", "tiny"]
        
        try:
            current_idx = model_hierarchy.index(self.current_model_size)
            if current_idx < len(model_hierarchy) - 1:
                new_size = model_hierarchy[current_idx + 1]
                logger.warning(f"Switching from {self.current_model_size} to {new_size} model")
                
                # Clear current model
                if self.predictor:
                    del self.predictor
                if self.image_predictor:
                    del self.image_predictor
                
                # Load smaller model
                self.current_model_size = new_size
                self._initialize_model()
                
                # Update all stream states
                for state in self.stream_states.values():
                    state.current_model_size = new_size
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
        
        return False
    
    def _empty_result(self, stream_id: str) -> VideoTrackingResult:
        """Create empty result."""
        frame_num = 0
        if stream_id in self.stream_states:
            frame_num = self.stream_states[stream_id].frame_count
        
        return VideoTrackingResult(
            masks=[],
            tracks=[],
            object_count=0,
            processing_time_ms=0,
            frame_number=frame_num
        )
    
    def _fallback_result(self, stream_id: str) -> VideoTrackingResult:
        """Create fallback result after errors."""
        if stream_id in self.stream_states:
            state = self.stream_states[stream_id]
            if state.last_good_result:
                # Return last good result with updated frame number
                result = state.last_good_result
                result.frame_number = state.frame_count
                return result
        
        return self._empty_result(stream_id)