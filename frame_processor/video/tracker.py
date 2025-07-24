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
import os

try:
    from sam2.build_sam import build_sam2_video_predictor, build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError:
    # Placeholder for development
    build_sam2_video_predictor = None
    build_sam2 = None
    SAM2ImagePredictor = None
    SAM2AutomaticMaskGenerator = None

from .base import VideoTracker, VideoTrackingResult, StreamState
from core.config import Config
from core.utils import get_logger, PerformanceTimer
from model_configs import get_model_config, get_model_by_size, MODEL_SIZE_MAP
from core.video_buffer import MemoryTree

logger = get_logger(__name__)


@dataclass
class SAM2StreamState(StreamState):
    """Extended stream state for SAM2 tracking."""
    inference_state: Any = None
    object_ids: List[int] = field(default_factory=list)
    memory_tree: MemoryTree = None
    last_good_result: Optional[VideoTrackingResult] = None
    current_model_size: str = "base"
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
        
        logger.info("Initializing SAM2RealtimeTracker...")
        
        # Get model configuration from unified system
        if hasattr(config, 'model_name') and config.model_name:
            # Use the same model as detection
            try:
                self.model_config = get_model_config(config.model_name)
                logger.info(f"Using model config from name: {config.model_name}")
            except ValueError as e:
                logger.warning(f"Failed to get model config: {e}")
                # Fallback to size-based selection
                self.model_config = get_model_by_size(getattr(config, 'sam2_model_size', 'base'))
        else:
            # Use size-based selection
            model_size = getattr(config, 'sam2_model_size', 'base')
            logger.info(f"Using size-based model selection: {model_size}")
            self.model_config = get_model_by_size(model_size)
        
        logger.info(f"Model config: {self.model_config.name}, checkpoint: {self.model_config.checkpoint_path}")
        
        # Initialize model
        self.current_model_size = self.model_config.name
        self.predictor = None
        self.image_predictor = None
        self.mask_generator = None
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
        logger.info("Starting SAM2 model initialization...")
        
        if build_sam2_video_predictor is None:
            logger.error("SAM2 not installed! build_sam2_video_predictor is None")
            logger.error("The frame processor will not work without SAM2")
            raise RuntimeError("SAM2 is required but not installed. Please install sam-2 package.")
        
        try:
            logger.info(f"Loading SAM2 model: {self.model_config.name} from {self.model_config.checkpoint_path}")
            logger.info(f"Config file: {self.model_config.config_file}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            
            # Check if model files exist
            import os
            if not os.path.exists(self.model_config.checkpoint_path):
                logger.error(f"Model checkpoint not found: {self.model_config.checkpoint_path}")
                raise FileNotFoundError(f"SAM2 model checkpoint not found at: {self.model_config.checkpoint_path}")
            
            # Build video predictor
            self.predictor = build_sam2_video_predictor(
                self.model_config.config_file,
                self.model_config.checkpoint_path,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                vos_optimized=True  # Critical for speed
            )
            
            # Compile model for additional speedup (PyTorch 2.0+)
            if self.config.enable_model_compilation and torch.__version__ >= "2.0":
                logger.info("Compiling SAM2 model for improved performance...")
                self.predictor = torch.compile(self.predictor)
            
            # Initialize image predictor for initial prompting
            logger.info(f"Predictor type: {type(self.predictor)}")
            logger.info(f"Predictor attributes: {dir(self.predictor)}")
            
            # Try different ways to get the model
            sam_model = None
            if hasattr(self.predictor, 'model'):
                logger.info("Found 'model' attribute")
                sam_model = self.predictor.model
                self.image_predictor = SAM2ImagePredictor(sam_model)
            elif hasattr(self.predictor, 'sam_model'):
                logger.info("Found 'sam_model' attribute")
                sam_model = self.predictor.sam_model
                self.image_predictor = SAM2ImagePredictor(sam_model)
            elif hasattr(self.predictor, '_model'):
                logger.info("Found '_model' attribute")
                sam_model = self.predictor._model
                self.image_predictor = SAM2ImagePredictor(sam_model)
            else:
                # Try to create model and image predictor with the same config
                logger.warning("Could not find model attribute in predictor, trying alternative approach")
                try:
                    # Build the base model
                    sam_model = build_sam2(
                        self.model_config.config_file,
                        self.model_config.checkpoint_path,
                        device='cuda' if torch.cuda.is_available() else 'cpu'
                    )
                    self.image_predictor = SAM2ImagePredictor(sam_model)
                    logger.info("Created image predictor using build_sam2")
                except Exception as e:
                    logger.error(f"Failed to create image predictor: {e}")
            
            logger.info(f"Image predictor initialized: {self.image_predictor is not None}")
            
            # Try to create automatic mask generator
            if SAM2AutomaticMaskGenerator:
                try:
                    # Use sam_model that we might have already created
                    if sam_model:
                        # Configure automatic mask generator with optimized settings
                        # Use video-specific thresholds for better quality
                        self.mask_generator = SAM2AutomaticMaskGenerator(
                            model=sam_model,
                            points_per_side=self.model_config.parameters.get('points_per_side', 32),  # Higher density for better quality
                            pred_iou_thresh=self.config.sam_video_pred_iou_thresh,  # Configurable via env var
                            stability_score_thresh=self.config.sam_video_stability_score_thresh,  # Configurable via env var
                            min_mask_region_area=self.config.sam_video_min_area,  # Configurable via env var
                            points_per_batch=64,  # Process in batches for speed
                            output_mode="binary_mask",  # Use binary masks for compatibility
                            crop_n_layers=0,  # Disable crop augmentation for speed
                            crop_overlap_ratio=0.0  # No overlap needed
                        )
                        logger.info("Created automatic mask generator successfully")
                    else:
                        logger.warning("No model available for automatic mask generator")
                except Exception as e:
                    logger.warning(f"Failed to create automatic mask generator: {e}")
            
            logger.info(f"SAM2 {self.current_model_size} model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SAM2 model: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    async def initialize_stream(self, stream_id: str, first_frame: np.ndarray, 
                                prompts: Optional[Dict] = None) -> VideoTrackingResult:
        """Initialize tracking for a new stream with automatic object discovery."""
        logger.info(f"=== SAM2 initialize_stream called for {stream_id} ===")
        logger.info(f"Frame shape: {first_frame.shape}, dtype: {first_frame.dtype}")
        logger.info(f"Predictor available: {self.predictor is not None}")
        logger.info(f"Image predictor available: {self.image_predictor is not None}")
        
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
                    # For live streaming, we need to use a different approach
                    # SAM2 video predictor expects a video file or folder, not live frames
                    # We'll initialize with the image predictor and handle tracking differently
                    try:
                        if self.predictor:
                            # Create a temporary state that we'll manage ourselves
                            # Instead of using init_state which expects video files
                            state.inference_state = {
                                'video_predictor': self.predictor,
                                'frames': [],
                                'frame_idx': 0,
                                'initialized': True
                            }
                    except torch.cuda.OutOfMemoryError:
                        logger.error(f"GPU OOM initializing stream {stream_id}")
                        # Try cleanup and retry once
                        await self._handle_gpu_oom()
                        if self.predictor:
                            state.inference_state = {
                                'video_predictor': self.predictor,
                                'frames': [],
                                'frame_idx': 0,
                                'initialized': True
                            }
                    
                    # Generate prompts if not provided
                    prompt_start = time.time()
                    if prompts is None:
                        logger.info("Generating initial prompts...")
                        prompts = await self._generate_initial_prompts(first_frame)
                    prompt_time = (time.time() - prompt_start) * 1000
                    
                    # Record prompt generation time
                    from core.performance_monitor import get_performance_monitor
                    monitor = get_performance_monitor()
                    monitor.record_timing('prompt_generation', prompt_time)
                    
                    logger.info(f"Prompts: {len(prompts.get('points', []))} points")
                    
                    # Add prompts and get initial masks
                    masks = []
                    mask_start = time.time()
                    if state.inference_state:
                        logger.info("Adding prompts to state...")
                        masks = await self._add_prompts_to_state(
                            state.inference_state, 
                            prompts, 
                            first_frame,
                            frame_idx=0
                        )
                        logger.info(f"Got {len(masks)} masks from prompts")
                        # Store initial masks for tracking
                        if masks:
                            state.inference_state['previous_masks'] = masks
                    else:
                        logger.error("No inference state available!")
                    
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
        logger.debug(f"=== SAM2 process_frame called for {stream_id}, timestamp: {timestamp} ===")
        
        if not self.validate_frame(frame):
            logger.error(f"Invalid frame for stream {stream_id}")
            return self._empty_result(stream_id)
        
        if stream_id not in self.stream_states:
            logger.info(f"Stream {stream_id} not in states, auto-initializing...")
            # Auto-initialize if needed
            return await self.initialize_stream(stream_id, frame)
        
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                async with self._locks[stream_id]:
                    with PerformanceTimer("sam2_frame_process", logger) as timer:
                        state = self.stream_states[stream_id]
                        state.update()  # Update frame count and timestamp
                        
                        # Check if we need to re-detect all objects
                        if state.frame_count % self.reprompt_interval == 0:
                            # Clear existing tracks and detect fresh
                            logger.info(f"Re-detecting all objects at frame {state.frame_count}")
                            state.object_ids.clear()
                            
                            # Run automatic mask generation
                            if self.mask_generator:
                                try:
                                    masks_result = self.mask_generator.generate(frame)
                                    
                                    # Convert and limit to top 5 objects by area
                                    masks = []
                                    max_tracks = int(os.getenv('MAX_TRACKS', '5'))
                                    
                                    # Sort by area and take top N
                                    sorted_masks = sorted(masks_result, 
                                                        key=lambda x: x.get('area', 0), 
                                                        reverse=True)[:max_tracks]
                                    
                                    for i, mask_info in enumerate(sorted_masks):
                                        if mask_info.get('area', 0) > self.min_object_area:
                                            seg = mask_info.get('segmentation')
                                            if seg is not None:
                                                # Calculate bbox
                                                y_coords, x_coords = np.where(seg)
                                                if len(y_coords) > 0:
                                                    bbox = [int(x_coords.min()), int(y_coords.min()), 
                                                           int(x_coords.max() - x_coords.min()), 
                                                           int(y_coords.max() - y_coords.min())]
                                                    
                                                    mask_data = {
                                                        "object_id": i,
                                                        "segmentation": seg,
                                                        "area": int(mask_info['area']),
                                                        "confidence": float(mask_info.get('predicted_iou', 0.9)),
                                                        "bbox": bbox,
                                                        "stability_score": float(mask_info.get('stability_score', 0.0))
                                                    }
                                                    masks.append(mask_data)
                                    
                                    logger.info(f"Detected {len(masks)} objects (from {len(masks_result)} total)")
                                except Exception as e:
                                    logger.error(f"Error in automatic mask generation: {e}")
                                    masks = []
                            else:
                                masks = []
                        else:
                            # Between detection intervals, just propagate existing masks
                            masks = []
                            sam_start = time.time()
                            if state.inference_state and self.predictor:
                                masks = await self._propagate_masks(
                                    state.inference_state,
                                    frame,
                                    state.frame_count
                                )
                        sam_time = (time.time() - sam_start) * 1000
                        
                        # Record SAM2 tracking time
                        from core.performance_monitor import get_performance_monitor
                        monitor = get_performance_monitor()
                        monitor.record_timing('sam2_tracking', sam_time)
                        
                        # Update memory tree
                        state.memory_tree.update(masks, timestamp)
                        
                        # Get best path from memory tree
                        best_path = state.memory_tree.get_best_path()
                        if best_path['confidence'] > 0.9:
                            masks = best_path.get('masks', masks)
                        
                        # Filter small objects for performance
                        masks = [m for m in masks if m.get('area', 0) >= self.min_object_area]
                        
                        # Store masks for next frame tracking
                        if state.inference_state and masks:
                            state.inference_state['previous_masks'] = masks
                        
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
        
        # Generate grid points with smaller margin for better coverage
        margin = 20
        y_coords = np.linspace(margin, h - margin, density)
        x_coords = np.linspace(margin, w - margin, density)
        
        points = []
        labels = []
        
        # Add positive points in a grid
        for y in y_coords:
            for x in x_coords:
                points.append([int(x), int(y)])
                labels.append(1)  # Positive point
        
        # Also add some negative points at the edges to help SAM2
        # These tell SAM2 where NOT to segment
        edge_points = [
            [margin//2, margin//2],  # Top-left corner
            [w - margin//2, margin//2],  # Top-right corner
            [margin//2, h - margin//2],  # Bottom-left corner
            [w - margin//2, h - margin//2],  # Bottom-right corner
        ]
        
        for point in edge_points:
            points.append(point)
            labels.append(0)  # Negative point
        
        logger.info(f"Generated {len(points)} prompts ({density*density} positive, {len(edge_points)} negative) for {w}x{h} frame")
        
        return {
            "points": np.array(points, dtype=np.float32),
            "labels": np.array(labels, dtype=np.int32)
        }
    
    async def _add_prompts_to_state(self, inference_state, prompts, frame, frame_idx):
        """Add prompts to SAM2 state and get masks using automatic mask generation."""
        logger.info(f"=== _add_prompts_to_state called, frame_idx: {frame_idx} ===")
        
        if not self.image_predictor:
            logger.error("No image predictor available!")
            return []
        
        try:
            # Use automatic mask generation instead of point prompts
            logger.info("Using automatic mask generation...")
            
            # Try automatic mask generation first
            if self.mask_generator:
                logger.info("Using automatic mask generation...")
                try:
                    masks_result = self.mask_generator.generate(frame)
                    logger.info(f"Generated {len(masks_result)} masks automatically")
                    
                    # Convert to our format
                    masks_list = []
                    for i, mask_info in enumerate(masks_result):
                        # Check for required fields - use configured threshold
                        if mask_info.get('area', 0) > self.config.sam_video_min_area:
                            # Handle segmentation format
                            if 'segmentation' in mask_info:
                                seg = mask_info['segmentation']
                                # Ensure it's a numpy array
                                if not isinstance(seg, np.ndarray):
                                    logger.warning(f"Unknown segmentation format: {type(seg)}")
                                    continue
                            else:
                                logger.warning("No segmentation in mask_info")
                                continue
                            
                            # Calculate bbox if not provided
                            if 'bbox' not in mask_info and seg.sum() > 0:
                                y_coords, x_coords = np.where(seg)
                                bbox = [int(x_coords.min()), int(y_coords.min()), 
                                       int(x_coords.max() - x_coords.min()), 
                                       int(y_coords.max() - y_coords.min())]  # x,y,w,h
                            else:
                                bbox = mask_info.get('bbox', [0, 0, 0, 0])
                            
                            mask_data = {
                                "object_id": i,
                                "segmentation": seg if isinstance(seg, np.ndarray) else seg.astype(bool),
                                "area": int(mask_info['area']),
                                "confidence": float(mask_info.get('predicted_iou', 0.9)),
                                "bbox": bbox,
                                "stability_score": float(mask_info.get('stability_score', 0.0))
                            }
                            masks_list.append(mask_data)
                    
                    if masks_list:
                        return masks_list
                except Exception as e:
                    logger.error(f"Automatic mask generation failed: {e}")
            
            # Fallback: Use grid of individual point prompts
            logger.info("Using individual point prompts as fallback...")
            self.image_predictor.set_image(frame)
            
            masks_list = []
            points = prompts["points"]
            labels = prompts["labels"]
            
            # Only process positive points individually
            positive_points = [(p, l) for p, l in zip(points, labels) if l == 1]
            
            for i, (point, label) in enumerate(positive_points[:20]):  # Limit to 20 points for performance
                try:
                    # Single point prediction
                    masks, scores, logits = self.image_predictor.predict(
                        point_coords=np.array([point]),
                        point_labels=np.array([label]),
                        multimask_output=True
                    )
                    
                    # Get best mask
                    if len(masks) > 0:
                        best_idx = np.argmax(scores)
                        best_mask = masks[best_idx]
                        best_score = scores[best_idx]
                        
                        if best_score > 0.8 and best_mask.sum() > 1000:
                            # Check if this mask overlaps too much with existing ones
                            is_duplicate = False
                            for existing in masks_list:
                                overlap = (existing['segmentation'] & best_mask).sum()
                                if overlap > 0.5 * min(existing['area'], best_mask.sum()):
                                    is_duplicate = True
                                    break
                            
                            if not is_duplicate:
                                mask_data = {
                                    "object_id": i,
                                    "segmentation": best_mask.astype(bool),
                                    "area": int(best_mask.sum()),
                                    "confidence": float(best_score)
                                }
                                masks_list.append(mask_data)
                except Exception as e:
                    logger.debug(f"Failed to process point {i}: {e}")
                    continue
            
            logger.info(f"Generated {len(masks_list)} masks from point prompts")
            return masks_list
            
        except Exception as e:
            logger.error(f"Error adding prompts: {e}")
            return []
    
    async def _propagate_masks(self, inference_state, frame, frame_idx):
        """Propagate masks to current frame."""
        # For live streaming with image predictor, we can't use video propagation
        # Instead, we'll use previous masks to guide current frame detection
        
        if not self.image_predictor:
            logger.error("No image predictor available for propagation!")
            return []
        
        
        try:
            # Get stored masks from previous frame if available
            if 'previous_masks' in inference_state and inference_state['previous_masks']:
                # Set image for current frame
                self.image_predictor.set_image(frame)
                
                masks_list = []
                for prev_mask in inference_state['previous_masks']:
                    # Get center of previous mask as prompt
                    seg = prev_mask['segmentation']
                    if seg.sum() > 0:
                        y_coords, x_coords = np.where(seg)
                        center_y = int(y_coords.mean())
                        center_x = int(x_coords.mean())
                        
                        # Use center as positive prompt
                        masks, scores, logits = self.image_predictor.predict(
                            point_coords=np.array([[center_x, center_y]]),
                            point_labels=np.array([1]),
                            multimask_output=False
                        )
                        
                        if len(masks) > 0 and scores[0] > 0.5:
                            mask_data = {
                                "object_id": prev_mask['object_id'],
                                "segmentation": masks[0].astype(bool),
                                "area": int(masks[0].sum()),
                                "confidence": float(scores[0])
                            }
                            masks_list.append(mask_data)
                
                # Store for next frame
                inference_state['previous_masks'] = masks_list
                return masks_list
            
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