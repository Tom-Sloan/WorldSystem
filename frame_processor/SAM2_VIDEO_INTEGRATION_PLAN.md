# SAM2 Video Tracking Integration Plan

## Overview

This document outlines the plan to transform the frame_processor from a frame-by-frame processing system to a true video understanding system using SAM2's native video tracking capabilities. The implementation will be modular and support video streams as the primary input format across all detectors.

## Background Research

### SAM2 Documentation and Resources

1. **Official SAM2 Repository**: https://github.com/facebookresearch/sam2
   - Core implementation and model checkpoints
   - Video predictor example notebook: https://github.com/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb

2. **Real-time Streaming Implementation**: https://github.com/Gy920/segment-anything-2-real-time
   - Community implementation for camera/stream processing
   - Provides `build_sam2_camera_predictor` for continuous frame processing

3. **Extended Projects**:
   - **Grounded-SAM-2**: https://github.com/IDEA-Research/Grounded-SAM-2 - Combines SAM2 with detection models
   - **AutoSeg-SAM2**: https://github.com/zrporz/AutoSeg-SAM2 - Automatic full video segmentation
   - **SAM2Long**: https://github.com/Mark12Ding/SAM2Long - Handles long videos with memory tree

4. **Documentation**:
   - Roboflow Tutorial: https://blog.roboflow.com/sam-2-video-segmentation/
   - Ultralytics Integration: https://docs.ultralytics.com/models/sam-2/

## Current System Analysis

### Existing Architecture

```
Current Flow:
H.264 Stream → H264StreamDecoder → Frame → Detector → Detections → IOU Tracker → Tracks → API Processing

Files:
- frame_processor/core/h264_decoder.py - Decodes H.264 to frames
- frame_processor/detection/sam.py - SAM2 single-frame detection only
- frame_processor/tracking/iou_tracker.py - Simple spatial matching
- frame_processor/pipeline/processor.py - Orchestrates pipeline
- frame_processor/main.py - RabbitMQ consumer
```

### Limitations
- Stateless frame processing - no temporal context
- SAM2 used only for single-frame detection
- IOU tracking loses objects during occlusions
- Each detector processes frames independently

## Proposed Architecture

### New Video-First Architecture

```
New Flow:
H.264 Stream → H264StreamDecoder → Frame Buffer → Video Tracker → Detections + Tracks → API Processing

Key Changes:
- Unified detection and tracking per video stream
- Stateful processing with temporal context
- All detectors support video streams
- Modular design for easy extension
```

## Implementation Plan

### Phase 1: Core Video Infrastructure

#### 1.1 Create Video Buffer System

**File**: `frame_processor/core/video_buffer.py` (NEW)
```python
class VideoBuffer:
    """Manages frame buffering for video-based processing."""
    def __init__(self, buffer_size: int = 30):
        self.buffers = {}  # Per stream buffers
        
    async def add_frame(self, stream_id: str, frame: np.ndarray, timestamp: int):
        """Add frame to stream buffer."""
        
    async def get_frames(self, stream_id: str, count: int) -> List[Frame]:
        """Get recent frames for processing."""
```

#### 1.2 Update H.264 Decoder

**File**: `frame_processor/core/h264_decoder.py`
```python
# Add video buffer integration
class H264StreamDecoder:
    def __init__(self, video_buffer: VideoBuffer = None):
        self.video_buffer = video_buffer
        
    async def process_stream_chunk(self, stream_id: str, chunk_data: bytes):
        # Existing decode logic...
        if self.video_buffer:
            await self.video_buffer.add_frame(stream_id, frame, timestamp)
```

### Phase 2: Video Tracker Base Classes

#### 2.1 Create Video Tracker Interface

**File**: `frame_processor/tracking/video_base.py` (NEW)
```python
from abc import ABC, abstractmethod

class VideoTracker(ABC):
    """Base class for video-aware trackers."""
    
    @abstractmethod
    async def initialize_stream(self, stream_id: str, first_frame: np.ndarray):
        """Initialize tracking for a new stream."""
        
    @abstractmethod
    async def process_frame(self, stream_id: str, frame: np.ndarray) -> TrackerOutput:
        """Process a single frame in the video stream."""
        
    @abstractmethod
    async def cleanup_stream(self, stream_id: str):
        """Clean up resources for a stream."""
```

#### 2.2 SAM2 Video Tracker Implementation

**File**: `frame_processor/tracking/sam2_video_tracker.py` (NEW)
```python
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAM2VideoTracker(VideoTracker):
    """SAM2-based video tracking with automatic prompting."""
    
    def __init__(self, config: Config):
        self.predictor = build_sam2_video_predictor(
            config.sam_model_cfg,
            config.sam_checkpoint_path,
            vos_optimized=True  # Enable VOS optimizations
        )
        self.inference_states = {}  # Per-stream state
        self.prompt_strategy = config.sam2_prompt_strategy
```

### Phase 3: Update Existing Components

#### 3.1 Modify Detection Base Class

**File**: `frame_processor/detection/base.py`
```python
class Detector(ABC):
    # Add video support methods
    @abstractmethod
    async def detect_video(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Detect objects in video frames (batch processing)."""
        
    def supports_video(self) -> bool:
        """Whether this detector supports video processing."""
        return False
```

#### 3.2 Update YOLO Detector for Video

**File**: `frame_processor/detection/yolo.py`
```python
class YOLODetector(Detector):
    def supports_video(self) -> bool:
        return True
        
    async def detect_video(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Process multiple frames efficiently."""
        # Use YOLO's batch processing
        results = self.model(frames, stream=True)
        return self._process_batch_results(results)
```

#### 3.3 Update SAM Detector for Video Mode

**File**: `frame_processor/detection/sam.py`
```python
class SAMDetector(Detector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add video predictor alongside image predictor
        if kwargs.get('enable_video_mode', False):
            self.video_predictor = build_sam2_video_predictor(...)
            
    def supports_video(self) -> bool:
        return hasattr(self, 'video_predictor')
```

### Phase 4: Pipeline Integration

#### 4.1 Update Component Factory

**File**: `frame_processor/pipeline/processor.py`
```python
class ComponentFactory:
    # Add video trackers
    VIDEO_TRACKERS: Dict[str, Type[VideoTracker]] = {
        "sam2_video": SAM2VideoTracker,
        "yolo_track": YOLOVideoTracker,  # Future
    }
    
    @classmethod
    def create_video_tracker(cls, config: Config) -> VideoTracker:
        """Create video tracker based on configuration."""
        tracker_type = config.video_tracker_type
        if tracker_type not in cls.VIDEO_TRACKERS:
            raise ValueError(f"Unknown video tracker: {tracker_type}")
        return cls.VIDEO_TRACKERS[tracker_type](config)
```

#### 4.2 Create Video-Aware Processor

**File**: `frame_processor/pipeline/video_processor.py` (NEW)
```python
class VideoProcessor:
    """Orchestrates video-based processing pipeline."""
    
    def __init__(self, config: Config):
        self.config = config
        self.video_buffer = VideoBuffer()
        self.video_tracker = ComponentFactory.create_video_tracker(config)
        
    async def process_stream_frame(self, stream_id: str, frame: np.ndarray):
        """Process a frame as part of a video stream."""
        # Add to buffer
        await self.video_buffer.add_frame(stream_id, frame, timestamp)
        
        # Process with video tracker
        result = await self.video_tracker.process_frame(stream_id, frame)
        
        return result
```

### Phase 5: Main Service Updates

#### 5.1 Update Main Processing Loop

**File**: `frame_processor/main.py`
```python
class FrameProcessorService:
    def __init__(self):
        # Add video processor
        self.video_processor = VideoProcessor(self.config) if self.config.video_mode else None
        
    async def process_stream_message(self, message: aio_pika.IncomingMessage):
        """Process H.264 stream chunks."""
        # Existing decode logic...
        
        if self.config.video_mode and self.video_processor:
            # Use video processor
            result = await self.video_processor.process_stream_frame(
                websocket_id, frame
            )
        else:
            # Fall back to frame-by-frame
            result = await self.processor.process_frame(frame, timestamp_ns)
```

### Phase 6: Configuration Updates

#### 6.1 Update Config Class

**File**: `frame_processor/core/config.py`
```python
class Config(BaseSettings):
    # Video processing mode
    video_mode: bool = Field(
        default=True,
        description="Enable video-aware processing"
    )
    
    video_tracker_type: str = Field(
        default="sam2_video",
        description="Video tracker to use (sam2_video, yolo_track)"
    )
    
    # SAM2 video configuration
    sam2_prompt_strategy: str = Field(
        default="auto_grid",
        description="Prompting strategy: auto_grid, motion, saliency, hybrid"
    )
    
    sam2_reprompt_interval: int = Field(
        default=30,
        description="Frames between automatic re-detection"
    )
    
    sam2_max_objects_per_stream: int = Field(
        default=50,
        description="Maximum tracked objects per stream"
    )
    
    sam2_stream_timeout: float = Field(
        default=60.0,
        description="Seconds before cleaning up inactive stream"
    )
    
    sam2_vos_optimized: bool = Field(
        default=True,
        description="Enable VOS optimizations for speed"
    )
    
    # Video buffer settings
    video_buffer_size: int = Field(
        default=30,
        description="Number of frames to buffer per stream"
    )
    
    video_processing_fps: int = Field(
        default=15,
        description="Target FPS for video processing"
    )
```

#### 6.2 Update Docker Compose

**File**: `docker-compose.yml`
```yaml
frame_processor:
    environment:
      # Existing variables...
      
      # Video processing configuration
      - VIDEO_MODE=${VIDEO_MODE:-true}
      - VIDEO_TRACKER_TYPE=${VIDEO_TRACKER_TYPE:-sam2_video}
      - VIDEO_BUFFER_SIZE=${VIDEO_BUFFER_SIZE:-30}
      - VIDEO_PROCESSING_FPS=${VIDEO_PROCESSING_FPS:-15}
      
      # SAM2 video configuration
      - SAM2_PROMPT_STRATEGY=${SAM2_PROMPT_STRATEGY:-auto_grid}
      - SAM2_REPROMPT_INTERVAL=${SAM2_REPROMPT_INTERVAL:-30}
      - SAM2_MAX_OBJECTS_PER_STREAM=${SAM2_MAX_OBJECTS_PER_STREAM:-50}
      - SAM2_STREAM_TIMEOUT=${SAM2_STREAM_TIMEOUT:-60}
      - SAM2_VOS_OPTIMIZED=${SAM2_VOS_OPTIMIZED:-true}
      
      # Update existing SAM2 configs to support video
      - SAM_VIDEO_MODE=${SAM_VIDEO_MODE:-true}
```

### Phase 7: Automatic Prompting Strategies

#### 7.1 Create Prompt Generator

**File**: `frame_processor/tracking/prompt_strategies.py` (NEW)
```python
class PromptStrategy(ABC):
    """Base class for automatic prompt generation."""
    
    @abstractmethod
    async def generate_prompts(self, frame: np.ndarray) -> List[Prompt]:
        """Generate prompts for object discovery."""

class GridPromptStrategy(PromptStrategy):
    """Sample points in a grid pattern."""
    
class MotionPromptStrategy(PromptStrategy):
    """Detect moving regions between frames."""
    
class SaliencyPromptStrategy(PromptStrategy):
    """Find visually prominent regions."""
    
class HybridPromptStrategy(PromptStrategy):
    """Combine multiple strategies."""
```

## Migration Strategy

### 1. Backward Compatibility
- Keep existing frame-by-frame processing as fallback
- Use `VIDEO_MODE` flag to enable/disable video processing
- Maintain existing API contracts

### 2. Testing Plan
1. **Unit Tests**: Test each new component in isolation
2. **Integration Tests**: Test video pipeline end-to-end
3. **A/B Testing**: Compare video vs frame processing
4. **Performance Tests**: Measure latency and throughput

### 3. Rollout Phases
1. **Development**: Test with simulator
2. **Staging**: Limited rollout with monitoring
3. **Production**: Gradual rollout with feature flags

## Performance Considerations

### GPU Memory Management
- Stream limits to prevent OOM
- Aggressive cleanup of inactive streams
- Configurable buffer sizes

### Processing Optimization
- Frame skipping for high FPS streams
- Batch processing where possible
- VOS optimizations enabled by default

### Monitoring
- Add metrics for video buffer usage
- Track stream lifecycle (creation/cleanup)
- Monitor prompt generation performance

## Benefits

1. **Temporal Consistency**: Smooth object tracking across frames
2. **Occlusion Handling**: Objects maintained through temporary occlusions
3. **Unified Pipeline**: Simpler architecture with video-first design
4. **Better Accuracy**: Leverages temporal context
5. **Modularity**: Easy to add new video trackers
6. **Future-Ready**: Supports interactive prompting

## Next Steps

1. Create video buffer system
2. Implement SAM2 video tracker
3. Update existing detectors for video support
4. Integrate into main pipeline
5. Add configuration and Docker updates
6. Comprehensive testing
7. Documentation updates

## Files to Modify/Create

### New Files
- `frame_processor/core/video_buffer.py`
- `frame_processor/tracking/video_base.py`
- `frame_processor/tracking/sam2_video_tracker.py`
- `frame_processor/pipeline/video_processor.py`
- `frame_processor/tracking/prompt_strategies.py`

### Modified Files
- `frame_processor/core/h264_decoder.py`
- `frame_processor/core/config.py`
- `frame_processor/detection/base.py`
- `frame_processor/detection/yolo.py`
- `frame_processor/detection/sam.py`
- `frame_processor/pipeline/processor.py`
- `frame_processor/main.py`
- `docker-compose.yml`
- `frame_processor/CLAUDE.md`

## References

1. SAM2 Paper: "SAM 2: Segment Anything in Images and Videos"
2. PyAV Documentation: https://pyav.org/docs/stable/
3. Docker Compose Environment Variables: https://docs.docker.com/compose/environment-variables/