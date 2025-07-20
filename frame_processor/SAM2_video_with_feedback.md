# SAM2 Video Integration Plan - Real-time Pipeline

## Overview

This document outlines the plan to transform the frame_processor from a frame-by-frame processing system to a real-time video tracking system using SAM2's native video capabilities with SAM2Long enhancements. The system will maintain 15+ FPS on RTX 3090 and feed tracked objects to Google Lens API for identification.

## Pipeline Architecture

```
H.264 Stream → H264StreamDecoder → Frame Buffer → SAM2 Video Tracker → Object Masks/Tracks → Image Enhancement → Google Lens API
                                          ↑                                      ↓
                                   Grid Prompts                            Object Identities
                                 (every N frames)
```

## Current System Analysis

### Existing Architecture (from current codebase)

```
Current Files:
- frame_processor/core/h264_decoder.py - Decodes H.264 to frames
- frame_processor/detection/sam.py - SAM2 single-frame detection only
- frame_processor/tracking/iou_tracker.py - Simple spatial matching
- frame_processor/pipeline/processor.py - Orchestrates pipeline
- frame_processor/main.py - RabbitMQ consumer
```

### Limitations to Address
- Stateless frame processing loses temporal context
- IOU tracking fails during occlusions
- No persistent object tracking for Google Lens queries
- Inefficient re-detection every frame

## Implementation Plan

### Phase 1: Core Video Infrastructure with SAM2Long

#### 1.1 Video Buffer with Memory Tree

**File**: `frame_processor/core/video_buffer.py` (NEW)
```python
class SAM2LongVideoBuffer:
    """Enhanced video buffer with SAM2Long memory tree for error prevention."""
    
    def __init__(self, buffer_size: int = 30, tree_branches: int = 3):
        self.buffers = {}  # Per stream buffers
        self.memory_trees = {}  # SAM2Long memory trees per stream
        self.tree_branches = tree_branches
        
    async def add_frame(self, stream_id: str, frame: np.ndarray, timestamp: int):
        """Add frame and maintain memory tree branches."""
        
    def get_optimal_memory_path(self, stream_id: str):
        """Select best segmentation path from memory tree."""
```

#### 1.2 SAM2 Video Tracker with Real-time Optimizations

**File**: `frame_processor/tracking/sam2_realtime_tracker.py` (NEW)
```python
from sam2.build_sam import build_sam2_video_predictor
import torch

class SAM2RealtimeTracker:
    """SAM2 tracker optimized for 15+ FPS on RTX 3090."""
    
    def __init__(self, config: Config):
        # Use smaller model for real-time performance
        model_cfg = "sam2_hiera_s.yaml"  # Small model for speed
        
        # Enable all optimizations
        self.predictor = build_sam2_video_predictor(
            model_cfg,
            config.sam_checkpoint_path,
            device='cuda',
            vos_optimized=True  # Critical for speed
        )
        
        # Compile for additional speedup
        self.predictor = torch.compile(self.predictor)
        
        # Memory trees for each stream (SAM2Long approach)
        self.memory_trees = {}
        
        # Grid prompt generator
        self.points_per_side = 16  # Balanced for speed/coverage
        self.reprompt_interval = 60  # Frames between re-prompting
```

### Phase 2: Automatic Prompting for Object Discovery

#### 2.1 Simple Grid Prompting

**File**: `frame_processor/tracking/grid_prompter.py` (NEW)
```python
class GridPrompter:
    """Generate grid prompts for automatic object discovery."""
    
    def generate_prompts(self, frame_shape, density=16):
        """Generate uniform grid of point prompts."""
        h, w = frame_shape[:2]
        # Reduce density for real-time performance
        y_coords = np.linspace(50, h-50, density)
        x_coords = np.linspace(50, w-50, density)
        
        points = []
        for y in y_coords:
            for x in x_coords:
                points.append([x, y])
        
        return np.array(points), np.ones(len(points))  # All positive prompts
```

### Phase 3: Integration with Enhancement Pipeline

#### 3.1 Enhanced Frame Processor

**File**: `frame_processor/pipeline/video_processor.py` (UPDATE)
```python
class VideoProcessor:
    """Orchestrates real-time video processing pipeline."""
    
    def __init__(self, config: Config):
        self.config = config
        self.video_buffer = SAM2LongVideoBuffer()
        self.sam2_tracker = SAM2RealtimeTracker(config)
        self.grid_prompter = GridPrompter()
        self.enhancer = ImageEnhancer()  # Existing enhancement
        
        # Track objects pending Google Lens queries
        self.pending_identifications = {}
        
    async def process_stream_frame(self, stream_id: str, frame: np.ndarray):
        """Process frame through tracking → enhancement → identification pipeline."""
        
        # Add to buffer
        await self.video_buffer.add_frame(stream_id, frame, timestamp)
        
        # Initialize or update tracking
        if stream_id not in self.sam2_tracker.memory_trees:
            # First frame - generate grid prompts
            prompts = self.grid_prompter.generate_prompts(frame.shape)
            masks = await self.sam2_tracker.initialize_stream(
                stream_id, frame, prompts
            )
        else:
            # Continue tracking with memory tree
            masks = await self.sam2_tracker.track_frame(stream_id, frame)
        
        # Process each tracked object
        enhanced_crops = []
        for obj_id, mask in enumerate(masks):
            if mask['area'] < self.config.min_object_area:
                continue  # Skip small objects for performance
                
            # Crop and enhance object
            crop = self.extract_object_crop(frame, mask)
            enhanced = await self.enhancer.enhance(crop)
            
            enhanced_crops.append({
                'object_id': f"{stream_id}_{obj_id}",
                'enhanced_image': enhanced,
                'mask': mask,
                'timestamp': timestamp
            })
        
        # Queue for Google Lens API (separate async process)
        await self.queue_for_identification(enhanced_crops)
        
        return {
            'frame_id': f"{stream_id}_{timestamp}",
            'object_count': len(masks),
            'masks': masks,
            'enhanced_crops': enhanced_crops
        }
```

### Phase 4: Google Lens Integration

#### 4.1 Asynchronous Identification Queue

**File**: `frame_processor/external/lens_identifier.py` (NEW)
```python
class LensIdentifier:
    """Manages Google Lens API calls for object identification."""
    
    def __init__(self, api_client):
        self.api_client = api_client
        self.identification_cache = {}  # Cache results by object appearance
        self.rate_limiter = RateLimiter(calls_per_second=10)
        
    async def identify_objects(self, enhanced_crops):
        """Identify objects using Google Lens API."""
        results = []
        
        for crop_data in enhanced_crops:
            obj_id = crop_data['object_id']
            
            # Check cache first (based on visual similarity)
            cached = self.check_cache(crop_data['enhanced_image'])
            if cached:
                results.append(cached)
                continue
            
            # Rate-limited API call
            async with self.rate_limiter:
                identity = await self.api_client.identify(
                    crop_data['enhanced_image']
                )
                
            # Cache result
            self.cache_result(crop_data['enhanced_image'], identity)
            results.append({
                'object_id': obj_id,
                'identity': identity,
                'confidence': identity.confidence
            })
            
        return results
```

### Phase 5: Performance Optimizations

#### 5.1 Real-time Configuration

**File**: `frame_processor/core/config.py` (UPDATE)
```python
class Config(BaseSettings):
    # Model selection for speed
    sam2_model_size: str = Field(
        default="small",
        description="Model size: tiny (fastest), small, base, large"
    )
    
    # Performance settings
    target_fps: int = Field(
        default=15,
        description="Target FPS for real-time processing"
    )
    
    processing_resolution: int = Field(
        default=720,
        description="Max resolution for processing (scales down if needed)"
    )
    
    # SAM2Long memory tree
    memory_tree_branches: int = Field(
        default=3,
        description="Number of hypothesis branches in memory tree"
    )
    
    # Prompting settings
    grid_prompt_density: int = Field(
        default=16,
        description="Points per side for grid prompting"
    )
    
    reprompt_interval: int = Field(
        default=60,
        description="Frames between re-prompting for new objects"
    )
    
    # Object filtering
    min_object_area: int = Field(
        default=1000,
        description="Minimum pixel area for tracking"
    )
    
    # Google Lens API
    lens_api_rate_limit: int = Field(
        default=10,
        description="Max API calls per second"
    )
    
    enable_identification_cache: bool = Field(
        default=True,
        description="Cache similar objects to reduce API calls"
    )
```

### Phase 6: Main Service Updates

#### 6.1 Streamlined Main Loop

**File**: `frame_processor/main.py` (UPDATE)
```python
class FrameProcessorService:
    def __init__(self):
        self.config = Config()
        self.video_processor = VideoProcessor(self.config)
        self.lens_identifier = LensIdentifier(self.config)
        
        # Performance monitoring
        self.fps_monitor = FPSMonitor(target=self.config.target_fps)
        
    async def process_stream_message(self, message: aio_pika.IncomingMessage):
        """Process H.264 stream chunks with performance monitoring."""
        
        stream_data = self.parse_message(message)
        
        # Decode H.264 chunk
        frames = await self.decoder.decode_chunk(
            stream_data.stream_id,
            stream_data.chunk_data
        )
        
        for frame in frames:
            # Monitor FPS
            with self.fps_monitor.measure():
                # Process through pipeline
                result = await self.video_processor.process_stream_frame(
                    stream_data.stream_id,
                    frame
                )
                
                # Async identification (non-blocking)
                asyncio.create_task(
                    self.identify_and_publish(result)
                )
            
            # Dynamic quality adjustment if FPS drops
            if self.fps_monitor.current_fps < self.config.target_fps:
                await self.adjust_quality_settings()
```

## Performance Considerations

### GPU Memory Management
- Use SAM2-Small model (balanced speed/quality)
- Limit active streams based on GPU memory
- Aggressive cleanup of completed streams

### Processing Optimization
- Grid prompting only every N frames
- Skip frames if falling behind real-time
- Batch Google Lens API calls
- Cache identification results

### Expected Performance
- **SAM2-Small on RTX 3090**: 15-20 FPS at 720p
- **SAM2-Tiny on RTX 3090**: 20-25 FPS at 720p
- **Memory usage**: ~4GB per active stream
- **Latency**: <100ms per frame

## Benefits Over Current System

1. **Temporal Consistency**: Objects maintain IDs across frames
2. **Occlusion Handling**: SAM2Long memory tree handles temporary occlusions
3. **Efficiency**: No redundant detection after initial discovery
4. **Real-time**: Optimized for consistent 15+ FPS
5. **Integrated Pipeline**: Seamless flow from tracking to identification

## Implementation Priority

1. **Week 1**: Core SAM2 video tracker with memory tree
2. **Week 2**: Grid prompting and real-time optimizations  
3. **Week 3**: Integration with enhancement pipeline
4. **Week 4**: Google Lens API integration and caching
5. **Week 5**: Performance tuning and monitoring

## Key Differences from Original Plan

- **Removed**: Backward compatibility, multiple detector support
- **Added**: SAM2Long memory tree from start, Google Lens integration
- **Simplified**: Single tracker approach, basic grid prompting
- **Optimized**: Real-time performance focus, smaller models
- **Integrated**: Direct pipeline to object identification