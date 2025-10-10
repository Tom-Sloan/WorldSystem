# Frame Processor - AI-Powered Object Detection and Tracking

## Overview

The frame_processor is a GPU-accelerated video processing service that performs real-time object detection, tracking, and identification on drone video streams. It represents the AI intelligence layer of WorldSystem, using state-of-the-art models like SAM2 (Segment Anything Model 2) for video object segmentation and Google Lens for object identification. The service enables augmented reality experiences by understanding what objects are in the scene and providing real-world scale information.

## What This Service Does

### Core Functionality
- **Video Object Tracking**: SAM2-based segmentation and tracking across frames
- **Object Identification**: Google Lens API integration for object recognition
- **Dimension Estimation**: Perplexity AI for real-world size lookup
- **Frame Enhancement**: Optional image quality improvement
- **Stream Management**: Per-stream decoder and tracking state
- **Batch Processing**: Efficient API call batching to reduce costs

### Key Features
- WebSocket video consumption from server
- GPU-accelerated ML inference
- Dynamic model switching on OOM
- Multi-level caching for API responses
- Real-time Rerun visualization
- Prometheus metrics integration
- Configurable processing profiles
- Rich terminal UI for debugging

## System Architecture Connection

### Data Flow
```
Server (WebSocket)
    ↓ Video Frames
Frame Processor
    ├── SAM2 Tracking (GPU)
    ├── Quality Scoring
    ├── API Enhancement
    │   ├── Google Lens (Object ID)
    │   └── Perplexity (Dimensions)
    └── Publishing
        ├── RabbitMQ (processed_frames)
        ├── Rerun (visualization)
        └── Local Output (optional)
```

### Communication Protocols
- **Input**: WebSocket video stream from server
- **Output**: 
  - RabbitMQ exchanges for processed results
  - Rerun for real-time visualization
  - Local file output (optional)
  - API results to scene_scaling_exchange

### Integration Points
1. **WebSocket Consumer**: Video frame reception
2. **RabbitMQ Publisher**: Result distribution
3. **External APIs**: Google Lens, Perplexity
4. **Rerun Visualization**: Real-time display
5. **Prometheus Metrics**: Performance monitoring

## Development History

### Evolution Timeline

1. **Initial Implementation** (Commits 5081161, 3511a04)
   - Basic frame processing structure
   - YOLO-based detection
   - RabbitMQ integration

2. **SAM Integration** (Commits 03176c0, d78532f, 44144c0)
   - Added Segment Anything Model
   - Improved segmentation quality
   - GPU acceleration

3. **H.264 Streaming** (Commit 52fcd0d)
   - H.264 decoder integration
   - PyAV for hardware acceleration
   - Stream-specific resource management

4. **SAM2 Video Upgrade** (Commits 3a9e2c4, 2a77bea)
   - Migrated to SAM2 for video
   - Temporal consistency
   - Object tracking across frames

5. **WebSocket Transition** (Commits 1b55dd8, 415cb86, ae9f9da)
   - Removed RTSP/YOLO dependencies
   - Direct WebSocket consumption
   - Simplified architecture

6. **Performance & Monitoring** (Commits 42641e8, 78e0a88)
   - Rich terminal UI
   - Enhanced visualization
   - Prometheus metrics

### Design Philosophy
- **GPU-First**: All ML inference on GPU
- **Modular Pipeline**: Pluggable components
- **Async Throughout**: Non-blocking I/O
- **Graceful Degradation**: Handle service failures

## Technical Details

### Technology Stack
- **Language**: Python 3.11+
- **ML Framework**: PyTorch 2.0+
- **Key Models**:
  - SAM2 (video object segmentation)
  - Google Lens (via SerpAPI)
  - Perplexity AI (dimension lookup)
- **Infrastructure**:
  - WebSocket for video input
  - RabbitMQ for messaging
  - Prometheus for metrics
  - Rerun for visualization

### Architecture Components

#### Core Pipeline (`main.py`)
1. WebSocket video consumer adapter
2. Async frame processing
3. Performance monitoring
4. Service orchestration

#### Video Processing (`video/`)
- `video_processor.py`: Main tracking pipeline
- `sam2_tracker.py`: SAM2 model wrapper
- Stream-specific state management
- GPU memory optimization

#### External Integrations (`external/`)
- `api_client.py`: Google Lens & Perplexity clients
- `lens_identifier.py`: Object identification logic
- `lens_batch_processor.py`: Batched API calls
- Multi-level caching system

#### Pipeline Components (`pipeline/`)
- `processor.py`: Main orchestrator
- `scorer.py`: Frame quality assessment
- `enhancer.py`: Image enhancement
- `publisher.py`: RabbitMQ publishing
- `output_manager.py`: Result handling

### Processing Pipeline

1. **Frame Reception**
   - WebSocket frames from server
   - Frame skip for performance
   - Timestamp synchronization

2. **SAM2 Tracking**
   - Point-based prompting
   - Mask propagation
   - Track management
   - Quality scoring

3. **Enhancement** (Optional)
   - Gamma correction
   - Contrast adjustment
   - Brightness optimization
   - Auto-adjustment mode

4. **API Processing**
   - Batch accumulation
   - Google Lens queries
   - Perplexity dimension lookup
   - Result caching

5. **Publishing**
   - Multi-exchange distribution
   - Visualization updates
   - Scene scaling info

### Configuration System

Environment-based configuration with profiles:

```python
# Profiles
CONFIG_PROFILE=performance|balanced|quality

# Model Settings
MODEL_NAME=sam2_tiny|sam2_small|sam2_base_plus|sam2_large
TARGET_FPS=15
PROCESSING_RESOLUTION=720

# API Configuration
USE_SERPAPI=true
USE_PERPLEXITY=true
SERPAPI_API_KEY=xxx
PERPLAXITY_KEY=xxx

# Enhancement
ENHANCEMENT_ENABLED=true
ENHANCEMENT_AUTO_ADJUST=true
```

### Performance Characteristics
- **Processing Rate**: 10-25 FPS (model dependent)
- **Latency**: 40-100ms per frame
- **GPU Memory**: 2-8GB (model size)
- **API Latency**: 200-500ms (cached: 0ms)
- **Batch Size**: 5-10 items optimal

## Algorithms and Implementation

### SAM2 Video Tracking
- Temporal consistency across frames
- Point-based initial prompting
- Mask propagation between frames
- Quality-based track management
- GPU-optimized inference

### Frame Quality Scoring
- Laplacian variance (sharpness)
- Histogram analysis (exposure)
- Motion blur detection
- Combined quality metric

### API Integration Strategy
1. **Batching**: Accumulate requests
2. **Caching**: Multi-level cache
3. **Fallback**: Handle API failures
4. **Throttling**: Rate limit compliance

## Challenges and Solutions

1. **GPU Memory Management**
   - Challenge: OOM with large models
   - Solution: Dynamic model switching, memory monitoring

2. **API Cost Control**
   - Challenge: Expensive external API calls
   - Solution: Batching, aggressive caching, quality filtering

3. **Real-time Performance**
   - Challenge: Complex pipeline under 100ms
   - Solution: Async processing, frame skipping, GPU optimization

4. **Stream Synchronization**
   - Challenge: Multiple concurrent streams
   - Solution: Per-stream state management, cleanup on disconnect

5. **Model Accuracy**
   - Challenge: Generic models for specific objects
   - Solution: Quality scoring, multiple prompts, API verification

## Debug and Monitoring

### Rich Terminal UI
Enabled with `ENABLE_RICH_TERMINAL=true`:
- Real-time performance metrics
- Component status display
- Event log with timestamps
- Frame processing breakdown

### Prometheus Metrics
- `frame_processor_frames_processed_total`
- `frame_processor_tracks_active`
- `frame_processor_processing_time_seconds`
- `frame_processor_api_calls_total`
- `frame_processor_gpu_memory_usage_bytes`

### Logging System
Structured logs to `/app/logs/`:
- `processing.log`: Main pipeline
- `api.log`: External API calls
- `errors.log`: Error tracking
- `metrics.log`: Performance data

### Rerun Visualization
- Frame display with masks
- Track ID overlay
- Performance graphs
- API results display

## Usage Examples

### Basic Usage
```bash
# Default balanced mode
docker-compose --profile frame_processor up

# Performance mode
CONFIG_PROFILE=performance docker-compose up frame_processor

# Quality mode with enhancement
CONFIG_PROFILE=quality ENHANCEMENT_ENABLED=true \
docker-compose up frame_processor
```

### Development
```bash
# Rich terminal UI
ENABLE_RICH_TERMINAL=true docker-compose run frame_processor

# Disable APIs for testing
USE_SERPAPI=false USE_PERPLEXITY=false \
docker-compose up frame_processor

# Custom model
MODEL_NAME=sam2_large TARGET_FPS=5 \
docker-compose up frame_processor
```

## Future Enhancements

### Planned Features
1. **Custom Model Training**
   - Domain-specific fine-tuning
   - Indoor object specialization
   - Drone perspective adaptation

2. **Edge Deployment**
   - Model quantization
   - TensorRT optimization
   - Reduced latency pipeline

3. **Advanced Tracking**
   - Multi-object relationships
   - Occlusion handling
   - Long-term re-identification

4. **Scene Understanding**
   - Room type classification
   - Object placement logic
   - Spatial relationships

### Research Directions
- Few-shot object learning
- Neural radiance field integration
- Semantic scene graphs
- Real-time style transfer

## Performance Optimization

### For Maximum Speed
```bash
CONFIG_PROFILE=performance
FRAME_PROCESSOR_SKIP=3
ENHANCEMENT_ENABLED=false
PROCESSING_RESOLUTION=480
```

### For Best Quality
```bash
CONFIG_PROFILE=quality
FRAME_PROCESSOR_SKIP=1
ENHANCEMENT_ENABLED=true
MODEL_NAME=sam2_large
```

### For API Cost Reduction
```bash
DETECTOR_CONFIDENCE=0.8
MIN_MASK_AREA=1000
QUALITY_THRESHOLD=0.7
```

This service provides the intelligence layer that transforms raw video into semantic understanding, enabling AR experiences with real-world context and scale.