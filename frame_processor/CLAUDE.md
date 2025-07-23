# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Frame Processor Service Overview

The frame_processor is a GPU-accelerated video processing service for the WorldSystem that performs real-time object detection, tracking, and identification. It processes H.264 video streams, applies ML models for analysis, and publishes results to enable AR/VR experiences with real-world scale.

## Architecture

The service uses a modular, factory-based architecture with async processing throughout:

```
frame_processor/
├── main.py                    # Async entry point with RabbitMQ consumer
├── core/                      # Core utilities and configuration
│   ├── config.py             # Pydantic-based configuration with profiles
│   ├── utils.py              # Logging, NTP sync, timing utilities
│   └── video_decoder.py      # H.264 stream decoder using PyAV
├── video/                     # Video processing with SAM2
│   ├── video_processor.py    # Main video tracking pipeline
│   └── sam2_tracker.py       # SAM2 model wrapper with GPU optimization
├── external/                  # External API integrations
│   └── api_client.py         # Google Lens (SerpAPI) and Perplexity clients
├── pipeline/                  # Processing pipeline components
│   ├── processor.py          # Main orchestrator
│   ├── scorer.py             # Frame quality scoring
│   ├── enhancer.py           # Image enhancement
│   └── publisher.py          # RabbitMQ publisher
├── visualization/             # Real-time visualization
│   └── rerun_client.py       # Rerun integration
└── models/                    # ML model weights (gitignored)
```

## Development Commands

### Build and Run

```bash
# Install models (required first time)
cd frame_processor
./install_models.sh

# Build the service
docker-compose build frame_processor

# Run with GPU support
docker-compose --profile frame_processor up

# Run without frame processor
docker compose up --detach $(docker compose config --services | grep -v frame_processor)

# Access service shell
docker-compose exec frame_processor bash

# View logs with filtering
docker logs -f worldsystem-frame_processor-1 2>&1 | grep -E "(Processing|FPS|tracks)"

# Enable rich terminal UI
ENABLE_RICH_TERMINAL=true docker-compose run frame_processor
```

### Testing and Monitoring

```bash
# Run tests (requires test dependencies)
pip install -r requirements-test.txt
pytest tests/

# Full pipeline test (from WorldSystem root)
./test_full_pipeline.sh

# Monitor health and metrics
curl http://localhost:5001/health/video        # Via server
curl http://localhost:8003/metrics              # Prometheus metrics

# Check video stream status
curl http://localhost:5001/video/status | jq

# Connect to Rerun viewer
# Service auto-connects to: rerun+http://localhost:9876/proxy
```

## Configuration

The service uses environment variables and configuration profiles:

### Model Configuration
- `MODEL_NAME`: SAM2 model variant (`sam2_tiny`, `sam2_small`, `sam2_base_plus`, `sam2_large`)
- `CONFIG_PROFILE`: Preset configurations (`performance`, `balanced`, `quality`)
- `TARGET_FPS`: Target processing frame rate (default: 15)
- `PROCESSING_RESOLUTION`: Max resolution for processing (default: 720)

### Processing Settings
- `DETECTOR_CONFIDENCE`: Detection confidence threshold (0.0-1.0)
- `TRACKER_MAX_LOST`: Frames before removing lost track
- `MIN_MASK_AREA`: Minimum mask area to consider valid
- `PROPAGATE_MAX_FRAMES`: Max frames for mask propagation

### API Integration
- `USE_SERPAPI`: Enable Google Lens identification
- `USE_PERPLEXITY`: Enable dimension lookup
- `USE_GCS`: Enable Google Cloud Storage for images
- `SERPAPI_API_KEY`: SerpAPI authentication
- `PERPLEXITY_KEY`: Perplexity AI key
- `GCS_BUCKET_NAME`: GCS bucket for uploads

### Enhancement Settings
- `ENHANCEMENT_ENABLED`: Enable frame enhancement
- `ENHANCEMENT_AUTO_ADJUST`: Auto-adjust enhancement parameters
- `ENHANCEMENT_GAMMA`: Gamma correction factor
- `ENHANCEMENT_ALPHA`: Contrast factor
- `ENHANCEMENT_BETA`: Brightness offset

### RabbitMQ Configuration
- **Input Exchanges**: `video_stream_exchange`, `analysis_mode_exchange`
- **Output Exchanges**: `processed_frames_exchange`, `api_results_exchange`, `scene_scaling_exchange`

## Key Technical Details

### H.264 Stream Processing
- Receives H.264 chunks via RabbitMQ from WebSocket streams
- Uses PyAV for hardware-accelerated decoding
- Maintains per-stream decoders and buffers
- Automatic cleanup on stream disconnect

### SAM2 Video Tracking
- Real-time video object segmentation and tracking
- GPU memory-aware model selection
- Dynamic model switching on OOM
- Compiled models for inference optimization
- Point-based prompting with quality scoring

### Processing Pipeline
1. **Stream Reception**: H.264 chunks from RabbitMQ
2. **Decoding**: PyAV decoder with buffering
3. **Tracking**: SAM2 video tracking with propagation
4. **Quality Scoring**: Frame selection based on sharpness/exposure
5. **Enhancement**: Optional image improvement
6. **API Processing**: Batched Google Lens + Perplexity
7. **Publishing**: Multi-exchange result distribution
8. **Visualization**: Real-time Rerun display

### Performance Optimizations
- GPU acceleration with CUDA 12.1+
- Model compilation with PyTorch 2.0+
- Stream-specific resource management
- Batch API processing to reduce costs
- Async I/O throughout the pipeline
- Memory-efficient mask storage

### Error Handling
- Graceful degradation when services unavailable
- Automatic GPU OOM recovery with model downgrade
- Stream cleanup on disconnect
- Retry logic with exponential backoff
- Structured logging to separate files

## Monitoring and Debugging

### Prometheus Metrics (port 8003)
- `frame_processor_frames_processed_total`
- `frame_processor_tracks_active`
- `frame_processor_processing_time_seconds`
- `frame_processor_api_calls_total`
- `frame_processor_gpu_memory_usage_bytes`
- `frame_processor_stream_active_count`

### Logging
- Structured JSON logs to `/app/logs/`
- Separate logs: processing, api, errors, metrics
- Log level configurable via `LOG_LEVEL`
- Rich terminal UI for development

### Rerun Visualization
- Automatic connection on startup
- Real-time frame and mask display
- Track visualization with IDs
- Performance metrics overlay
- Stream health indicators

## Integration with WorldSystem

The frame_processor is a critical component in the real-time pipeline:

1. **Receives** H.264 video streams from Android app via server
2. **Processes** frames with SAM2 for object tracking
3. **Identifies** objects using Google Lens API
4. **Estimates** dimensions via Perplexity AI
5. **Publishes** results for AR overlay and 3D reconstruction
6. **Enables** real-world scale calculation for SLAM3R

### Dependencies
- **Required**: RabbitMQ, Server (for streams)
- **Optional**: Rerun, External APIs (Lens, Perplexity)
- **GPU**: NVIDIA runtime with CUDA 12.1+

### Configuration Profiles
- **performance**: Optimized for speed (tiny model, 480p, 25fps)
- **balanced**: Default balanced settings (small model, 720p, 15fps)
- **quality**: Maximum quality (base+ model, 1080p, 10fps)