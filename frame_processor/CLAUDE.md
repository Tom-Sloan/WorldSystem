# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Frame Processor Service Overview

The frame_processor is a GPU-accelerated video processing service with a modular architecture that:
- Performs real-time object detection and tracking
- Identifies objects using Google Lens API
- Estimates object dimensions via Perplexity AI
- Calculates real-world scene scale
- Provides real-time visualization via Rerun
- Publishes results to multiple RabbitMQ exchanges

## Architecture

The service follows a clean, modular architecture:

```
frame_processor/
├── main.py                    # Async entry point with RabbitMQ consumer
├── core/                      # Core utilities
│   ├── config.py             # Pydantic-based configuration
│   └── utils.py              # Logging, NTP sync, timing utilities
├── detection/                 # Detection algorithms (factory pattern)
│   ├── base.py              # Abstract detector interface
│   └── yolo.py              # YOLOv11 implementation
├── tracking/                  # Tracking algorithms (factory pattern)
│   ├── base.py              # Abstract tracker interface
│   └── iou_tracker.py       # IOU-based tracker with quality scoring
├── external/                  # External API integrations
│   └── api_client.py        # GCS, SerpAPI, Perplexity clients
├── pipeline/                  # Processing pipeline
│   ├── processor.py         # Main orchestrator
│   ├── scorer.py            # Frame quality scoring
│   ├── enhancer.py          # Image enhancement
│   └── publisher.py         # RabbitMQ publisher
└── visualization/             # Real-time visualization
    ├── rerun_client.py      # Rerun integration
    └── enhanced_visualizer.py
```

## Development Commands

```bash
# Build the service
docker-compose build frame_processor

# Run with GPU support
docker-compose --profile frame_processor up

# Run without specific services
docker compose up --detach $(docker compose config --services | grep -v frame_processor)

# View logs
docker logs -f worldsystem-frame_processor-1

# Monitor metrics
curl http://localhost:8003/metrics

# Connect to Rerun viewer (automatic at startup)
# Service connects to: rerun+http://localhost:9876/proxy
```

## Configuration

### Algorithm Selection
- `DETECTOR_TYPE`: Detection algorithm (`yolo`, future: `detectron2`, `grounding_dino`)
- `TRACKER_TYPE`: Tracking algorithm (`iou`, future: `sort`, `deep_sort`)

### Detection Settings
- `DETECTOR_MODEL`: Model path (default: `yolov11l.pt`)
- `DETECTOR_CONFIDENCE`: Confidence threshold (0.0-1.0)
- `DETECTOR_DEVICE`: `cuda` or `cpu`
- `DETECTOR_BATCH_SIZE`: Batch size for inference

### Tracking Settings
- `TRACKER_IOU_THRESHOLD`: IOU matching threshold (0.0-1.0)
- `TRACKER_MAX_LOST`: Frames before removing lost track
- `PROCESS_AFTER_SECONDS`: Delay before API processing
- `REPROCESS_INTERVAL_SECONDS`: Interval between reprocessing

### API Integration
- `USE_GCS`: Enable Google Cloud Storage upload
- `USE_SERPAPI`: Enable Google Lens identification
- `USE_PERPLEXITY`: Enable dimension lookup
- `GCS_BUCKET_NAME`: GCS bucket for uploads
- `SERPAPI_KEY`: SerpAPI authentication
- `PERPLEXITY_API_KEY`: Perplexity AI key

### Enhancement Settings
- `ENHANCEMENT_ENABLED`: Enable image enhancement
- `ENHANCEMENT_AUTO_ADJUST`: Auto-adjust parameters
- `ENHANCEMENT_GAMMA`: Gamma correction factor
- `ENHANCEMENT_ALPHA`: Contrast factor
- `ENHANCEMENT_BETA`: Brightness offset

### RabbitMQ Exchanges
- **Input**: `video_frames_exchange`, `analysis_mode_exchange`, `imu_data_exchange`
- **Output**: `processed_frames_exchange`, `scene_scaling_exchange`, `api_results_exchange`

## Key Components

### Detection Module
- Abstract base class in `detection/base.py` for easy extension
- YOLO detector supports GPU acceleration and batch processing
- Returns list of `Detection` objects with bbox, confidence, class

### Tracking Module  
- Abstract base class in `tracking/base.py` for algorithm swapping
- IOU tracker maintains tracks with quality scoring
- Memory-optimized: stores only ROIs, not full frames
- Integrated frame selection based on sharpness, exposure, centering

### API Pipeline
1. **Frame Selection**: Best quality frame after 1.5s delay
2. **Enhancement**: Optional auto-adjusting enhancement
3. **GCS Upload**: Temporary storage with signed URLs
4. **Google Lens**: Object identification via SerpAPI
5. **Perplexity**: Dimension query with structured extraction
6. **Scene Scaling**: Real-world scale calculation

### Processing Flow
1. Async message consumption from RabbitMQ
2. Object detection (configurable algorithm)
3. Track association and management
4. Quality-based frame selection
5. Timed API processing pipeline
6. Multi-exchange result publishing
7. Real-time Rerun visualization

## Monitoring & Debugging

### Prometheus Metrics (port 8003)
- `frame_processor_frames_processed_total`
- `frame_processor_yolo_detections_total`
- `frame_processor_processing_time_seconds`
- `frame_processor_active_tracks`
- `frame_processor_api_calls_total`
- `frame_processor_ntp_time_offset_seconds`

### Logging
- Structured JSON logs to `/app/logs/`
- Separate logs: detections, metrics, errors, api
- Configurable via `LOG_LEVEL` environment variable

### Rerun Visualization
- Automatic connection on startup
- Real-time frame display with annotations
- Track visualization with IDs
- Performance metrics overlay

## Adding New Components

### New Detector
1. Create class in `detection/` inheriting from `BaseDetector`
2. Implement `detect()` method returning `List[Detection]`
3. Register in `detection/__init__.py` factory

### New Tracker
1. Create class in `tracking/` inheriting from `BaseTracker`
2. Implement `update()` and track management methods
3. Register in `tracking/__init__.py` factory

### New API Integration
1. Add client methods to `external/api_client.py`
2. Add feature flag to `core/config.py`
3. Integrate into `pipeline/processor.py` workflow

## Performance Considerations

- GPU acceleration auto-detected and preferred
- Batch processing support for efficiency
- Async I/O throughout the pipeline
- ROI-only storage reduces memory usage
- API response caching minimizes costs
- Configurable worker threads for parallel processing
- NTP synchronization for accurate timestamps