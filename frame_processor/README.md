# Enhanced Frame Processor

This service processes video frames with object detection, tracking, and dimension estimation to provide real-world scaling for 3D reconstruction.

> **Update**: The frame processor now uses WebSocket for video input instead of RabbitMQ, providing lower latency and simplified architecture. See [WEBSOCKET_UNIFIED.md](WEBSOCKET_UNIFIED.md) for details.

## Features

- **WebSocket Video Input**: Direct H.264 stream consumption via WebSocket
- **SAM2 Video Tracking**: Advanced video object segmentation and tracking
- **Object Tracking**: IOU-based tracking across frames
- **Quality Scoring**: Selects best frames based on sharpness, exposure, size, and centering
- **Image Enhancement**: Improves image quality before API processing
- **Google Lens Integration**: Identifies objects using SerpAPI
- **Dimension Extraction**: Uses Perplexity AI to get real-world object dimensions
- **Scene Scaling**: Calculates weighted average scale for 3D reconstruction
- **Rerun Visualization**: Real-time monitoring with custom layout

## Directory Structure

```
frame_processor/
├── main.py                # Main unified processor with WebSocket support
├── modules/                # Processing modules
│   ├── __init__.py
│   ├── tracker.py         # Object tracking
│   ├── frame_scorer.py    # Frame quality scoring
│   ├── enhancement.py     # Image enhancement
│   ├── api_client.py      # External API integration
│   └── scene_scaler.py    # Scene scale calculation
├── docs/                  # Documentation
│   ├── API_INTEGRATION_GUIDE.md
│   └── implementation.md
├── Dockerfile
├── requirements.txt
├── CLAUDE.md              # AI assistant instructions
└── worldsystem-23f7306a1a75.json  # Google Cloud credentials
```

## Configuration

Set these environment variables in the main `.env` file:

```env
# API Keys
SERPAPI_API_KEY=your_serpapi_key
PPLX_API_KEY=your_perplexity_key

# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=/app/worldsystem-23f7306a1a75.json
GCS_BUCKET_NAME=worldsystem-frame-processor

# Processing
PROCESS_AFTER_SECONDS=1.5      # Time before processing object
REPROCESS_INTERVAL_SECONDS=3.0  # Time between reprocessing
IOU_THRESHOLD=0.3              # Tracking IOU threshold
MAX_LOST_FRAMES=10             # Frames before removing lost track

# Enhancement
ENHANCEMENT_ENABLED=true
ENHANCEMENT_GAMMA=1.2
ENHANCEMENT_ALPHA=1.3
ENHANCEMENT_BETA=20

# Scene Scaling
MIN_CONFIDENCE_FOR_SCALING=0.7
DIMENSION_CACHE_EXPIRY_DAYS=30
```

## How It Works

1. **Detection & Tracking**: YOLO detects objects, tracker maintains their history
2. **Quality Assessment**: Each frame is scored for quality (sharpness, exposure, etc.)
3. **Timed Processing**: After 1.5s of tracking, best frame is selected
4. **Enhancement**: Image is enhanced for better API recognition
5. **Identification**: Google Lens identifies the object
6. **Dimension Lookup**: Perplexity AI retrieves real-world dimensions
7. **Scale Calculation**: Weighted average scale is computed and published

## API Usage

The service publishes scene scale information to the `scene_scaling_exchange`:

```json
{
  "scale_factor": 0.1234,        // meters per reconstruction unit
  "units_per_meter": 8.1,        // reconstruction units per meter
  "confidence": 0.85,            // confidence of scale estimate
  "num_estimates": 3,            // number of objects used
  "timestamp_ns": 1234567890     // nanosecond timestamp
}
```

## Development

To revert to the original simple processor:
```bash
cp frame_processor_original.py frame_processor.py
```

To test modules individually:
```python
from modules.tracker import ObjectTracker
tracker = ObjectTracker()
```

## Performance

- GPU acceleration for YOLO and enhancement
- Caching for API responses (30 days)
- Efficient frame selection reduces API calls
- Prometheus metrics on port 8003