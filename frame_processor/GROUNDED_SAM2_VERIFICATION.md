# Grounded SAM2 Tracking Integration

## Status

✅ **Fully integrated** with WebSocket pipeline as of latest commit.

## Implementation Details

The Grounded SAM2 tracker has been integrated into the main frame processor pipeline with the following key features:

### 1. Continuous Object Detection
- Performs full object detection every N frames (configurable)
- Uses GroundingDINO for open-vocabulary detection
- Automatically detects new objects entering the scene

### 2. Consistent ID Management
- Assigns unique IDs to each detected object
- Maintains ID consistency across frames using IOU matching
- Tracks objects even when temporarily occluded

### 3. WebSocket Integration
- Fully compatible with the existing WebSocket video streaming pipeline
- Works with the same `/ws/video/consume` endpoint
- Publishes results through the same RabbitMQ exchanges

## Configuration

To enable Grounded SAM2 tracking, set the following environment variables:

```bash
# Enable Grounded SAM2 tracker
VIDEO_TRACKER_TYPE=grounded_sam2

# Detection settings
GROUNDED_DETECTION_INTERVAL=30  # Detect new objects every 30 frames
GROUNDED_TEXT_PROMPT="person. car. object."  # What to detect
GROUNDED_BOX_THRESHOLD=0.25  # Detection confidence threshold
GROUNDED_TEXT_THRESHOLD=0.2  # Text matching threshold
GROUNDED_IOU_THRESHOLD=0.5  # IOU for matching detections to tracks

# Other standard settings still apply
WS_HOST=server
WS_PORT=5001
FRAME_PROCESSOR_SKIP=1
```

## Usage Examples

### 1. Track All Objects (Default)
```bash
docker-compose run -e VIDEO_TRACKER_TYPE=grounded_sam2 frame_processor
```

### 2. Track Specific Objects
```bash
docker-compose run \
  -e VIDEO_TRACKER_TYPE=grounded_sam2 \
  -e GROUNDED_TEXT_PROMPT="person. face. hand." \
  frame_processor
```

### 3. High-Frequency Detection (Every 10 frames)
```bash
docker-compose run \
  -e VIDEO_TRACKER_TYPE=grounded_sam2 \
  -e GROUNDED_DETECTION_INTERVAL=10 \
  frame_processor
```

## Architecture

```
WebSocket Video Stream
        ↓
   main.py (WebSocketVideoAdapter)
        ↓
   VideoProcessor
        ↓
   GroundedSAM2Tracker ← (NEW)
        ↓
   RabbitMQ Publisher
```

## Key Differences from SAM2 Realtime Tracker

| Feature | SAM2 Realtime | Grounded SAM2 |
|---------|---------------|---------------|
| Detection Method | Point prompts | Text prompts |
| New Objects | Manual prompting | Auto-detection |
| Use Case | Known scenes | Dynamic scenes |
| Performance | Faster | More comprehensive |

## Testing

To verify the implementation:

1. Start the system:
```bash
./start.sh
```

2. Enable Grounded SAM2:
```bash
export VIDEO_TRACKER_TYPE=grounded_sam2
docker-compose --profile frame_processor up
```

3. Stream video through WebSocket (via Android app or simulation)

4. Monitor logs for detection events:
```bash
docker logs -f worldsystem-frame_processor-1 2>&1 | grep -E "(Detected|tracks|GroundedSAM2)"
```

## Performance Considerations

- Detection interval affects performance significantly
- Lower intervals (e.g., 10) provide better new object detection but use more GPU
- Higher intervals (e.g., 60) are more efficient but may miss fast-moving objects
- Recommended: 30 frames for balanced performance

## Future Improvements

- [ ] Dynamic detection intervals based on scene complexity
- [ ] Custom text prompts per stream
- [ ] Integration with scene context from SLAM3R