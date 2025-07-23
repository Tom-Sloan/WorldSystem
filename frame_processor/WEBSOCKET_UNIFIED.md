# WebSocket Unified Frame Processor

## Overview

The frame processor has been unified to use WebSocket for video input while maintaining RabbitMQ for result publishing. This eliminates the complexity of having two separate implementations.

## Architecture

```
Android/Simulator
    ↓ WebSocket (H.264)
Server (/ws/video)
    ↓ WebSocket broadcast (/ws/video/consume)
Frame Processor (main.py)
    ├─→ Video Processing (SAM2)
    └─→ RabbitMQ Publishing (results)
```

## Key Changes

### 1. Video Input
- **Before**: RabbitMQ queue (`video_stream_exchange`)
- **After**: WebSocket endpoint (`/ws/video/consume`)

### 2. Single Implementation
- **Before**: Two files (`main.py`, `websocket_frame_processor.py`)
- **After**: One unified `main.py`

### 3. Docker Configuration
- **Before**: Two Dockerfiles
- **After**: Single `Dockerfile` with all dependencies

## Configuration

### Environment Variables
```bash
# WebSocket Configuration
WS_HOST=server              # Server hostname
WS_PORT=5001               # Server port
FRAME_PROCESSOR_SKIP=1     # Process every Nth frame

# RabbitMQ Configuration (for publishing)
RABBITMQ_URL=amqp://admin:admin@rabbitmq:5672/

# Other settings remain the same
MODEL_NAME=sam2_tiny
TARGET_FPS=15
PROCESSING_RESOLUTION=720
```

## Running the Unified Version

### Build and Run
```bash
# Build the frame processor
docker-compose --profile frame_processor build

# Run with GPU support
docker-compose --profile frame_processor up

# View logs
docker logs -f worldsystem-frame_processor-1
```

### Testing
1. Start core services (server, rabbitmq)
2. Run frame processor
3. Send video from app/simulator
4. Monitor logs and metrics

## Implementation Details

### WebSocketVideoAdapter
A new adapter class bridges the WebSocket consumer with the existing async pipeline:

```python
class WebSocketVideoAdapter(WebSocketVideoConsumer):
    def process_frame(self, frame, frame_number):
        # Bridge to async pipeline
        self.frame_queue.put_nowait((frame, frame_number, timestamp))
```

### Processing Flow
1. WebSocket receives H.264 stream
2. Base class decodes to frames
3. Adapter queues frames for async processing
4. Existing pipeline processes frames
5. Results published to RabbitMQ

## Benefits

1. **Simplified Architecture**: One implementation to maintain
2. **Lower Latency**: Direct WebSocket connection for video
3. **Preserved Features**: All existing functionality intact
4. **Clean Migration**: Minimal changes to core logic

## Monitoring

### Metrics
- Same Prometheus metrics at http://localhost:8003/metrics
- Performance dashboard still available
- WebSocket connection status in logs

### Health Check
```bash
# Check if processing frames
curl http://localhost:8003/metrics | grep frames_processed

# View active tracks
curl http://localhost:8003/metrics | grep active_tracks
```

## Troubleshooting

### WebSocket Connection Issues
1. Check server is running: `docker ps | grep server`
2. Verify endpoint exists: `curl http://localhost:5001/health`
3. Check logs: `docker logs worldsystem-frame_processor-1`

### No Frames Processing
1. Ensure video is being sent to server
2. Check WebSocket connection in logs
3. Verify GPU is available: `docker exec worldsystem-frame_processor-1 nvidia-smi`

## Migration Notes

### From RabbitMQ Version
- No changes needed in configuration
- Video input automatically uses WebSocket
- All other features remain the same

### From Separate WebSocket Version
- Remove `websocket_frame_processor.py`
- Remove `Dockerfile.websocket`
- Use standard `docker-compose` commands

## Future Improvements

1. Add WebSocket reconnection with exponential backoff
2. Support multiple WebSocket streams
3. Add stream quality adaptation
4. Implement WebRTC for even lower latency