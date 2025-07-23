# WebSocket Frame Processor with Rerun Visualization

This implementation allows the frame_processor to consume H.264 video streams directly via WebSocket, bypassing RabbitMQ for video data. It includes Rerun visualization for testing and monitoring.

## Architecture

```
Android/Simulator
    ↓ WebSocket (H.264)
Server (/ws/video)
    ↓ WebSocket broadcast
Frame Processor (/ws/video/consume)
    ↓ Decode + Track
    ├─→ Rerun (visualization)
    └─→ RabbitMQ (results)
```

## Features

- **Direct WebSocket consumption** of H.264 video streams
- **Rerun visualization** showing:
  - Raw H.264 stream (`video/h264_stream`)
  - Decoded frames (`video/decoded_frame`)
  - Tracked objects (`video/tracked_frame`)
  - Individual masks (`tracks/mask_*`)
- **SAM2 video tracking** with persistent object IDs
- **Prometheus metrics** for monitoring
- **Automatic reconnection** on connection loss

## Quick Start

### 1. Start Rerun Viewer (Optional)
```bash
# Install Rerun if not already installed
pip install rerun-sdk

# Start Rerun viewer
rerun --port 9876
```

### 2. Run the Test
```bash
# Run the automated test script
./test_websocket_frame_processor.sh
```

### 3. Send Video
Start sending video from the Android app or simulator to see:
- Real-time H.264 stream in Rerun
- Object detection and tracking
- Performance metrics

## Manual Setup

### Build and Run
```bash
# Build the WebSocket frame processor
docker-compose -f docker-compose.websocket.yml build

# Start the service
docker-compose -f docker-compose.websocket.yml up -d

# View logs
docker logs -f worldsystem-frame_processor_websocket
```

### Configuration
Environment variables:
- `WS_HOST`: Server hostname (default: server)
- `WS_PORT`: Server port (default: 5001)
- `FRAME_PROCESSOR_SKIP`: Process every Nth frame (default: 1)
- `MODEL_NAME`: SAM2 model variant (default: sam2_tiny)
- `TARGET_FPS`: Target processing FPS (default: 15)
- `PROCESSING_RESOLUTION`: Max resolution (default: 720)

## Monitoring

### Rerun Viewer
Open http://localhost:9876 to see:
- Live video stream
- Object tracking visualization
- Performance statistics

### Prometheus Metrics
```bash
curl http://localhost:8003/metrics | grep frame_processor
```

### Logs
```bash
# Stream logs
docker logs -f worldsystem-frame_processor_websocket

# Filter for specific info
docker logs worldsystem-frame_processor_websocket 2>&1 | grep -E "(Connected|Processed|FPS)"
```

## Differences from RabbitMQ Version

| Feature | RabbitMQ Version | WebSocket Version |
|---------|-----------------|-------------------|
| Video Input | RabbitMQ queue | Direct WebSocket |
| Latency | Higher (queue overhead) | Lower (direct stream) |
| Buffering | Automatic (RabbitMQ) | Minimal |
| Scalability | Better (multiple consumers) | Single connection |
| Complexity | Higher | Lower |

## Troubleshooting

### No Video in Rerun
1. Check server is running: `docker ps | grep server`
2. Verify WebSocket connection: Check logs for "Connected to WebSocket"
3. Ensure video is being sent from app/simulator

### High GPU Memory Usage
1. Use smaller model: Set `MODEL_NAME=sam2_tiny`
2. Reduce resolution: Set `PROCESSING_RESOLUTION=480`
3. Increase frame skip: Set `FRAME_PROCESSOR_SKIP=2`

### Connection Issues
1. Check server endpoint exists: `curl http://localhost:5001/health`
2. Verify network: Services must be on same Docker network
3. Check firewall/ports: Port 5001 must be accessible

## Development

### Running Locally (without Docker)
```bash
cd frame_processor
python websocket_frame_processor.py
```

### Adding Features
The WebSocket frame processor inherits from `WebSocketVideoConsumer` base class:
- Override `process_frame()` to handle decoded frames
- Use `self.logger` for consistent logging
- Frame skip is handled automatically by base class

## Next Steps

To fully integrate WebSocket streaming:
1. Update all services to use WebSocket instead of RabbitMQ for video
2. Add Grounded-SAM-2 for open-vocabulary detection
3. Implement adaptive quality based on network conditions
4. Add WebRTC support for lower latency