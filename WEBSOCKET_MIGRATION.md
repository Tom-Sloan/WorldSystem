# WebSocket Video Streaming Migration

## Overview
We have successfully migrated from the complex RTSP/FFmpeg/MediaMTX architecture to a simpler WebSocket-based video streaming system.

## Architecture Changes

### Before (RTSP-based):
```
Android/Simulator → WebSocket → Server → FFmpeg → MediaMTX → RTSP → Consumers
                                  ↓
                              (complex subprocess management)
```

### After (WebSocket-based):
```
Android/Simulator → WebSocket → Server → WebSocket → Consumers
                         ↓         ↓          ↓
                    (custom headers) (strip headers) (pure H.264)
```

## Key Changes Made

### 1. Server (server/main.py)
- **Removed**: All RTSP/FFmpeg related code
  - `start_rtsp_server()`, `stop_rtsp_server()`, `check_ffmpeg()`
  - `rtsp_process`, `rtsp_queue`, `rtsp_thread` global variables
  - RTSP health check endpoints
- **Added**: WebSocket broadcasting functionality
  - `WebSocketVideoManager` class for managing video consumers
  - `/ws/video/consume` endpoint for services to consume H.264 stream
  - Broadcasting of stripped H.264 data to all connected consumers
  - New health endpoints: `/health/video` and `/video/status`

### 2. Common Base Class (common/websocket_video_consumer.py)
- **Created**: New base class for WebSocket video consumers
  - Handles WebSocket connection and reconnection
  - H.264 decoding using PyAV
  - Frame queuing and processing
  - OpenCV-compatible `WebSocketVideoCapture` class for drop-in replacement

### 3. Frame Processor (frame_processor/grounded_sam2_processor.py)
- **Changed**: Inherits from `WebSocketVideoConsumer` instead of `RTSPConsumer`
- **Updated**: Uses `VIDEO_STREAM_URL` environment variable for WebSocket URL

### 4. Storage Service (storage/rtsp_storage.py)
- **Changed**: Inherits from `WebSocketVideoConsumer` instead of `RTSPConsumer`
- **Updated**: Uses `VIDEO_STREAM_URL` environment variable for WebSocket URL
- **Disabled**: FFmpeg recording mode (now uses decoded frames directly)

### 5. Docker Configuration
- **docker-compose.yml**:
  - Removed MediaMTX service entirely
  - Updated environment variables for frame_processor and storage:
    - Added `VIDEO_STREAM_URL=ws://127.0.0.1:5001/ws/video/consume`
    - Removed RTSP_HOST and RTSP_PORT
  - Updated health checks to use new endpoints
- **storage/Dockerfile**:
  - Added dependencies: `websockets` and `av`

## Benefits of WebSocket Architecture

1. **Simplicity**: No FFmpeg subprocess management or MediaMTX configuration
2. **Reliability**: Direct WebSocket connection without intermediate processes
3. **Performance**: Lower latency, no RTSP protocol overhead
4. **Flexibility**: Easy to add new consumers without RTSP server limits
5. **Debugging**: Simpler to debug WebSocket connections vs RTSP

## Testing

### Test Scripts Created:
1. **test_websocket_streaming.py**: Tests consuming H.264 from WebSocket
2. **test_websocket_producer.py**: Simulates Android app sending H.264 with headers

### Running Tests:
```bash
# Start the server
docker-compose up server

# In another terminal, test consumer
python test_websocket_streaming.py

# Test producer (simulates Android)
python test_websocket_producer.py

# Start all services
docker-compose up
```

## Environment Variables

### For Consumers (frame_processor, storage, slam3r):
```yaml
VIDEO_STREAM_URL: ws://127.0.0.1:5001/ws/video/consume
VIDEO_STREAM_TYPE: websocket  # Optional, for future use
```

## Monitoring

Check video streaming status:
```bash
# Video streaming health
curl http://localhost:5001/health/video

# Detailed video status
curl http://localhost:5001/video/status
```

## Next Steps

1. Update SLAM3R service to use WebSocket (similar to frame_processor/storage)
2. Update monitoring dashboards to use new metrics:
   - `h264_consumers_connected`
   - `h264_frames_relayed_total`
   - `h264_bytes_relayed_total`
3. Test with real Android app and simulator
4. Consider adding WebSocket authentication/authorization if needed