# H.264 Video Streaming Implementation

## Overview
The system now supports H.264 video streaming in addition to JPEG image streaming. This allows for more efficient bandwidth usage when streaming from phones/drones.

## Changes Made

### 1. Server (`/ws/video` endpoint)
- New WebSocket endpoint for H.264 video streams
- Uses PyAV library for H.264 decoding
- Converts decoded frames to JPEG for compatibility with existing pipeline
- Located in: `server/src/core/h264_handler_pyav.py`

### 2. Simulation
- New script: `simulation/simulate_video_stream.py`
- Extracts H.264 streams from video files
- Sends raw H.264 data over WebSocket

### 3. Storage
- New module: `storage/video_storage.py`
- Saves video segments (10-second MP4 files) instead of individual frames
- Controlled by environment variables:
  - `SAVE_SIMULATION_DATA`: Whether to save simulated data
  - `SAVE_INDIVIDUAL_FRAMES`: Whether to save frames as images (legacy mode)

## Usage

### Server
The server automatically detects whether incoming data is H.264 or JPEG:
- H.264 streams: Connect to `/ws/video`
- JPEG images: Connect to `/ws/phone` (legacy)

### Testing
```bash
# Test H.264 streaming with a video file
cd simulation
python simulate_video_stream.py path/to/video.mp4
```

### Configuration
In `docker-compose.yml`:
```yaml
storage:
  environment:
    SAVE_SIMULATION_DATA: "false"  # Don't save simulated data by default
    SAVE_INDIVIDUAL_FRAMES: "false"  # Use video storage instead of images
```

## Technical Details
- H.264 decoding uses PyAV's `CodecContext.parse()` method
- Supports standard Annex B format with NAL units
- Thread-safe stream management
- Automatic cleanup on disconnect