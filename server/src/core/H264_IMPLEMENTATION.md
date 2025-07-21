# H.264 Video Streaming Implementation

## Overview

The server now supports H.264 video streaming on the `/ws/video` WebSocket endpoint. This replaces the previous JPEG image streaming approach.

## Current Implementation

The server uses **PyAV's CodecContext.parse()** method (`h264_handler_pyav.py`) for H.264 decoding. This approach was chosen after testing multiple solutions because:

1. It properly handles raw H.264 streams from real devices
2. It supports standard Annex B format with NAL units
3. It's the recommended approach in PyAV documentation
4. It successfully decoded 5000+ frames in testing

## How It Works

1. **Client connects** to `/ws/video` WebSocket endpoint
2. **Client sends** raw H.264 stream data in binary messages
3. **Server parses** the stream using `codec_context.parse(data)`
4. **Server decodes** packets into frames
5. **Server converts** frames to JPEG and publishes to RabbitMQ

## Key Features

- Thread-safe stream management
- Multi-threaded decoding for performance
- Automatic cleanup on disconnect
- Progress logging and statistics
- Support for multiple concurrent streams

## Usage Example

```python
# Client sends raw H.264 data
websocket.send_bytes(h264_data)

# Server automatically handles:
# - NAL unit parsing
# - Frame decoding
# - JPEG conversion
# - RabbitMQ publishing
```

## Statistics Endpoint

GET `/h264/stats` returns current streaming statistics:
- Active streams
- Bytes received
- Frames decoded
- Errors per stream

## Configuration

The server uses these environment variables:
- `SAVE_SIMULATION_DATA`: Whether to save simulated video data
- `SAVE_INDIVIDUAL_FRAMES`: Whether to save frames as images (legacy)

## Archived Implementations

Previous H.264 handler attempts have been moved to `archive_h264_handlers/`:
- `h264_handler.py`: Initial custom NAL unit parser
- `h264_handler_simple.py`: Buffer-based approach
- `h264_handler_file.py`: Temporary file approach
- `h264_handler_ffmpeg.py`: FFmpeg subprocess approach

These were archived because PyAV's parse() method proved most reliable.