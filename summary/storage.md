# Storage - WebSocket Data Persistence Service

## Overview

The storage service is a Python-based microservice responsible for persisting real-time video streams from the drone to disk. It implements an efficient chunking system that saves video in time-based segments, handles idle timeouts gracefully, and provides robust recording capabilities with minimal frame loss. The service consumes H.264 video streams via WebSocket and stores them as video files for later processing or archival.

## What This Service Does

### Core Functionality
- **Real-time Video Recording**: Captures and saves H.264 video streams to disk
- **Time-based Chunking**: Splits recordings into manageable chunks (default 60 seconds)
- **Idle Timeout Handling**: Automatically saves partial chunks when stream pauses
- **Dynamic FPS Support**: Adapts to varying frame rates from the source
- **Metadata Tracking**: Maintains detailed session and chunk information
- **Zero Frame Loss**: Records every frame without skipping for complete data capture

### Key Features
- Configurable chunk duration and idle timeout
- Support for both H.264 and MJPEG codecs
- Atomic metadata updates for crash recovery
- Prometheus metrics for monitoring
- RabbitMQ event publishing for storage notifications
- Efficient frame queuing with backpressure handling

## System Architecture Connection

### Data Flow
```
Server (WebSocket)
    ↓ ws://server:5001/ws/video/consume
Storage Service
    ↓ Video Files
Disk Storage (/app/recordings)
    └── session_id/
        ├── metadata.json
        └── video_segments/
            ├── chunk_0001.mp4
            ├── chunk_0002.mp4
            └── ...
```

### Communication Protocol
- **Input**: H.264 video frames via WebSocket
- **Output**: 
  - Video files on disk
  - Storage events via RabbitMQ
  - Prometheus metrics

### Integration Points
1. **WebSocket Consumer**: Connects to server's video stream endpoint
2. **File System**: Writes to configurable storage path
3. **RabbitMQ Publisher**: Sends storage events to `storage_events_exchange`
4. **Prometheus**: Exposes metrics on port 8005

## Development History

### Evolution Timeline

1. **Initial Implementation** (Commit 54521ff - "Refactor data storage service")
   - Basic storage functionality
   - Simple file writing

2. **Monitoring Addition** (Commit fdabc5c)
   - Added Prometheus metrics
   - Performance tracking

3. **Timestamp Enhancement** (Commit e99d473, cdf4218)
   - Improved timestamp handling
   - Multi-format support
   - NTP synchronization awareness

4. **H.264 Streaming Support** (Commit 3ba7b09)
   - Major refactor for H.264 video
   - WebSocket integration
   - Chunk-based storage system
   - Idle timeout implementation

### Design Decisions
- **Chunking Strategy**: Time-based chunks for manageable file sizes
- **Idle Handling**: Automatic partial chunk saving to prevent data loss
- **Codec Flexibility**: Support for both H.264 (compression) and MJPEG (compatibility)
- **Metadata Persistence**: JSON metadata for session reconstruction

## Technical Details

### Technology Stack
- **Language**: Python 3.11+
- **Video Processing**: OpenCV (cv2)
- **Async Framework**: asyncio with aio_pika
- **WebSocket**: Custom WebSocketVideoConsumer
- **Metrics**: Prometheus client
- **Container**: Docker with Python base image

### Project Structure
```
storage/
├── websocket_storage.py    # Main storage service
├── Dockerfile             # Container configuration
└── (uses common/websocket_video_consumer.py)
```

### Key Components

1. **WebSocketStorageProcessor**
   - Extends WebSocketVideoConsumer
   - Manages video writing lifecycle
   - Handles chunking logic
   - Implements idle timeout

2. **Chunk Management**
   - Time-based segmentation
   - Configurable duration
   - Automatic rollover
   - Partial chunk support

3. **Metadata System**
   ```json
   {
     "session_id": "20240114_123456",
     "start_time": 1705239296.123,
     "chunks": [
       {
         "index": 1,
         "filename": "chunk_0001.mp4",
         "duration": 60.5,
         "frame_count": 1815,
         "size_bytes": 12345678,
         "partial": false
       }
     ],
     "total_frames": 5400,
     "estimated_fps": 29.97
   }
   ```

4. **Idle Timeout Mechanism**
   - Configurable timeout period
   - Minimum frame threshold
   - Automatic partial chunk saving
   - Timer-based implementation

### Configuration
- **Environment Variables**:
  - `STORAGE_PATH`: Base directory for recordings
  - `VIDEO_CHUNK_DURATION_SECONDS`: Chunk size (default: 60)
  - `VIDEO_IDLE_TIMEOUT_SECONDS`: Idle timeout (default: 10.0)
  - `MIN_CHUNK_FRAMES`: Minimum frames to save (default: 30)
  - `USE_H264_CODEC`: Enable H.264 codec (default: false)
  - `VIDEO_STREAM_URL`: WebSocket source URL
  - `METRICS_PORT`: Prometheus metrics port

### Performance Characteristics
- **Frame Processing**: Zero-copy from WebSocket to disk
- **Memory Usage**: Bounded by frame queue size
- **Disk I/O**: Sequential writes for optimal performance
- **CPU Usage**: Minimal (mainly video encoding)

## Features and Capabilities

### Storage Features
1. **Session Management**
   - Unique session IDs with timestamps
   - Organized directory structure
   - Atomic metadata updates

2. **Video Chunking**
   - Configurable chunk duration
   - Seamless chunk transitions
   - No frame loss between chunks

3. **Reliability**
   - Idle timeout handling
   - Graceful shutdown with final chunk saving
   - Crash recovery via metadata

4. **Monitoring**
   - Real-time metrics via Prometheus
   - Storage event notifications
   - Progress logging

### Metrics Exposed
- `storage_video_chunks_saved_total`: Total chunks saved
- `storage_video_frames_written_total`: Total frames written
- `storage_video_chunk_duration_seconds`: Chunk duration histogram
- `storage_current_chunk_size_bytes`: Current chunk size
- `storage_actual_fps`: Detected frame rate
- `storage_queue_size`: Frame queue depth
- `storage_partial_chunks_saved_total`: Partial chunks due to idle

## Challenges and Solutions

1. **Variable Frame Rates**
   - Challenge: Source FPS varies with network conditions
   - Solution: Dynamic FPS calculation with smoothing

2. **Stream Interruptions**
   - Challenge: Network drops cause incomplete recordings
   - Solution: Idle timeout with partial chunk saving

3. **Large File Sizes**
   - Challenge: Hours of recording create unwieldy files
   - Solution: Time-based chunking system

4. **Crash Recovery**
   - Challenge: Service crashes could lose recording state
   - Solution: Atomic metadata updates, session reconstruction

5. **Backpressure Handling**
   - Challenge: Disk I/O slower than incoming frames
   - Solution: Frame queuing with monitoring

## Storage Organization

### Directory Structure
```
/app/recordings/
└── 20240114_123456/              # Session ID
    ├── metadata.json              # Session metadata
    └── video_segments/
        ├── chunk_0001.mp4         # 60-second chunk
        ├── chunk_0002.mp4
        └── chunk_0003.mp4         # Partial chunk (idle timeout)
```

### Metadata Schema
- Session-level: ID, timestamps, codec, statistics
- Chunk-level: Index, duration, frame count, size
- Recovery info: Partial chunks, last update time

## Future Enhancements
- Cloud storage integration (S3, GCS)
- Real-time compression optimization
- Multi-stream recording support
- Thumbnail generation
- Automatic old session cleanup
- Stream health monitoring
- Adaptive bitrate recording
- Integration with video processing pipeline