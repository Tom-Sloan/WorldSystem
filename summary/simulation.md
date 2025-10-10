# Simulation - Video Streaming Test Utilities

## Overview

The simulation service is a Python-based testing and development tool that simulates drone video streaming by reading pre-recorded video files and sending them to the server via WebSocket. It replicates the exact H.264 streaming protocol used by the Android app, making it invaluable for testing the entire processing pipeline without requiring a physical drone or the Android application.

## What This Service Does

### Core Functionality
- **H.264 Stream Simulation**: Converts video files to H.264 streams matching Android app format
- **WebSocket Streaming**: Sends video data using the same protocol as the real drone app
- **Real-time Playback**: Maintains proper frame timing to simulate live streaming
- **Multiple Modes**: Supports single video or sequential segment streaming
- **Protocol Compliance**: Implements Android packet format with proper NAL unit handling

### Key Features
- Automatic H.264 extraction or re-encoding
- Frame rate detection and timing control
- SPS/PPS parameter set handling
- Keyframe detection and proper packet typing
- Progress tracking and bandwidth monitoring
- Configurable streaming modes

## System Architecture Connection

### Data Flow
```
Test Video Files
    ↓ FFmpeg (H.264 extraction)
Simulation Service
    ↓ WebSocket (Android packet format)
Server (/ws/video endpoint)
    ↓ 
Processing Pipeline (same as real drone)
```

### Communication Protocol
- **Output**: H.264 packets via WebSocket
- **Format**: Android-compatible packet structure
- **Endpoint**: `ws://server:5001/ws/video`
- **Packet Types**:
  - SPS (0x01): Sequence Parameter Set
  - PPS (0x02): Picture Parameter Set
  - Keyframe (0x03): IDR frames
  - Frame (0x04): Non-IDR frames
  - Config (0x05): Configuration data

### Integration Points
1. **WebSocket Client**: Connects to server's video endpoint
2. **FFmpeg**: Video processing and H.264 extraction
3. **Test Data**: Pre-recorded videos and segments
4. **Docker Network**: Runs in host network mode

## Development History

### Evolution Timeline

1. **Initial Creation** (Commit 7b59a93)
   - Basic simulation framework
   - Simple data streaming

2. **H.264 Implementation** (Commit 3ba7b09)
   - Major refactor for H.264 support
   - WebSocket endpoint integration
   - Segment-based streaming

3. **Protocol Enhancement** (Commit 52fcd0d)
   - Android packet format implementation
   - NAL unit parsing
   - Proper timing control

4. **Stabilization** (Commit 06a8db5)
   - Cleaned up and fixed simulation
   - Improved error handling
   - Better logging

### Design Philosophy
- **Accurate Simulation**: Replicate real drone behavior exactly
- **Easy Testing**: Simple to use for development and debugging
- **Flexible Modes**: Support various testing scenarios
- **Production Protocol**: Use actual Android packet format

## Technical Details

### Technology Stack
- **Language**: Python 3.11+
- **Video Processing**: FFmpeg (via subprocess)
- **WebSocket**: websockets library
- **Async Framework**: asyncio
- **Container**: Docker with FFmpeg

### Project Structure
```
simulation/
├── simulate_video_stream.py   # Main simulation script
├── test_stream.sh            # Testing helper script
├── Dockerfile               # Container configuration
└── test_data/              # Video files
    ├── test_video.mp4      # Single test video
    └── 20250617_211214_segments/  # Segment collection
        ├── 0000_segment_000.mp4
        ├── 0010_segment_001.mp4
        └── ...
```

### Key Components

1. **VideoStreamSimulator Class**
   - WebSocket connection management
   - H.264 stream extraction
   - NAL unit parsing
   - Packet creation and sending

2. **Android Packet Format**
   ```python
   # 21-byte header for frames
   [4 bytes] Total packet size (big-endian)
   [1 byte]  Packet type
   [8 bytes] Timestamp (microseconds)
   [4 bytes] Data size
   [4 bytes] Flags
   [N bytes] H.264 data
   
   # 13-byte header for config (SPS/PPS)
   [4 bytes] Total packet size
   [1 byte]  Packet type
   [8 bytes] Timestamp
   [N bytes] SPS/PPS data
   ```

3. **NAL Unit Processing**
   - Start code detection (0x00000001 or 0x000001)
   - NAL type extraction
   - Proper sequencing (SPS → PPS → IDR → P-frames)

4. **Timing Control**
   - FPS detection from video metadata
   - Frame duration calculation
   - Real-time playback simulation
   - Sleep-based rate limiting

### Configuration
- **Environment Variables**:
  - `SERVER_WS_URL`: WebSocket endpoint (default: ws://127.0.0.1:5001/ws/video)
  - `SIMULATION_MODE`: "video" or "segments"
  
### Streaming Modes

1. **Video Mode**
   - Streams single test_video.mp4
   - Good for basic testing
   - Continuous playback

2. **Segments Mode**
   - Streams all segment files sequentially
   - 2-second pause between segments
   - Simulates real drone recording patterns

## Features and Capabilities

### Video Processing
- Automatic codec detection
- H.264 extraction from containers
- Re-encoding for non-H.264 sources
- Annex B format conversion

### Streaming Features
- Accurate frame timing
- Bandwidth monitoring
- Progress tracking
- Error recovery
- Connection management

### Testing Capabilities
- Single video testing
- Multi-segment simulation
- Performance benchmarking
- Protocol verification
- Pipeline integration testing

## Usage Examples

### Basic Usage
```bash
# Stream single video
docker-compose run simulator

# Stream segments
SIMULATION_MODE=segments docker-compose run simulator
```

### Development Testing
```bash
# Test with custom video
cp my_test_video.mp4 simulation/test_data/test_video.mp4
docker-compose build simulator
docker-compose run simulator

# Monitor streaming
docker logs -f worldsystem-simulator-1
```

## Challenges and Solutions

1. **Protocol Accuracy**
   - Challenge: Match Android app's exact packet format
   - Solution: Reverse-engineered protocol, byte-perfect implementation

2. **Timing Accuracy**
   - Challenge: Simulate real-time streaming behavior
   - Solution: FPS detection, frame duration calculation, sleep-based control

3. **H.264 Compatibility**
   - Challenge: Various video formats and codecs
   - Solution: FFmpeg integration for universal conversion

4. **NAL Unit Handling**
   - Challenge: Proper H.264 stream parsing
   - Solution: Custom NAL unit parser with start code detection

5. **Connection Stability**
   - Challenge: WebSocket timeouts during processing
   - Solution: Increased timeouts, proper error handling

## Test Data

### Included Videos
- `test_video.mp4`: Default test video for basic testing
- Segments collection: 36 video segments from real drone flight
  - Each segment ~10 seconds
  - Sequential numbering
  - Real-world test case

### Adding Test Data
1. Place videos in `test_data/` directory
2. Ensure H.264 codec or compatible format
3. Update docker-compose volume mapping if needed

## Future Enhancements
- Multiple concurrent stream simulation
- Variable bitrate testing
- Network condition simulation
- Automated test scenarios
- Performance profiling
- Custom packet injection
- Error condition testing