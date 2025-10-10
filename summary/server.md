# Server - FastAPI Hub with WebSocket/RabbitMQ Orchestration

## Overview

The server is the central nervous system of the WorldSystem project, implemented as a FastAPI application that orchestrates all real-time communication between services. It provides WebSocket endpoints for drone/phone connections, video streaming, and web viewer communication, while using RabbitMQ for asynchronous message distribution to processing services. The server handles video streaming, IMU data, control commands, and system coordination.

## What This Service Does

### Core Functionality
- **WebSocket Management**: Maintains persistent connections with Android app, website, and consumers
- **Video Stream Routing**: Receives H.264 video and broadcasts to multiple consumers
- **Message Distribution**: Publishes data to RabbitMQ exchanges for service consumption
- **Time Synchronization**: NTP client for consistent timestamps across the system
- **Control Relay**: Forwards control commands from website to drone
- **Health Monitoring**: Provides health check and metrics endpoints

### Key Features
- Dual WebSocket architecture for producers and consumers
- Weak reference management for automatic cleanup
- H.264 packet parsing and routing
- IMU data buffering and deduplication
- Prometheus metrics and OpenTelemetry tracing
- File upload handling for CSV data
- Real-time bandwidth monitoring

## System Architecture Connection

### Data Flow
```
Android App                    Website
    ↓ WebSocket                 ↓ WebSocket
    Server (FastAPI)
    ├── Video Broadcasting (WebSocket)
    │   └── Frame Processor, Storage
    └── RabbitMQ Publishing
        └── All Processing Services
```

### Communication Protocols

#### WebSocket Endpoints
1. **`/ws/phone`**: Android app connection
   - Receives: Video frames, IMU data, status
   - Sends: Control commands from website

2. **`/ws/video`**: Video stream producer
   - Receives: H.264 packets from Android
   - Broadcasts to consumers

3. **`/ws/video/consume`**: Video stream consumer
   - Sends: H.264 frames to services
   - Used by: frame_processor, storage

4. **`/ws/viewer`**: Website connection
   - Receives: Control commands
   - Sends: Processed data, trajectories

### RabbitMQ Exchanges
All exchanges use fanout type for one-to-many distribution:
- `video_frames_exchange`: Decoded JPEG frames
- `video_stream_exchange`: Raw H.264 streams
- `imu_data_exchange`: Sensor data
- `processed_frames_exchange`: Analysis results
- `trajectory_data_exchange`: SLAM poses
- `restart_exchange`: System restart signals
- `ply_fanout_exchange`: 3D point clouds

## Development History

### Evolution Timeline

1. **Initial Architecture**
   - Basic WebSocket server
   - Simple message routing
   - Direct phone-to-viewer communication

2. **RabbitMQ Integration**
   - Added asynchronous message distribution
   - Fanout exchanges for scalability
   - Service decoupling

3. **Video Streaming Enhancement**
   - H.264 WebSocket broadcasting
   - Consumer management with weak references
   - SPS/PPS caching for new consumers

4. **Time Synchronization**
   - NTP client implementation
   - Multiple server fallback
   - Nanosecond precision timestamps

5. **Performance Optimization**
   - IMU buffering and deduplication
   - Bandwidth monitoring
   - Prometheus metrics

### Major Features Added
- WebSocket video broadcasting for multiple consumers
- Automatic consumer cleanup on disconnect
- Frame latency tracking
- Health check endpoints
- OpenTelemetry tracing
- File upload support

## Technical Details

### Technology Stack
- **Framework**: FastAPI with async/await
- **WebSocket**: Native FastAPI WebSocket support
- **Message Queue**: aio_pika (async RabbitMQ client)
- **Monitoring**: Prometheus, OpenTelemetry
- **Time Sync**: ntplib
- **Container**: Docker with Python 3.11

### Project Structure
```
server/
├── main.py                  # Main FastAPI application
├── src/
│   ├── api/
│   │   └── routes.py       # REST API endpoints
│   ├── config/
│   │   └── settings.py     # Configuration management
│   └── core/
│       ├── analyzers/      # Frame analysis modules
│       ├── h264_handler_pyav.py
│       └── model_loader.py
├── models/                 # 3D model files (OBJ/MTL)
└── uploads/               # Uploaded files storage
```

### Key Components

1. **WebSocketVideoManager**
   - Manages video consumer connections
   - Broadcasts H.264 frames efficiently
   - Caches SPS/PPS for new consumers
   - Uses weak references for auto-cleanup

2. **NTP Time Synchronization**
   - Multiple server fallback
   - 60-second sync interval
   - Nanosecond precision
   - Offset tracking

3. **IMU Data Processing**
   - Buffering for out-of-order packets
   - Deduplication by timestamp
   - Periodic flush (50ms)
   - Coordinate transformation

4. **RabbitMQ Publishing**
   - Robust connection with retry
   - Fanout exchanges for broadcast
   - JSON and binary message support
   - Metadata in headers

### Configuration
- **Environment Variables**:
  - `RABBITMQ_URL`: Message queue connection
  - `API_PORT`: Server port (default: 5001)
  - `BIND_HOST`: Bind address (0.0.0.0)
  - `NTP_SERVER`: Custom NTP server
  - Exchange names for each data type

### Performance Characteristics
- **WebSocket Connections**: Handles 10+ simultaneous
- **Video Throughput**: 10+ Mbps per stream
- **Message Rate**: 1000+ msgs/sec
- **Latency**: <10ms internal routing
- **Memory**: Bounded by connection count

## Features and Capabilities

### Connection Management
- Automatic reconnection handling
- Connection state tracking
- Graceful disconnect handling
- Resource cleanup on shutdown

### Data Routing
1. **Video Streams**
   - H.264 packet parsing
   - Multi-consumer broadcasting
   - Bandwidth monitoring
   - Frame counting

2. **IMU Data**
   - Timestamp deduplication
   - Buffer management
   - Coordinate transformation
   - Rate monitoring

3. **Control Commands**
   - Bidirectional relay
   - Command types: movement, camera, flight mode
   - JSON message validation

### Monitoring and Health
- `/health`: Basic health check
- `/health/video`: Video streaming status
- `/video/status`: Consumer statistics
- `/metrics`: Prometheus metrics
- `/h264/test`: Codec availability

### File Handling
- Chunked file upload support
- CSV data storage
- Progress tracking
- Atomic file operations

## Challenges and Solutions

1. **Connection Management**
   - Challenge: Memory leaks from disconnected clients
   - Solution: Weak references for automatic cleanup

2. **Video Broadcasting**
   - Challenge: Efficient multi-consumer streaming
   - Solution: Single parse, multiple send architecture

3. **Time Synchronization**
   - Challenge: Consistent timestamps across services
   - Solution: NTP with multiple server fallback

4. **Message Ordering**
   - Challenge: Out-of-order IMU packets
   - Solution: Buffering with periodic sorted flush

5. **Scalability**
   - Challenge: Growing number of services
   - Solution: RabbitMQ fanout for decoupling

## API Endpoints

### WebSocket Endpoints
- `/ws/phone`: Android app connection
- `/ws/video`: Video producer
- `/ws/video/consume`: Video consumer
- `/ws/viewer`: Website connection

### REST Endpoints
- `/health`: Service health
- `/health/video`: Video system health
- `/video/status`: Streaming statistics
- `/metrics`: Prometheus metrics
- `/h264/test`: Decoder test
- `/api/*`: Additional REST APIs

## Message Formats

### Video Frame (Binary)
```
[8 bytes] Timestamp (nanoseconds)
[N bytes] JPEG image data
```

### H.264 Packet (Binary)
```
[4 bytes] Total size
[1 byte]  Packet type
[8 bytes] Timestamp
[4 bytes] Data size
[4 bytes] Flags
[N bytes] H.264 data
```

### IMU Data (JSON)
```json
{
  "type": "imu_data",
  "timestamp": 1234567890123456789,
  "accelerometer": {"x": 0.1, "y": 0.2, "z": 9.8},
  "gyroscope": {"x": 0.01, "y": 0.02, "z": 0.03}
}
```

## Future Enhancements
- WebRTC support for lower latency
- Horizontal scaling with multiple instances
- Redis for shared state
- GraphQL subscriptions
- Authentication and authorization
- Rate limiting
- Connection pooling
- Message persistence