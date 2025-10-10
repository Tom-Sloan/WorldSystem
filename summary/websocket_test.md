# WebSocketTest - Android Drone Control Application

## Overview

WebSocketTest is an Android application that serves as the primary data acquisition interface for the WorldSystem project. It connects to DJI drones, captures real-time video streams and sensor data, and transmits this information via WebSocket to the server for 3D reconstruction processing.

## What This Service Does

### Core Functionality
- **DJI Drone Integration**: Connects to DJI drones using the DJI Mobile SDK v5
- **Real-time Video Streaming**: Captures H.264 encoded video from drone camera at 30fps
- **Sensor Data Collection**: Gathers IMU data (accelerometer, gyroscope, magnetometer) and GPS coordinates
- **WebSocket Communication**: Streams video and sensor data to the server in real-time
- **Virtual Stick Control**: Provides manual flight control through on-screen joysticks
- **Time Synchronization**: Uses NTP (Network Time Protocol) to ensure accurate timestamps across the distributed system

### Key Features
- Dual WebSocket connections: one for control/status, one for video streaming
- Automatic reconnection with exponential backoff
- Message queuing for reliability when disconnected
- Frame skipping capabilities for bandwidth optimization
- Support for both text and binary WebSocket messages

## System Architecture Connection

### Data Flow
```
DJI Drone
    ↓ (USB/WiFi)
Android App (WebSocketTest)
    ↓ WebSocket (H.264 video + IMU)
Server (FastAPI)
    ↓ RabbitMQ
Processing Pipeline (SLAM3R, Frame Processor, etc.)
```

### Communication Protocols
- **Video Stream**: H.264 encoded video sent as binary WebSocket frames
- **Sensor Data**: JSON-formatted IMU and GPS data
- **Control Commands**: JSON messages for drone control (takeoff, land, movement)

### Integration Points
1. **Server Connection**: Default WebSocket server at `ws://134.117.167.139:5001`
2. **Video Endpoint**: `/ws/video` for H.264 streaming
3. **Control Endpoint**: `/ws/phone` for control and sensor data

## Development History

### Initial Development (Commit fff75f5 - "added android app")
- First introduction of the Android application to the WorldSystem project
- Basic DJI SDK integration and WebSocket communication
- Initial virtual stick control implementation

### Major Milestones

1. **WebSocket Integration** (Commit 1b55dd8)
   - Implemented dual WebSocket architecture
   - Added singleton pattern for connection management
   - Introduced message queuing for reliability

2. **H.264 Video Streaming** (Commit 52fcd0d)
   - Transitioned from frame-by-frame to H.264 streaming
   - Implemented YUV420 to H.264 encoding
   - Added frame skipping for performance optimization

3. **RTSP to WebSocket Transition** (Commit 415cb86)
   - Removed RTSP streaming in favor of WebSocket
   - Simplified communication architecture
   - Improved real-time performance

4. **Cleanup and Optimization** (Recent commits)
   - Removed unnecessary dependencies (YOLO, NKSR)
   - Streamlined codebase for production use
   - Enhanced connection stability

### Technical Evolution
- **Architecture**: Evolved from simple socket communication to robust WebSocket with auto-reconnection
- **Video Pipeline**: Progressed from JPEG frames → RTSP → H.264 WebSocket streaming
- **Control System**: Advanced from basic commands to full virtual stick control
- **Reliability**: Added NTP synchronization, message queuing, and connection state management

## Technical Details

### Technology Stack
- **Language**: Kotlin (primary) with Java compatibility
- **Min SDK**: 31 (Android 12)
- **Target SDK**: 34 (Android 14)
- **Architecture**: arm64-v8a only (DJI SDK requirement)
- **Key Dependencies**:
  - DJI Mobile SDK v5
  - OkHttp for WebSocket
  - Android MediaCodec for H.264 encoding

### Key Components

1. **WebSocket Management** (`websocket/` package)
   - `WebsocketContainer`: Singleton managing control WebSocket
   - `VideoWebsocketContainer`: Dedicated H.264 video streaming
   - `WebsocketMessageHandler`: Message processing and routing

2. **DJI Integration**
   - `DJIAircraftApplication`: Main application extending DJIApplication
   - `MSDKManagerVM`: SDK lifecycle management
   - `LiveStreamVM`: Video capture and encoding
   - `VirtualStickVM`: Flight control interface

3. **UI Components**
   - `FirstFragment`: Main control interface
   - `OnScreenJoystick`: Custom joystick for drone control
   - Real-time status displays

### Configuration
- **API Keys**: Stored in `gradle.properties`
  - `AIRCRAFT_API_KEY`: DJI SDK authentication
  - `GMAP_API_KEY`: Google Maps integration
  - `MAPLIBRE_TOKEN`: Map rendering

### Permissions Required
- Network and Internet access
- Storage (for temporary video buffering)
- Location services
- USB host/accessory (for drone connection)
- Audio recording (for drone audio if available)

## Challenges and Solutions

1. **Real-time Performance**
   - Challenge: Maintaining 30fps video streaming while processing sensor data
   - Solution: Separate WebSocket connections, frame skipping, efficient H.264 encoding

2. **Connection Reliability**
   - Challenge: Maintaining stable connection in varying network conditions
   - Solution: Auto-reconnection, message queuing, connection state management

3. **Time Synchronization**
   - Challenge: Ensuring synchronized timestamps across distributed system
   - Solution: NTP client implementation with offset calculation

4. **DJI SDK Limitations**
   - Challenge: arm64-v8a only support, complex initialization
   - Solution: Careful architecture configuration, robust error handling

## Future Considerations
- Migration to DJI SDK v6 when available
- WebRTC integration for lower latency
- Multi-drone support
- Enhanced telemetry data collection
- AR overlay capabilities in the app itself