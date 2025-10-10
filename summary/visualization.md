# Visualization - Rerun-based 3D Visualization Service

## Overview

The visualization service is a Python-based microservice that provides real-time 3D visualization of the WorldSystem's processing pipeline using Rerun, a multimodal data visualization tool. It consumes processed frames, segmentation results, and tracking data from RabbitMQ and creates rich, interactive visualizations for debugging and monitoring the system's performance.

## What This Service Does

### Core Functionality
- **Real-time Data Visualization**: Displays processed video frames with detected objects and segmentation masks
- **Object Tracking Visualization**: Shows object trajectories and tracking history over time
- **Multi-view Layouts**: Provides different viewing modes (video only, segmentation only, or both)
- **Performance Monitoring**: Tracks FPS, processing times, and detection statistics
- **Gallery Management**: Maintains a rolling gallery of enhanced object detections
- **Timeline Visualization**: Creates temporal views of object appearances and movements

### Key Features
- Rerun integration for powerful 3D/2D visualization
- Blueprint-based layout management
- Support for both JSON and binary (JPEG) message formats
- Automatic reconnection to RabbitMQ
- Efficient base64 image decoding
- Object-specific timeline tracking

## System Architecture Connection

### Data Flow
```
Frame Processor
    ↓ RabbitMQ (processed_frames_exchange)
Visualization Service
    ↓ Rerun (port 9876)
Rerun Viewer (Browser/Native)
```

### Communication Protocol
- **Input**: Processed frames via RabbitMQ fanout exchange
- **Output**: Rerun visualizations via gRPC connection
- **Message Types**:
  - `processed_frame`: Video frames with detections
  - `segmentation_result`: Object masks and classifications
  - `tracking_update`: Object tracking information
  - `analysis_result`: API analysis results

### Integration Points
1. **RabbitMQ Consumer**: Subscribes to `processed_frames_exchange`
2. **Rerun Connection**: gRPC connection to Rerun viewer on port 9876
3. **Message Formats**: Supports both JSON and binary JPEG with metadata

## Development History

### Evolution
The visualization service was developed as part of the architectural shift towards decoupled services:

1. **Initial Development** (Commit 78e0a88 - "updated visualization")
   - Basic Rerun integration
   - Simple frame logging

2. **Architecture Decoupling** (Commit 867a18c)
   - Separated visualization from SLAM3R processing
   - Improved performance by removing bottlenecks
   - Enhanced maintainability through service isolation

3. **Enhanced Functionality** (Commit 5d42284)
   - Added camera setup for better 3D visualization
   - Implemented debug logging capabilities
   - Test grid points for connection verification

4. **Performance Optimization** (Commit 9bc7c17)
   - Decoupled visualization from core processing
   - Reduced latency in the pipeline
   - Improved resource utilization

### Design Philosophy
- **Separation of Concerns**: Visualization is independent of processing
- **Performance**: Non-blocking visualization that doesn't slow down core pipeline
- **Flexibility**: Multiple view modes and layouts
- **Debugging**: Rich visual debugging capabilities

## Technical Details

### Technology Stack
- **Language**: Python 3.11+
- **Visualization**: Rerun SDK
- **Message Queue**: aio_pika (async RabbitMQ client)
- **Image Processing**: OpenCV, NumPy
- **Container**: Docker with Python base image

### Project Structure
```
visualization/
├── main.py                    # Service entry point
├── visualization_handler.py   # Message processing and Rerun logging
├── blueprint_manager.py       # Layout and view management
├── rerun_client.py           # Rerun connection utilities
├── enhanced_visualizer.py    # Advanced visualization features
├── requirements.txt          # Python dependencies
└── Dockerfile               # Container configuration
```

### Key Components

1. **VisualizationService** (`main.py`)
   - RabbitMQ connection management
   - Message consumption and routing
   - Service lifecycle management

2. **VisualizationHandler** (`visualization_handler.py`)
   - Base64 image decoding
   - Mask extraction and processing
   - Rerun logging orchestration
   - FPS and statistics tracking

3. **BlueprintManager** (`blueprint_manager.py`)
   - View mode management (VIDEO_ONLY, SEGMENTATION_ONLY, BOTH)
   - Layout configuration
   - Dynamic blueprint updates

4. **Message Processing**
   - Supports both JSON and binary formats
   - Extracts metadata from message headers
   - Handles various data types (frames, masks, detections)

### Configuration
- **Environment Variables**:
  - `RABBITMQ_URL`: RabbitMQ connection string
  - `RERUN_PORT`: Rerun viewer port (default: 9876)
  - `FRAME_PROCESSOR_OUTPUTS_EXCHANGE`: Input exchange name
  - `LOG_LEVEL`: Logging verbosity

### Data Structures
```python
# Processed Frame Message
{
    'type': 'processed_frame',
    'frame': np.ndarray,  # RGB image
    'timestamp_ns': int,
    'frame_number': int,
    'processing_time_ms': float,
    'detection_count': int,
    'class_summary': {'person': 2, 'chair': 3, ...}
}

# Segmentation Result
{
    'type': 'segmentation_result',
    'masks': base64_string,
    'mask_info': [
        {
            'instance_id': int,
            'class_name': str,
            'confidence': float,
            'bbox': [x, y, w, h]
        }
    ]
}
```

## Features and Capabilities

### Visualization Types
1. **Video Stream**: Real-time processed video display
2. **Segmentation Masks**: Object masks with class labels
3. **Detection Boxes**: Bounding boxes with confidence scores
4. **Object Gallery**: Grid of detected objects
5. **Timeline View**: Temporal object tracking
6. **Statistics Panel**: FPS, detection counts, processing times

### View Modes
- **VIDEO_ONLY**: Just the processed video stream
- **SEGMENTATION_ONLY**: Only segmentation masks
- **BOTH**: Split view with video and segmentation

### Advanced Features
- Object-specific timelines
- Enhanced gallery with FIFO management
- Real-time statistics tracking
- Multi-format message support
- Efficient image decoding pipeline

## Challenges and Solutions

1. **Performance Impact**
   - Challenge: Visualization affecting processing performance
   - Solution: Decoupled architecture with async message handling

2. **Large Data Volumes**
   - Challenge: High-resolution frames at 30fps
   - Solution: Efficient base64 decoding, optional downsampling

3. **Connection Reliability**
   - Challenge: Maintaining stable connections to both RabbitMQ and Rerun
   - Solution: Robust reconnection logic, graceful degradation

4. **Message Format Flexibility**
   - Challenge: Supporting various message formats from different services
   - Solution: Content-type based routing, header metadata extraction

## Future Enhancements
- 3D point cloud visualization
- Multi-camera support
- Recording and playback capabilities
- Web-based Rerun viewer integration
- Custom visualization plugins
- Performance profiling overlays
- AR/VR visualization modes