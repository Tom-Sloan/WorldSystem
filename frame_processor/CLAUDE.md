# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Frame Processor Service Overview

The frame_processor is a GPU-accelerated video processing service that:
- Receives raw video frames from RabbitMQ
- Optionally applies YOLO object detection
- Publishes processed frames with annotations
- Provides real-time visualization via Rerun
- Exposes Prometheus metrics for monitoring
- Uses NTP time synchronization for accurate timestamps

## Architecture & Integration

This service is part of the WorldSystem microservices architecture:
- **Input**: Raw video frames from `sensor_data` exchange (routing key: `sensor.video`)
- **Output**: Processed frames to `processing_results` exchange (routing key: `result.frames.yolo`)
- **Control**: Analysis mode updates from `control_commands` exchange (routing key: `control.analysis_mode`)
- **Monitoring**: Prometheus metrics on port 8003
- **Visualization**: Rerun viewer integration on port 9090

## Key Dependencies

- **ultralytics**: YOLOv8 object detection model
- **opencv-python-headless**: Image processing
- **pika**: RabbitMQ client
- **rerun-sdk==0.23.2**: Real-time visualization
- **prometheus_client**: Metrics collection
- **ntplib**: NTP time synchronization
- **NVIDIA CUDA**: GPU acceleration (requires nvidia-docker)

## Development Commands

```bash
# Build the service
docker-compose build frame_processor

# Run with the full stack
docker-compose up

# View logs
docker logs -f worldsystem-frame_processor-1

# Access metrics
curl http://localhost:8003/metrics

# Connect to Rerun viewer
# The service connects to rerun+http://localhost:9876/proxy
```

## Configuration (Environment Variables)

- `RABBITMQ_URL`: RabbitMQ connection URL (default: "amqp://rabbitmq")
- `VIDEO_FRAMES_EXCHANGE`: Input exchange name
- `PROCESSED_FRAMES_EXCHANGE`: Output exchange name  
- `ANALYSIS_MODE_EXCHANGE`: Control exchange name
- `INITIAL_ANALYSIS_MODE`: Starting mode ("none" or "yolo")
- `RERUN_ENABLED`: Enable Rerun visualization ("true"/"false")
- `RERUN_VIEWER_ADDRESS`: Rerun viewer address
- `RERUN_CONNECT_URL`: Rerun gRPC connection URL
- `NTP_SERVER`: NTP server for time sync (default: "pool.ntp.org")
- `METRICS_PORT`: Prometheus metrics port (default: 8003)

## Code Architecture

The service follows a single-file architecture (`frame_processor.py`) with:
1. **Metrics Setup**: Prometheus counters, gauges, and histograms
2. **NTP Synchronization**: Time offset tracking and periodic sync
3. **Rerun Initialization**: Visualization pipeline setup
4. **RabbitMQ Connection**: Exchange/queue declarations and bindings
5. **YOLO Model Loading**: GPU-aware model initialization
6. **Message Callbacks**: Frame processing and mode switching handlers

Key functions:
- `sync_ntp_time()`: Updates NTP time offset
- `get_ntp_time_ns()`: Returns NTP-synchronized time in nanoseconds
- `frame_callback()`: Main processing pipeline for video frames
- `analysis_mode_callback()`: Handles dynamic mode switching

## Message Flow

1. Raw frames arrive with headers containing:
   - `timestamp_ns`: Original capture timestamp
   - `server_received`: Server receipt timestamp
   - `ntp_time`: NTP-synchronized timestamp
   - Resolution and dimension metadata

2. Processing includes:
   - Optional YOLO detection (if mode="yolo")
   - Frame annotation with bounding boxes
   - Rerun logging with synchronized timestamps
   - Prometheus metric updates

3. Output frames include all original metadata plus:
   - `processing_time_ms`: Processing duration
   - `ntp_offset`: Current NTP offset
   - Updated dimensions if resized

## Monitoring & Debugging

Prometheus metrics available:
- `frame_processor_frames_processed_total`: Frame count
- `frame_processor_yolo_detections_total`: Detection count
- `frame_processor_processing_time_ms`: Processing latency
- `frame_processor_frame_size_bytes`: Output frame size
- `frame_processor_connection_status`: RabbitMQ connection
- `frame_processor_ntp_time_offset_seconds`: Time sync offset

## Performance Considerations

- GPU acceleration is automatically detected and used when available
- JPEG compression quality is set to 85 for balance of quality/size
- Heartbeat timeout is set to 3600s for long-running connections
- NTP sync occurs every 60 seconds in a background thread
- Rerun data is compressed for efficient transmission

## Testing & Development

When modifying the frame processor:
1. Ensure GPU support if testing YOLO mode
2. Monitor processing time metrics for performance regression
3. Verify NTP synchronization is working correctly
4. Check Rerun viewer for visual debugging
5. Validate message headers are preserved through pipeline