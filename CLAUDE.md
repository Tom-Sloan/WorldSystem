# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. Don't build docker containers unless asked. 

## Project Overview

WorldSystem is a real-time 3D reconstruction and visualization system for drone-based room mapping. It combines SLAM, neural reconstruction, and real-time visualization to create 3D models of indoor spaces with the goal of overlaying fantasy elements in augmented reality.

## High-Level Architecture

The system follows a microservices architecture with three primary communication patterns:

1. **WebSocket Streaming**: Real-time video/IMU data from Android to server, and server to website
2. **RabbitMQ Message Bus**: Asynchronous fanout distribution to processing services
3. **Shared Memory IPC**: Zero-copy data transfer between SLAM3R and Mesh Service

### Data Flow Pipeline

```
Android App (30fps video + IMU)
    ↓ WebSocket
Server (FastAPI hub)
    ↓ RabbitMQ Fanout
┌─────────────┬──────────────┬───────────────┐
Frame Processor    SLAM3R       Storage Service
(SAM2/YOLO)    (Pose Est.)     (Disk persist)
    ↓              ↓                 
RabbitMQ    Shared Memory
    ↓              ↓
Website ← ─ ─ Mesh Service → Rerun
         WebSocket  (TSDF+MC)
```

### Key Architectural Patterns

**Server (Python/FastAPI)**:
- Async/await throughout with aio_pika for RabbitMQ
- WeakSet consumer management for automatic cleanup
- NTP time synchronization across distributed services
- Multiple WebSocket endpoints: `/ws/phone`, `/ws/viewer`, `/ws/video`

**Mesh Service (C++/CUDA)**:
- Shared memory reader using POD structs for C++ compatibility
- GPU-accelerated TSDF fusion and Marching Cubes
- Adaptive quality based on camera velocity
- Memory pool allocation with 1GB default

**Frontend (React/Three.js)**:
- WebSocketContext for pub/sub message distribution
- React Three Fiber for 3D visualization
- WebXR support for AR/VR experiences
- Tab-based lazy loading for performance

**Failure Handling**:
- RabbitMQ auto-reconnection with exponential backoff
- Graceful degradation when optional services unavailable
- Message acknowledgment only after successful processing
- Weak references for WebSocket consumers to prevent memory leaks

## Build and Development Commands

### Quick Start

```bash
./start.sh  # Generates .env, rebuilds, and starts all services
```

### Docker Commands

```bash
# Full system
docker-compose build
docker-compose up

# Selective deployment
docker-compose --profile slam3r up
docker-compose --profile frame_processor up

# Run without specific service
docker compose up --detach $(docker compose config --services | grep -v slam3r)

# Clean rebuild
docker-compose down --remove-orphans --volumes --rmi local
docker-compose build --no-cache

# Access service shells
docker-compose exec website sh
docker-compose exec server bash

# View filtered logs
docker logs worldsystem-frame_processor-1 2>&1 | grep -E "(FPS|tracks)"
```

### Service-Specific Development

**Website (React)**:
```bash
cd website
npm run dev      # Development server (port 3001)
npm run build    # Production build
npm run lint     # ESLint checking
npm run preview  # Preview production build
```

**Mesh Service (C++/CUDA)**:
```bash
cd mesh_service
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)                    # Parallel build
make VERBOSE=1                     # Verbose output for debugging
./test_build.sh                    # Build verification
python3 ../test_mesh_service.py    # Integration test
```

**Frame Processor**:
```bash
cd frame_processor
./install_models.sh               # Download ML models
./test_integration.sh             # Run integration tests
# Enable rich terminal UI
ENABLE_RICH_TERMINAL=true docker-compose run frame_processor
```

**SLAM3R**:
```bash
cd slam3r/SLAM3R_engine/scripts
./demo_wild.sh                    # Run with sample data
./demo_vis_wild.sh               # Run with visualization
```

### Testing Commands

```bash
# Full test suite
python run_tests.py

# Pipeline testing
./test_full_pipeline.sh                              # End-to-end WebSocket test
cd simulation && ./test_stream.sh                    # H.264 streaming test

# Integration tests
python tests/test_slam3r_mesh_integration.py         # SLAM→Mesh integration
python tests/integration/test_full_integration.py    # Full system test
python tests/check_shared_memory.py                  # Shared memory status

# Component tests
cd website && npm run lint
cd mesh_service && python3 test_mesh_service.py
cd frame_processor && ./test_integration.sh

# WebSocket testing
python test_websocket_producer.py
python test_websocket_streaming.py

# Monitoring endpoints
curl http://localhost:5001/health/video             # Video health check
curl http://localhost:5001/video/status | jq        # Streaming status (JSON)
```

### Monitoring URLs (when running)

- RabbitMQ Management: http://localhost:15672 (guest/guest)
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- Jaeger Tracing: http://localhost:16686
- Rerun Viewer: http://localhost:9876
- Website: http://localhost:3001

## Key Technical Details

### RabbitMQ Exchanges

All exchanges use fanout type for one-to-many distribution:
- `video_frames_exchange` - Decoded video frames with metadata
- `imu_data_exchange` - IMU sensor data (accel, gyro, magnetometer)
- `slam3r_keyframe_exchange` - Camera poses and point clouds
- `processed_frames_exchange` - Video frames with detections/segmentation
- `scene_scaling_exchange` - Real-world scale information
- `restart_exchange` - System-wide restart notifications

### WebSocket Message Format

```json
{
  "type": "video_frame|imu_data|slam_update|mesh_update",
  "timestamp": 1234567890123,  // Unix timestamp in ms
  "data": { /* type-specific payload */ }
}
```

### Shared Memory Structure

Located at `/dev/shm/slam3r_keyframes`, uses POD struct:
```cpp
struct SharedKeyframe {
    uint64_t timestamp_ns;
    uint32_t point_count;
    uint32_t color_channels;
    float pose_matrix[16];    // Row-major 4x4
    float bbox[6];            // min/max bounds
    // Variable arrays follow
};
```

### Service Configuration

All services configured via environment variables:
- **Server**: `SERVER_PORT`, `RABBITMQ_URL`, `ENABLE_NTP_SYNC`
- **Mesh Service**: `MESH_*` prefix (see mesh_service/CONFIG.md)
- **Frame Processor**: `DETECTOR_TYPE`, `USE_SERPAPI`, `USE_PERPLEXITY`
- **Website**: `VITE_WS_URL`, `VITE_API_URL`
- **Performance**: `STREAM_CLEANUP_TIMEOUT_SECONDS`, `SLAM3R_CONF_THRES_L2W`

### GPU Requirements

GPU-accelerated services (require CUDA):
- `slam3r` - Neural SLAM processing
- `mesh_service` - TSDF fusion and mesh generation
- `frame_processor` - SAM2/YOLO detection

### Message Queue Patterns

**Reliability**:
- Exclusive queues with service prefixes (e.g., `mesh_service_video_frames`)
- Prefetch count of 1 for backpressure control
- Message acknowledgment after successful processing
- Auto-reconnection with exponential backoff

**Caching Strategy**:
- Multi-level caching: memory → disk with fallback
- Cache location discovery with write tests
- MD5-based cache keys for API responses
- Automatic cache loading on startup

### Development Guidelines

**Code Style**:
- Python: PEP 8, type hints, docstrings for public APIs
- C++: Modern C++17, RAII, avoid raw pointers
- JavaScript: ESLint config, functional components, hooks

**Performance Considerations**:
- Shared memory for high-bandwidth data (>10MB/s)
- RabbitMQ for loosely-coupled async communication
- WebSockets for real-time bidirectional updates
- GPU memory management critical for ML services

**Error Handling**:
- Services should gracefully degrade (e.g., mesh without colors)
- Use structured logging with correlation IDs
- Implement retry logic with exponential backoff
- Monitor memory usage, especially GPU memory

**Debugging Tools**:
- Rich terminal UI for frame processor: `ENABLE_RICH_TERMINAL=true`
- Performance timers with microsecond precision
- Structured JSON logging to separate files
- OpenTelemetry tracing with Jaeger

## Modifiable Directories

When implementing features, modify code in:
- `server/`, `website/`, `storage/` - Core services
- `mesh_service/src/` - Mesh generation algorithms
- `frame_processor/` - Video processing pipeline
- `slam3r/` - SLAM processor files (not SLAM3R_engine)
- `simulation/`, `fantasy/` - Additional features
- `nginx/`, `docker/` - Infrastructure configs
- Configuration files at root level

Avoid modifying:
- `slam3r/SLAM3R_engine/` - Third-party SLAM implementation
- Generated files in `build/` directories
- Model files in `*/models/`