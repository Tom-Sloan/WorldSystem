# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

You are an expert programmer. Make sure to make clean, maintainable code.

Do not leave placeholders, TODOs, without specifically asking for permission.

Do not use magic numbers where possible. Minimize their use.

## Project Overview

WorldSystem is a real-time 3D reconstruction and visualization system for drone-based room mapping. It combines SLAM, neural reconstruction, and real-time visualization to create 3D models of indoor spaces with the goal of overlaying fantasy elements in augmented reality.

## Architecture

The system uses a microservices architecture with the following data flow:
1. **Android App** → **Server** (WebSocket): Streams video (30fps) and IMU data
2. **Server** → **RabbitMQ** → **Frame Processor/SLAM/Storage**: Distributes data
3. **SLAM3R** → **Shared Memory** → **Mesh Service**: Camera poses via zero-copy IPC
4. **Mesh Service** → **Rerun**: Real-time 3D visualization (TSDF + Marching Cubes)
5. **Server** → **Website** (WebSocket): Real-time updates and mesh display

Key services:
- **Server**: Central hub for data routing (FastAPI, Python)
- **SLAM3R**: Camera pose estimation (C++/Python bindings)
- **Mesh Service**: GPU-accelerated real-time mesh generation (CUDA C++)
- **Website**: Real-time 3D visualization (React/Three.js)
- **Frame Processor**: Video processing with YOLO detection
- **Fantasy Builder**: Adds game-like elements to 3D models (WIP)
- **Storage**: Persistent storage for images and IMU data

## Build and Development Commands

### Quick Start
```bash
./start.sh  # Builds and starts all services
```

### Docker Commands
- Full build: `docker-compose build`
- Single service: `docker-compose build --no-cache <service_name>`
- Run all services: `docker-compose up`
- Run with specific profile: `docker-compose --profile slam3r up`
- Run without specific service: `docker compose up --detach $(docker compose config --services | grep -v slam3r)`

### Development Commands
- Website: `cd website && npm run dev`
- Website build: `cd website && npm run build`

### Testing Commands
- Website: `cd website && npm run lint`
- Mesh Service: `cd mesh_service && python3 test_mesh_service.py`
- Integration tests: `cd tests && python test_*.py`

### Monitoring URLs (when running)
- RabbitMQ: http://localhost:15672
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- Jaeger: http://localhost:16686
- Rerun: http://localhost:9876

## Code Guidelines

When a file becomes too long, split it into smaller files. When a function becomes too long, split it into smaller functions.

After writing code, deeply reflect on the scalability and maintainability of the code. Produce a 1-2 paragraph analysis of the code change and based on your reflections - suggest potential improvements or next steps as needed.

### Style Guidelines
- **Python**: PEP 8, 4-space indentation, docstrings for all functions
- **JavaScript/React**: ESLint config, camelCase, alphabetical imports
- **Error handling**: try/except in Python, catch and log in JavaScript

### Modifiable Directories
Only modify code in:
- `slam3r/aaa/` - Custom algorithm implementations
- `website/`, `server/`, `storage/` - Core services
- `simulation/`, `fantasy/` - Additional features
- `nginx/`, `docker/` - Infrastructure
- `assets/`, `Drone_Camera_Imu_Config/`, `Drone_Calibration/` - Resources
- Configuration files: `docker-compose.yml`, Dockerfiles, `README.md`, `prometheus.yml`

## Key Technical Details

### WebSocket Communication
- Server receives data from Android on port 5001
- Website connects to server WebSocket for real-time updates
- Messages use JSON format with type field for routing

### SLAM Integration
- SLAM3R service writes keyframes to shared memory (/dev/shm/slam3r_keyframes)
- Mesh Service reads from shared memory for zero-copy performance
- Camera poses must be synchronized with image timestamps
- Shared memory uses POD structs for C++/Python compatibility

### 3D Reconstruction Pipeline
1. Images saved to disk by storage service
2. SLAM3R processes images → camera poses → shared memory
3. Mesh Service reads poses + images → TSDF volume → Marching Cubes → mesh
4. Real-time visualization via Rerun (9876) and website

### Service Dependencies
- GPU-required services: frame_processor, slam3r, mesh_service
- All services communicate through RabbitMQ
- Services use environment variables for configuration
- Docker networks: backend_network (internal), monitoring
- Mesh Service configuration: All env vars prefixed with MESH_ (see mesh_service/CONFIG.md)