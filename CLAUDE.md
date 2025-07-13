# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WorldSystem is a real-time 3D reconstruction and visualization system for drone-based room mapping. It combines SLAM, neural reconstruction, and real-time visualization to create 3D models of indoor spaces with the goal of overlaying fantasy elements in augmented reality.

## Architecture

The system uses a microservices architecture with the following data flow:
1. **Android App** → **Server** (WebSocket): Streams video (30fps) and IMU data
2. **Server** → **RabbitMQ** → **Frame Processor/SLAM3R/Storage**: Distributes data
3. **SLAM3R** → **Server/Website**: Provides camera poses and dense point clouds/meshes
4. **Server** → **Website** (WebSocket): Real-time visualization

Key services:
- **Server**: Central hub for data routing (FastAPI, Python)
- **SLAM3R**: Neural SLAM with integrated dense reconstruction (PyTorch)
- **Website**: Real-time 3D visualization (React/Three.js)
- **Frame Processor**: Video processing with YOLO detection
- **Fantasy Builder**: Adds game-like elements to 3D models (WIP)
- **Storage**: Persistent storage service with API for data retrieval
- **Monitoring Stack**: Prometheus, Grafana, Jaeger, cAdvisor

## Build and Development Commands

### Environment Setup
```bash
# Website environment
cp website/.env.example website/.env

# Configure CUDA paths and display in root .env
```

### Quick Start
```bash
./start.sh  # Builds and starts all services
```

### Docker Commands
```bash
# Full build
docker-compose build

# Single service rebuild  
docker-compose build --no-cache <service_name>

# Run all services
docker-compose up

# Run with SLAM3R profile
docker-compose --profile slam3r up

# Build with branch tag
./docker-compose-build.sh

# Run with branch tag
./docker-compose-up.sh
```

### Development Commands
```bash
# Website development (port 5173)
cd website && npm run dev

# Website build
cd website && npm run build

# Website preview production build (port 3000)
cd website && npm run preview

# Website linting
cd website && npm run lint

# SLAM3R demo
cd slam3r && bash scripts/demo_wild.sh

# SLAM3R visualization demo
cd slam3r && bash scripts/demo_vis_wild.sh

# SLAM3R Replica dataset demo
cd slam3r && bash scripts/demo_replica.sh

# SLAM3R evaluation on Replica
cd slam3r && bash scripts/eval_replica.sh

# SLAM3R training
cd slam3r && bash scripts/train_i2p.sh  # Image-to-Points model
cd slam3r && bash scripts/train_l2w.sh  # Local-to-World model

# Run simulation
./simulation/run_simulation.sh

# Test segment detection
python test_segment_detection.py

# Test segment reset
python test_segment_reset.py
```

### Monitoring URLs (when running)
- RabbitMQ Management: http://localhost:15672
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- Jaeger UI: http://localhost:16686

## Code Guidelines

When a file becomes too long, split it into smaller files. When a function becomes too long, split it into smaller functions.

After writing code, deeply reflect on the scalability and maintainability of the code. Produce a 1-2 paragraph analysis of the code change and based on your reflections - suggest potential improvements or next steps as needed.

### Style Guidelines
- **Python**: PEP 8, 4-space indentation, docstrings for all functions
- **JavaScript/React**: ESLint config, camelCase, alphabetical imports
- **Error handling**: try/except in Python, catch and log in JavaScript

### Modifiable Directories
Only modify code in:
- `slam3r/` - SLAM3R neural SLAM implementation
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

### SLAM3R Integration
- SLAM3R processes RGB frames directly from RabbitMQ
- Outputs camera poses and dense point clouds in real-time
- Optional mesh generation with Open3D
- Publishes results to RabbitMQ for visualization
- Runs at ~25fps target performance
- Automatic segment detection handles video boundaries

### 3D Reconstruction Pipeline
1. Images received from RabbitMQ by SLAM3R
2. SLAM3R neural networks: Image-to-Points → Local-to-World transformation
3. Outputs dense point clouds and optional meshes
4. Website receives and displays results via WebSocket

### RabbitMQ Message Flow
- Exchange: `video_stream` (fanout)
- Queues: `slam3r_queue`, `frame_processor_queue`, `storage_queue`
- Messages contain base64-encoded images and metadata
- Automatic reconnection on connection failures

### Service Dependencies
- Frame Processor, SLAM3R require GPU access (CUDA)
- All services communicate through RabbitMQ
- Services use environment variables for configuration
- Docker networks: backend_network (internal), monitoring
- Host networking mode used for all services

### Frontend Technology Stack
- React with Vite bundler
- Three.js via React Three Fiber for 3D rendering
- TailwindCSS for styling
- ESLint for code quality (no TypeScript)
- WebSocket for real-time updates