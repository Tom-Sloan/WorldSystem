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

## Build and Development Commands

### Quick Start
```bash
./start.sh  # Builds and starts all services
```

### Docker Commands
- Full build: `docker-compose build`
- Single service: `docker-compose build --no-cache <service_name>`
- Run all services: `docker-compose up`
- Run with SLAM3R profile: `docker-compose --profile slam3r up`

### Development Commands
- Website: `cd website && npm run dev`
- Website build: `cd website && npm run build`
- SLAM3R demo: `cd slam3r && bash scripts/demo_wild.sh`

### Testing Commands
- Website: `cd website && npm run lint`
- SLAM3R evaluation: `cd slam3r && bash scripts/eval_replica.sh`

### Monitoring URLs (when running)
- RabbitMQ: http://localhost:15672
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- Jaeger: http://localhost:16686

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

### 3D Reconstruction Pipeline
1. Images received from RabbitMQ by SLAM3R
2. SLAM3R neural networks: Image-to-Points → Local-to-World transformation
3. Outputs dense point clouds and optional meshes
4. Website receives and displays results via WebSocket

### Service Dependencies
- Frame Processor, SLAM, Reconstruction require GPU access
- All services communicate through RabbitMQ
- Services use environment variables for configuration
- Docker networks: backend_network (internal), monitoring