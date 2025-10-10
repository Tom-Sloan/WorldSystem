# Mesh Service

Real-time 3D mesh generation service for the WorldSystem project using GPU-accelerated TSDF integration and marching cubes extraction.

## Overview

The mesh service converts point clouds from SLAM3R into 3D meshes in real-time. It processes incoming keyframes via shared memory, generates meshes using TSDF (Truncated Signed Distance Function) with marching cubes, and streams results to visualization clients.

### Key Features

- GPU-accelerated TSDF integration
- Real-time marching cubes mesh extraction  
- Adaptive quality based on camera velocity
- Spatial deduplication for efficiency
- Comprehensive runtime configuration
- Shared memory IPC with SLAM3R
- WebSocket streaming to clients
- Optional Open3D integration for high-quality normals

### Architecture

```text
SLAM3R → Shared Memory → Mesh Service → WebSocket → Viewers
                            ↓
                        RabbitMQ → Storage/Analytics
```

## Quick Start

```bash
# Start with default settings (fast mode)
docker-compose up mesh_service

# Start with Open3D support (if built with USE_OPEN3D=ON)
MESH_NORMAL_PROVIDER=1 docker-compose up mesh_service

# Monitor performance
docker logs mesh_service | grep FPS
```

## Configuration

All configuration is done through environment variables with the `MESH_` prefix.

### Core Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `MESH_VOXEL_SIZE` | 0.05 | TSDF voxel size in meters |
| `MESH_TRUNCATION_DISTANCE` | 0.15 | TSDF truncation distance |
| `MESH_MAX_VERTICES` | 5000000 | Maximum vertices per mesh |
| `MESH_NORMAL_PROVIDER` | 0 | Normal estimation provider (0=camera, 1=open3d) |

### Memory Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MESH_MEMORY_POOL_SIZE` | 1073741824 (1GB) | GPU memory pool size |
| `MESH_MAX_GPU_MEMORY` | 4294967296 (4GB) | Maximum GPU memory limit |

### Performance Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `MESH_CAMERA_VELOCITY_THRESHOLD` | 0.1 | Camera velocity threshold (m/s) |
| `MESH_DEBUG_SAVE_INTERVAL` | 10 | Save debug output every N frames |
| `MESH_FPS_LOG_INTERVAL` | 30 | Log FPS every N frames |

### Scene Bounds

| Variable | Default | Description |
|----------|---------|-------------|
| `TSDF_VOLUME_MIN` | -5,-5,0 | Minimum volume bounds (x,y,z) |
| `TSDF_VOLUME_MAX` | 30,5,5 | Maximum volume bounds (x,y,z) |

See the full configuration reference in the Environment Variables section below.

## Build Instructions

### Docker Build

```bash
# Standard build (without Open3D)
docker-compose build mesh_service

# Build with Open3D support
docker-compose build --build-arg USE_OPEN3D=ON mesh_service
```

### Local Build

```bash
cd mesh_service
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_OPEN3D=OFF ..
make -j$(nproc)
```

### Build Options

- `USE_OPEN3D`: Enable Open3D for high-quality normal estimation (OFF by default)
- `CMAKE_BUILD_TYPE`: Release or Debug

## Algorithm Details

### Current Implementations

#### 1. TSDF + Marching Cubes (Default)
NVIDIA's marching cubes implementation with TSDF integration for real-time performance:
- **TSDF Integration**: Points are integrated into a voxel grid with truncated signed distance values
- **Marching Cubes**: Extracts isosurface from TSDF volume to generate triangle mesh
- **Normal Estimation**: Camera-based (fast) or Open3D KD-tree (quality)

#### 2. Open3D Poisson Reconstruction (Optional)
High-quality surface reconstruction when Open3D is available:
- **Octree Construction**: Adaptive octree based on point density
- **Indicator Function**: Solves for indicator function using point normals
- **Iso-surface Extraction**: Extracts watertight surface at specified iso-value
- **Density Filtering**: Removes low-confidence regions

### Performance Characteristics

- **Frame Rate**: 25-50 FPS depending on point cloud size
- **Latency**: <50ms per frame
- **GPU Memory**: 500MB-1GB typical usage

### Normal Estimation Providers

1. **Camera-based (0)**: Fast GPU implementation
   - Uses ray direction from camera position
   - 0ms overhead (integrated in pipeline)
   - Lower quality but real-time
   - Best for moving camera scenarios

2. **Open3D (1)**: High quality, KD-tree based
   - K-nearest neighbors (k=30) or radius search
   - 40-60ms for 50k points
   - Requires CPU-GPU transfers
   - Optional compile-time dependency
   - Best for static reconstruction

## Development

### Project Structure

```text
mesh_service/
├── include/           # Headers
│   ├── algorithms/    # Reconstruction algorithms
│   ├── config/        # Configuration headers
│   └── normal_providers/  # Normal estimation
├── src/              # Implementation
│   ├── algorithms/   # Algorithm implementations
│   └── normal_providers/  # Provider implementations
├── external/         # Third-party code
│   └── nvidia_mc/    # NVIDIA marching cubes
└── debug_output/     # Debug PLY files (runtime)
```

### Adding New Features

1. **New Configuration Parameter**:
   ```cpp
   // In mesh_service_config.h
   static constexpr float DEFAULT_MY_PARAM = 1.0f;
   
   // In configuration_manager.cpp
   CONFIG_FLOAT("MESH_MY_PARAM", DEFAULT_MY_PARAM);
   ```

2. **New Normal Provider**:
   - Implement `INormalProvider` interface
   - Add to factory in `normal_provider_factory.cpp`
   - Currently supports camera-based and Open3D providers

### Debug Output

Debug files are saved to `/debug_output` every N frames:

- `pointcloud_XXXXXX.ply`: Raw input point clouds
- `mesh_XXXXXX.ply`: Generated meshes
- `tsdf_slice_XXXXXX.txt`: TSDF volume slices

View with MeshLab or similar tools.

## Testing

### Integration Test

```bash
# Run test script
cd mesh_service
python3 test_mesh_service.py

# Check shared memory
python3 ../tests/check_shared_memory.py
```

### Performance Test

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check frame rate
docker logs mesh_service | grep FPS
```

## Troubleshooting

### Common Issues

1. **No mesh output**
   - Check if points are within TSDF volume bounds
   - Verify shared memory is accessible
   - Check GPU memory availability

2. **Poor mesh quality**
   - Reduce voxel size for more detail
   - Enable Open3D normal estimation
   - Check point cloud quality from SLAM3R

3. **Low performance**
   - Increase voxel size
   - Disable debug saves
   - Check GPU utilization

### Debug Commands

```bash
# Check service health
curl http://localhost:8006/health

# View logs with timestamps
docker logs -t mesh_service

# Access container shell
docker exec -it mesh_service bash
```

## Environment Variables Reference

### Algorithm Selection

- `MESH_ALGORITHM`: Algorithm selection (currently only TSDF)
- `MESH_NORMAL_PROVIDER`: Normal provider (0=camera, 1=open3d)

### TSDF Parameters

- `TSDF_VOXEL_SIZE`: Voxel size in meters (0.01-0.2)
- `TSDF_TRUNCATION_DISTANCE`: Truncation distance (0.05-0.5)
- `TSDF_VOLUME_MIN`: Min bounds as "x,y,z"
- `TSDF_VOLUME_MAX`: Max bounds as "x,y,z"

### Memory Management

- `MESH_MEMORY_POOL_SIZE`: GPU memory pool (bytes)
- `MESH_MEMORY_BLOCK_SIZE`: Block allocation size
- `MESH_MAX_GPU_MEMORY`: Maximum GPU memory

### Performance

- `MESH_CAMERA_VELOCITY_THRESHOLD`: Velocity threshold for quality
- `MESH_FPS_LOG_INTERVAL`: FPS logging frequency
- `MESH_DEBUG_SAVE_INTERVAL`: Debug save frequency

### Normal Estimation

- `MESH_NORMAL_K_NEIGHBORS`: K neighbors for KD-tree methods
- `MESH_NORMAL_SEARCH_RADIUS`: Radius for normal estimation

### Scene Processing

- `MESH_MAX_SCENE_COORDINATE`: Filter outliers beyond this
- `MESH_CONFIDENCE_THRESHOLD`: Min confidence for points
- `MESH_SIMPLIFICATION_RATIO`: Mesh simplification (0-1)

## Future Enhancements

1. **GPU Poisson Reconstruction**: Custom CUDA implementation
   - Currently available via Open3D integration
   - Native GPU version planned for better performance

2. **Incremental Updates**: Only process changed regions
   - Octree-based spatial indexing
   - 90% overlap detection

3. **Advanced Normal Estimation**: 
   - GPU-based KD-tree implementation
   - Bilateral filtering for noise reduction

### Enabling Poisson Reconstruction

Poisson reconstruction is available when built with Open3D support:

```bash
# Build with Open3D
docker-compose build --build-arg USE_OPEN3D=ON mesh_service

# Run with Poisson enabled
MESH_ALGORITHM=OPEN3D_POISSON docker-compose up mesh_service
```

## Performance Optimization Tips

1. **For Speed**:
   - Larger voxel size (0.08-0.1)
   - Camera-based normals
   - Aggressive simplification

2. **For Quality**:
   - Smaller voxel size (0.02-0.04)
   - Open3D normals
   - Less simplification

3. **For Large Scenes**:
   - Increase memory pools
   - Enable spatial deduplication
   - Use streaming compression

## Docker Compose Configuration

```yaml
mesh_service:
  build:
    context: ./mesh_service
    args:
      USE_OPEN3D: "${USE_OPEN3D:-OFF}"
  environment:
    # Core settings
    - MESH_VOXEL_SIZE=${MESH_VOXEL_SIZE:-0.05}
    - MESH_NORMAL_PROVIDER=${MESH_NORMAL_PROVIDER:-0}
    # Memory
    - MESH_MEMORY_POOL_SIZE=${MESH_MEMORY_POOL_SIZE:-1073741824}
    # Performance
    - MESH_DEBUG_SAVE_INTERVAL=${MESH_DEBUG_SAVE_INTERVAL:-10}
  volumes:
    - /dev/shm:/dev/shm
    - ./mesh_service/debug_output:/debug_output
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

## File Reference

This section documents the purpose of each file in the mesh service codebase.

### Core Service Files

#### Main Entry Point

- **`src/main.cpp`** - Primary service entry point. Initializes RabbitMQ consumer, WebSocket server, shared memory reader, and orchestrates the mesh generation pipeline

#### Mesh Generation Pipeline

- **`src/mesh_generator.cu`** - Core mesh generation orchestrator. Manages the entire pipeline from point cloud input to mesh output
- **`src/simple_tsdf.cu`** - TSDF (Truncated Signed Distance Function) volume implementation. Integrates point clouds into voxel grid

#### Algorithm Implementation

- **`src/algorithms/nvidia_marching_cubes.cu`** - NVIDIA's marching cubes implementation wrapper. Extracts isosurface from TSDF volume
- **`src/algorithms/open3d_poisson.cpp`** - Open3D Poisson reconstruction (currently disabled for performance)
- **`src/algorithm_selector.cpp`** - Runtime algorithm selection based on scene complexity and camera motion
- **`include/algorithms/algorithm_base.h`** - Base interface for all reconstruction algorithms
- **`include/algorithms/nvidia_marching_cubes.h`** - Header for NVIDIA marching cubes
- **`include/algorithms/open3d_poisson.h`** - Header for Open3D Poisson

#### Normal Estimation

- **`src/normal_estimation.cu`** - GPU-based normal estimation (currently disabled for performance)
- **`src/normal_provider_factory.cpp`** - Factory for creating normal estimation providers
- **`src/normal_providers/camera_based_normal_provider.cu`** - Fast camera-based normal estimation
- **`src/normal_providers/open3d_normal_provider.cpp`** - High-quality Open3D-based normal estimation
- **`include/normal_provider.h`** - Normal provider interface
- **`include/normal_providers/*.h`** - Headers for specific normal providers

### Communication & Streaming

#### RabbitMQ Integration
- **`src/rabbitmq_consumer.cpp`** - RabbitMQ consumer for receiving messages from other services
- **`include/rabbitmq_consumer.h`** - RabbitMQ consumer interface
- **Used for**: Receiving keyframe notifications, publishing mesh updates to storage/analytics

#### WebSocket Server
- **`src/websocket_server.cpp`** - WebSocket server for real-time mesh streaming to web clients
- **`include/websocket_server.h`** - WebSocket server interface
- **Used for**: Streaming mesh updates to visualization clients (website)

#### Shared Memory
- **`src/shared_memory.cpp`** - Shared memory reader for zero-copy IPC with SLAM3R
- **`include/shared_memory.h`** - Shared memory data structures and interface
- **Used for**: High-performance point cloud transfer from SLAM3R without serialization overhead

### Visualization & Monitoring

#### Rerun Integration
- **`src/rerun_publisher.cpp`** - Publishes mesh data to Rerun visualization tool
- **`include/rerun_publisher.h`** - Rerun publisher interface
- **Used for**: Real-time 3D visualization and debugging in Rerun viewer

#### Metrics & Monitoring
- **`src/metrics.cpp`** - Prometheus metrics for performance monitoring
- **`include/metrics.h`** - Metrics collection interface
- **Used for**: Tracking FPS, processing time, memory usage for Grafana dashboards

### Advanced Algorithms (Currently Disabled)

#### GPU-Accelerated Algorithms
- **`src/gpu_octree.cu`** - GPU octree for spatial indexing (for future incremental updates)
- **`src/gpu_poisson_reconstruction.cu`** - Custom GPU Poisson implementation (disabled)
- **`include/gpu_octree.h`** - GPU octree interface
- **`include/gpu_poisson_reconstruction.h`** - GPU Poisson interface

#### CPU Algorithms
- **`src/octree.cpp`** - CPU octree implementation (backup for GPU version)

#### Neural Reconstruction

### Configuration & Management

#### Configuration System
- **`src/configuration_manager.cpp`** - Runtime configuration management via environment variables
- **`include/config/configuration_manager.h`** - Configuration manager interface
- **`include/config/mesh_service_config.h`** - Default configuration values
- **`include/config/normal_provider_config.h`** - Normal estimation configuration
- **`include/config/poisson_config.h`** - Poisson-specific configuration

### External Dependencies

#### NVIDIA Marching Cubes
- **`external/nvidia_mc/marchingCubes_kernel.cu`** - NVIDIA's marching cubes CUDA kernels
- **`external/nvidia_mc/tables.h`** - Marching cubes lookup tables
- **`external/nvidia_mc/defines.h`** - NVIDIA MC definitions

### Build & Deployment

#### Build System
- **`CMakeLists.txt`** - CMake build configuration
- **`Dockerfile`** - Docker container definition with all dependencies
- **`entrypoint.sh`** - Docker container entry point script

#### Testing
- **`test_build.sh`** - Build verification script
- **`test_mesh_service.py`** - Python integration test for mesh service

#### Development
- **`.gitignore`** - Git ignore patterns for build artifacts and debug output

### Purpose Summary by Category

**Point Cloud Processing**:

- TSDF integration: `simple_tsdf.cu`
- Normal estimation: `normal_*.cu/cpp` files
- Mesh generation: `mesh_generator.cu`

**Mesh Algorithms**:

- Marching Cubes: `nvidia_marching_cubes.cu` (primary, real-time)
- Poisson: `*poisson*.cpp/cu` (high quality, disabled)

**Data Flow**:

- Input: `shared_memory.cpp` (from SLAM3R)
- Processing: `mesh_generator.cu` + selected algorithm
- Output: `websocket_server.cpp` (to viewers), `rabbitmq_consumer.cpp` (to storage)

**Monitoring**:

- Performance: `metrics.cpp` (Prometheus)
- Visualization: `rerun_publisher.cpp` (Rerun)
- Debug: Files saved to `debug_output/`

## License

Part of the WorldSystem project. See root LICENSE file.
