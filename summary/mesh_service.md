# Mesh Service - GPU-Accelerated 3D Mesh Generation

## Overview

The mesh service is a high-performance C++/CUDA microservice that converts real-time point clouds from SLAM3R into 3D triangle meshes. It implements GPU-accelerated TSDF (Truncated Signed Distance Function) integration with marching cubes extraction, enabling real-time mesh generation at 25-50 FPS. The service acts as the bridge between raw SLAM reconstruction and viewable 3D models.

## What This Service Does

### Core Functionality
- **TSDF Integration**: Accumulates point clouds into a voxel-based distance field
- **Mesh Extraction**: Generates triangle meshes using NVIDIA's marching cubes
- **Normal Estimation**: Computes surface normals for rendering
- **Real-time Streaming**: Delivers meshes via WebSocket to visualization clients
- **Memory Management**: Efficient GPU memory pooling and allocation
- **Quality Adaptation**: Adjusts mesh quality based on camera velocity

### Key Features
- Zero-copy shared memory IPC with SLAM3R
- GPU-accelerated processing pipeline
- Configurable voxel resolution (0.01-0.2m)
- Multiple normal estimation providers
- Comprehensive runtime configuration
- Debug output for analysis
- Prometheus metrics integration

## System Architecture Connection

### Data Flow
```
SLAM3R (Point Clouds)
    ↓ Shared Memory (/dev/shm/slam3r_keyframes)
Mesh Service
    ├── TSDF Integration (GPU)
    ├── Marching Cubes (GPU)
    └── Normal Estimation
        ↓ WebSocket
    Website/Viewers
```

### Communication Protocols
- **Input**: Shared memory keyframes from SLAM3R
- **Output**: 
  - WebSocket mesh streaming (port 8080)
  - RabbitMQ notifications
  - Rerun visualization
  - Debug PLY files

### Integration Points
1. **Shared Memory Reader**: Zero-copy point cloud transfer
2. **WebSocket Server**: Real-time mesh streaming
3. **RabbitMQ Consumer**: Service coordination
4. **Rerun Publisher**: 3D visualization
5. **Prometheus Metrics**: Performance monitoring

## Development History

### Evolution Timeline

1. **Initial Architecture** (Commit 867a18c)
   - Decoupled mesh service from SLAM3R
   - Eliminated CPU bottlenecks
   - Introduced shared memory IPC

2. **Base Implementation** (Commit d58c97f)
   - Basic TSDF integration
   - NVIDIA marching cubes wrapper
   - Initial WebSocket streaming

3. **Building Success** (Commit 9972ff8)
   - Resolved CUDA compilation issues
   - Integrated external dependencies

4. **Enhanced Functionality** (Commits 1705107, 1e93faf)
   - Extensive debug logging
   - Camera-based normal estimation
   - Corrupted point filtering
   - POD-compliant shared memory

5. **Visualization Integration** (Commit 5d42284)
   - Rerun camera setup
   - Debug grid visualization
   - Improved debugging tools

6. **Performance Optimization** (Commit ac1772c)
   - Prometheus metrics
   - Health checks
   - Resource management

### Design Philosophy
- **GPU-First**: All heavy computation on GPU
- **Zero-Copy**: Shared memory for high-bandwidth data
- **Configurable**: Runtime adjustment via environment variables
- **Observable**: Comprehensive metrics and logging

## Technical Details

### Technology Stack
- **Language**: C++17 with CUDA
- **GPU Computing**: CUDA 11.8+
- **Build System**: CMake 3.18+
- **Dependencies**:
  - NVIDIA marching cubes
  - Open3D (optional)
  - RabbitMQ C client
  - WebSocket++ 
  - Prometheus C++ client
  - Rerun SDK

### Architecture Components

#### Core Pipeline (`mesh_generator.cu`)
1. Receives point cloud from shared memory
2. Validates and filters points
3. Integrates into TSDF volume
4. Extracts mesh with marching cubes
5. Estimates normals
6. Streams via WebSocket

#### TSDF Implementation (`simple_tsdf.cu`)
- Voxel grid representation
- GPU kernel for integration
- Weighted averaging
- Truncation for efficiency
- Camera position tracking

#### Algorithm Implementations
1. **NVIDIA Marching Cubes** (Primary)
   - GPU-accelerated extraction
   - 256 case lookup table
   - Triangle generation kernels
   - Vertex deduplication

2. **Open3D Poisson** (Optional)
   - High-quality reconstruction
   - CPU-based (slower)
   - Watertight meshes

#### Normal Estimation Providers
1. **Camera-based** (Fast)
   - Ray direction from camera
   - GPU implementation
   - 0ms overhead

2. **Open3D KD-tree** (Quality)
   - K-nearest neighbors
   - Higher accuracy
   - 40-60ms overhead

### Configuration System

All parameters configurable via environment variables:

```bash
# Core Parameters
MESH_VOXEL_SIZE=0.05          # Resolution in meters
MESH_TRUNCATION_DISTANCE=0.15  # TSDF truncation
MESH_MAX_VERTICES=5000000      # Vertex limit
MESH_NORMAL_PROVIDER=0         # 0=camera, 1=open3d

# Memory Configuration  
MESH_MEMORY_POOL_SIZE=1073741824  # 1GB pool
MESH_MAX_GPU_MEMORY=4294967296    # 4GB limit

# Performance Tuning
MESH_CAMERA_VELOCITY_THRESHOLD=0.1  # m/s
MESH_DEBUG_SAVE_INTERVAL=10         # frames
MESH_FPS_LOG_INTERVAL=30            # frames

# Scene Bounds
TSDF_VOLUME_MIN=-5,-5,0
TSDF_VOLUME_MAX=30,5,5
```

### Shared Memory Structure

POD-compliant structure for C++ compatibility:

```cpp
struct SharedKeyframe {
    uint64_t timestamp_ns;
    uint32_t point_count;
    uint32_t color_channels;
    float pose_matrix[16];
    float bbox[6];
    // Variable-length arrays follow
};
```

### Performance Characteristics
- **Frame Rate**: 25-50 FPS typical
- **Latency**: <50ms per frame
- **GPU Memory**: 500MB-1GB usage
- **CPU Usage**: Minimal (GPU-focused)
- **Network**: 1-10 MB/s mesh streaming

## Algorithms and Implementation

### TSDF (Truncated Signed Distance Function)
- Discretizes 3D space into voxel grid
- Stores signed distance to nearest surface
- Truncates values beyond threshold
- Weighted integration for noise reduction

### Marching Cubes
- Classifies voxels by corner values
- 256 possible configurations
- Generates triangles from lookup table
- GPU parallel processing

### Optimization Techniques
1. **Spatial Deduplication**: Skip redundant regions
2. **Adaptive Quality**: Reduce detail when moving fast
3. **Memory Pooling**: Reuse GPU allocations
4. **Streaming Compression**: Efficient mesh encoding

## Challenges and Solutions

1. **GPU Memory Management**
   - Challenge: Limited GPU memory for large scenes
   - Solution: Configurable bounds, memory pooling

2. **Real-time Performance**
   - Challenge: 50ms budget per frame
   - Solution: GPU acceleration, adaptive quality

3. **Point Cloud Noise**
   - Challenge: Sensor noise creates artifacts
   - Solution: TSDF weighted averaging, filtering

4. **Shared Memory Synchronization**
   - Challenge: Race conditions with SLAM3R
   - Solution: Lock-free circular buffer design

5. **Network Bandwidth**
   - Challenge: Large meshes for streaming
   - Solution: Compression, LOD selection

## Debug and Monitoring

### Debug Outputs
Saved to `/debug_output/`:
- `pointcloud_XXXXXX.ply`: Input point clouds
- `mesh_XXXXXX.ply`: Generated meshes
- `tsdf_slice_XXXXXX.txt`: TSDF volume slices

### Metrics Exposed
- `mesh_fps`: Frames per second
- `mesh_processing_time_ms`: Per-frame latency
- `mesh_vertex_count`: Mesh complexity
- `mesh_gpu_memory_bytes`: Memory usage
- `mesh_points_processed_total`: Cumulative points

### Health Endpoints
- `/health`: Service status
- `/metrics`: Prometheus metrics
- WebSocket status on connection

## Usage Examples

### Basic Usage
```bash
# Default configuration
docker-compose up mesh_service

# High quality mode
MESH_VOXEL_SIZE=0.02 MESH_NORMAL_PROVIDER=1 \
docker-compose up mesh_service

# Fast mode for demos
MESH_VOXEL_SIZE=0.1 MESH_DEBUG_SAVE_INTERVAL=0 \
docker-compose up mesh_service
```

### Development
```bash
# Local build
cd mesh_service
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)

# Run tests
./test_build.sh
python3 test_mesh_service.py
```

## Future Enhancements

### Planned Features
1. **GPU Poisson Reconstruction**
   - Native CUDA implementation
   - Better quality than marching cubes
   - Watertight guarantees

2. **Incremental Updates**
   - Only process changed regions
   - Octree spatial indexing
   - 90% computation reduction

3. **Multi-resolution Support**
   - LOD generation
   - Progressive transmission
   - Bandwidth adaptation

4. **Texture Mapping**
   - UV coordinate generation
   - Color texture atlases
   - PBR material support

### Research Directions
- Neural implicit representations
- Learned mesh compression
- Semantic mesh segmentation
- Real-time mesh optimization

## Performance Optimization

### For Maximum Speed
```bash
MESH_VOXEL_SIZE=0.1
MESH_NORMAL_PROVIDER=0
MESH_DEBUG_SAVE_INTERVAL=0
MESH_SIMPLIFICATION_RATIO=0.5
```

### For Best Quality
```bash
MESH_VOXEL_SIZE=0.02
MESH_NORMAL_PROVIDER=1
MESH_SIMPLIFICATION_RATIO=1.0
USE_OPEN3D=ON
```

### For Large Scenes
```bash
MESH_MEMORY_POOL_SIZE=2147483648  # 2GB
TSDF_VOLUME_MIN=-10,-10,-2
TSDF_VOLUME_MAX=10,10,5
```

This service represents the critical link between raw SLAM output and user-viewable 3D content, enabling real-time visualization of the reconstruction process.