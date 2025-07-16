# Mesh Service Configuration Guide

This document describes all configuration parameters available for the mesh service. The configuration system allows runtime parameter tuning without recompilation.

## Configuration System Overview

The mesh service uses a hierarchical configuration system:
1. **Default values** - Hardcoded in `include/config/mesh_service_config.h`
2. **Environment variables** - Override defaults at runtime
3. **Docker Compose** - Set environment variables for containers

## Environment Variable Reference

All mesh service configuration parameters use the prefix `MESH_` to avoid conflicts with other services.

### Memory Configuration

| Variable | Default | Range | Description |
|----------|---------|-------|-------------|
| `MESH_MEMORY_POOL_SIZE` | 1073741824 (1GB) | 512MB-8GB | GPU memory pool size for mesh generation |
| `MESH_MEMORY_BLOCK_SIZE` | 67108864 (64MB) | 32MB-256MB | Memory block allocation size |
| `MESH_OCTREE_POOL_SIZE` | 536870912 (512MB) | 256MB-2GB | Memory pool for octree operations |
| `MESH_SOLVER_POOL_SIZE` | 268435456 (256MB) | 128MB-1GB | Memory pool for Poisson solver |
| `MESH_MAX_GPU_MEMORY` | 4294967296 (4GB) | 2GB-24GB | Maximum GPU memory limit |

### Algorithm Parameters

| Variable | Default | Range | Description |
|----------|---------|-------|-------------|
| `MESH_NORMAL_K_NEIGHBORS` | 30 | 10-50 | Number of neighbors for normal estimation |
| `MESH_TRUNCATION_DISTANCE` | 0.15 | 0.05-0.5 | TSDF truncation distance in meters |
| `MESH_MAX_VERTICES` | 5000000 | 1M-20M | Maximum vertices per mesh |
| `MESH_SIMPLIFICATION_RATIO` | 0.1 | 0.01-1.0 | Mesh simplification ratio (0.1 = keep 10%) |
| `MESH_MAX_TSDF_WEIGHT` | 100.0 | 10-1000 | Maximum weight for TSDF integration |
| `MESH_CONFIDENCE_THRESHOLD` | 0.1 | 0.0-1.0 | Minimum confidence for point inclusion |
| `MESH_INFLUENCE_RADIUS` | 0.1 | 0.05-0.5 | Influence radius for spatial operations |
| `MESH_VOXEL_SIZE` | 0.05 | 0.01-0.2 | TSDF voxel size in meters |
| `MESH_ISO_VALUE` | 0.0 | -0.1-0.1 | Iso-surface value for marching cubes |

### Scene Configuration

| Variable | Default | Range | Description |
|----------|---------|-------|-------------|
| `MESH_MAX_SCENE_COORDINATE` | 1000.0 | 100-10000 | Maximum scene coordinate (filters outliers) |
| `MESH_OCTREE_SCENE_SIZE` | 10.0 | 5-100 | Octree scene size in meters |
| `MESH_OCTREE_MAX_DEPTH` | 8 | 4-12 | Maximum octree depth |
| `MESH_OCTREE_LEAF_SIZE` | 64 | 32-256 | Octree leaf node size |
| `MESH_OVERLAP_THRESHOLD` | 0.9 | 0.5-1.0 | Spatial overlap threshold for deduplication |

### Performance Thresholds

| Variable | Default | Range | Description |
|----------|---------|-------|-------------|
| `MESH_CAMERA_VELOCITY_THRESHOLD` | 0.1 | 0.01-1.0 | Camera velocity threshold (m/s) |
| `MESH_VELOCITY_THRESHOLD_HIGH` | 2.0 | 1.0-5.0 | High velocity threshold for algorithm switching |
| `MESH_VELOCITY_THRESHOLD_LOW` | 0.5 | 0.1-2.0 | Low velocity threshold for quality mode |
| `MESH_TIME_DELTA_THRESHOLD` | 0.001 | 0.0001-0.01 | Minimum time delta for velocity calculation |
| `MESH_VELOCITY_SMOOTH_FACTOR` | 0.8 | 0.5-0.95 | Exponential smoothing factor for velocity |
| `MESH_VELOCITY_CURRENT_FACTOR` | 0.2 | 0.05-0.5 | Weight for current velocity (1 - smooth_factor) |
| `MESH_SWITCH_STABILITY_FRAMES` | 5 | 1-20 | Frames before algorithm switch |
| `MESH_POINT_COUNT_THRESHOLD` | 100000 | 10000-1M | Point count threshold for complexity |
| `MESH_COMPLEXITY_THRESHOLD` | 0.7 | 0.1-1.0 | Scene complexity threshold |

### Debug Configuration

| Variable | Default | Range | Description |
|----------|---------|-------|-------------|
| `MESH_FPS_LOG_INTERVAL` | 30 | 10-300 | Log FPS every N frames |
| `MESH_DEBUG_SAVE_INTERVAL` | 10 | 5-100 | Save debug output every N frames |
| `MESH_DEBUG_PRINT_LIMIT` | 5 | 1-20 | Number of items to print in debug logs |
| `MESH_MAX_POINTS_TO_SCAN` | 1000000 | 10000-10M | Max points to scan for bounds checking |
| `MESH_DEBUG_VOXEL_LIMIT` | 10 | 5-100 | Number of voxels to debug |
| `MESH_DEBUG_POINT_LIMIT` | 10 | 5-100 | Number of points to debug |
| `MESH_DEBUG_CONFIG` | false | true/false | Log configuration at startup |
| `MESH_DEBUG_OUTPUT_DIR` | /debug_output | path | Directory for debug file output |

### TSDF Volume Configuration (Legacy)

These parameters are kept for compatibility with the existing TSDF implementation:

| Variable | Default | Range | Description |
|----------|---------|-------|-------------|
| `TSDF_VOLUME_MIN` | -5,-5,0 | x,y,z | Minimum volume bounds (comma-separated) |
| `TSDF_VOLUME_MAX` | 30,5,5 | x,y,z | Maximum volume bounds (comma-separated) |
| `TSDF_VOXEL_SIZE` | 0.05 | 0.01-0.2 | TSDF voxel size (overrides MESH_VOXEL_SIZE) |
| `TSDF_TRUNCATION_DISTANCE` | 0.10 | 0.05-0.5 | TSDF truncation distance |

## Usage Examples

### 1. High-Performance Configuration (Drone)

```yaml
mesh_service:
  environment:
    # Optimize for speed
    - MESH_VOXEL_SIZE=0.08              # Larger voxels for speed
    - MESH_SIMPLIFICATION_RATIO=0.05    # More aggressive simplification
    - MESH_NORMAL_K_NEIGHBORS=20        # Fewer neighbors
    - MESH_DEBUG_SAVE_INTERVAL=50       # Less frequent debug saves
    - MESH_VELOCITY_THRESHOLD_HIGH=3.0  # Higher threshold for fast drones
```

### 2. High-Quality Configuration (Ground Robot)

```yaml
mesh_service:
  environment:
    # Optimize for quality
    - MESH_VOXEL_SIZE=0.02              # Smaller voxels for detail
    - MESH_SIMPLIFICATION_RATIO=0.2     # Keep more vertices
    - MESH_NORMAL_K_NEIGHBORS=40        # More neighbors for smoother normals
    - MESH_TRUNCATION_DISTANCE=0.2      # Larger truncation for smoother surfaces
    - MESH_VELOCITY_THRESHOLD_LOW=0.2   # Lower threshold for slow robots
```

### 3. Limited GPU Memory Configuration

```yaml
mesh_service:
  environment:
    # Reduce memory usage
    - MESH_MEMORY_POOL_SIZE=536870912   # 512MB pool
    - MESH_MAX_VERTICES=2000000          # 2M vertices max
    - MESH_OCTREE_POOL_SIZE=268435456   # 256MB octree
    - MESH_MAX_GPU_MEMORY=2147483648    # 2GB limit
```

### 4. Debug Configuration

```yaml
mesh_service:
  environment:
    # Enable detailed debugging
    - MESH_DEBUG_CONFIG=true             # Log all config values
    - MESH_DEBUG_PRINT_LIMIT=20          # Print more debug info
    - MESH_DEBUG_SAVE_INTERVAL=1         # Save every frame
    - MESH_FPS_LOG_INTERVAL=1            # Log FPS every frame
    - MESH_DEBUG_OUTPUT_DIR=/workspace/debug
```

## Configuration Validation

The configuration manager validates parameters at startup. Invalid configurations will cause the service to exit with an error message. Common validation rules:

- Memory sizes must be positive and block size ≤ pool size
- Ratios must be in range [0, 1]
- Thresholds must be positive
- Octree depth must be in range [1, 12]

## Adding New Configuration Parameters

To add a new configurable parameter:

1. Add the default value to `mesh_service_config.h`:
```cpp
static constexpr float DEFAULT_MY_PARAM = 1.0f;
```

2. Load it in `configuration_manager.cpp`:
```cpp
if (const char* val = loadEnv("MESH_MY_PARAM")) {
    float_params_["MESH_MY_PARAM"] = stringToNumber<float>(val, 
        config::MyConfig::DEFAULT_MY_PARAM);
}
```

3. Use it in your code:
```cpp
float my_param = CONFIG_FLOAT("MESH_MY_PARAM", 
    config::MyConfig::DEFAULT_MY_PARAM);
```

4. Add to docker-compose.yml:
```yaml
- MESH_MY_PARAM=${MESH_MY_PARAM:-1.0}
```

5. Document it in this file.

## Performance Impact

Most configuration parameters have minimal runtime overhead as they are loaded once at startup. However, some parameters can significantly impact performance:

- **Voxel size**: Smaller values increase memory usage and processing time cubically
- **Normal K neighbors**: Linear impact on normal estimation time
- **Max vertices**: Affects memory allocation and mesh extraction time
- **Debug intervals**: Frequent saves can impact real-time performance

## Troubleshooting

### Service won't start
- Check logs for configuration validation errors
- Ensure memory parameters don't exceed available GPU memory
- Verify environment variable syntax in docker-compose.yml

### Poor performance
- Increase voxel size
- Reduce max vertices
- Increase simplification ratio
- Disable debug saves

### Poor quality
- Decrease voxel size
- Increase normal K neighbors
- Reduce simplification ratio
- Increase truncation distance

### Out of memory
- Reduce memory pool sizes
- Increase voxel size
- Reduce max vertices
- Enable more aggressive simplification