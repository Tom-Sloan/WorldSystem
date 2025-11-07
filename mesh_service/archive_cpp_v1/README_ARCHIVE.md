# Mesh Service C++ v1.0 - ARCHIVED

**Date Archived:** 2025-11-07
**Status:** Not in active use
**Replaced By:** Python v2.0 implementation (see ../python/)

## Why This Was Archived

The C++ v1.0 implementation was deprecated due to critical performance and memory issues:

### Problems with C++ v1.0
- **Memory leak:** 5-10 GB memory usage after 5 minutes ❌
- **Poor rendering performance:** <1 FPS in Rerun viewer ❌
- **Root cause:** Unlimited entity accumulation in Rerun
- **Complex codebase:** ~4,500+ lines of CUDA/C++ code
- **Difficult to maintain:** Multiple GPU algorithms with intertwined dependencies

### Python v2.0 Improvements
- **Constant memory:** ~50 MB ✅
- **Smooth rendering:** 60 FPS in Rerun ✅
- **Fast processing:** 3-6 ms per keyframe ✅
- **Single entity:** Efficient rendering strategy ✅
- **Maintainable:** Clean architecture with Open3D integration ✅

## What's Included

This archive contains the complete C++ v1.0 implementation:

### Core Components
- **src/main.cpp** - Simple RabbitMQ consumer (235 lines)
- **src/shared_memory.cpp** - Shared memory reader for SLAM3R keyframes
- **src/rabbitmq_consumer.cpp** - RabbitMQ integration (19KB)
- **src/rerun_publisher.cpp** - Rerun visualization (11KB)

### GPU-Accelerated Mesh Generation (Unused)
- **src/mesh_generator.cu** (32KB) - Main mesh generation pipeline
- **src/simple_tsdf.cu** (29KB) - TSDF fusion algorithm
- **src/gpu_poisson_reconstruction.cu** (27KB) - Poisson surface reconstruction
- **src/gpu_octree.cu** (15KB) - GPU-based octree
- **src/algorithms/nvidia_marching_cubes.cu** (54KB) - Marching cubes implementation
- **src/algorithms/open3d_poisson.cpp** (8.8KB) - Open3D Poisson wrapper

### Configuration & Utilities
- **src/algorithm_selector.cpp** (11KB) - Dynamic algorithm selection
- **src/configuration_manager.cpp** (11KB) - Configuration management
- **src/metrics.cpp** (9.6KB) - Performance metrics
- **src/normal_provider_factory.cpp** - Normal estimation
- **src/normal_providers/** - Camera-based and Open3D normal providers

### External Dependencies
- **external/nvidia_mc/** - NVIDIA marching cubes tables and kernels

### Build System
- **CMakeLists.txt** - CMake build configuration
  - Only builds simple main.cpp + shared_memory.cpp
  - Complex GPU features not integrated into build

## Should You Use This?

**No.** This code is archived for reference only. Use the Python v2.0 implementation instead.

## If You Need to Reference This Code

### Build Instructions (Historical)
```bash
cd archive_cpp_v1
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

**Note:** This only builds the simple implementation (main.cpp). The advanced features (TSDF, Poisson, Marching Cubes) were never fully integrated into the build system.

### Dependencies Required
- C++17 compiler
- CUDA toolkit
- Open3D
- RabbitMQ-C library
- msgpack-c
- Rerun C++ SDK

## Migration Notes

If you're looking at this code to understand the migration:

**Key changes in v2.0:**
1. Replaced TSDF/Marching Cubes with Open3D voxel downsampling
2. Single Rerun entity instead of per-keyframe entities
3. RabbitMQ consumer in Python with asyncio
4. Shared memory reader using ctypes/numpy
5. Removed WebSocket server (handled by main server now)

## Contact

For questions about this archived code, refer to git history:
```bash
git log --follow -- mesh_service/archive_cpp_v1/
```

Last active commit: 7303616 (July 2024)
