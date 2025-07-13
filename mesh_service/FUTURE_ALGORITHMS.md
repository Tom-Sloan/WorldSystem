# Re-enabling Poisson and NKSR Algorithms

This document describes how to re-enable the GPU Poisson and NKSR algorithms that have been commented out for TSDF-only operation.

## Current State

The mesh_service is configured to use only TSDF with Marching Cubes for optimal real-time performance in indoor drone mapping scenarios. The other algorithms remain in the codebase but are disabled.

## Re-enabling GPU Poisson Reconstruction

### 1. Update CMakeLists.txt
Uncomment the following source files:
```cmake
src/gpu_poisson_reconstruction.cu
src/poisson_reconstruction.cpp
```

### 2. Update mesh_generator.cu
In the `Impl` class constructor, uncomment:
```cpp
// Mesh generation components
std::unique_ptr<GPUPoissonReconstruction> gpu_poisson;
std::unique_ptr<PoissonReconstruction> poisson;
std::unique_ptr<IncrementalPoissonReconstruction> incremental_poisson;

// In constructor
gpu_poisson = std::make_unique<GPUPoissonReconstruction>();
poisson = std::make_unique<PoissonReconstruction>();
incremental_poisson = std::make_unique<IncrementalPoissonReconstruction>();

// Configuration
incremental_poisson->initialize(10.0f, 8);
gpu_poisson->initialize(10.0f, 8);
GPUPoissonReconstruction::Parameters gpu_poisson_params;
// ... (full configuration block)
```

### 3. Uncomment the method
Uncomment the entire `generateIncrementalPoissonMesh()` method.

### 4. Re-enable algorithm switching
In `generateIncrementalMesh()`, uncomment the switch statement and adaptive selection logic.

### 5. Docker environment variables
Add to docker-compose.yml:
```yaml
# Poisson configuration
- POISSON_OCTREE_DEPTH=${POISSON_OCTREE_DEPTH:-8}
- POISSON_POINT_WEIGHT=${POISSON_POINT_WEIGHT:-4.0}
- POISSON_SOLVER_ITERATIONS=${POISSON_SOLVER_ITERATIONS:-8}
```

## Re-enabling NKSR (Neural Kernel Surface Reconstruction)

### 1. Update CMakeLists.txt
Uncomment:
```cmake
src/nksr_reconstruction.cu
```

### 2. Update mesh_generator.cu
In the `Impl` class, uncomment:
```cpp
std::unique_ptr<NKSRReconstruction> nksr;

// In constructor
nksr = std::make_unique<NKSRReconstruction>();

// Configuration
NKSRReconstruction::Parameters nksr_params;
// ... (full configuration block)
```

### 3. Uncomment the method
Uncomment the entire `generateNKSRMesh()` method.

### 4. Add CUDA libraries
Ensure CMakeLists.txt links:
```cmake
${CUDA_cublas_LIBRARY}
${CUDA_cusparse_LIBRARY}
```

### 5. Docker environment variables
```yaml
# NKSR configuration
- NKSR_CHUNK_SIZE=${NKSR_CHUNK_SIZE:-500000}
- NKSR_SUPPORT_RADIUS=${NKSR_SUPPORT_RADIUS:-0.1}
- NKSR_MAX_GPU_MEMORY=${NKSR_MAX_GPU_MEMORY:-4294967296}  # 4GB
```

## Re-enabling Motion-Adaptive Quality

### 1. In mesh_generator.cu
Change `setQualityAdaptive(false)` back to `true` in main.cpp

### 2. Uncomment velocity-based switching
```cpp
if (pImpl->quality_adaptive) {
    if (pImpl->camera_velocity > 0.5f) {
        pImpl->method = MeshMethod::TSDF_MARCHING_CUBES;
    } else {
        pImpl->method = MeshMethod::INCREMENTAL_POISSON;
    }
}
```

### 3. Add environment variables
```yaml
- VELOCITY_THRESHOLD_HIGH=${VELOCITY_THRESHOLD_HIGH:-0.5}
- VELOCITY_THRESHOLD_LOW=${VELOCITY_THRESHOLD_LOW:-0.3}
```

## Testing After Re-enabling

### 1. Build Test
```bash
cd mesh_service
mkdir build && cd build
cmake ..
make -j8
```

### 2. Runtime Test
Create a test configuration:
```yaml
MESH_METHOD=INCREMENTAL_POISSON  # or NKSR
ENABLE_ADAPTIVE_QUALITY=true
```

### 3. Performance Verification
- Monitor GPU memory usage
- Check mesh generation FPS
- Verify algorithm switching works smoothly

## Performance Considerations

### GPU Memory Requirements
- **TSDF Only**: ~500MB-1GB
- **+ GPU Poisson**: +2-4GB
- **+ NKSR**: +4-8GB (with chunking)

### Processing Time (per frame)
- **TSDF**: 20-30ms
- **GPU Poisson**: 100-200ms
- **NKSR**: 200-500ms (depends on chunk size)

## Dependencies to Verify

1. **CGAL**: Required for CPU Poisson fallback
2. **cuBLAS/cuSPARSE**: Required for NKSR solver
3. **Thrust**: Already included with CUDA

## Known Issues When Re-enabling

1. **Memory Allocation**: The memory pool size may need adjustment
2. **Algorithm Switching**: May cause visual artifacts - consider implementing smooth transitions
3. **Parameter Tuning**: Default parameters may not be optimal for all scenes

## Recommended Approach

1. Start by re-enabling only GPU Poisson
2. Test thoroughly with your specific use case
3. Only add NKSR if you need to handle very large scenes (>10M points)
4. Consider keeping TSDF as the default with manual switching rather than automatic

## Code Sections to Review

When re-enabling, pay special attention to:
- Memory management in the constructor/destructor
- Stream synchronization between algorithms
- Parameter validation from environment variables
- Error handling for algorithm initialization failures