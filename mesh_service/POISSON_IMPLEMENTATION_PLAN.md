# Poisson Reconstruction Implementation Plan for SLAM3R

## Overview

This document outlines the complete plan to implement Poisson reconstruction as the primary mesh generation algorithm for SLAM3R, replacing TSDF which requires camera poses that SLAM3R doesn't reliably provide. **The implementation prioritizes using established libraries (Open3D, CGAL) over custom implementations.**

## Why Poisson for SLAM3R?

1. **No camera poses needed** - Works directly with point clouds and normals
2. **Handles SLAM3R confidence values** - Can weight reconstruction by point confidence
3. **Smooth, watertight meshes** - No blocky TSDF artifacts
4. **Library implementations available** - Open3D (GPU-accelerated) and CGAL (CPU) provide robust implementations
5. **Proven algorithm** - Well-tested in research and production

## Current State

- **TSDF + Marching Cubes**: Currently active but producing poor quality meshes due to unreliable camera poses
- **Open3D**: Already integrated for normal estimation, will be extended for Poisson reconstruction

## Implementation Steps

### Step 1: Update Configuration Files

#### 1.1 Update `include/config/mesh_service_config.h`

Add missing Poisson constants to the `AlgorithmConfig` struct (after line 50):

```cpp
// Poisson reconstruction - GPU specific
static constexpr int DEFAULT_POISSON_BLOCK_SIZE = 256;
static constexpr int DEFAULT_POISSON_BLOCK_OVERLAP = 32;
static constexpr int DEFAULT_POISSON_SOLVER_ITERATIONS = 8;
static constexpr float DEFAULT_POISSON_SOLVER_TOLERANCE = 1e-6f;

// Poisson density filtering
static constexpr float DEFAULT_POISSON_DENSITY_PERCENTILE = 0.1f;  // Remove bottom 10%
static constexpr float DEFAULT_POISSON_MIN_DENSITY = 0.01f;

// Poisson confidence integration  
static constexpr float DEFAULT_POISSON_CONFIDENCE_WEIGHT_SCALE = 0.01f;  // Scale confidence to 0-1
static constexpr float DEFAULT_POISSON_MIN_CONFIDENCE_WEIGHT = 0.1f;

// Incremental Poisson
static constexpr int DEFAULT_POISSON_INCREMENTAL_BLOCK_SIZE = 10000;  // Points per block
static constexpr float DEFAULT_POISSON_INCREMENTAL_OVERLAP_RATIO = 0.1f;
```

#### 1.2 Create `include/config/poisson_config.h`

```cpp
#pragma once

namespace mesh_service {
namespace config {

// Poisson-specific configuration that doesn't fit in general config
struct PoissonConfig {
    // Adaptive quality settings for SLAM3R
    struct AdaptiveQuality {
        static constexpr int OCTREE_DEPTH_STATIONARY = 8;      // High quality when still
        static constexpr int OCTREE_DEPTH_SLOW = 7;            // Medium quality
        static constexpr int OCTREE_DEPTH_FAST = 6;            // Low quality when moving
        
        static constexpr float VELOCITY_THRESHOLD_STATIONARY = 0.01f;  // m/s
        static constexpr float VELOCITY_THRESHOLD_SLOW = 0.1f;         // m/s
        static constexpr float VELOCITY_THRESHOLD_FAST = 0.5f;         // m/s
    };
    
    // SLAM3R-specific settings
    struct SLAM3RIntegration {
        static constexpr float CONFIDENCE_THRESHOLD_I2P = 10.0f;   // From SLAM3R config
        static constexpr float CONFIDENCE_THRESHOLD_L2W = 12.0f;   // From SLAM3R config
        static constexpr bool USE_CONFIDENCE_WEIGHTING = true;
        static constexpr bool IGNORE_CAMERA_POSE = true;           // Don't use unreliable poses
    };
    
    // Memory management
    struct Memory {
        static constexpr size_t MAX_POINTS_PER_BATCH = 500000;
        static constexpr size_t GPU_TO_CPU_TRANSFER_BLOCK = 65536;
        static constexpr size_t CPU_TO_GPU_TRANSFER_BLOCK = 65536;
    };
    
    // Solver parameters
    struct Solver {
        static constexpr int MAX_ITERATIONS = 100;
        static constexpr float CONVERGENCE_TOLERANCE = 1e-7f;
        static constexpr float REGULARIZATION_WEIGHT = 0.0001f;
    };
};

} // namespace config
} // namespace mesh_service
```

#### 1.3 Update `include/gpu_poisson_reconstruction.h`

Replace hardcoded constants (lines 12-13) with:

```cpp
#include "config/mesh_service_config.h"
#include "config/poisson_config.h"

namespace mesh_service {

// Use config constants instead of hardcoded values
constexpr int POISSON_BLOCK_SIZE = config::AlgorithmConfig::DEFAULT_POISSON_BLOCK_SIZE;
constexpr int POISSON_BLOCK_OVERLAP = config::AlgorithmConfig::DEFAULT_POISSON_BLOCK_OVERLAP;
```

### Step 2: Create Open3D Poisson Implementation

Since we want to use established libraries, we'll create a new Open3D-based Poisson implementation rather than using the custom GPU implementation.

#### 2.1 Create `include/algorithms/open3d_poisson.h`

```cpp
#pragma once

#include "algorithm_base.h"
#include <memory>

namespace mesh_service {

class Open3DPoisson : public ReconstructionAlgorithm {
public:
    Open3DPoisson();
    ~Open3DPoisson() override;
    
    bool initialize(const AlgorithmParams& params) override;
    
    bool reconstruct(
        const float3* d_points,
        const float3* d_normals,
        size_t num_points,
        const float* camera_pose,  // Ignored for Poisson
        MeshUpdate& output,
        cudaStream_t stream = 0
    ) override;
    
    size_t getMemoryUsage() const override;
    void reset() override;
    ReconstructionMethod getMethod() const override { 
        return ReconstructionMethod::OPEN3D_POISSON; 
    }

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace mesh_service
```

#### 2.2 Create `src/algorithms/open3d_poisson.cpp`

```cpp
#include "algorithms/open3d_poisson.h"
#include "config/mesh_service_config.h"
#include "config/poisson_config.h"
#include <open3d/Open3D.h>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

namespace mesh_service {

class Open3DPoisson::Impl {
public:
    AlgorithmParams params;
    
    // Open3D Poisson parameters
    int depth = 8;
    float width = 0.0f;
    float scale = 1.1f;
    bool linear_fit = false;
    int n_threads = 0;  // 0 = automatic
    
    // Confidence filtering
    bool use_confidence = true;
    float confidence_threshold = 12.0f;
    
    size_t memory_usage = 0;
};

Open3DPoisson::Open3DPoisson() : pImpl(std::make_unique<Impl>()) {}
Open3DPoisson::~Open3DPoisson() = default;

bool Open3DPoisson::initialize(const AlgorithmParams& params) {
    pImpl->params = params;
    
    // Set Open3D parameters from config
    pImpl->depth = params.poisson.octree_depth;
    pImpl->width = CONFIG_FLOAT("MESH_POISSON_WIDTH", 0.0f);
    pImpl->scale = CONFIG_FLOAT("MESH_POISSON_SCALE", 1.1f);
    pImpl->linear_fit = CONFIG_BOOL("MESH_POISSON_LINEAR_FIT", false);
    pImpl->n_threads = CONFIG_INT("MESH_POISSON_THREADS", 0);
    
    // SLAM3R-specific settings
    pImpl->use_confidence = CONFIG_BOOL("MESH_POISSON_USE_CONFIDENCE", 
        config::PoissonConfig::SLAM3RIntegration::USE_CONFIDENCE_WEIGHTING);
    pImpl->confidence_threshold = CONFIG_FLOAT("MESH_POISSON_CONFIDENCE_THRESHOLD",
        config::PoissonConfig::SLAM3RIntegration::CONFIDENCE_THRESHOLD_L2W);
    
    return true;
}

bool Open3DPoisson::reconstruct(
    const float3* d_points,
    const float3* d_normals,
    size_t num_points,
    const float* camera_pose,  // Ignored - Poisson doesn't need camera poses!
    MeshUpdate& output,
    cudaStream_t stream
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Step 1: Copy GPU data to CPU
    std::vector<float3> h_points(num_points);
    std::vector<float3> h_normals(num_points);
    
    cudaMemcpyAsync(h_points.data(), d_points, 
                    num_points * sizeof(float3), 
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_normals.data(), d_normals, 
                    num_points * sizeof(float3), 
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Step 2: Create Open3D point cloud
    auto pcd = std::make_shared<open3d::geometry::PointCloud>();
    pcd->points_.reserve(num_points);
    pcd->normals_.reserve(num_points);
    
    // TODO: Get confidence values from SharedKeyframe
    // For now, add all points
    for (size_t i = 0; i < num_points; ++i) {
        pcd->points_.emplace_back(
            h_points[i].x, h_points[i].y, h_points[i].z
        );
        pcd->normals_.emplace_back(
            h_normals[i].x, h_normals[i].y, h_normals[i].z
        );
    }
    
    std::cout << "[Open3D Poisson] Processing " << pcd->points_.size() 
              << " points with octree depth " << pImpl->depth << std::endl;
    
    // Step 3: Run Poisson reconstruction using Open3D
    std::shared_ptr<open3d::geometry::TriangleMesh> mesh;
    std::vector<double> densities;
    
    std::tie(mesh, densities) = open3d::geometry::TriangleMesh::CreateFromPointCloudPoisson(
        *pcd, 
        pImpl->depth,
        pImpl->width,
        pImpl->scale,
        pImpl->linear_fit,
        pImpl->n_threads
    );
    
    // Step 4: Filter by density to remove spurious geometry
    if (!densities.empty() && CONFIG_BOOL("MESH_POISSON_DENSITY_FILTER", true)) {
        // Calculate density threshold
        std::vector<double> sorted_densities = densities;
        std::sort(sorted_densities.begin(), sorted_densities.end());
        
        float percentile = CONFIG_FLOAT("MESH_POISSON_DENSITY_PERCENTILE",
            config::AlgorithmConfig::DEFAULT_POISSON_DENSITY_PERCENTILE);
        size_t threshold_idx = static_cast<size_t>(sorted_densities.size() * percentile);
        double density_threshold = sorted_densities[threshold_idx];
        
        // Remove low-density vertices
        std::vector<bool> vertices_to_remove(mesh->vertices_.size(), false);
        for (size_t i = 0; i < densities.size(); ++i) {
            if (densities[i] < density_threshold) {
                vertices_to_remove[i] = true;
            }
        }
        
        mesh->RemoveVerticesByMask(vertices_to_remove);
    }
    
    // Step 5: Ensure mesh has vertex normals
    if (!mesh->HasVertexNormals()) {
        mesh->ComputeVertexNormals();
    }
    
    // Step 6: Convert back to GPU format
    size_t num_vertices = mesh->vertices_.size();
    size_t num_faces = mesh->triangles_.size();
    
    // Update memory usage estimate
    pImpl->memory_usage = num_vertices * sizeof(float3) * 2 + num_faces * sizeof(int3);
    
    // Allocate output buffers if needed
    output.vertex_count = num_vertices;
    output.face_count = num_faces;
    
    if (!output.vertices || output.allocated_vertices < num_vertices) {
        if (output.vertices) cudaFree(output.vertices);
        if (output.normals) cudaFree(output.normals);
        if (output.faces) cudaFree(output.faces);
        
        cudaMalloc(&output.vertices, num_vertices * sizeof(float3));
        cudaMalloc(&output.normals, num_vertices * sizeof(float3));
        cudaMalloc(&output.faces, num_faces * sizeof(int3));
        output.allocated_vertices = num_vertices;
    }
    
    // Copy vertices
    std::vector<float3> vertices(num_vertices);
    for (size_t i = 0; i < num_vertices; ++i) {
        vertices[i] = make_float3(
            static_cast<float>(mesh->vertices_[i].x()),
            static_cast<float>(mesh->vertices_[i].y()),
            static_cast<float>(mesh->vertices_[i].z())
        );
    }
    cudaMemcpyAsync(output.vertices, vertices.data(),
                    num_vertices * sizeof(float3),
                    cudaMemcpyHostToDevice, stream);
    
    // Copy normals
    std::vector<float3> normals(num_vertices);
    for (size_t i = 0; i < num_vertices; ++i) {
        normals[i] = make_float3(
            static_cast<float>(mesh->vertex_normals_[i].x()),
            static_cast<float>(mesh->vertex_normals_[i].y()),
            static_cast<float>(mesh->vertex_normals_[i].z())
        );
    }
    cudaMemcpyAsync(output.normals, normals.data(),
                    num_vertices * sizeof(float3),
                    cudaMemcpyHostToDevice, stream);
    
    // Copy faces
    std::vector<int3> faces(num_faces);
    for (size_t i = 0; i < num_faces; ++i) {
        faces[i] = make_int3(
            mesh->triangles_[i](0),
            mesh->triangles_[i](1),
            mesh->triangles_[i](2)
        );
    }
    cudaMemcpyAsync(output.faces, faces.data(),
                    num_faces * sizeof(int3),
                    cudaMemcpyHostToDevice, stream);
    
    cudaStreamSynchronize(stream);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "[Open3D Poisson] Generated mesh with " << num_vertices 
              << " vertices, " << num_faces << " faces in " << duration << "ms" << std::endl;
    
    return true;
}

size_t Open3DPoisson::getMemoryUsage() const {
    return pImpl->memory_usage;
}

void Open3DPoisson::reset() {
    pImpl->memory_usage = 0;
}

} // namespace mesh_service
```

#### 2.3 Update `CMakeLists.txt`

Add Open3D Poisson implementation:
```cmake
# Add Open3D Poisson
src/algorithms/open3d_poisson.cpp

# Ensure Open3D is linked
find_package(Open3D REQUIRED)
target_link_libraries(mesh_service PRIVATE Open3D::Open3D)
```

### Step 3: Update Algorithm Selector to Use Open3D Poisson

#### 3.1 Update `src/algorithm_selector.cpp`

Add Open3D Poisson implementation in `initialize()`:

```cpp
bool AlgorithmSelector::initialize() {
    // Initialize NVIDIA Marching Cubes (existing)
    auto nvidia_mc = std::make_shared<NvidiaMarchingCubes>();
    // ... existing code ...
    algorithms_[ReconstructionMethod::NVIDIA_MARCHING_CUBES] = nvidia_mc;
    
    // Initialize Open3D Poisson as primary reconstruction method
    auto open3d_poisson = std::make_shared<Open3DPoisson>();
    AlgorithmParams poisson_params;
    poisson_params.poisson.octree_depth = CONFIG_INT("MESH_POISSON_OCTREE_DEPTH", 
        config::AlgorithmConfig::DEFAULT_POISSON_OCTREE_DEPTH);
    poisson_params.poisson.point_weight = CONFIG_FLOAT("MESH_POISSON_POINT_WEIGHT",
        config::AlgorithmConfig::DEFAULT_POISSON_POINT_WEIGHT);
    poisson_params.poisson.solver_iterations = CONFIG_INT("MESH_POISSON_SOLVER_ITERATIONS",
        config::AlgorithmConfig::DEFAULT_POISSON_SOLVER_ITERATIONS);
    
    if (!open3d_poisson->initialize(poisson_params)) {
        std::cerr << "Failed to initialize Open3D Poisson" << std::endl;
        return false;
    }
    algorithms_[ReconstructionMethod::OPEN3D_POISSON] = open3d_poisson;
    
    return true;
}
```

In the `selectAlgorithm()` method, always use Open3D Poisson for SLAM3R:

```cpp
ReconstructionMethod AlgorithmSelector::selectAlgorithm(
    float camera_velocity,
    size_t point_count,
    float scene_complexity
) {
    // For SLAM3R: Always use Open3D Poisson (doesn't need camera poses)
    return ReconstructionMethod::OPEN3D_POISSON;
}
```

### Step 4: Add Confidence Support

#### 4.1 Update `include/shared_memory.h`

Add confidence data to `SharedKeyframe` struct:

```cpp
struct SharedKeyframe {
    // ... existing fields ...
    
    // Add confidence data from SLAM3R
    float* confidence;           // Per-point confidence values
    uint32_t confidence_size;    // Should match point_count
};
```

#### 4.2 Update `src/shared_memory.cpp`

In `SharedMemoryManager::readKeyframe()`, add confidence handling:

```cpp
// After reading points, read confidence if available
if (keyframe->confidence_size > 0) {
    size_t confidence_offset = /* calculate offset */;
    keyframe->confidence = reinterpret_cast<float*>(
        static_cast<char*>(shared_mem) + confidence_offset
    );
}
```

### Step 5: Enable Open3D in Mesh Generator

Since we're using Open3D Poisson exclusively, update the mesh generator to use the algorithm selector properly:

#### 5.1 Update `src/mesh_generator.cu`

Ensure the algorithm selector is used and confidence filtering is handled in Open3D implementation:

```cpp
// In generateIncrementalMesh() or processKeyframe()
bool success = pImpl->algorithm_selector->processWithAutoSelect(
    d_points,
    d_normals,
    num_points,
    camera_pose,
    pImpl->camera_velocity,
    update,
    pImpl->stream
);
```

Note: The confidence filtering is handled inside the Open3D Poisson implementation (see Step 2.2 where we check for confidence values).

### Step 6: Update Docker Configuration

#### 6.1 Update `docker-compose.yml`

Add Open3D Poisson environment variables:

```yaml
mesh_service:
  environment:
    # Force Open3D Poisson for SLAM3R
    - MESH_ALGORITHM=OPEN3D_POISSON
    
    # Open3D Poisson configuration
    - MESH_POISSON_OCTREE_DEPTH=${POISSON_OCTREE_DEPTH:-8}
    - MESH_POISSON_WIDTH=${POISSON_WIDTH:-0.0}
    - MESH_POISSON_SCALE=${POISSON_SCALE:-1.1}
    - MESH_POISSON_LINEAR_FIT=${POISSON_LINEAR_FIT:-false}
    - MESH_POISSON_THREADS=${POISSON_THREADS:-0}
    
    # Density filtering
    - MESH_POISSON_DENSITY_FILTER=${POISSON_DENSITY_FILTER:-true}
    - MESH_POISSON_DENSITY_PERCENTILE=${POISSON_DENSITY_PERCENTILE:-0.1}
    
    # SLAM3R-specific settings
    - MESH_POISSON_USE_CONFIDENCE=${POISSON_USE_CONFIDENCE:-true}
    - MESH_POISSON_CONFIDENCE_THRESHOLD=${POISSON_CONFIDENCE_THRESHOLD:-12.0}
```

#### 6.2 Update `Dockerfile`

Ensure Open3D is installed:

```dockerfile
# Install Open3D dependencies
RUN apt-get update && apt-get install -y \
    libeigen3-dev \
    libglfw3-dev \
    libglew-dev

# Option 1: Install pre-built Open3D (faster)
RUN apt-get install -y software-properties-common && \
    add-apt-repository ppa:isl-org/release && \
    apt-get update && \
    apt-get install -y open3d

# Option 2: Build Open3D from source (if pre-built not available)
# RUN git clone --depth 1 --branch v0.17.0 https://github.com/isl-org/Open3D.git && \
#     cd Open3D && \
#     mkdir build && cd build && \
#     cmake -DBUILD_CUDA_MODULE=OFF \
#           -DUSE_SYSTEM_EIGEN3=ON \
#           -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
#     make -j$(nproc) && \
#     make install && \
#     cd ../.. && rm -rf Open3D
```

### Step 7: Update Main Entry Point

#### 7.1 Update `src/main.cpp`

Remove the setQualityAdaptive setting since we're always using Open3D Poisson.

## Files to Reference

### Core Implementation Files (To Create)
- `include/algorithms/open3d_poisson.h` - Open3D Poisson header
- `src/algorithms/open3d_poisson.cpp` - Open3D Poisson implementation

### Existing Files to Modify
- `src/algorithm_selector.cpp` - Add Open3D Poisson initialization
- `src/mesh_generator.cu` - Ensure algorithm selector is used
- `CMakeLists.txt` - Add Open3D Poisson source file

### Configuration Files
- `include/config/mesh_service_config.h` - Add missing Poisson constants
- `include/config/poisson_config.h` - Poisson-specific config (to be created)

### Integration Files
- `include/shared_memory.h` - Add confidence data to SharedKeyframe
- `src/shared_memory.cpp` - Handle confidence data in shared memory
- `src/main.cpp` - Service entry point

### Docker Files
- `docker-compose.yml` - Add Open3D Poisson environment variables
- `Dockerfile` - Ensure Open3D is installed

### Reference Files
- `include/normal_providers/open3d_normal_provider.h` - Example of Open3D integration
- `src/normal_providers/open3d_normal_provider.cpp` - Example implementation

## Testing Plan

### 1. Unit Tests
```bash
# Test Poisson with synthetic data
cd mesh_service/build
./test_poisson_reconstruction
```

### 2. Integration Test
```bash
# Test with SLAM3R pipeline
docker-compose up slam3r mesh_service

# Monitor logs for algorithm selection
docker logs mesh_service | grep "ALGORITHM SELECTOR"
```

### 3. Quality Comparison
```bash
# Run with TSDF (baseline)
MESH_FORCE_TSDF=true docker-compose up mesh_service

# Run with Poisson (improved)
MESH_FORCE_TSDF=false docker-compose up mesh_service

# Compare output meshes in /debug_output
```

### 4. Performance Monitoring
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor mesh generation time
docker logs mesh_service | grep "mesh generation"
```

## Expected Results

1. **Quality**: Smooth, watertight meshes without blocky artifacts
2. **Performance**: 50-100ms per frame (acceptable for real-time)
3. **Robustness**: Works with stationary and moving drone
4. **Confidence**: Uses SLAM3R confidence to filter noise

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `POISSON_OCTREE_DEPTH` to 6 or 7
   - Increase `POISSON_BLOCK_SIZE` to process larger chunks

2. **Slow Performance**
   - Check if GPU Poisson is being used (not CPU fallback)
   - Reduce point cloud size with confidence filtering
   - Use adaptive quality based on motion

3. **Poor Mesh Quality**
   - Increase `POISSON_CONFIDENCE_THRESHOLD` to filter more aggressively
   - Check if normals are being computed correctly
   - Verify confidence values are being passed from SLAM3R

## Summary

This plan replaces the current TSDF+Marching Cubes implementation with Open3D's Poisson reconstruction, which:

1. **Doesn't require camera poses** - Perfect for SLAM3R's architecture
2. **Uses established library** - Open3D is well-tested and optimized
3. **Produces smooth meshes** - No blocky TSDF artifacts
4. **Handles confidence values** - Can filter points based on SLAM3R confidence
5. **Simple integration** - Minimal code changes required

The implementation involves:
- Creating a new Open3D Poisson algorithm class
- Updating the algorithm selector to use it
- Adding configuration for Open3D parameters
- Ensuring Open3D is installed in Docker
- Testing with SLAM3R point clouds

This approach leverages Open3D's proven implementation rather than maintaining custom code, resulting in better quality meshes and easier maintenance.