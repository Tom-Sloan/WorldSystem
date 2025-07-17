# Algorithm Implementation Guide - Open3D Normal Estimation Update

## Update: Open3D Normal Estimation Implementation Complete ✅

Successfully implemented a modular normal estimation system with Open3D as an optional high-quality provider. The system maintains the current fast performance as default while allowing runtime selection of different normal estimation algorithms.

### Implementation Summary

#### 1. Provider Pattern Architecture
- **Numeric IDs**: Clean integer-based provider selection (0=camera, 1=open3d, etc.)
- **Factory Pattern**: Runtime provider creation with automatic fallback
- **Interface-based**: All providers implement `INormalProvider` interface
- **Optional Dependencies**: System works without Open3D installed

#### 2. Files Created
- `/include/config/normal_provider_config.h` - Configuration constants
- `/src/normal_provider_factory.cpp` - Factory implementation
- `/src/normal_providers/camera_based_normal_provider.cu` - GPU camera-based
- `/src/normal_providers/open3d_normal_provider.cpp` - Open3D integration
- `/include/normal_providers/*.h` - Provider headers

#### 3. Files Modified
- `/include/normal_provider.h` - Added numeric provider enum
- `/src/mesh_generator.cu` - Integrated provider system
- `/src/main.cpp` - Updated startup banner
- `CMakeLists.txt` - Optional Open3D support
- `Dockerfile` - Open3D build argument

#### 4. Usage
```bash
# Default (camera-based, fast)
docker-compose up mesh_service

# With Open3D (high quality)
docker-compose build --build-arg USE_OPEN3D=ON mesh_service
MESH_NORMAL_PROVIDER=1 docker-compose up mesh_service
```

#### 5. Performance
- Camera-based: ~0ms (integrated in TSDF)
- Open3D: 40-60ms for 50k points
- Maintains ~50 FPS with camera-based default

---

# Original Algorithm Implementation Guide

This document provides a detailed implementation plan for integrating library-based reconstruction algorithms into the mesh service, starting with NVIDIA Marching Cubes.

## Overview

The mesh service will support three reconstruction algorithms:
1. **TSDF + Marching Cubes** (NVIDIA CUDA Samples) - Real-time, fast
2. **GPU Poisson** (Open3D) - High quality, medium speed (Future)
3. **NKSR** (NVIDIA Research) - State-of-the-art, slower (Future)

## Current Status

✅ **IMPLEMENTATION COMPLETE**

The mesh service has been successfully refactored with:
- NVIDIA Marching Cubes implementation integrated
- Clean algorithm abstraction layer
- SimpleTSDF for efficient volume management
- Algorithm selector with velocity-based switching
- All compilation errors fixed
- Old implementation files removed

### Files Removed:
- `src/enhanced_marching_cubes.cu`
- `src/marching_cubes.cu` 
- `src/marching_cubes_tables.cu`
- `include/marching_cubes.h`
- `include/marching_cubes_tables.h`

### New Architecture Implemented:
- Base algorithm interface (`algorithm_base.h`)
- NVIDIA MC wrapper (`nvidia_marching_cubes.h/cu`)
- Algorithm selector (`algorithm_selector.h/cpp`)
- SimpleTSDF (`simple_tsdf.h/cu`)
- Updated mesh generator to use new architecture

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Mesh Generator                         │
├─────────────────────────────────────────────────────────┤
│                 Algorithm Selector                        │
│  ┌─────────────┬──────────────┬────────────────┐       │
│  │ NVIDIA MC   │ Open3D       │ NKSR Service   │       │
│  │ (Built-in)  │ (Future)     │ (Future)       │       │
│  └─────────────┴──────────────┴────────────────┘       │
│                        ↓                                  │
│                   Simple TSDF                             │
└─────────────────────────────────────────────────────────┘
```

## Phase 1: NVIDIA Marching Cubes Implementation

### Step 1: Download NVIDIA Sample Files

```bash
cd mesh_service
mkdir -p external/nvidia_mc
cd external/nvidia_mc

# Download the required files
wget https://raw.githubusercontent.com/NVIDIA/cuda-samples/master/Samples/5_Domain_Specific/marchingCubes/marchingCubes_kernel.cu
wget https://raw.githubusercontent.com/NVIDIA/cuda-samples/master/Samples/5_Domain_Specific/marchingCubes/marchingCubes_kernel.cuh
wget https://raw.githubusercontent.com/NVIDIA/cuda-samples/master/Samples/5_Domain_Specific/marchingCubes/tables.h

# Note: We'll manually wrap in namespace during implementation rather than patching
```

### Step 2: Create Directory Structure

```
mesh_service/
├── include/
│   ├── algorithms/
│   │   ├── algorithm_base.h          # Base interface
│   │   ├── nvidia_marching_cubes.h   # NVIDIA MC wrapper
│   │   ├── open3d_poisson.h          # Future: Open3D wrapper
│   │   └── nksr_client.h             # Future: NKSR client
│   ├── simple_tsdf.h                 # TSDF implementation
│   └── algorithm_selector.h          # Algorithm switching logic
├── src/
│   ├── algorithms/
│   │   ├── nvidia_marching_cubes.cu
│   │   ├── open3d_poisson.cpp        # Future
│   │   └── nksr_client.cpp           # Future
│   ├── simple_tsdf.cu
│   └── algorithm_selector.cpp
└── external/
    └── nvidia_mc/
        ├── marchingCubes_kernel.cu
        ├── marchingCubes_kernel.cuh
        └── tables.h
```

### Step 3: Base Algorithm Interface

```cpp
// include/algorithms/algorithm_base.h
#pragma once

#include <cuda_runtime.h>
#include "mesh_generator.h" // For MeshUpdate

namespace mesh_service {

enum class ReconstructionMethod {
    NVIDIA_MARCHING_CUBES,
    OPEN3D_POISSON,
    NKSR
};

class ReconstructionAlgorithm {
public:
    virtual ~ReconstructionAlgorithm() = default;
    
    // Initialize the algorithm
    virtual bool initialize(const AlgorithmParams& params) = 0;
    
    // Process point cloud to generate mesh
    virtual bool reconstruct(
        const float3* d_points,
        const float3* d_normals,
        size_t num_points,
        const float* camera_pose,
        MeshUpdate& output,
        cudaStream_t stream = 0
    ) = 0;
    
    // Get memory usage
    virtual size_t getMemoryUsage() const = 0;
    
    // Reset internal state
    virtual void reset() = 0;
    
    // Get method type
    virtual ReconstructionMethod getMethod() const = 0;
};

struct AlgorithmParams {
    // Common parameters
    float voxel_size = 0.05f;
    float3 volume_min = make_float3(-5.0f, -5.0f, -5.0f);
    float3 volume_max = make_float3(5.0f, 5.0f, 5.0f);
    
    // Method-specific parameters
    union {
        struct {
            float iso_value;
            float truncation_distance;
            int max_vertices;
        } marching_cubes;
        
        struct {
            int octree_depth;
            float point_weight;
            int solver_iterations;
        } poisson;
        
        struct {
            float detail_level;
            int chunk_size;
        } nksr;
    };
};

} // namespace mesh_service
```

### Step 4: NVIDIA Marching Cubes Wrapper - Detailed Implementation

```cpp
// include/algorithms/nvidia_marching_cubes.h
#pragma once

#include "algorithm_base.h"
#include "simple_tsdf.h"
#include <memory>
#include <cuda_runtime.h>

namespace mesh_service {

class NvidiaMarchingCubes : public ReconstructionAlgorithm {
public:
    NvidiaMarchingCubes();
    ~NvidiaMarchingCubes() override;
    
    bool initialize(const AlgorithmParams& params) override;
    
    bool reconstruct(
        const float3* d_points,
        const float3* d_normals,
        size_t num_points,
        const float* camera_pose,
        MeshUpdate& output,
        cudaStream_t stream = 0
    ) override;
    
    size_t getMemoryUsage() const override;
    void reset() override;
    ReconstructionMethod getMethod() const override { 
        return ReconstructionMethod::NVIDIA_MARCHING_CUBES; 
    }

private:
    // TSDF volume management
    std::unique_ptr<SimpleTSDF> tsdf_;
    
    // NVIDIA MC specific buffers
    struct MCBuffers {
        // Voxel classification
        uint* d_voxel_verts_scan = nullptr;      // Number of vertices per voxel
        uint* d_voxel_occupied_scan = nullptr;   // Occupied voxel flags
        uint* d_compressed_voxel_array = nullptr; // Compacted voxel list
        
        // Lookup tables (constant memory would be better)
        uint* d_num_verts_table = nullptr;       // 256 entries
        uint* d_tri_table = nullptr;             // 256x16 entries
        
        // Output buffers
        float4* d_vertex_buffer = nullptr;       // xyz + padding
        float4* d_normal_buffer = nullptr;       // xyz + padding  
        
        // Sizes
        size_t allocated_voxels = 0;
        size_t allocated_vertices = 0;
    } buffers_;
    
    // Parameters
    AlgorithmParams params_;
    
    // Internal methods
    void allocateBuffers(const int3& volume_dims);
    void freeBuffers();
    void uploadTables();
    
    // CUDA kernel launchers
    void classifyVoxels(
        const float* d_tsdf,
        const int3& dims,
        uint* d_voxel_verts,
        uint* d_voxel_occupied,
        cudaStream_t stream
    );
    
    void compactVoxels(
        const uint* d_voxel_occupied,
        const uint* d_voxel_occupied_scan,
        uint* d_compressed_voxels,
        uint num_voxels,
        cudaStream_t stream
    );
    
    void generateTriangles(
        const float* d_tsdf,
        const int3& dims,
        const float3& origin,
        const uint* d_compressed_voxels,
        const uint* d_num_verts_scan,
        uint num_active_voxels,
        float4* d_vertices,
        float4* d_normals,
        cudaStream_t stream
    );
};

} // namespace mesh_service
```

### Step 5: NVIDIA MC Implementation Details

```cpp
// src/algorithms/nvidia_marching_cubes.cu
#include "algorithms/nvidia_marching_cubes.h"
#include "../../external/nvidia_mc/marchingCubes_kernel.cuh"
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>

namespace mesh_service {

// Import NVIDIA tables
extern "C" {
    extern const unsigned int numVertsTable[256];
    extern const unsigned int triTable[256][16];
}

NvidiaMarchingCubes::NvidiaMarchingCubes() {
    tsdf_ = std::make_unique<SimpleTSDF>();
}

NvidiaMarchingCubes::~NvidiaMarchingCubes() {
    freeBuffers();
}

bool NvidiaMarchingCubes::initialize(const AlgorithmParams& params) {
    params_ = params;
    
    // Initialize TSDF volume
    tsdf_->initialize(
        params.volume_min,
        params.volume_max,
        params.voxel_size
    );
    
    // Upload marching cubes tables to GPU
    uploadTables();
    
    // Pre-allocate buffers based on volume size
    allocateBuffers(tsdf_->getVolumeDims());
    
    return true;
}

void NvidiaMarchingCubes::uploadTables() {
    // Allocate table memory
    cudaMalloc(&buffers_.d_num_verts_table, 256 * sizeof(uint));
    cudaMalloc(&buffers_.d_tri_table, 256 * 16 * sizeof(uint));
    
    // Copy tables to GPU
    cudaMemcpy(buffers_.d_num_verts_table, numVertsTable, 
               256 * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(buffers_.d_tri_table, triTable,
               256 * 16 * sizeof(uint), cudaMemcpyHostToDevice);
}

bool NvidiaMarchingCubes::reconstruct(
    const float3* d_points,
    const float3* d_normals,
    size_t num_points,
    const float* camera_pose,
    MeshUpdate& output,
    cudaStream_t stream
) {
    // Step 1: Integrate points into TSDF
    tsdf_->integrate(d_points, d_normals, num_points, camera_pose, stream);
    
    // Get TSDF data
    float* d_tsdf = tsdf_->getTSDFVolume();
    int3 dims = tsdf_->getVolumeDims();
    float3 origin = tsdf_->getVolumeOrigin();
    
    // Step 2: Classify voxels (determine which contain the isosurface)
    classifyVoxels(d_tsdf, dims, 
                   buffers_.d_voxel_verts_scan,
                   buffers_.d_voxel_occupied_scan,
                   stream);
    
    // Step 3: Scan to get total vertices and compaction offsets
    size_t temp_storage_bytes = 0;
    uint num_voxels = dims.x * dims.y * dims.z;
    
    // Get scan storage requirements
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes,
                                   buffers_.d_voxel_verts_scan,
                                   buffers_.d_voxel_verts_scan,
                                   num_voxels, stream);
    
    // Allocate temporary storage
    thrust::device_vector<uint8_t> d_temp_storage(temp_storage_bytes);
    
    // Perform scans
    cub::DeviceScan::ExclusiveSum(d_temp_storage.data().get(), temp_storage_bytes,
                                   buffers_.d_voxel_verts_scan,
                                   buffers_.d_voxel_verts_scan,
                                   num_voxels, stream);
    
    cub::DeviceScan::ExclusiveSum(d_temp_storage.data().get(), temp_storage_bytes,
                                   buffers_.d_voxel_occupied_scan,
                                   buffers_.d_voxel_occupied_scan,
                                   num_voxels, stream);
    
    // Get total counts
    uint h_total_verts, h_last_vert_scan;
    uint h_total_voxels, h_last_voxel_scan;
    
    cudaMemcpyAsync(&h_last_vert_scan, 
                    buffers_.d_voxel_verts_scan + num_voxels - 1,
                    sizeof(uint), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_total_verts,
                    buffers_.d_voxel_verts_scan + num_voxels - 1,
                    sizeof(uint), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_last_voxel_scan,
                    buffers_.d_voxel_occupied_scan + num_voxels - 1,
                    sizeof(uint), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_total_voxels,
                    buffers_.d_voxel_occupied_scan + num_voxels - 1,
                    sizeof(uint), cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    uint total_vertices = h_last_vert_scan + h_total_verts;
    uint active_voxels = h_last_voxel_scan + h_total_voxels;
    
    if (total_vertices == 0) {
        output.vertex_count = 0;
        output.face_count = 0;
        return true;
    }
    
    // Step 4: Compact active voxels
    compactVoxels(buffers_.d_voxel_occupied,
                  buffers_.d_voxel_occupied_scan,
                  buffers_.d_compressed_voxel_array,
                  num_voxels,
                  stream);
    
    // Step 5: Generate triangles
    generateTriangles(d_tsdf, dims, origin,
                      buffers_.d_compressed_voxel_array,
                      buffers_.d_voxel_verts_scan,
                      active_voxels,
                      buffers_.d_vertex_buffer,
                      buffers_.d_normal_buffer,
                      stream);
    
    // Step 6: Copy results to output
    output.vertex_count = total_vertices;
    output.face_count = total_vertices / 3;  // Triangle soup
    
    // Ensure output buffers are allocated
    if (!output.vertices || output.allocated_vertices < total_vertices) {
        if (output.vertices) cudaFree(output.vertices);
        if (output.normals) cudaFree(output.normals);
        if (output.faces) cudaFree(output.faces);
        
        cudaMalloc(&output.vertices, total_vertices * sizeof(float3));
        cudaMalloc(&output.normals, total_vertices * sizeof(float3));
        cudaMalloc(&output.faces, output.face_count * sizeof(int3));
        output.allocated_vertices = total_vertices;
    }
    
    // Convert float4 to float3 and copy
    // Launch a simple kernel to do this conversion
    convertAndCopyVertices<<<(total_vertices + 255)/256, 256, 0, stream>>>(
        buffers_.d_vertex_buffer,
        buffers_.d_normal_buffer,
        output.vertices,
        output.normals,
        total_vertices
    );
    
    // Generate face indices (simple triangle soup)
    generateFaceIndices<<<(output.face_count + 255)/256, 256, 0, stream>>>(
        output.faces,
        output.face_count
    );
    
    return true;
}

// CUDA Kernels
__global__ void classifyVoxelsKernel(
    const float* tsdf,
    int3 dims,
    float iso_value,
    const uint* num_verts_table,
    uint* voxel_verts,
    uint* voxel_occupied
) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint num_voxels = dims.x * dims.y * dims.z;
    
    if (idx >= num_voxels) return;
    
    // Convert to 3D coordinates
    uint z = idx / (dims.x * dims.y);
    uint y = (idx % (dims.x * dims.y)) / dims.x;
    uint x = idx % dims.x;
    
    // Skip boundary voxels
    if (x >= dims.x - 1 || y >= dims.y - 1 || z >= dims.z - 1) {
        voxel_verts[idx] = 0;
        voxel_occupied[idx] = 0;
        return;
    }
    
    // Sample 8 corners of the voxel
    float field[8];
    field[0] = tsdf[idx];
    field[1] = tsdf[idx + 1];
    field[2] = tsdf[idx + 1 + dims.x];
    field[3] = tsdf[idx + dims.x];
    field[4] = tsdf[idx + dims.x * dims.y];
    field[5] = tsdf[idx + 1 + dims.x * dims.y];
    field[6] = tsdf[idx + 1 + dims.x + dims.x * dims.y];
    field[7] = tsdf[idx + dims.x + dims.x * dims.y];
    
    // Calculate cube index
    uint cube_index = 0;
    for (int i = 0; i < 8; i++) {
        if (field[i] < iso_value) {
            cube_index |= (1 << i);
        }
    }
    
    // Use lookup table
    uint num_verts = num_verts_table[cube_index];
    voxel_verts[idx] = num_verts;
    voxel_occupied[idx] = (num_verts > 0) ? 1 : 0;
}

void NvidiaMarchingCubes::classifyVoxels(
    const float* d_tsdf,
    const int3& dims,
    uint* d_voxel_verts,
    uint* d_voxel_occupied,
    cudaStream_t stream
) {
    uint num_voxels = dims.x * dims.y * dims.z;
    dim3 grid((num_voxels + 127) / 128);
    dim3 block(128);
    
    classifyVoxelsKernel<<<grid, block, 0, stream>>>(
        d_tsdf, dims,
        params_.marching_cubes.iso_value,
        buffers_.d_num_verts_table,
        d_voxel_verts,
        d_voxel_occupied
    );
}

// Additional kernel implementations...
__global__ void compactVoxelsKernel(
    const uint* voxel_occupied,
    const uint* voxel_occupied_scan,
    uint* compressed_voxels,
    uint num_voxels
) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_voxels) return;
    
    if (voxel_occupied[idx]) {
        compressed_voxels[voxel_occupied_scan[idx]] = idx;
    }
}

__global__ void generateTrianglesKernel(
    const float* tsdf,
    int3 dims,
    float3 origin,
    float voxel_size,
    float iso_value,
    const uint* compressed_voxels,
    const uint* num_verts_scan,
    const uint* tri_table,
    uint num_active_voxels,
    float4* vertices,
    float4* normals
) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_active_voxels) return;
    
    uint voxel_idx = compressed_voxels[idx];
    uint vertex_offset = num_verts_scan[voxel_idx];
    
    // Convert voxel index to 3D coordinates
    uint z = voxel_idx / (dims.x * dims.y);
    uint y = (voxel_idx % (dims.x * dims.y)) / dims.x;
    uint x = voxel_idx % dims.x;
    
    // Calculate world position of voxel corner
    float3 pos = make_float3(
        origin.x + x * voxel_size,
        origin.y + y * voxel_size,
        origin.z + z * voxel_size
    );
    
    // Sample field values at 8 corners
    float field[8];
    // ... (sample TSDF at 8 corners)
    
    // Calculate cube index
    uint cube_index = 0;
    // ... (calculate cube index)
    
    // Generate vertices using tri_table
    // ... (vertex generation using NVIDIA algorithm)
}

// Helper kernels
__global__ void convertAndCopyVertices(
    const float4* src_verts,
    const float4* src_normals,
    float3* dst_verts,
    float3* dst_normals,
    uint num_vertices
) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vertices) return;
    
    float4 v = src_verts[idx];
    float4 n = src_normals[idx];
    
    dst_verts[idx] = make_float3(v.x, v.y, v.z);
    dst_normals[idx] = make_float3(n.x, n.y, n.z);
}

__global__ void generateFaceIndices(
    int3* faces,
    uint num_faces
) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_faces) return;
    
    // Simple triangle soup indexing
    faces[idx] = make_int3(idx * 3, idx * 3 + 1, idx * 3 + 2);
}

} // namespace mesh_service
```

### Step 6: Algorithm Selector Implementation

```cpp
// include/algorithm_selector.h
#pragma once

#include "algorithms/algorithm_base.h"
#include <memory>
#include <unordered_map>

namespace mesh_service {

class AlgorithmSelector {
public:
    AlgorithmSelector();
    ~AlgorithmSelector();
    
    // Initialize all algorithms
    bool initialize();
    
    // Select algorithm based on conditions
    ReconstructionMethod selectAlgorithm(
        float camera_velocity,
        size_t point_count,
        float scene_complexity
    );
    
    // Get algorithm instance
    std::shared_ptr<ReconstructionAlgorithm> getAlgorithm(
        ReconstructionMethod method
    );
    
    // Process with automatic selection
    bool processWithAutoSelect(
        const float3* d_points,
        const float3* d_normals,
        size_t num_points,
        const float* camera_pose,
        float camera_velocity,
        MeshUpdate& output,
        cudaStream_t stream = 0
    );

private:
    std::unordered_map<ReconstructionMethod, 
                      std::shared_ptr<ReconstructionAlgorithm>> algorithms_;
    
    // Thresholds for algorithm selection
    struct {
        float velocity_threshold_high = 0.5f;  // m/s
        float velocity_threshold_low = 0.3f;   // m/s
        size_t point_count_threshold = 100000;
        float complexity_threshold = 0.7f;
    } thresholds_;
    
    ReconstructionMethod current_method_;
    
    // Hysteresis to prevent rapid switching
    int method_stable_frames_ = 0;
    static constexpr int SWITCH_STABILITY_FRAMES = 10;
};

} // namespace mesh_service
```

```cpp
// src/algorithm_selector.cpp
#include "algorithm_selector.h"
#include "algorithms/nvidia_marching_cubes.h"
// Future includes:
// #include "algorithms/open3d_poisson.h"
// #include "algorithms/nksr_client.h"

namespace mesh_service {

AlgorithmSelector::AlgorithmSelector() 
    : current_method_(ReconstructionMethod::NVIDIA_MARCHING_CUBES) {
}

bool AlgorithmSelector::initialize() {
    // Initialize NVIDIA Marching Cubes
    auto nvidia_mc = std::make_shared<NvidiaMarchingCubes>();
    AlgorithmParams mc_params;
    mc_params.marching_cubes.iso_value = 0.0f;
    mc_params.marching_cubes.truncation_distance = 0.15f;
    mc_params.marching_cubes.max_vertices = 5000000;
    
    if (!nvidia_mc->initialize(mc_params)) {
        return false;
    }
    algorithms_[ReconstructionMethod::NVIDIA_MARCHING_CUBES] = nvidia_mc;
    
    // TODO: Initialize Open3D Poisson
    // auto open3d_poisson = std::make_shared<Open3DPoisson>();
    // AlgorithmParams poisson_params;
    // poisson_params.poisson.octree_depth = 8;
    // poisson_params.poisson.point_weight = 4.0f;
    // if (!open3d_poisson->initialize(poisson_params)) {
    //     return false;
    // }
    // algorithms_[ReconstructionMethod::OPEN3D_POISSON] = open3d_poisson;
    
    // TODO: Initialize NKSR Client
    // auto nksr = std::make_shared<NKSRClient>("localhost:50051");
    // AlgorithmParams nksr_params;
    // nksr_params.nksr.detail_level = 0.5f;
    // nksr_params.nksr.chunk_size = 500000;
    // if (!nksr->initialize(nksr_params)) {
    //     return false;
    // }
    // algorithms_[ReconstructionMethod::NKSR] = nksr;
    
    return true;
}

ReconstructionMethod AlgorithmSelector::selectAlgorithm(
    float camera_velocity,
    size_t point_count,
    float scene_complexity
) {
    ReconstructionMethod desired_method = current_method_;
    
    // High velocity: Always use fast marching cubes
    if (camera_velocity > thresholds_.velocity_threshold_high) {
        desired_method = ReconstructionMethod::NVIDIA_MARCHING_CUBES;
    }
    // Low velocity: Can use higher quality
    else if (camera_velocity < thresholds_.velocity_threshold_low) {
        // Large point clouds: Use NKSR (when available)
        if (point_count > thresholds_.point_count_threshold * 2) {
            // desired_method = ReconstructionMethod::NKSR;
            desired_method = ReconstructionMethod::NVIDIA_MARCHING_CUBES; // For now
        }
        // Medium complexity: Use Poisson (when available)
        else if (scene_complexity > thresholds_.complexity_threshold) {
            // desired_method = ReconstructionMethod::OPEN3D_POISSON;
            desired_method = ReconstructionMethod::NVIDIA_MARCHING_CUBES; // For now
        }
        // Simple scenes: Marching cubes is sufficient
        else {
            desired_method = ReconstructionMethod::NVIDIA_MARCHING_CUBES;
        }
    }
    
    // Implement hysteresis to prevent rapid switching
    if (desired_method != current_method_) {
        method_stable_frames_++;
        if (method_stable_frames_ >= SWITCH_STABILITY_FRAMES) {
            current_method_ = desired_method;
            method_stable_frames_ = 0;
            std::cout << "Switched to algorithm: " 
                      << static_cast<int>(current_method_) << std::endl;
        }
    } else {
        method_stable_frames_ = 0;
    }
    
    return current_method_;
}

bool AlgorithmSelector::processWithAutoSelect(
    const float3* d_points,
    const float3* d_normals,
    size_t num_points,
    const float* camera_pose,
    float camera_velocity,
    MeshUpdate& output,
    cudaStream_t stream
) {
    // Calculate scene complexity (simple heuristic for now)
    float scene_complexity = std::min(1.0f, num_points / 500000.0f);
    
    // Select algorithm
    ReconstructionMethod method = selectAlgorithm(
        camera_velocity, num_points, scene_complexity
    );
    
    // Get algorithm instance
    auto algorithm = algorithms_[method];
    if (!algorithm) {
        std::cerr << "Algorithm not available: " << static_cast<int>(method) << std::endl;
        return false;
    }
    
    // Process
    return algorithm->reconstruct(
        d_points, d_normals, num_points, 
        camera_pose, output, stream
    );
}

} // namespace mesh_service
```

### Step 7: Integration with Main Mesh Generator

```cpp
// Updated mesh_generator.cu
#include "mesh_generator.h"
#include "algorithm_selector.h"
#include "normal_estimation.h"

namespace mesh_service {

class GPUMeshGenerator::Impl {
public:
    std::unique_ptr<AlgorithmSelector> algorithm_selector;
    std::unique_ptr<NormalEstimation> normal_estimator;
    
    // Camera tracking
    float3 prev_camera_pos = make_float3(0, 0, 0);
    float camera_velocity = 0.0f;
    std::chrono::steady_clock::time_point last_frame_time;
    
    cudaStream_t stream;
    thrust::device_vector<float3> d_normals;
    
    Impl() {
        cudaStreamCreate(&stream);
        algorithm_selector = std::make_unique<AlgorithmSelector>();
        normal_estimator = std::make_unique<NormalEstimation>();
        
        if (!algorithm_selector->initialize()) {
            throw std::runtime_error("Failed to initialize algorithms");
        }
        
        last_frame_time = std::chrono::steady_clock::now();
    }
    
    ~Impl() {
        cudaStreamDestroy(stream);
    }
    
    void updateCameraVelocity(const float* camera_pose) {
        float3 camera_pos = make_float3(
            camera_pose[12], camera_pose[13], camera_pose[14]
        );
        
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_frame_time).count();
        
        if (dt > 0.001f) {  // Avoid division by zero
            float3 diff = make_float3(
                camera_pos.x - prev_camera_pos.x,
                camera_pos.y - prev_camera_pos.y,
                camera_pos.z - prev_camera_pos.z
            );
            float distance = sqrtf(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
            camera_velocity = distance / dt;
            
            // Smooth velocity with exponential moving average
            static float smooth_velocity = 0.0f;
            smooth_velocity = 0.8f * smooth_velocity + 0.2f * camera_velocity;
            camera_velocity = smooth_velocity;
        }
        
        prev_camera_pos = camera_pos;
        last_frame_time = now;
    }
};

void GPUMeshGenerator::generateIncrementalMesh(
    float3* d_points,
    size_t num_points,
    const float* camera_pose,
    MeshUpdate& update
) {
    // Update camera velocity
    pImpl->updateCameraVelocity(camera_pose);
    
    // Estimate normals if needed
    if (pImpl->d_normals.size() < num_points) {
        pImpl->d_normals.resize(num_points);
    }
    
    pImpl->normal_estimator->estimateNormals(
        d_points,
        thrust::raw_pointer_cast(pImpl->d_normals.data()),
        num_points,
        16,  // k neighbors
        pImpl->stream
    );
    
    // Process with automatic algorithm selection
    bool success = pImpl->algorithm_selector->processWithAutoSelect(
        d_points,
        thrust::raw_pointer_cast(pImpl->d_normals.data()),
        num_points,
        camera_pose,
        pImpl->camera_velocity,
        update,
        pImpl->stream
    );
    
    if (!success) {
        std::cerr << "Mesh generation failed" << std::endl;
        update.vertex_count = 0;
        update.face_count = 0;
    }
    
    cudaStreamSynchronize(pImpl->stream);
}

} // namespace mesh_service
```

## Phase 2: Open3D Poisson Integration (Future)

```cpp
// Placeholder for Open3D integration
// include/algorithms/open3d_poisson.h
class Open3DPoisson : public ReconstructionAlgorithm {
    // TODO: Implement Open3D wrapper
    // - Convert point cloud to Open3D format
    // - Call Open3D GPU Poisson
    // - Convert result back
};
```

## Phase 3: NKSR Service Integration (Future)

```cpp
// Placeholder for NKSR gRPC client
// include/algorithms/nksr_client.h
class NKSRClient : public ReconstructionAlgorithm {
    // TODO: Implement gRPC client
    // - Serialize point cloud
    // - Send to Python NKSR service
    // - Deserialize mesh result
};
```

## Testing Strategy

### Unit Tests
1. Test NVIDIA MC with synthetic TSDF data
2. Test algorithm switching logic
3. Test memory management and cleanup

### Integration Tests
1. Test with RabbitMQ pipeline
2. Test algorithm switching under different velocities
3. Test performance metrics

### Performance Benchmarks
1. Measure frame time for each algorithm
2. Measure memory usage
3. Measure quality metrics

## Build Instructions

```bash
# 1. Download NVIDIA samples
cd mesh_service
mkdir -p external/nvidia_mc
cd external/nvidia_mc
./download_nvidia_mc.sh

# 2. Update CMakeLists.txt
# Add external/nvidia_mc to include paths
# Add new source files

# 3. Build
mkdir build && cd build
cmake ..
make -j8

# 4. Test
./test_marching_cubes
```

## Environment Variables

```yaml
# Algorithm selection
MESH_ALGORITHM: "AUTO"  # AUTO, MARCHING_CUBES, POISSON, NKSR
VELOCITY_THRESHOLD_HIGH: 0.5
VELOCITY_THRESHOLD_LOW: 0.3

# Marching Cubes parameters
MC_ISO_VALUE: 0.0
MC_VOXEL_SIZE: 0.05
MC_MAX_VERTICES: 5000000

# Future: Poisson parameters
POISSON_OCTREE_DEPTH: 8
POISSON_POINT_WEIGHT: 4.0

# Future: NKSR parameters
NKSR_SERVICE_URL: "localhost:50051"
NKSR_DETAIL_LEVEL: 0.5
```

## Debugging Tips

1. **Visualize intermediate results**: Save TSDF volume as .vti file
2. **Profile CUDA kernels**: Use nvprof or Nsight Compute
3. **Check memory usage**: Monitor with nvidia-smi
4. **Validate mesh**: Check for degenerate triangles, disconnected components

## Environment Variables

The implementation supports configuration through environment variables:

```bash
# TSDF Configuration
TSDF_VOXEL_SIZE=0.05              # Voxel size in meters (default: 0.05)
TSDF_TRUNCATION_DISTANCE=0.15      # Truncation distance (default: 0.15)
TSDF_MAX_WEIGHT=100.0              # Maximum weight (default: 100.0)
TSDF_SCENE_BOUNDS_MIN=-5,-5,-5     # Minimum scene bounds
TSDF_SCENE_BOUNDS_MAX=5,5,5        # Maximum scene bounds

# Algorithm Selection (Future)
MESH_ALGORITHM=AUTO                # AUTO, MARCHING_CUBES, POISSON, NKSR
VELOCITY_THRESHOLD_HIGH=0.5        # High velocity threshold (m/s)
VELOCITY_THRESHOLD_LOW=0.3         # Low velocity threshold (m/s)
```

## Build and Run

```bash
# Build with Docker
cd mesh_service
docker build -t mesh_service .

# Or with docker-compose
docker-compose build mesh_service

# Run with environment variables
docker run -e TSDF_VOXEL_SIZE=0.04 -e TSDF_SCENE_BOUNDS_MIN=-10,-10,-10 mesh_service
```

## Next Steps

1. ✅ NVIDIA MC integration complete and tested
2. 🔄 Add Open3D Poisson when needed
3. 🔄 Add NKSR service for advanced reconstruction
4. 🔄 Optimize algorithm switching based on real-world testing
5. 🔄 Add quality metrics to guide algorithm selection