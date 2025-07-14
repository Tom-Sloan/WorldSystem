#pragma once

#include "algorithm_base.h"
#include "../simple_tsdf.h"
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