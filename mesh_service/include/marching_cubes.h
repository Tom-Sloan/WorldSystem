#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>

namespace mesh_service {

struct MeshUpdate;

// GPU-accelerated Marching Cubes implementation
class MarchingCubesGPU {
public:
    struct Parameters {
        float voxel_size = 0.05f;      // Size of each voxel in meters
        float iso_value = 0.0f;        // Isosurface value to extract
        float truncation_distance = 0.1f; // TSDF truncation distance
        bool use_color_weighting = true;
        int max_voxels_per_dim = 512;  // Maximum voxel grid dimension
    };

    MarchingCubesGPU();
    ~MarchingCubesGPU();

    // Set parameters
    void setParameters(const Parameters& params);

    // Initialize voxel grid
    void initializeGrid(float3 min_bounds, float3 max_bounds);

    // Integrate points into TSDF
    void integrateTSDF(
        float3* d_points,
        float3* d_normals,
        size_t num_points,
        const float* camera_pose,
        cudaStream_t stream
    );

    // Extract mesh using marching cubes
    void extractMesh(
        MeshUpdate& output,
        cudaStream_t stream
    );

    // Reset grid
    void reset();

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Incremental TSDF fusion with marching cubes
class IncrementalTSDFFusion {
public:
    struct VoxelBlock {
        static constexpr int BLOCK_SIZE = 8;  // 8x8x8 voxels per block
        int3 block_pos;                       // Block position in voxel coordinates
        float* tsdf_values = nullptr;         // TSDF values
        float* weights = nullptr;             // Integration weights
        uint8_t* colors = nullptr;            // RGB colors
        bool allocated = false;
        bool needs_update = false;
        uint64_t last_update = 0;
    };

    IncrementalTSDFFusion();
    ~IncrementalTSDFFusion();

    // Set voxel size
    void setVoxelSize(float size);

    // Allocate blocks on demand
    void allocateBlock(int3 block_pos, cudaStream_t stream);

    // Integrate new depth data
    void integrateDepthMap(
        float* d_depth,
        int width, int height,
        const float* intrinsics,
        const float* pose,
        cudaStream_t stream
    );

    // Integrate point cloud
    void integratePointCloud(
        float3* d_points,
        size_t num_points,
        const float* pose,
        cudaStream_t stream
    );

    // Extract mesh from dirty blocks
    void extractMeshIncremental(
        MeshUpdate& output,
        cudaStream_t stream
    );

    // Get memory usage
    size_t getMemoryUsage() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// CUDA kernels for marching cubes
namespace cuda {

// Marching cubes lookup tables
extern __constant__ int edge_table[256];
extern __constant__ int tri_table[256][16];
extern __constant__ int vertex_offset[8][3];

// TSDF integration kernel
__global__ void integrateTSDFKernel(
    float* tsdf_volume,
    float* weight_volume,
    int3 volume_size,
    float voxel_size,
    float truncation_distance,
    float3* points,
    int num_points,
    float* world_to_volume
);

// Marching cubes kernel
__global__ void marchingCubesKernel(
    float* tsdf_volume,
    int3 volume_size,
    float iso_value,
    float voxel_size,
    float3 volume_origin,
    float* vertices,
    int* triangles,
    int* vertex_count,
    int* triangle_count,
    int max_vertices
);

// Normal estimation kernel
__global__ void computeNormalsKernel(
    float* tsdf_volume,
    int3 volume_size,
    float voxel_size,
    float* vertices,
    float* normals,
    int num_vertices
);

} // namespace cuda

} // namespace mesh_service