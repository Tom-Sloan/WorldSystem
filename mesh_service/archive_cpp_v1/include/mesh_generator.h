#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include "shared_memory.h"

namespace mesh_service {

enum class MeshMethod {
    INCREMENTAL_POISSON,  // Primary - best for streaming
    TSDF_MARCHING_CUBES  // Fast preview
};

struct MeshUpdate {
    std::string keyframe_id;
    uint64_t timestamp_ns;
    std::vector<float> vertices;    // x,y,z triplets
    std::vector<uint32_t> faces;    // Triangle indices
    std::vector<uint8_t> vertex_colors;  // RGB triplets
    std::vector<int> updated_regions;    // Octree node IDs that changed
};

class GPUMeshGenerator {
public:
    GPUMeshGenerator();
    ~GPUMeshGenerator();
    
    // Generate incremental mesh from keyframe
    void generateIncrementalMesh(
        const SharedKeyframe* keyframe,
        MeshUpdate& update
    );
    
    // Generate mesh from point cloud directly
    void generateMesh(
        float3* d_points,
        size_t num_points,
        const float* camera_pose,
        MeshUpdate& update
    );
    
    // Set mesh generation method
    void setMethod(MeshMethod method);
    
    // Set quality parameters
    void setQualityAdaptive(bool adaptive);
    void setSimplificationRatio(float ratio);
    
    // Get current camera velocity (for adaptive quality)
    void updateCameraVelocity(const float pose_matrix[16]);
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
    
    // Internal mesh generation methods
    void generateIncrementalPoissonMesh(
        const SharedKeyframe* keyframe,
        float3* d_points,
        float3* d_normals,
        MeshUpdate& update
    );
    
    
    void generateMarchingCubesMesh(
        const SharedKeyframe* keyframe,
        float3* d_points,
        float3* d_normals,
        MeshUpdate& update
    );
};

// CUDA kernels
namespace kernels {

__global__ void updateOctreeRegions(
    float3* new_points,
    int num_points,
    void* octree_nodes,
    int* dirty_flags
);

__global__ void processIPSRBlock(
    float3* points,
    uint32_t num_points,
    void* block,
    float* d_implicit_function
);

__device__ uint64_t computeSpatialHash(float3 min, float3 max);

} // namespace kernels
} // namespace mesh_service