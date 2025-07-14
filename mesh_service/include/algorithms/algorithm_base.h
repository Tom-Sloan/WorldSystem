#pragma once

#include <cuda_runtime.h>
#include "../mesh_generator.h" // For MeshUpdate

namespace mesh_service {

enum class ReconstructionMethod {
    NVIDIA_MARCHING_CUBES,
    OPEN3D_POISSON,
    NKSR
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
    
    // Default constructor initializes marching cubes params
    AlgorithmParams() {
        marching_cubes.iso_value = 0.0f;
        marching_cubes.truncation_distance = 0.15f;
        marching_cubes.max_vertices = 5000000;
    }
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

} // namespace mesh_service