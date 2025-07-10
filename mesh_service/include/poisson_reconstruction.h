#pragma once

#include <vector>
#include <memory>
#include <cuda_runtime.h>

namespace mesh_service {

// Forward declaration
struct MeshUpdate;

// Poisson Surface Reconstruction implementation
class PoissonReconstruction {
public:
    struct Parameters {
        int octree_depth = 8;          // Controls reconstruction detail level
        float samples_per_node = 1.5f;  // Minimum samples per octree node
        float point_weight = 4.0f;      // Importance of interpolation at points
        int solver_iterations = 8;      // Conjugate gradient iterations
        float confidence_threshold = 0.1f; // Minimum confidence for surface
        bool adaptive = true;           // Enable adaptive octree refinement
    };

    PoissonReconstruction();
    ~PoissonReconstruction();

    // Set reconstruction parameters
    void setParameters(const Parameters& params);

    // Main reconstruction function
    void reconstruct(
        const float* points,
        const float* normals,
        size_t num_points,
        MeshUpdate& output
    );

    // GPU-accelerated version using CUDA
    void reconstructGPU(
        float3* d_points,
        float3* d_normals,
        size_t num_points,
        cudaStream_t stream,
        MeshUpdate& output
    );

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Incremental Poisson Surface Reconstruction for streaming
class IncrementalPoissonReconstruction {
public:
    struct Block {
        float center[3];
        float size;
        std::vector<size_t> point_indices;
        bool is_dirty = true;
        uint64_t last_update = 0;
    };

    IncrementalPoissonReconstruction();
    ~IncrementalPoissonReconstruction();

    // Initialize spatial grid
    void initialize(float scene_size, int grid_resolution);

    // Add new points incrementally
    void addPoints(
        const float* points,
        const float* normals,
        size_t num_points,
        uint64_t timestamp
    );

    // Update only dirty blocks
    void updateDirtyBlocks(MeshUpdate& output);

    // Force full reconstruction
    void forceFullUpdate(MeshUpdate& output);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace mesh_service