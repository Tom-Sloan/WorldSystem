#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include "gpu_octree.h"
#include "shared_memory.h"

namespace mesh_service {

// Block size for incremental processing
constexpr int POISSON_BLOCK_SIZE = 256;
constexpr int POISSON_BLOCK_OVERLAP = 32;  // Overlap for smooth boundaries

// Poisson octree node with function values
struct PoissonNode {
    float3 center;
    float size;
    
    // Function values at corners (8 values)
    float corner_values[8];
    
    // Gradient at center
    float3 gradient;
    
    // Weight (confidence)
    float weight;
    
    // For incremental updates
    uint32_t last_update;
    bool needs_update;
    
    // Links to octree structure
    int octree_node_idx;
};

// Block structure for incremental processing
struct PoissonBlock {
    int3 block_coord;  // Block coordinates in grid
    float3 min_bound;
    float3 max_bound;
    
    // Nodes in this block
    int node_start;
    int node_count;
    
    // Neighbor blocks (6-connected)
    int neighbors[6];
    
    // Status flags
    bool is_dirty;
    bool is_processing;
    uint32_t last_update;
};

// GPU-accelerated Incremental Poisson Surface Reconstruction
class GPUPoissonReconstruction {
public:
    struct Parameters {
        int octree_depth = 8;
        float point_weight = 4.0f;
        float samples_per_node = 1.5f;
        int solver_iterations = 8;
        float iso_value = 0.0f;
        bool adaptive = true;
        int block_size = POISSON_BLOCK_SIZE;
        float confidence_threshold = 0.1f;
    };
    
    GPUPoissonReconstruction();
    ~GPUPoissonReconstruction();
    
    // Initialize the reconstruction volume
    void initialize(float scene_size, int grid_resolution);
    
    // Set parameters
    void setParameters(const Parameters& params) { params_ = params; }
    
    // Incremental reconstruction from new points
    void addPoints(const float3* points, const float3* normals, 
                  const uint8_t* colors, int num_points,
                  GPUOctree* octree, cudaStream_t stream = 0);
    
    // Extract mesh from implicit function
    void extractMesh(std::vector<float>& vertices,
                    std::vector<uint32_t>& faces,
                    std::vector<uint8_t>& colors,
                    const std::vector<int>& dirty_blocks,
                    cudaStream_t stream = 0);
    
    // Get dirty blocks that need mesh extraction
    void getDirtyBlocks(std::vector<int>& dirty_block_indices);
    void clearDirtyBlocks(const std::vector<int>& cleared_blocks);
    
private:
    Parameters params_;
    
    // Device memory for Poisson nodes
    PoissonNode* d_nodes;
    int max_nodes;
    int node_count;
    
    // Block management
    PoissonBlock* d_blocks;
    int3 grid_dims;
    int total_blocks;
    
    // Solver data
    float* d_divergence;      // Divergence field
    float* d_solution;        // Solution (implicit function)
    float* d_weights;         // Point weights
    float3* d_gradients;      // Gradient field
    
    // Temporary buffers for solver
    float* d_r;  // Residual
    float* d_p;  // Search direction
    float* d_Ap; // A * p
    
    // For mesh extraction
    float3* d_mesh_vertices;
    uint32_t* d_mesh_faces;
    uint8_t* d_mesh_colors;
    int* d_vertex_counter;
    int* d_face_counter;
    
    // Memory pool
    void* solver_memory_pool;
    size_t solver_pool_size;
    
    // Helper methods
    void allocateMemory(float scene_size);
    void freeMemory();
    void updateDivergenceField(const float3* points, const float3* normals,
                              int num_points, const std::vector<int>& affected_blocks,
                              cudaStream_t stream);
    void solvePoisson(const std::vector<int>& blocks_to_solve, cudaStream_t stream);
    void extractMeshFromBlocks(const std::vector<int>& blocks, cudaStream_t stream);
};

// CUDA kernels for Poisson reconstruction
namespace poisson_kernels {

// Compute divergence field from oriented points
__global__ void computeDivergence(
    const float3* points,
    const float3* normals,
    int num_points,
    const PoissonNode* nodes,
    int num_nodes,
    float* divergence,
    float3* gradients,
    float* weights,
    float point_weight
);

// Update divergence incrementally
__global__ void updateDivergenceIncremental(
    const float3* new_points,
    const float3* new_normals,
    int num_new_points,
    const PoissonBlock* blocks,
    const int* affected_block_indices,
    int num_affected_blocks,
    PoissonNode* nodes,
    float* divergence,
    float3* gradients,
    float* weights,
    float point_weight
);

// Conjugate gradient solver step
__global__ void cgSolverStep(
    const float* divergence,
    float* solution,
    float* residual,
    float* p,
    float* Ap,
    const PoissonNode* nodes,
    const PoissonBlock* blocks,
    const int* active_blocks,
    int num_active_blocks,
    float* alpha,
    float* beta,
    float* residual_norm
);

// Apply Laplacian operator
__global__ void applyLaplacian(
    const float* x,
    float* Ax,
    const PoissonNode* nodes,
    const PoissonBlock* blocks,
    const int* active_blocks,
    int num_active_blocks
);

// Extract iso-surface using marching cubes
__global__ void extractIsoSurface(
    const float* solution,
    const PoissonNode* nodes,
    const PoissonBlock* blocks,
    const int* blocks_to_extract,
    int num_blocks,
    float iso_value,
    float3* vertices,
    uint32_t* faces,
    int* vertex_counter,
    int* face_counter
);

// Stitch block boundaries
__global__ void stitchBlockBoundaries(
    float3* vertices,
    uint32_t* faces,
    int num_vertices,
    const PoissonBlock* blocks,
    const int* boundary_blocks,
    int num_boundary_blocks,
    float epsilon
);

// Helper device functions
__device__ float basisFunction(float t);
__device__ float3 basisGradient(const float3& p, const float3& center, float size);
__device__ float evaluateFunction(const float3& p, const PoissonNode* nodes, 
                                 const float* solution, int num_nodes);
__device__ int findContainingNode(const float3& p, const PoissonNode* nodes, 
                                  int num_nodes);

} // namespace poisson_kernels

} // namespace mesh_service