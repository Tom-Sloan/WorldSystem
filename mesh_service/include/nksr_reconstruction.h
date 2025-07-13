#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include "gpu_octree.h"
#include "shared_memory.h"

namespace mesh_service {

// Neural Kernel Surface Reconstruction
// Based on NVIDIA's 2023 paper: "Neural Kernel Surface Reconstruction"
// Optimized for out-of-core processing of large scenes

// Kernel function types for NKSR
enum class KernelType {
    WENDLAND_C2,     // Compact support, C2 continuous
    GAUSSIAN_RBF,    // Gaussian radial basis function
    THIN_PLATE,      // Thin plate spline
    NEURAL_ADAPTIVE  // Learned kernel from neural network
};

// Neural kernel parameters
struct NeuralKernelParams {
    float support_radius = 0.1f;     // Kernel support radius
    float regularization = 0.001f;   // Regularization strength
    int polynomial_degree = 1;       // Polynomial degree for reproduction
    bool use_confidence = true;      // Use point confidence weights
    float outlier_threshold = 3.0f;  // Outlier rejection threshold
};

// Chunk for out-of-core processing
struct ProcessingChunk {
    int chunk_id;
    float3 min_bound;
    float3 max_bound;
    
    // Points in this chunk
    thrust::device_vector<float3> d_points;
    thrust::device_vector<float3> d_normals;
    thrust::device_vector<float> d_confidences;
    thrust::device_vector<uint8_t> d_colors;
    
    // Solver data
    thrust::device_vector<float> d_weights;      // RBF weights
    thrust::device_vector<float> d_poly_coeffs;  // Polynomial coefficients
    
    // Status
    bool is_loaded;
    bool is_solved;
    bool needs_update;
    uint32_t last_access;
};

// GPU-accelerated NKSR implementation
class NKSRReconstruction {
public:
    struct Parameters {
        KernelType kernel_type = KernelType::WENDLAND_C2;
        NeuralKernelParams kernel_params;
        
        // Out-of-core settings
        size_t max_gpu_memory = 4ULL * 1024 * 1024 * 1024;  // 4GB
        size_t chunk_size = 1000000;  // Points per chunk
        float chunk_overlap = 0.1f;    // 10% overlap between chunks
        
        // Solver settings
        int max_iterations = 100;
        float convergence_threshold = 1e-6f;
        bool use_multigrid = true;
        int multigrid_levels = 3;
        
        // Mesh extraction
        float iso_value = 0.0f;
        float marching_cubes_resolution = 0.01f;
        bool extract_confidence = true;
    };
    
    NKSRReconstruction();
    ~NKSRReconstruction();
    
    // Initialize the reconstruction system
    void initialize(const Parameters& params);
    
    // Add points to the reconstruction (streaming interface)
    void addPointStream(const float3* points, const float3* normals,
                       const float* confidences, const uint8_t* colors,
                       size_t num_points, cudaStream_t stream = 0);
    
    // Process chunks (can be called asynchronously)
    void processChunks(int num_chunks_to_process = -1);
    
    // Extract mesh from solved chunks
    void extractMesh(std::vector<float>& vertices,
                    std::vector<uint32_t>& faces,
                    std::vector<uint8_t>& colors,
                    std::vector<float>& confidence_map,
                    const float* query_bounds = nullptr);
    
    // Memory management
    size_t getCurrentGPUMemoryUsage() const;
    void evictLRUChunks(size_t bytes_to_free);
    void clearAllChunks();
    
    // Get reconstruction progress
    float getProgress() const;
    size_t getNumProcessedPoints() const { return total_points_processed_; }
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
    
    Parameters params_;
    size_t total_points_processed_;
    
    // Chunk management
    void createChunks(const float3* all_points, size_t num_points);
    void loadChunkToGPU(ProcessingChunk& chunk);
    void evictChunkFromGPU(ProcessingChunk& chunk);
    
    // NKSR solver
    void solveChunk(ProcessingChunk& chunk, cudaStream_t stream);
    void evaluateImplicitFunction(const ProcessingChunk& chunk,
                                 const float3* query_points,
                                 float* values,
                                 size_t num_queries,
                                 cudaStream_t stream);
};

// CUDA kernels for NKSR
namespace nksr_kernels {

// Wendland C2 kernel function and its derivative
__device__ float wendlandC2(float r, float h);
__device__ float wendlandC2Derivative(float r, float h);

// Build system matrix for RBF interpolation
__global__ void buildSystemMatrix(
    const float3* points,
    const float3* normals,
    const float* confidences,
    int num_points,
    float* A_matrix,
    float* b_vector,
    float support_radius,
    float regularization,
    int polynomial_degree
);

// Solve using iterative method (Conjugate Gradient)
__global__ void conjugateGradientStep(
    const float* A_matrix,
    const float* b_vector,
    float* x_vector,
    float* r_vector,
    float* p_vector,
    float* Ap_vector,
    int num_unknowns,
    float* alpha,
    float* beta,
    float* residual_norm
);

// Evaluate implicit function at query points
__global__ void evaluateRBF(
    const float3* rbf_centers,
    const float* rbf_weights,
    const float* poly_coeffs,
    int num_centers,
    int poly_degree,
    const float3* query_points,
    float* output_values,
    int num_queries,
    float support_radius,
    KernelType kernel_type
);

// Extract iso-surface using adaptive marching cubes
__global__ void adaptiveMarchingCubes(
    const ProcessingChunk* chunk,
    float iso_value,
    float base_resolution,
    const float* confidence_field,
    float3* vertices,
    uint3* faces,
    float* vertex_confidence,
    int* vertex_counter,
    int* face_counter,
    int max_vertices
);

// Stitch chunk boundaries for watertight mesh
__global__ void stitchChunkBoundaries(
    const float3* vertices,
    int num_vertices,
    const ProcessingChunk* chunks,
    int num_chunks,
    int* vertex_mapping,
    float epsilon
);

// Neural network kernel evaluation (placeholder for learned kernels)
__device__ float neuralKernel(
    const float3& x1, const float3& x2,
    const float* network_params,
    int param_count
);

// Outlier detection and removal
__global__ void detectOutliers(
    const float3* points,
    const float3* normals,
    int num_points,
    const float3* neighbor_points,
    const int* neighbor_indices,
    const int* neighbor_counts,
    float* outlier_scores,
    float threshold
);

// Confidence estimation from point density and normal consistency
__global__ void estimateConfidence(
    const float3* points,
    const float3* normals,
    int num_points,
    const int* neighbor_counts,
    const float3* neighbor_normals,
    float* confidences,
    float density_sigma,
    float normal_sigma
);

} // namespace nksr_kernels

// Helper class for managing large point clouds
class PointCloudStreamer {
public:
    PointCloudStreamer(size_t chunk_size, float overlap_ratio);
    
    void addPoints(const float3* points, const float3* normals,
                  const float* confidences, size_t num_points);
    
    bool hasNextChunk() const;
    void getNextChunk(ProcessingChunk& chunk);
    
private:
    struct ChunkInfo {
        size_t start_idx;
        size_t end_idx;
        float3 min_bound;
        float3 max_bound;
    };
    
    std::vector<ChunkInfo> chunks_;
    std::vector<float3> all_points_;
    std::vector<float3> all_normals_;
    std::vector<float> all_confidences_;
    
    size_t chunk_size_;
    float overlap_ratio_;
    size_t current_chunk_;
};

} // namespace mesh_service