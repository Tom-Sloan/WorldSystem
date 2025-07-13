#include "nksr_reconstruction.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <queue>
#include <chrono>

namespace mesh_service {

// Constants
constexpr int THREADS_PER_BLOCK = 256;
constexpr float PI = 3.14159265358979323846f;

// Wendland C2 kernel implementation
__device__ float nksr_kernels::wendlandC2(float r, float h) {
    if (r >= h) return 0.0f;
    float t = r / h;
    float val = 1.0f - t;
    val = val * val * val * val;  // (1-t)^4
    val *= (4.0f * t + 1.0f);
    return val;
}

__device__ float nksr_kernels::wendlandC2Derivative(float r, float h) {
    if (r >= h || r < 1e-10f) return 0.0f;
    float t = r / h;
    float val = -20.0f * t * (1.0f - t) * (1.0f - t) * (1.0f - t) / h;
    return val;
}

// Build system matrix for RBF interpolation
__global__ void nksr_kernels::buildSystemMatrix(
    const float3* points,
    const float3* normals,
    const float* confidences,
    int num_points,
    float* A_matrix,
    float* b_vector,
    float support_radius,
    float regularization,
    int polynomial_degree) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_points) return;
    
    float3 p_i = points[row];
    float3 n_i = normals[row];
    float conf_i = confidences ? confidences[row] : 1.0f;
    
    // Build row of A matrix
    for (int col = 0; col < num_points; col++) {
        float3 p_j = points[col];
        float3 diff = make_float3(p_j.x - p_i.x, p_j.y - p_i.y, p_j.z - p_i.z);
        float r = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
        
        float kernel_val = wendlandC2(r, support_radius);
        
        // Add regularization on diagonal
        if (row == col) {
            kernel_val += regularization / conf_i;
        }
        
        A_matrix[row * num_points + col] = kernel_val * conf_i;
    }
    
    // Build polynomial part (if degree >= 0)
    if (polynomial_degree >= 0) {
        // Constant term
        A_matrix[row * num_points + num_points] = conf_i;
        A_matrix[(num_points + 1) * num_points + row] = conf_i;
    }
    if (polynomial_degree >= 1) {
        // Linear terms
        A_matrix[row * num_points + num_points + 1] = p_i.x * conf_i;
        A_matrix[row * num_points + num_points + 2] = p_i.y * conf_i;
        A_matrix[row * num_points + num_points + 3] = p_i.z * conf_i;
        
        A_matrix[(num_points + 1) * num_points + row] = p_i.x * conf_i;
        A_matrix[(num_points + 2) * num_points + row] = p_i.y * conf_i;
        A_matrix[(num_points + 3) * num_points + row] = p_i.z * conf_i;
    }
    
    // Build b vector (normal constraints)
    // We want gradient of implicit function to match normals
    b_vector[row] = 0.0f;  // f(p_i) = 0 for points on surface
    
    // Add gradient constraints
    if (row < num_points / 2) {  // Use half points for gradient constraints
        float offset = support_radius * 0.1f;
        float3 p_plus = make_float3(
            p_i.x + offset * n_i.x,
            p_i.y + offset * n_i.y,
            p_i.z + offset * n_i.z
        );
        
        // Evaluate gradient at offset point
        b_vector[num_points + row] = offset * conf_i;  // Target value at offset
    }
}

// Iterative conjugate gradient solver
__global__ void nksr_kernels::conjugateGradientStep(
    const float* A_matrix,
    const float* b_vector,
    float* x_vector,
    float* r_vector,
    float* p_vector,
    float* Ap_vector,
    int num_unknowns,
    float* alpha,
    float* beta,
    float* residual_norm) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_unknowns) return;
    
    // Compute Ap = A * p
    float sum = 0.0f;
    for (int j = 0; j < num_unknowns; j++) {
        sum += A_matrix[tid * num_unknowns + j] * p_vector[j];
    }
    Ap_vector[tid] = sum;
    
    __syncthreads();
    
    // Compute alpha = (r^T * r) / (p^T * Ap)
    if (tid == 0) {
        float rTr = 0.0f;
        float pTAp = 0.0f;
        
        for (int i = 0; i < num_unknowns; i++) {
            rTr += r_vector[i] * r_vector[i];
            pTAp += p_vector[i] * Ap_vector[i];
        }
        
        *alpha = rTr / (pTAp + 1e-10f);
    }
    
    __syncthreads();
    
    float alpha_val = *alpha;
    
    // Update x and r
    x_vector[tid] += alpha_val * p_vector[tid];
    r_vector[tid] -= alpha_val * Ap_vector[tid];
    
    __syncthreads();
    
    // Compute new residual norm and beta
    if (tid == 0) {
        float new_rTr = 0.0f;
        float old_rTr = *residual_norm * *residual_norm;
        
        for (int i = 0; i < num_unknowns; i++) {
            new_rTr += r_vector[i] * r_vector[i];
        }
        
        *residual_norm = sqrtf(new_rTr);
        *beta = new_rTr / (old_rTr + 1e-10f);
    }
    
    __syncthreads();
    
    // Update p
    float beta_val = *beta;
    p_vector[tid] = r_vector[tid] + beta_val * p_vector[tid];
}

// Evaluate RBF implicit function
__global__ void nksr_kernels::evaluateRBF(
    const float3* rbf_centers,
    const float* rbf_weights,
    const float* poly_coeffs,
    int num_centers,
    int poly_degree,
    const float3* query_points,
    float* output_values,
    int num_queries,
    float support_radius,
    KernelType kernel_type) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_queries) return;
    
    float3 query = query_points[tid];
    float value = 0.0f;
    
    // Sum RBF contributions
    for (int i = 0; i < num_centers; i++) {
        float3 center = rbf_centers[i];
        float3 diff = make_float3(
            query.x - center.x,
            query.y - center.y,
            query.z - center.z
        );
        float r = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
        
        float kernel_val = 0.0f;
        switch (kernel_type) {
            case KernelType::WENDLAND_C2:
                kernel_val = wendlandC2(r, support_radius);
                break;
            case KernelType::GAUSSIAN_RBF:
                kernel_val = expf(-r * r / (support_radius * support_radius));
                break;
            case KernelType::THIN_PLATE:
                kernel_val = (r > 1e-10f) ? r * r * logf(r) : 0.0f;
                break;
            default:
                kernel_val = wendlandC2(r, support_radius);
        }
        
        value += rbf_weights[i] * kernel_val;
    }
    
    // Add polynomial contribution
    if (poly_coeffs != nullptr) {
        if (poly_degree >= 0) {
            value += poly_coeffs[0];  // Constant
        }
        if (poly_degree >= 1) {
            value += poly_coeffs[1] * query.x;
            value += poly_coeffs[2] * query.y;
            value += poly_coeffs[3] * query.z;
        }
    }
    
    output_values[tid] = value;
}

// Confidence estimation based on local point density and normal consistency
__global__ void nksr_kernels::estimateConfidence(
    const float3* points,
    const float3* normals,
    int num_points,
    const int* neighbor_counts,
    const float3* neighbor_normals,
    float* confidences,
    float density_sigma,
    float normal_sigma) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_points) return;
    
    float3 my_normal = normals[tid];
    int num_neighbors = neighbor_counts[tid];
    
    // Density-based confidence
    float expected_neighbors = 30.0f;  // Expected number of neighbors
    float density_conf = expf(-fabsf(num_neighbors - expected_neighbors) / density_sigma);
    
    // Normal consistency confidence
    float normal_conf = 1.0f;
    if (num_neighbors > 0) {
        float avg_dot = 0.0f;
        for (int i = 0; i < num_neighbors; i++) {
            float3 neighbor_normal = neighbor_normals[tid * 32 + i];  // Assuming max 32 neighbors
            float dot_prod = my_normal.x * neighbor_normal.x + 
                           my_normal.y * neighbor_normal.y + 
                           my_normal.z * neighbor_normal.z;
            avg_dot += dot_prod;
        }
        avg_dot /= num_neighbors;
        normal_conf = expf((avg_dot - 1.0f) / normal_sigma);
    }
    
    confidences[tid] = density_conf * normal_conf;
}

// NKSR Implementation class
class NKSRReconstruction::Impl {
public:
    Parameters params;
    
    // Chunk storage
    std::vector<ProcessingChunk> chunks;
    std::queue<int> processing_queue;
    size_t current_gpu_memory = 0;
    
    // CUDA resources
    cublasHandle_t cublas_handle;
    cusparseHandle_t cusparse_handle;
    cudaStream_t compute_stream;
    cudaStream_t transfer_stream;
    
    // Solver workspace
    thrust::device_vector<float> d_A_matrix;
    thrust::device_vector<float> d_b_vector;
    thrust::device_vector<float> d_x_vector;
    thrust::device_vector<float> d_r_vector;
    thrust::device_vector<float> d_p_vector;
    thrust::device_vector<float> d_Ap_vector;
    
    // Output mesh buffers
    thrust::device_vector<float3> d_mesh_vertices;
    thrust::device_vector<uint3> d_mesh_faces;
    thrust::device_vector<float> d_vertex_confidence;
    
    Impl() {
        cublasCreate(&cublas_handle);
        cusparseCreate(&cusparse_handle);
        cudaStreamCreate(&compute_stream);
        cudaStreamCreate(&transfer_stream);
    }
    
    ~Impl() {
        cublasDestroy(cublas_handle);
        cusparseDestroy(cusparse_handle);
        cudaStreamDestroy(compute_stream);
        cudaStreamDestroy(transfer_stream);
    }
    
    void allocateSolverWorkspace(size_t num_points) {
        size_t matrix_size = num_points * num_points;
        if (params.kernel_params.polynomial_degree >= 0) {
            matrix_size += 4 * num_points + 16;  // Extra space for polynomial
        }
        
        d_A_matrix.resize(matrix_size);
        d_b_vector.resize(num_points + 4);
        d_x_vector.resize(num_points + 4);
        d_r_vector.resize(num_points + 4);
        d_p_vector.resize(num_points + 4);
        d_Ap_vector.resize(num_points + 4);
        
        // Initialize
        thrust::fill(d_x_vector.begin(), d_x_vector.end(), 0.0f);
    }
};

// NKSRReconstruction implementation
NKSRReconstruction::NKSRReconstruction() 
    : pImpl(std::make_unique<Impl>()), total_points_processed_(0) {
}

NKSRReconstruction::~NKSRReconstruction() = default;

void NKSRReconstruction::initialize(const Parameters& params) {
    params_ = params;
    pImpl->params = params;
}

void NKSRReconstruction::addPointStream(const float3* points, const float3* normals,
                                       const float* confidences, const uint8_t* colors,
                                       size_t num_points, cudaStream_t stream) {
    // Create chunks from input points
    createChunks(points, num_points);
    
    // Process each chunk
    for (auto& chunk : pImpl->chunks) {
        // Copy data to chunk
        size_t chunk_size = chunk.d_points.size();
        
        chunk.d_points.resize(chunk_size);
        chunk.d_normals.resize(chunk_size);
        
        cudaMemcpyAsync(chunk.d_points.data().get(), 
                       points + chunk.chunk_id * params_.chunk_size,
                       chunk_size * sizeof(float3),
                       cudaMemcpyHostToDevice, stream);
        
        cudaMemcpyAsync(chunk.d_normals.data().get(),
                       normals + chunk.chunk_id * params_.chunk_size,
                       chunk_size * sizeof(float3),
                       cudaMemcpyHostToDevice, stream);
        
        if (confidences) {
            chunk.d_confidences.resize(chunk_size);
            cudaMemcpyAsync(chunk.d_confidences.data().get(),
                           confidences + chunk.chunk_id * params_.chunk_size,
                           chunk_size * sizeof(float),
                           cudaMemcpyHostToDevice, stream);
        } else {
            // Estimate confidences
            chunk.d_confidences.resize(chunk_size);
            thrust::fill(chunk.d_confidences.begin(), chunk.d_confidences.end(), 1.0f);
        }
        
        if (colors) {
            chunk.d_colors.resize(chunk_size * 3);
            cudaMemcpyAsync(chunk.d_colors.data().get(),
                           colors + chunk.chunk_id * params_.chunk_size * 3,
                           chunk_size * 3 * sizeof(uint8_t),
                           cudaMemcpyHostToDevice, stream);
        }
        
        chunk.is_loaded = true;
        chunk.needs_update = true;
        pImpl->processing_queue.push(chunk.chunk_id);
    }
    
    total_points_processed_ += num_points;
}

void NKSRReconstruction::createChunks(const float3* all_points, size_t num_points) {
    // Simple chunking strategy - divide points into fixed-size chunks
    size_t num_chunks = (num_points + params_.chunk_size - 1) / params_.chunk_size;
    
    pImpl->chunks.resize(num_chunks);
    
    for (size_t i = 0; i < num_chunks; i++) {
        ProcessingChunk& chunk = pImpl->chunks[i];
        chunk.chunk_id = i;
        
        size_t start_idx = i * params_.chunk_size;
        size_t end_idx = std::min(start_idx + params_.chunk_size, num_points);
        size_t chunk_points = end_idx - start_idx;
        
        // Compute bounds
        thrust::host_vector<float3> h_points(chunk_points);
        cudaMemcpy(h_points.data(), all_points + start_idx, 
                  chunk_points * sizeof(float3), cudaMemcpyHostToHost);
        
        chunk.min_bound = h_points[0];
        chunk.max_bound = h_points[0];
        
        for (size_t j = 1; j < chunk_points; j++) {
            chunk.min_bound.x = fminf(chunk.min_bound.x, h_points[j].x);
            chunk.min_bound.y = fminf(chunk.min_bound.y, h_points[j].y);
            chunk.min_bound.z = fminf(chunk.min_bound.z, h_points[j].z);
            
            chunk.max_bound.x = fmaxf(chunk.max_bound.x, h_points[j].x);
            chunk.max_bound.y = fmaxf(chunk.max_bound.y, h_points[j].y);
            chunk.max_bound.z = fmaxf(chunk.max_bound.z, h_points[j].z);
        }
        
        // Expand bounds for overlap
        float3 expansion = make_float3(
            (chunk.max_bound.x - chunk.min_bound.x) * params_.chunk_overlap,
            (chunk.max_bound.y - chunk.min_bound.y) * params_.chunk_overlap,
            (chunk.max_bound.z - chunk.min_bound.z) * params_.chunk_overlap
        );
        
        chunk.min_bound.x -= expansion.x;
        chunk.min_bound.y -= expansion.y;
        chunk.min_bound.z -= expansion.z;
        chunk.max_bound.x += expansion.x;
        chunk.max_bound.y += expansion.y;
        chunk.max_bound.z += expansion.z;
        
        chunk.is_loaded = false;
        chunk.is_solved = false;
        chunk.needs_update = true;
        chunk.last_access = 0;
    }
}

void NKSRReconstruction::processChunks(int num_chunks_to_process) {
    if (num_chunks_to_process < 0) {
        num_chunks_to_process = pImpl->processing_queue.size();
    }
    
    int processed = 0;
    while (!pImpl->processing_queue.empty() && processed < num_chunks_to_process) {
        int chunk_id = pImpl->processing_queue.front();
        pImpl->processing_queue.pop();
        
        ProcessingChunk& chunk = pImpl->chunks[chunk_id];
        
        if (!chunk.is_loaded) {
            loadChunkToGPU(chunk);
        }
        
        if (chunk.needs_update) {
            solveChunk(chunk, pImpl->compute_stream);
            chunk.needs_update = false;
            chunk.is_solved = true;
        }
        
        processed++;
        
        // Check memory usage and evict if necessary
        if (getCurrentGPUMemoryUsage() > params_.max_gpu_memory * 0.9f) {
            evictLRUChunks(params_.max_gpu_memory * 0.1f);
        }
    }
}

void NKSRReconstruction::solveChunk(ProcessingChunk& chunk, cudaStream_t stream) {
    size_t num_points = chunk.d_points.size();
    if (num_points == 0) return;
    
    // Allocate solver workspace
    pImpl->allocateSolverWorkspace(num_points);
    
    // Build system matrix
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((num_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    
    nksr_kernels::buildSystemMatrix<<<grid, block, 0, stream>>>(
        chunk.d_points.data().get(),
        chunk.d_normals.data().get(),
        chunk.d_confidences.size() > 0 ? chunk.d_confidences.data().get() : nullptr,
        num_points,
        pImpl->d_A_matrix.data().get(),
        pImpl->d_b_vector.data().get(),
        params_.kernel_params.support_radius,
        params_.kernel_params.regularization,
        params_.kernel_params.polynomial_degree
    );
    
    // Initialize conjugate gradient
    thrust::copy(pImpl->d_b_vector.begin(), pImpl->d_b_vector.end(), pImpl->d_r_vector.begin());
    thrust::copy(pImpl->d_r_vector.begin(), pImpl->d_r_vector.end(), pImpl->d_p_vector.begin());
    
    float residual_norm = thrust::transform_reduce(
        pImpl->d_r_vector.begin(), pImpl->d_r_vector.end(),
        [] __device__ (float x) { return x * x; },
        0.0f, thrust::plus<float>()
    );
    residual_norm = sqrtf(residual_norm);
    
    // Iterative solve
    thrust::device_vector<float> d_alpha(1);
    thrust::device_vector<float> d_beta(1);
    thrust::device_vector<float> d_residual_norm(1);
    d_residual_norm[0] = residual_norm;
    
    for (int iter = 0; iter < params_.max_iterations; iter++) {
        if (residual_norm < params_.convergence_threshold) break;
        
        nksr_kernels::conjugateGradientStep<<<grid, block, 0, stream>>>(
            pImpl->d_A_matrix.data().get(),
            pImpl->d_b_vector.data().get(),
            pImpl->d_x_vector.data().get(),
            pImpl->d_r_vector.data().get(),
            pImpl->d_p_vector.data().get(),
            pImpl->d_Ap_vector.data().get(),
            num_points,
            d_alpha.data().get(),
            d_beta.data().get(),
            d_residual_norm.data().get()
        );
        
        cudaMemcpyAsync(&residual_norm, d_residual_norm.data().get(), sizeof(float),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    
    // Store solution
    chunk.d_weights.resize(num_points);
    thrust::copy(pImpl->d_x_vector.begin(), pImpl->d_x_vector.begin() + num_points,
                chunk.d_weights.begin());
    
    if (params_.kernel_params.polynomial_degree >= 0) {
        chunk.d_poly_coeffs.resize(4);
        thrust::copy(pImpl->d_x_vector.begin() + num_points,
                    pImpl->d_x_vector.begin() + num_points + 4,
                    chunk.d_poly_coeffs.begin());
    }
}

void NKSRReconstruction::extractMesh(std::vector<float>& vertices,
                                    std::vector<uint32_t>& faces,
                                    std::vector<uint8_t>& colors,
                                    std::vector<float>& confidence_map,
                                    const float* query_bounds) {
    // Determine extraction bounds
    float3 min_bound, max_bound;
    if (query_bounds) {
        min_bound = make_float3(query_bounds[0], query_bounds[1], query_bounds[2]);
        max_bound = make_float3(query_bounds[3], query_bounds[4], query_bounds[5]);
    } else {
        // Use bounds from all chunks
        min_bound = pImpl->chunks[0].min_bound;
        max_bound = pImpl->chunks[0].max_bound;
        
        for (const auto& chunk : pImpl->chunks) {
            min_bound.x = fminf(min_bound.x, chunk.min_bound.x);
            min_bound.y = fminf(min_bound.y, chunk.min_bound.y);
            min_bound.z = fminf(min_bound.z, chunk.min_bound.z);
            
            max_bound.x = fmaxf(max_bound.x, chunk.max_bound.x);
            max_bound.y = fmaxf(max_bound.y, chunk.max_bound.y);
            max_bound.z = fmaxf(max_bound.z, chunk.max_bound.z);
        }
    }
    
    // Generate query grid for marching cubes
    float resolution = params_.marching_cubes_resolution;
    int3 grid_size = make_int3(
        (int)((max_bound.x - min_bound.x) / resolution) + 1,
        (int)((max_bound.y - min_bound.y) / resolution) + 1,
        (int)((max_bound.z - min_bound.z) / resolution) + 1
    );
    
    size_t total_queries = grid_size.x * grid_size.y * grid_size.z;
    thrust::device_vector<float3> d_query_points(total_queries);
    thrust::device_vector<float> d_query_values(total_queries);
    
    // Generate query points
    auto query_gen = [min_bound, resolution, grid_size] __device__ (int idx) {
        int x = idx % grid_size.x;
        int y = (idx / grid_size.x) % grid_size.y;
        int z = idx / (grid_size.x * grid_size.y);
        
        return make_float3(
            min_bound.x + x * resolution,
            min_bound.y + y * resolution,
            min_bound.z + z * resolution
        );
    };
    
    thrust::transform(thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(total_queries),
                     d_query_points.begin(),
                     query_gen);
    
    // Evaluate implicit function at query points
    for (const auto& chunk : pImpl->chunks) {
        if (!chunk.is_solved) continue;
        
        thrust::device_vector<float> d_chunk_values(total_queries);
        
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid((total_queries + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        
        nksr_kernels::evaluateRBF<<<grid, block>>>(
            chunk.d_points.data().get(),
            chunk.d_weights.data().get(),
            chunk.d_poly_coeffs.size() > 0 ? chunk.d_poly_coeffs.data().get() : nullptr,
            chunk.d_points.size(),
            params_.kernel_params.polynomial_degree,
            d_query_points.data().get(),
            d_chunk_values.data().get(),
            total_queries,
            params_.kernel_params.support_radius,
            params_.kernel_type
        );
        
        // Accumulate values
        thrust::transform(d_query_values.begin(), d_query_values.end(),
                         d_chunk_values.begin(), d_query_values.begin(),
                         thrust::plus<float>());
    }
    
    // Extract mesh using marching cubes
    // This is simplified - in practice we'd use the enhanced marching cubes
    // For now, just create a simple mesh
    vertices.clear();
    faces.clear();
    
    // Convert grid values to mesh
    thrust::host_vector<float> h_values = d_query_values;
    
    for (int z = 0; z < grid_size.z - 1; z++) {
        for (int y = 0; y < grid_size.y - 1; y++) {
            for (int x = 0; x < grid_size.x - 1; x++) {
                // Check if iso-surface crosses this voxel
                int idx = x + y * grid_size.x + z * grid_size.x * grid_size.y;
                
                float vals[8];
                vals[0] = h_values[idx];
                vals[1] = h_values[idx + 1];
                vals[2] = h_values[idx + grid_size.x];
                vals[3] = h_values[idx + grid_size.x + 1];
                vals[4] = h_values[idx + grid_size.x * grid_size.y];
                vals[5] = h_values[idx + grid_size.x * grid_size.y + 1];
                vals[6] = h_values[idx + grid_size.x * grid_size.y + grid_size.x];
                vals[7] = h_values[idx + grid_size.x * grid_size.y + grid_size.x + 1];
                
                // Simple test - if values have different signs, there's a surface
                bool has_surface = false;
                for (int i = 1; i < 8; i++) {
                    if ((vals[0] > params_.iso_value) != (vals[i] > params_.iso_value)) {
                        has_surface = true;
                        break;
                    }
                }
                
                if (has_surface) {
                    // Add a simple triangle (placeholder)
                    float3 p = make_float3(
                        min_bound.x + x * resolution,
                        min_bound.y + y * resolution,
                        min_bound.z + z * resolution
                    );
                    
                    int base_idx = vertices.size() / 3;
                    
                    vertices.push_back(p.x);
                    vertices.push_back(p.y);
                    vertices.push_back(p.z);
                    
                    vertices.push_back(p.x + resolution);
                    vertices.push_back(p.y);
                    vertices.push_back(p.z);
                    
                    vertices.push_back(p.x);
                    vertices.push_back(p.y + resolution);
                    vertices.push_back(p.z);
                    
                    faces.push_back(base_idx);
                    faces.push_back(base_idx + 1);
                    faces.push_back(base_idx + 2);
                    
                    // Add dummy confidence
                    if (params_.extract_confidence) {
                        confidence_map.push_back(1.0f);
                        confidence_map.push_back(1.0f);
                        confidence_map.push_back(1.0f);
                    }
                }
            }
        }
    }
}

size_t NKSRReconstruction::getCurrentGPUMemoryUsage() const {
    size_t total = 0;
    
    for (const auto& chunk : pImpl->chunks) {
        if (chunk.is_loaded) {
            total += chunk.d_points.size() * sizeof(float3);
            total += chunk.d_normals.size() * sizeof(float3);
            total += chunk.d_confidences.size() * sizeof(float);
            total += chunk.d_colors.size() * sizeof(uint8_t);
            total += chunk.d_weights.size() * sizeof(float);
            total += chunk.d_poly_coeffs.size() * sizeof(float);
        }
    }
    
    // Add solver workspace
    total += pImpl->d_A_matrix.size() * sizeof(float);
    total += pImpl->d_b_vector.size() * sizeof(float);
    total += pImpl->d_x_vector.size() * sizeof(float);
    total += pImpl->d_r_vector.size() * sizeof(float);
    total += pImpl->d_p_vector.size() * sizeof(float);
    total += pImpl->d_Ap_vector.size() * sizeof(float);
    
    return total;
}

void NKSRReconstruction::evictLRUChunks(size_t bytes_to_free) {
    // Sort chunks by last access time
    std::vector<std::pair<uint32_t, int>> access_times;
    for (size_t i = 0; i < pImpl->chunks.size(); i++) {
        if (pImpl->chunks[i].is_loaded) {
            access_times.push_back({pImpl->chunks[i].last_access, i});
        }
    }
    
    std::sort(access_times.begin(), access_times.end());
    
    size_t freed = 0;
    for (const auto& pair : access_times) {
        if (freed >= bytes_to_free) break;
        
        ProcessingChunk& chunk = pImpl->chunks[pair.second];
        size_t chunk_size = chunk.d_points.size() * sizeof(float3) * 4;  // Rough estimate
        
        evictChunkFromGPU(chunk);
        freed += chunk_size;
    }
}

void NKSRReconstruction::loadChunkToGPU(ProcessingChunk& chunk) {
    // Implementation depends on where data is stored
    chunk.is_loaded = true;
    chunk.last_access = std::chrono::steady_clock::now().time_since_epoch().count();
}

void NKSRReconstruction::evictChunkFromGPU(ProcessingChunk& chunk) {
    chunk.d_points.clear();
    chunk.d_points.shrink_to_fit();
    chunk.d_normals.clear();
    chunk.d_normals.shrink_to_fit();
    chunk.d_confidences.clear();
    chunk.d_confidences.shrink_to_fit();
    chunk.d_colors.clear();
    chunk.d_colors.shrink_to_fit();
    chunk.d_weights.clear();
    chunk.d_weights.shrink_to_fit();
    chunk.d_poly_coeffs.clear();
    chunk.d_poly_coeffs.shrink_to_fit();
    
    chunk.is_loaded = false;
}

float NKSRReconstruction::getProgress() const {
    if (pImpl->chunks.empty()) return 0.0f;
    
    int solved = 0;
    for (const auto& chunk : pImpl->chunks) {
        if (chunk.is_solved) solved++;
    }
    
    return (float)solved / pImpl->chunks.size();
}

void NKSRReconstruction::clearAllChunks() {
    for (auto& chunk : pImpl->chunks) {
        evictChunkFromGPU(chunk);
    }
    pImpl->chunks.clear();
    while (!pImpl->processing_queue.empty()) {
        pImpl->processing_queue.pop();
    }
}

} // namespace mesh_service