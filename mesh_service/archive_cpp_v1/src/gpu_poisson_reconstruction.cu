#include "gpu_poisson_reconstruction.h"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

namespace mesh_service {

// Constants
constexpr int THREADS_PER_BLOCK = 256;
constexpr int MAX_VERTICES_PER_BLOCK = 65536;
constexpr int MAX_FACES_PER_BLOCK = 131072;

// Basis function for Poisson reconstruction
__device__ float poisson_kernels::basisFunction(float t) {
    float abs_t = fabsf(t);
    if (abs_t >= 1.0f) return 0.0f;
    
    // B-spline basis
    float t2 = abs_t * abs_t;
    float t3 = t2 * abs_t;
    return (2.0f * t3 - 3.0f * t2 + 1.0f);
}

__device__ float3 poisson_kernels::basisGradient(const float3& p, const float3& center, float size) {
    float3 normalized = make_float3(
        (p.x - center.x) / size,
        (p.y - center.y) / size,
        (p.z - center.z) / size
    );
    
    // Derivative of B-spline basis
    float3 grad;
    float abs_x = fabsf(normalized.x);
    float abs_y = fabsf(normalized.y);
    float abs_z = fabsf(normalized.z);
    
    if (abs_x < 1.0f) {
        float sign = (normalized.x > 0) ? 1.0f : -1.0f;
        grad.x = sign * (6.0f * normalized.x * abs_x - 6.0f * abs_x) / size;
    } else {
        grad.x = 0.0f;
    }
    
    if (abs_y < 1.0f) {
        float sign = (normalized.y > 0) ? 1.0f : -1.0f;
        grad.y = sign * (6.0f * normalized.y * abs_y - 6.0f * abs_y) / size;
    } else {
        grad.y = 0.0f;
    }
    
    if (abs_z < 1.0f) {
        float sign = (normalized.z > 0) ? 1.0f : -1.0f;
        grad.z = sign * (6.0f * normalized.z * abs_z - 6.0f * abs_z) / size;
    } else {
        grad.z = 0.0f;
    }
    
    return grad;
}

__device__ float poisson_kernels::evaluateFunction(const float3& p, const PoissonNode* nodes,
                                                  const float* solution, int num_nodes) {
    float value = 0.0f;
    
    // Find nearby nodes and accumulate their contributions
    for (int i = 0; i < num_nodes; i++) {
        const PoissonNode& node = nodes[i];
        
        float3 local = make_float3(
            (p.x - node.center.x) / node.size,
            (p.y - node.center.y) / node.size,
            (p.z - node.center.z) / node.size
        );
        
        if (fabsf(local.x) < 1.0f && fabsf(local.y) < 1.0f && fabsf(local.z) < 1.0f) {
            float basis = basisFunction(local.x) * basisFunction(local.y) * basisFunction(local.z);
            value += solution[i] * basis;
        }
    }
    
    return value;
}

__device__ int poisson_kernels::findContainingNode(const float3& p, const PoissonNode* nodes,
                                                   int num_nodes) {
    // Simple linear search - could be optimized with spatial structure
    for (int i = 0; i < num_nodes; i++) {
        const PoissonNode& node = nodes[i];
        float half_size = node.size * 0.5f;
        
        if (p.x >= node.center.x - half_size && p.x <= node.center.x + half_size &&
            p.y >= node.center.y - half_size && p.y <= node.center.y + half_size &&
            p.z >= node.center.z - half_size && p.z <= node.center.z + half_size) {
            return i;
        }
    }
    return -1;
}

// Kernel to compute divergence field from oriented points
__global__ void poisson_kernels::computeDivergence(
    const float3* points,
    const float3* normals,
    int num_points,
    const PoissonNode* nodes,
    int num_nodes,
    float* divergence,
    float3* gradients,
    float* weights,
    float point_weight) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_points) return;
    
    float3 point = points[tid];
    float3 normal = normals[tid];
    
    // Find nodes affected by this point
    for (int node_idx = 0; node_idx < num_nodes; node_idx++) {
        const PoissonNode& node = nodes[node_idx];
        
        float3 local = make_float3(
            (point.x - node.center.x) / node.size,
            (point.y - node.center.y) / node.size,
            (point.z - node.center.z) / node.size
        );
        
        // Check if point influences this node
        if (fabsf(local.x) < 1.0f && fabsf(local.y) < 1.0f && fabsf(local.z) < 1.0f) {
            float basis = basisFunction(local.x) * basisFunction(local.y) * basisFunction(local.z);
            float3 grad = basisGradient(point, node.center, node.size);
            
            // Atomic add to divergence
            float div_contribution = point_weight * basis * (normal.x * grad.x + normal.y * grad.y + normal.z * grad.z);
            atomicAdd(&divergence[node_idx], div_contribution);
            
            // Atomic add to gradient
            atomicAdd(&gradients[node_idx].x, point_weight * normal.x * basis);
            atomicAdd(&gradients[node_idx].y, point_weight * normal.y * basis);
            atomicAdd(&gradients[node_idx].z, point_weight * normal.z * basis);
            
            // Atomic add to weight
            atomicAdd(&weights[node_idx], point_weight * basis);
        }
    }
}

// Kernel for incremental divergence update
__global__ void poisson_kernels::updateDivergenceIncremental(
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
    float point_weight) {
    
    int block_tid = blockIdx.x;
    if (block_tid >= num_affected_blocks) return;
    
    int block_idx = affected_block_indices[block_tid];
    const PoissonBlock& block = blocks[block_idx];
    
    int point_tid = threadIdx.x;
    
    // Process points in this block
    for (int p = point_tid; p < num_new_points; p += blockDim.x) {
        float3 point = new_points[p];
        
        // Check if point is in this block
        if (point.x >= block.min_bound.x && point.x <= block.max_bound.x &&
            point.y >= block.min_bound.y && point.y <= block.max_bound.y &&
            point.z >= block.min_bound.z && point.z <= block.max_bound.z) {
            
            float3 normal = new_normals[p];
            
            // Update nodes in this block
            for (int n = 0; n < block.node_count; n++) {
                int node_idx = block.node_start + n;
                PoissonNode& node = nodes[node_idx];
                
                float3 local = make_float3(
                    (point.x - node.center.x) / node.size,
                    (point.y - node.center.y) / node.size,
                    (point.z - node.center.z) / node.size
                );
                
                if (fabsf(local.x) < 1.0f && fabsf(local.y) < 1.0f && fabsf(local.z) < 1.0f) {
                    float basis = basisFunction(local.x) * basisFunction(local.y) * basisFunction(local.z);
                    float3 grad = basisGradient(point, node.center, node.size);
                    
                    float div_contribution = point_weight * basis * (normal.x * grad.x + normal.y * grad.y + normal.z * grad.z);
                    atomicAdd(&divergence[node_idx], div_contribution);
                    
                    atomicAdd(&gradients[node_idx].x, point_weight * normal.x * basis);
                    atomicAdd(&gradients[node_idx].y, point_weight * normal.y * basis);
                    atomicAdd(&gradients[node_idx].z, point_weight * normal.z * basis);
                    
                    atomicAdd(&weights[node_idx], point_weight * basis);
                    
                    // Mark node as needing update
                    node.needs_update = true;
                }
            }
        }
    }
}

// Apply Laplacian operator for Poisson solver
__global__ void poisson_kernels::applyLaplacian(
    const float* x,
    float* Ax,
    const PoissonNode* nodes,
    const PoissonBlock* blocks,
    const int* active_blocks,
    int num_active_blocks) {
    
    int block_tid = blockIdx.x;
    if (block_tid >= num_active_blocks) return;
    
    int block_idx = active_blocks[block_tid];
    const PoissonBlock& block = blocks[block_idx];
    
    int local_tid = threadIdx.x;
    
    // Process nodes in this block
    for (int n = local_tid; n < block.node_count; n += blockDim.x) {
        int node_idx = block.node_start + n;
        const PoissonNode& node = nodes[node_idx];
        
        float laplacian = 0.0f;
        
        // Central difference approximation
        float h = node.size;
        float h2_inv = 1.0f / (h * h);
        
        // 7-point stencil
        laplacian = -6.0f * x[node_idx] * h2_inv;
        
        // Add neighbor contributions
        // This is simplified - in practice we'd look up actual neighbor indices
        for (int dim = 0; dim < 3; dim++) {
            for (int dir = -1; dir <= 1; dir += 2) {
                int neighbor_idx = node_idx + dir * (dim == 0 ? 1 : (dim == 1 ? block.node_count : block.node_count * block.node_count));
                if (neighbor_idx >= 0 && neighbor_idx < block.node_start + block.node_count) {
                    laplacian += x[neighbor_idx] * h2_inv;
                }
            }
        }
        
        Ax[node_idx] = laplacian;
    }
}

// Conjugate gradient solver step
__global__ void poisson_kernels::cgSolverStep(
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
    float* residual_norm) {
    
    // This is a simplified version - full CG would require multiple kernels
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_nodes = 0;
    
    for (int b = 0; b < num_active_blocks; b++) {
        total_nodes += blocks[active_blocks[b]].node_count;
    }
    
    if (tid >= total_nodes) return;
    
    // Compute residual = b - Ax
    residual[tid] = divergence[tid] - Ap[tid];
    
    // Update solution: x = x + alpha * p
    solution[tid] += (*alpha) * p[tid];
    
    // Update search direction: p = r + beta * p
    p[tid] = residual[tid] + (*beta) * p[tid];
}

// Extract iso-surface using marching cubes
__global__ void poisson_kernels::extractIsoSurface(
    const float* solution,
    const PoissonNode* nodes,
    const PoissonBlock* blocks,
    const int* blocks_to_extract,
    int num_blocks,
    float iso_value,
    float3* vertices,
    uint32_t* faces,
    int* vertex_counter,
    int* face_counter) {
    
    extern __shared__ float shared_values[];
    
    int block_tid = blockIdx.x;
    if (block_tid >= num_blocks) return;
    
    int block_idx = blocks_to_extract[block_tid];
    const PoissonBlock& block = blocks[block_idx];
    
    int tid = threadIdx.x;
    
    // Process cells in this block
    // Each thread processes one cell
    int cells_per_dim = POISSON_BLOCK_SIZE / 8;  // Assuming 8x8x8 cells per block
    int total_cells = cells_per_dim * cells_per_dim * cells_per_dim;
    
    for (int cell_idx = tid; cell_idx < total_cells; cell_idx += blockDim.x) {
        // Convert linear index to 3D coordinates
        int cx = cell_idx % cells_per_dim;
        int cy = (cell_idx / cells_per_dim) % cells_per_dim;
        int cz = cell_idx / (cells_per_dim * cells_per_dim);
        
        // Get corner values
        float corner_vals[8];
        int cube_index = 0;
        
        for (int corner = 0; corner < 8; corner++) {
            int dx = corner & 1;
            int dy = (corner >> 1) & 1;
            int dz = (corner >> 2) & 1;
            
            float3 corner_pos = make_float3(
                block.min_bound.x + (cx + dx) * (block.max_bound.x - block.min_bound.x) / cells_per_dim,
                block.min_bound.y + (cy + dy) * (block.max_bound.y - block.min_bound.y) / cells_per_dim,
                block.min_bound.z + (cz + dz) * (block.max_bound.z - block.min_bound.z) / cells_per_dim
            );
            
            corner_vals[corner] = evaluateFunction(corner_pos, nodes, solution, block.node_start + block.node_count);
            
            if (corner_vals[corner] < iso_value) {
                cube_index |= (1 << corner);
            }
        }
        
        // Skip if cube is entirely inside or outside
        if (cube_index == 0 || cube_index == 255) continue;
        
        // Generate triangles (simplified - full marching cubes needs lookup tables)
        // This is a placeholder for the actual marching cubes algorithm
        if (cube_index != 0 && cube_index != 255) {
            int base_vertex = atomicAdd(vertex_counter, 3);
            int face_idx = atomicAdd(face_counter, 1);
            
            if (base_vertex < MAX_VERTICES_PER_BLOCK * num_blocks && 
                face_idx < MAX_FACES_PER_BLOCK * num_blocks) {
                
                // Add dummy triangle for now
                float cell_size = (block.max_bound.x - block.min_bound.x) / cells_per_dim;
                vertices[base_vertex] = make_float3(
                    block.min_bound.x + cx * cell_size,
                    block.min_bound.y + cy * cell_size,
                    block.min_bound.z + cz * cell_size
                );
                vertices[base_vertex + 1] = vertices[base_vertex];
                vertices[base_vertex + 1].x += cell_size;
                vertices[base_vertex + 2] = vertices[base_vertex];
                vertices[base_vertex + 2].y += cell_size;
                
                faces[face_idx * 3] = base_vertex;
                faces[face_idx * 3 + 1] = base_vertex + 1;
                faces[face_idx * 3 + 2] = base_vertex + 2;
            }
        }
    }
}

// Stitch block boundaries to ensure watertight mesh
__global__ void poisson_kernels::stitchBlockBoundaries(
    float3* vertices,
    uint32_t* faces,
    int num_vertices,
    const PoissonBlock* blocks,
    const int* boundary_blocks,
    int num_boundary_blocks,
    float epsilon) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_vertices) return;
    
    float3 vertex = vertices[tid];
    
    // Check if vertex is on a block boundary
    for (int b = 0; b < num_boundary_blocks; b++) {
        const PoissonBlock& block = blocks[boundary_blocks[b]];
        
        // Check each face of the block
        bool on_boundary = false;
        
        if (fabsf(vertex.x - block.min_bound.x) < epsilon || 
            fabsf(vertex.x - block.max_bound.x) < epsilon) {
            on_boundary = true;
        }
        if (fabsf(vertex.y - block.min_bound.y) < epsilon || 
            fabsf(vertex.y - block.max_bound.y) < epsilon) {
            on_boundary = true;
        }
        if (fabsf(vertex.z - block.min_bound.z) < epsilon || 
            fabsf(vertex.z - block.max_bound.z) < epsilon) {
            on_boundary = true;
        }
        
        if (on_boundary) {
            // Snap to exact boundary position
            if (fabsf(vertex.x - block.min_bound.x) < epsilon) vertex.x = block.min_bound.x;
            if (fabsf(vertex.x - block.max_bound.x) < epsilon) vertex.x = block.max_bound.x;
            if (fabsf(vertex.y - block.min_bound.y) < epsilon) vertex.y = block.min_bound.y;
            if (fabsf(vertex.y - block.max_bound.y) < epsilon) vertex.y = block.max_bound.y;
            if (fabsf(vertex.z - block.min_bound.z) < epsilon) vertex.z = block.min_bound.z;
            if (fabsf(vertex.z - block.max_bound.z) < epsilon) vertex.z = block.max_bound.z;
            
            vertices[tid] = vertex;
        }
    }
}

// GPUPoissonReconstruction implementation
GPUPoissonReconstruction::GPUPoissonReconstruction() 
    : d_nodes(nullptr), d_blocks(nullptr), node_count(0), total_blocks(0) {
    // Initialize with default parameters
}

GPUPoissonReconstruction::~GPUPoissonReconstruction() {
    freeMemory();
}

void GPUPoissonReconstruction::initialize(float scene_size, int grid_resolution) {
    // Calculate grid dimensions
    grid_dims = make_int3(grid_resolution, grid_resolution, grid_resolution);
    total_blocks = grid_dims.x * grid_dims.y * grid_dims.z;
    
    // Calculate maximum nodes
    max_nodes = total_blocks * (POISSON_BLOCK_SIZE / 8) * (POISSON_BLOCK_SIZE / 8) * (POISSON_BLOCK_SIZE / 8);
    
    allocateMemory(scene_size);
    
    // Initialize blocks on device
    dim3 block(256);
    dim3 grid((total_blocks + 255) / 256);
    
    // Simple initialization kernel (would be defined separately)
    // For now, we'll do it on CPU and copy
    std::vector<PoissonBlock> h_blocks(total_blocks);
    float block_size = scene_size / grid_resolution;
    
    for (int z = 0; z < grid_dims.z; z++) {
        for (int y = 0; y < grid_dims.y; y++) {
            for (int x = 0; x < grid_dims.x; x++) {
                int idx = x + y * grid_dims.x + z * grid_dims.x * grid_dims.y;
                PoissonBlock& block = h_blocks[idx];
                
                block.block_coord = make_int3(x, y, z);
                block.min_bound = make_float3(
                    -scene_size/2 + x * block_size,
                    -scene_size/2 + y * block_size,
                    -scene_size/2 + z * block_size
                );
                block.max_bound = make_float3(
                    block.min_bound.x + block_size,
                    block.min_bound.y + block_size,
                    block.min_bound.z + block_size
                );
                
                block.node_start = idx * (POISSON_BLOCK_SIZE / 8) * (POISSON_BLOCK_SIZE / 8) * (POISSON_BLOCK_SIZE / 8);
                block.node_count = (POISSON_BLOCK_SIZE / 8) * (POISSON_BLOCK_SIZE / 8) * (POISSON_BLOCK_SIZE / 8);
                
                // Set up neighbors
                for (int i = 0; i < 6; i++) block.neighbors[i] = -1;
                if (x > 0) block.neighbors[0] = idx - 1;
                if (x < grid_dims.x - 1) block.neighbors[1] = idx + 1;
                if (y > 0) block.neighbors[2] = idx - grid_dims.x;
                if (y < grid_dims.y - 1) block.neighbors[3] = idx + grid_dims.x;
                if (z > 0) block.neighbors[4] = idx - grid_dims.x * grid_dims.y;
                if (z < grid_dims.z - 1) block.neighbors[5] = idx + grid_dims.x * grid_dims.y;
                
                block.is_dirty = false;
                block.is_processing = false;
                block.last_update = 0;
            }
        }
    }
    
    cudaMemcpy(d_blocks, h_blocks.data(), total_blocks * sizeof(PoissonBlock), cudaMemcpyHostToDevice);
}

void GPUPoissonReconstruction::allocateMemory(float scene_size) {
    // Allocate node storage
    cudaMalloc(&d_nodes, max_nodes * sizeof(PoissonNode));
    
    // Allocate block storage
    cudaMalloc(&d_blocks, total_blocks * sizeof(PoissonBlock));
    
    // Allocate solver arrays
    cudaMalloc(&d_divergence, max_nodes * sizeof(float));
    cudaMalloc(&d_solution, max_nodes * sizeof(float));
    cudaMalloc(&d_weights, max_nodes * sizeof(float));
    cudaMalloc(&d_gradients, max_nodes * sizeof(float3));
    
    // Allocate solver temporaries
    cudaMalloc(&d_r, max_nodes * sizeof(float));
    cudaMalloc(&d_p, max_nodes * sizeof(float));
    cudaMalloc(&d_Ap, max_nodes * sizeof(float));
    
    // Allocate mesh extraction buffers
    int max_vertices = MAX_VERTICES_PER_BLOCK * total_blocks;
    int max_faces = MAX_FACES_PER_BLOCK * total_blocks;
    
    cudaMalloc(&d_mesh_vertices, max_vertices * sizeof(float3));
    cudaMalloc(&d_mesh_faces, max_faces * 3 * sizeof(uint32_t));
    cudaMalloc(&d_mesh_colors, max_vertices * 3 * sizeof(uint8_t));
    cudaMalloc(&d_vertex_counter, sizeof(int));
    cudaMalloc(&d_face_counter, sizeof(int));
    
    // Allocate solver memory pool
    solver_pool_size = 256 * 1024 * 1024;  // 256MB
    cudaMalloc(&solver_memory_pool, solver_pool_size);
    
    // Initialize arrays
    cudaMemset(d_divergence, 0, max_nodes * sizeof(float));
    cudaMemset(d_solution, 0, max_nodes * sizeof(float));
    cudaMemset(d_weights, 0, max_nodes * sizeof(float));
    cudaMemset(d_gradients, 0, max_nodes * sizeof(float3));
}

void GPUPoissonReconstruction::freeMemory() {
    if (d_nodes) cudaFree(d_nodes);
    if (d_blocks) cudaFree(d_blocks);
    if (d_divergence) cudaFree(d_divergence);
    if (d_solution) cudaFree(d_solution);
    if (d_weights) cudaFree(d_weights);
    if (d_gradients) cudaFree(d_gradients);
    if (d_r) cudaFree(d_r);
    if (d_p) cudaFree(d_p);
    if (d_Ap) cudaFree(d_Ap);
    if (d_mesh_vertices) cudaFree(d_mesh_vertices);
    if (d_mesh_faces) cudaFree(d_mesh_faces);
    if (d_mesh_colors) cudaFree(d_mesh_colors);
    if (d_vertex_counter) cudaFree(d_vertex_counter);
    if (d_face_counter) cudaFree(d_face_counter);
    if (solver_memory_pool) cudaFree(solver_memory_pool);
}

void GPUPoissonReconstruction::addPoints(const float3* points, const float3* normals,
                                        const uint8_t* colors, int num_points,
                                        GPUOctree* octree, cudaStream_t stream) {
    if (num_points == 0) return;
    
    // Find affected blocks
    std::vector<int> affected_blocks;
    
    // This would normally be done on GPU
    // For now, simplified version
    for (int i = 0; i < total_blocks; i++) {
        affected_blocks.push_back(i);
    }
    
    // Upload affected block indices
    int* d_affected_blocks;
    cudaMalloc(&d_affected_blocks, affected_blocks.size() * sizeof(int));
    cudaMemcpy(d_affected_blocks, affected_blocks.data(), 
               affected_blocks.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    // Update divergence field incrementally
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(affected_blocks.size());
    
    poisson_kernels::updateDivergenceIncremental<<<grid, block, 0, stream>>>(
        points, normals, num_points,
        d_blocks, d_affected_blocks, affected_blocks.size(),
        d_nodes, d_divergence, d_gradients, d_weights,
        params_.point_weight
    );
    
    // Mark blocks as dirty
    for (int idx : affected_blocks) {
        // This would be done on GPU in practice
        PoissonBlock block;
        cudaMemcpy(&block, &d_blocks[idx], sizeof(PoissonBlock), cudaMemcpyDeviceToHost);
        block.is_dirty = true;
        cudaMemcpy(&d_blocks[idx], &block, sizeof(PoissonBlock), cudaMemcpyHostToDevice);
    }
    
    cudaFree(d_affected_blocks);
}

void GPUPoissonReconstruction::extractMesh(std::vector<float>& vertices,
                                          std::vector<uint32_t>& faces,
                                          std::vector<uint8_t>& colors,
                                          const std::vector<int>& dirty_blocks,
                                          cudaStream_t stream) {
    if (dirty_blocks.empty()) return;
    
    // Reset counters
    cudaMemset(d_vertex_counter, 0, sizeof(int));
    cudaMemset(d_face_counter, 0, sizeof(int));
    
    // Upload dirty block indices
    int* d_dirty_blocks;
    cudaMalloc(&d_dirty_blocks, dirty_blocks.size() * sizeof(int));
    cudaMemcpy(d_dirty_blocks, dirty_blocks.data(),
               dirty_blocks.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    // Solve Poisson equation for dirty blocks
    solvePoisson(dirty_blocks, stream);
    
    // Extract mesh from solution
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(dirty_blocks.size());
    
    size_t shared_mem = POISSON_BLOCK_SIZE * sizeof(float);
    poisson_kernels::extractIsoSurface<<<grid, block, shared_mem, stream>>>(
        d_solution, d_nodes, d_blocks, d_dirty_blocks, dirty_blocks.size(),
        params_.iso_value, d_mesh_vertices, d_mesh_faces,
        d_vertex_counter, d_face_counter
    );
    
    // Stitch block boundaries
    int num_vertices;
    cudaMemcpy(&num_vertices, d_vertex_counter, sizeof(int), cudaMemcpyDeviceToHost);
    
    dim3 stitch_grid((num_vertices + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    poisson_kernels::stitchBlockBoundaries<<<stitch_grid, block, 0, stream>>>(
        d_mesh_vertices, d_mesh_faces, num_vertices,
        d_blocks, d_dirty_blocks, dirty_blocks.size(),
        0.001f  // epsilon for boundary matching
    );
    
    // Copy results to host
    int num_faces;
    cudaMemcpy(&num_faces, d_face_counter, sizeof(int), cudaMemcpyDeviceToHost);
    
    vertices.resize(num_vertices * 3);
    faces.resize(num_faces * 3);
    
    // Convert float3 to float array
    std::vector<float3> temp_vertices(num_vertices);
    cudaMemcpy(temp_vertices.data(), d_mesh_vertices, 
               num_vertices * sizeof(float3), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < num_vertices; i++) {
        vertices[i * 3] = temp_vertices[i].x;
        vertices[i * 3 + 1] = temp_vertices[i].y;
        vertices[i * 3 + 2] = temp_vertices[i].z;
    }
    
    cudaMemcpy(faces.data(), d_mesh_faces, 
               num_faces * 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // Copy colors if available
    if (d_mesh_colors) {
        colors.resize(num_vertices * 3);
        cudaMemcpy(colors.data(), d_mesh_colors,
                   num_vertices * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    }
    
    cudaFree(d_dirty_blocks);
}

void GPUPoissonReconstruction::solvePoisson(const std::vector<int>& blocks_to_solve,
                                           cudaStream_t stream) {
    // Simplified conjugate gradient solver
    // In practice, this would be a full multigrid or PCG solver
    
    for (int iter = 0; iter < params_.solver_iterations; iter++) {
        // Apply Laplacian
        int* d_active_blocks;
        cudaMalloc(&d_active_blocks, blocks_to_solve.size() * sizeof(int));
        cudaMemcpy(d_active_blocks, blocks_to_solve.data(),
                   blocks_to_solve.size() * sizeof(int), cudaMemcpyHostToDevice);
        
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(blocks_to_solve.size());
        
        poisson_kernels::applyLaplacian<<<grid, block, 0, stream>>>(
            d_solution, d_Ap, d_nodes, d_blocks,
            d_active_blocks, blocks_to_solve.size()
        );
        
        cudaFree(d_active_blocks);
    }
}

void GPUPoissonReconstruction::getDirtyBlocks(std::vector<int>& dirty_block_indices) {
    // Copy blocks to host and check dirty flags
    std::vector<PoissonBlock> h_blocks(total_blocks);
    cudaMemcpy(h_blocks.data(), d_blocks, 
               total_blocks * sizeof(PoissonBlock), cudaMemcpyDeviceToHost);
    
    dirty_block_indices.clear();
    for (int i = 0; i < total_blocks; i++) {
        if (h_blocks[i].is_dirty) {
            dirty_block_indices.push_back(i);
        }
    }
}

void GPUPoissonReconstruction::clearDirtyBlocks(const std::vector<int>& cleared_blocks) {
    // Mark blocks as clean
    for (int idx : cleared_blocks) {
        PoissonBlock block;
        cudaMemcpy(&block, &d_blocks[idx], sizeof(PoissonBlock), cudaMemcpyDeviceToHost);
        block.is_dirty = false;
        cudaMemcpy(&d_blocks[idx], &block, sizeof(PoissonBlock), cudaMemcpyHostToDevice);
    }
}

} // namespace mesh_service