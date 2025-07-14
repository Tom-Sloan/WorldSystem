#include "normal_estimation.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <cub/cub.cuh>
#include <iostream>

namespace mesh_service {

namespace cuda {

__device__ void computeCovariance3x3(
    float3* neighbors,
    int count,
    float3 centroid,
    float* covariance
) {
    // Initialize covariance matrix
    for (int i = 0; i < 9; i++) {
        covariance[i] = 0.0f;
    }
    
    // Compute covariance
    for (int i = 0; i < count; i++) {
        float3 diff = make_float3(
            neighbors[i].x - centroid.x,
            neighbors[i].y - centroid.y,
            neighbors[i].z - centroid.z
        );
        
        // Upper triangular part
        covariance[0] += diff.x * diff.x;
        covariance[1] += diff.x * diff.y;
        covariance[2] += diff.x * diff.z;
        covariance[4] += diff.y * diff.y;
        covariance[5] += diff.y * diff.z;
        covariance[8] += diff.z * diff.z;
    }
    
    // Normalize
    float inv_count = 1.0f / count;
    for (int i = 0; i < 9; i++) {
        covariance[i] *= inv_count;
    }
    
    // Fill lower triangular part
    covariance[3] = covariance[1];
    covariance[6] = covariance[2];
    covariance[7] = covariance[5];
}

__device__ float3 computeSmallestEigenvector(float* cov) {
    // Power iteration method to find smallest eigenvalue/eigenvector
    // For 3x3 symmetric matrix
    
    float3 v = make_float3(1.0f, 0.0f, 0.0f);
    
    // Inverse power iteration
    for (int iter = 0; iter < 10; iter++) {
        // Compute determinant for inverse
        float det = cov[0] * (cov[4] * cov[8] - cov[5] * cov[7]) -
                   cov[1] * (cov[3] * cov[8] - cov[5] * cov[6]) +
                   cov[2] * (cov[3] * cov[7] - cov[4] * cov[6]);
        
        if (fabsf(det) < 1e-6f) {
            // Matrix is singular, return current estimate
            break;
        }
        
        // Compute cofactor matrix (transpose of adjugate)
        float adj[9];
        adj[0] = cov[4] * cov[8] - cov[5] * cov[7];
        adj[1] = cov[2] * cov[7] - cov[1] * cov[8];
        adj[2] = cov[1] * cov[5] - cov[2] * cov[4];
        adj[3] = cov[5] * cov[6] - cov[3] * cov[8];
        adj[4] = cov[0] * cov[8] - cov[2] * cov[6];
        adj[5] = cov[2] * cov[3] - cov[0] * cov[5];
        adj[6] = cov[3] * cov[7] - cov[4] * cov[6];
        adj[7] = cov[1] * cov[6] - cov[0] * cov[7];
        adj[8] = cov[0] * cov[4] - cov[1] * cov[3];
        
        // Multiply by inverse (adj/det)
        float3 new_v;
        new_v.x = (adj[0] * v.x + adj[1] * v.y + adj[2] * v.z) / det;
        new_v.y = (adj[3] * v.x + adj[4] * v.y + adj[5] * v.z) / det;
        new_v.z = (adj[6] * v.x + adj[7] * v.y + adj[8] * v.z) / det;
        
        // Normalize
        float len = sqrtf(new_v.x * new_v.x + new_v.y * new_v.y + new_v.z * new_v.z);
        if (len > 0.0f) {
            v = make_float3(new_v.x / len, new_v.y / len, new_v.z / len);
        }
    }
    
    return v;
}

__global__ void estimateNormalsPCA(
    float3* points,
    int num_points,
    int* neighbor_indices,
    int k_neighbors,
    float3* normals
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    // float3 query_point = points[idx];  // Will be used when neighbor collection is implemented
    
    // Collect neighbors (simplified - assumes pre-computed indices)
    float3 neighbors[64];  // Max k_neighbors
    int actual_neighbors = min(k_neighbors, 64);
    
    // Compute centroid
    float3 centroid = make_float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < actual_neighbors; i++) {
        int n_idx = neighbor_indices[idx * k_neighbors + i];
        if (n_idx >= 0 && n_idx < num_points) {
            neighbors[i] = points[n_idx];
            centroid.x += neighbors[i].x;
            centroid.y += neighbors[i].y;
            centroid.z += neighbors[i].z;
        }
    }
    
    centroid.x /= actual_neighbors;
    centroid.y /= actual_neighbors;
    centroid.z /= actual_neighbors;
    
    // Compute covariance matrix
    float covariance[9];
    computeCovariance3x3(neighbors, actual_neighbors, centroid, covariance);
    
    // Get normal as smallest eigenvector
    float3 normal = computeSmallestEigenvector(covariance);
    
    // Store result
    normals[idx] = normal;
}

__global__ void estimateNormalsTangentPlane(
    float3* points,
    int num_points,
    float search_radius,
    float3* normals
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    float3 query = points[idx];
    float radius_sq = search_radius * search_radius;
    
    // Find neighbors within radius
    float3 neighbors[64];
    int neighbor_count = 0;
    
    for (int i = 0; i < num_points && neighbor_count < 64; i++) {
        if (i == idx) continue;
        
        float3 diff = make_float3(
            points[i].x - query.x,
            points[i].y - query.y,
            points[i].z - query.z
        );
        
        float dist_sq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
        
        if (dist_sq < radius_sq) {
            neighbors[neighbor_count++] = points[i];
        }
    }
    
    if (neighbor_count < 3) {
        // Not enough neighbors, use default normal
        normals[idx] = make_float3(0.0f, 0.0f, 1.0f);
        return;
    }
    
    // Compute centroid
    float3 centroid = make_float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < neighbor_count; i++) {
        centroid.x += neighbors[i].x;
        centroid.y += neighbors[i].y;
        centroid.z += neighbors[i].z;
    }
    centroid.x /= neighbor_count;
    centroid.y /= neighbor_count;
    centroid.z /= neighbor_count;
    
    // Compute covariance and normal
    float covariance[9];
    computeCovariance3x3(neighbors, neighbor_count, centroid, covariance);
    normals[idx] = computeSmallestEigenvector(covariance);
}

__global__ void orientNormals(
    float3* points,
    float3* normals,
    int num_points,
    float3 view_point
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    float3 point = points[idx];
    float3 normal = normals[idx];
    
    // Vector from point to viewpoint
    float3 view_dir = make_float3(
        view_point.x - point.x,
        view_point.y - point.y,
        view_point.z - point.z
    );
    
    // Dot product
    float dot = normal.x * view_dir.x + normal.y * view_dir.y + normal.z * view_dir.z;
    
    // Flip normal if pointing away from viewpoint
    if (dot < 0.0f) {
        normals[idx] = make_float3(-normal.x, -normal.y, -normal.z);
    }
}

} // namespace cuda

// NormalEstimation implementation
class NormalEstimation::Impl {
public:
    Parameters params;
    thrust::device_vector<int> d_neighbor_indices;
    thrust::device_vector<float> d_distances;
    
    void buildNeighborIndices(float3* d_points, size_t num_points) {
        // Simplified: for each point, find k nearest neighbors
        // In production, use a spatial data structure like KD-tree
        
        d_neighbor_indices.resize(num_points * params.k_neighbors);
        d_distances.resize(num_points * params.k_neighbors);
        
        // For now, just use first k points as neighbors
        thrust::fill(d_neighbor_indices.begin(), d_neighbor_indices.end(), -1);
        
        for (size_t i = 0; i < num_points; i++) {
            for (int k = 0; k < params.k_neighbors && k < num_points; k++) {
                d_neighbor_indices[i * params.k_neighbors + k] = (i + k + 1) % num_points;
            }
        }
    }
};

NormalEstimation::NormalEstimation() : pImpl(std::make_unique<Impl>()) {}
NormalEstimation::~NormalEstimation() = default;

void NormalEstimation::setParameters(const Parameters& params) {
    pImpl->params = params;
}

void NormalEstimation::estimateNormals(
    float3* d_points,
    size_t num_points,
    float3* d_normals,
    cudaStream_t stream
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Build neighbor indices
    pImpl->buildNeighborIndices(d_points, num_points);
    
    // Launch kernel based on method
    dim3 block(256);
    dim3 grid((num_points + block.x - 1) / block.x);
    
    switch (pImpl->params.method) {
        case PCA:
            cuda::estimateNormalsPCA<<<grid, block, 0, stream>>>(
                d_points,
                num_points,
                pImpl->d_neighbor_indices.data().get(),
                pImpl->params.k_neighbors,
                d_normals
            );
            break;
            
        case TANGENT_PLANE:
            cuda::estimateNormalsTangentPlane<<<grid, block, 0, stream>>>(
                d_points,
                num_points,
                pImpl->params.search_radius,
                d_normals
            );
            break;
            
        default:
            // Fallback to PCA
            cuda::estimateNormalsPCA<<<grid, block, 0, stream>>>(
                d_points,
                num_points,
                pImpl->d_neighbor_indices.data().get(),
                pImpl->params.k_neighbors,
                d_normals
            );
    }
    
    // Orient normals if requested
    if (pImpl->params.orient_normals) {
        float3 view_point = make_float3(
            pImpl->params.view_point[0],
            pImpl->params.view_point[1],
            pImpl->params.view_point[2]
        );
        
        cuda::orientNormals<<<grid, block, 0, stream>>>(
            d_points,
            d_normals,
            num_points,
            view_point
        );
    }
    
    cudaStreamSynchronize(stream);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "Normal estimation completed in " << duration << "ms" << std::endl;
}

void NormalEstimation::estimateNormalsIndexed(
    float3* d_points,
    size_t num_points,
    void* spatial_index,
    float3* d_normals,
    cudaStream_t stream
) {
    // Use pre-built spatial index for faster neighbor search
    // Implementation depends on the spatial index type
    estimateNormals(d_points, num_points, d_normals, stream);
}

} // namespace mesh_service