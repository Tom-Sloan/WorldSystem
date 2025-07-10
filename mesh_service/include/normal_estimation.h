#pragma once

#include <cuda_runtime.h>
#include <memory>

namespace mesh_service {

// GPU-accelerated normal estimation
class NormalEstimation {
public:
    enum Method {
        PCA,           // Principal Component Analysis
        TANGENT_PLANE, // Local tangent plane fitting
        MLS,           // Moving Least Squares
        QUADRIC        // Quadric fitting
    };

    struct Parameters {
        Method method = PCA;
        int k_neighbors = 30;        // Number of neighbors for estimation
        float search_radius = 0.1f;  // Search radius (meters)
        bool orient_normals = true;  // Orient normals consistently
        float view_point[3] = {0, 0, 0}; // Viewpoint for orientation
    };

    NormalEstimation();
    ~NormalEstimation();

    void setParameters(const Parameters& params);

    // Estimate normals for point cloud
    void estimateNormals(
        float3* d_points,
        size_t num_points,
        float3* d_normals,
        cudaStream_t stream
    );

    // Estimate normals with pre-built spatial index
    void estimateNormalsIndexed(
        float3* d_points,
        size_t num_points,
        void* spatial_index,  // KD-tree or octree
        float3* d_normals,
        cudaStream_t stream
    );

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// CUDA kernels for normal estimation
namespace cuda {

// PCA-based normal estimation kernel
__global__ void estimateNormalsPCA(
    float3* points,
    int num_points,
    int* neighbor_indices,
    int k_neighbors,
    float3* normals
);

// Tangent plane fitting kernel
__global__ void estimateNormalsTangentPlane(
    float3* points,
    int num_points,
    float search_radius,
    float3* normals
);

// Normal orientation kernel
__global__ void orientNormals(
    float3* points,
    float3* normals,
    int num_points,
    float3 view_point
);

// Compute covariance matrix
__device__ void computeCovariance3x3(
    float3* neighbors,
    int count,
    float3 centroid,
    float* covariance
);

// Compute smallest eigenvector (normal)
__device__ float3 computeSmallestEigenvector(float* covariance);

} // namespace cuda

} // namespace mesh_service