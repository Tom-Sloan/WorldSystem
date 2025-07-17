#ifdef HAS_OPEN3D

#include "normal_providers/open3d_normal_provider.h"
#include "config/configuration_manager.h"
#include "config/normal_provider_config.h"
#include <open3d/Open3D.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <chrono>

namespace mesh_service {

// Implementation class to hide Open3D types from header
class Open3DNormalProvider::Impl {
public:
    int k_neighbors;
    float search_radius;
    bool fast_normal_computation;
    bool use_knn;  // true for KNN, false for radius search
    
    Impl() 
        : k_neighbors(CONFIG_INT("MESH_NORMAL_K_NEIGHBORS", 
                                config::NormalProviderConfig::DEFAULT_OPEN3D_K_NEIGHBORS))
        , search_radius(CONFIG_FLOAT("MESH_NORMAL_SEARCH_RADIUS",
                                    config::NormalProviderConfig::DEFAULT_OPEN3D_SEARCH_RADIUS))
        , fast_normal_computation(CONFIG_BOOL("MESH_OPEN3D_FAST_NORMAL",
                                            config::NormalProviderConfig::DEFAULT_OPEN3D_FAST_NORMAL_COMPUTATION))
        , use_knn(true)  // Default to KNN
    {
        std::cout << "[OPEN3D PROVIDER] Initialized with k=" << k_neighbors 
                  << ", fast_computation=" << (fast_normal_computation ? "true" : "false") << std::endl;
    }
};

Open3DNormalProvider::Open3DNormalProvider() 
    : pImpl(std::make_unique<Impl>()) {
}

Open3DNormalProvider::~Open3DNormalProvider() = default;

bool Open3DNormalProvider::isAvailable() const {
    // Check if Open3D is properly linked and available
    try {
        // Try to create a small point cloud to verify Open3D works
        auto test_pcd = std::make_shared<open3d::geometry::PointCloud>();
        test_pcd->points_.push_back(Eigen::Vector3d(0, 0, 0));
        return true;
    } catch (...) {
        return false;
    }
}

void Open3DNormalProvider::setKNeighbors(int k) {
    pImpl->k_neighbors = k;
    pImpl->use_knn = true;
}

void Open3DNormalProvider::setSearchRadius(float radius) {
    pImpl->search_radius = radius;
    pImpl->use_knn = false;
}

void Open3DNormalProvider::setFastNormalComputation(bool fast) {
    pImpl->fast_normal_computation = fast;
}

bool Open3DNormalProvider::estimateNormals(
    const float3* d_points,
    size_t num_points,
    float3* d_normals,
    cudaStream_t stream
) {
    if (!d_points || !d_normals || num_points == 0) {
        std::cerr << "[OPEN3D PROVIDER] Invalid input parameters" << std::endl;
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Step 1: Copy points from GPU to CPU
    std::vector<float3> h_points(num_points);
    cudaError_t err = cudaMemcpyAsync(h_points.data(), d_points, 
                                      num_points * sizeof(float3), 
                                      cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        std::cerr << "[OPEN3D PROVIDER] Failed to copy points from GPU: " 
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << "[OPEN3D PROVIDER] Stream synchronization failed: " 
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    auto copy_d2h_time = std::chrono::high_resolution_clock::now();
    
    // Step 2: Convert to Open3D point cloud
    auto pcd = std::make_shared<open3d::geometry::PointCloud>();
    pcd->points_.reserve(num_points);
    
    for (size_t i = 0; i < num_points; ++i) {
        pcd->points_.emplace_back(
            h_points[i].x, 
            h_points[i].y, 
            h_points[i].z
        );
    }
    
    auto convert_time = std::chrono::high_resolution_clock::now();
    
    // Step 3: Estimate normals using Open3D
    if (pImpl->use_knn) {
        pcd->EstimateNormals(
            open3d::geometry::KDTreeSearchParamKNN(pImpl->k_neighbors),
            pImpl->fast_normal_computation
        );
    } else {
        pcd->EstimateNormals(
            open3d::geometry::KDTreeSearchParamRadius(pImpl->search_radius),
            pImpl->fast_normal_computation
        );
    }
    
    auto normal_time = std::chrono::high_resolution_clock::now();
    
    // Optional: Orient normals towards camera if configured
    if (CONFIG_BOOL("MESH_ORIENT_NORMALS_TO_CAMERA",
                    config::NormalProviderConfig::DEFAULT_ORIENT_NORMALS_TO_CAMERA)) {
        // Get camera position from environment or use default
        float cam_x = CONFIG_FLOAT("MESH_CAMERA_POS_X", 0.0f);
        float cam_y = CONFIG_FLOAT("MESH_CAMERA_POS_Y", 0.0f);
        float cam_z = CONFIG_FLOAT("MESH_CAMERA_POS_Z", 0.0f);
        
        pcd->OrientNormalsTowardsCameraLocation(Eigen::Vector3d(cam_x, cam_y, cam_z));
    }
    
    // Step 4: Copy normals back to GPU
    std::vector<float3> h_normals(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        const auto& n = pcd->normals_[i];
        h_normals[i] = make_float3(
            static_cast<float>(n[0]), 
            static_cast<float>(n[1]), 
            static_cast<float>(n[2])
        );
    }
    
    err = cudaMemcpyAsync(d_normals, h_normals.data(),
                          num_points * sizeof(float3),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::cerr << "[OPEN3D PROVIDER] Failed to copy normals to GPU: " 
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << "[OPEN3D PROVIDER] Stream synchronization failed: " 
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    auto copy_h2d_time = std::chrono::high_resolution_clock::now();
    
    // Log timing information if enabled
    if (CONFIG_BOOL("MESH_LOG_NORMAL_TIMING", 
                    config::NormalProviderConfig::DEFAULT_LOG_NORMAL_TIMING)) {
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            copy_h2d_time - start_time).count();
        auto d2h_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            copy_d2h_time - start_time).count();
        auto convert_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            convert_time - copy_d2h_time).count();
        auto normal_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            normal_time - convert_time).count();
        auto h2d_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            copy_h2d_time - normal_time).count();
        
        std::cout << "[OPEN3D PROVIDER] Normal estimation timing breakdown:" << std::endl;
        std::cout << "  - GPU->CPU transfer: " << d2h_ms << " ms" << std::endl;
        std::cout << "  - Point cloud conversion: " << convert_ms << " ms" << std::endl;
        std::cout << "  - Normal computation: " << normal_ms << " ms" << std::endl;
        std::cout << "  - CPU->GPU transfer: " << h2d_ms << " ms" << std::endl;
        std::cout << "  - Total time: " << total_ms << " ms" << std::endl;
        std::cout << "  - Points processed: " << num_points << std::endl;
        std::cout << "  - Method: " << (pImpl->use_knn ? "KNN" : "Radius") 
                  << " (k=" << pImpl->k_neighbors << ", r=" << pImpl->search_radius << ")" << std::endl;
    }
    
    return true;
}

} // namespace mesh_service

#endif // HAS_OPEN3D