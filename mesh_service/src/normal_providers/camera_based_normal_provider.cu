#include "normal_providers/camera_based_normal_provider.h"
#include "config/configuration_manager.h"
#include "config/normal_provider_config.h"
#include <iostream>
#include <cuda_runtime.h>

namespace mesh_service {

// CUDA kernel for camera-based normal computation
__global__ void computeCameraBasedNormalsKernel(
    const float3* points,
    float3* normals,
    size_t num_points,
    float3 camera_position
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    float3 point = points[idx];
    float3 cam_to_point = make_float3(
        point.x - camera_position.x,
        point.y - camera_position.y,
        point.z - camera_position.z
    );
    
    float dist = sqrtf(cam_to_point.x * cam_to_point.x + 
                       cam_to_point.y * cam_to_point.y + 
                       cam_to_point.z * cam_to_point.z);
    
    if (dist > CONFIG_FLOAT("MESH_CAMERA_FALLBACK_DISTANCE", 
                           config::NormalProviderConfig::DEFAULT_CAMERA_FALLBACK_DISTANCE)) {
        // Normalize the direction vector to get the normal
        normals[idx] = make_float3(
            cam_to_point.x / dist,
            cam_to_point.y / dist,
            cam_to_point.z / dist
        );
    } else {
        // Point is too close to camera, use default up vector
        normals[idx] = make_float3(0.0f, 0.0f, 
                                  CONFIG_FLOAT("MESH_CAMERA_DEFAULT_Z",
                                              config::NormalProviderConfig::DEFAULT_CAMERA_DEFAULT_Z));
    }
    
    // Debug output for first few normals
    if (idx < CONFIG_INT("MESH_DEBUG_NORMAL_COUNT", 
                        config::NormalProviderConfig::DEFAULT_DEBUG_NORMAL_COUNT)) {
        printf("[CAMERA NORMAL] Point %d: pos=[%.3f,%.3f,%.3f], normal=[%.3f,%.3f,%.3f], dist=%.3f\n",
               idx, point.x, point.y, point.z, 
               normals[idx].x, normals[idx].y, normals[idx].z, dist);
    }
}

// Constructor
CameraBasedNormalProvider::CameraBasedNormalProvider(const float3& camera_pos)
    : camera_position_(camera_pos) {
    if (CONFIG_BOOL("MESH_LOG_NORMAL_TIMING", 
                    config::NormalProviderConfig::DEFAULT_LOG_NORMAL_TIMING)) {
        std::cout << "[CAMERA NORMAL PROVIDER] Initialized with camera position: ["
                  << camera_position_.x << ", " 
                  << camera_position_.y << ", " 
                  << camera_position_.z << "]" << std::endl;
    }
}

// Estimate normals implementation
bool CameraBasedNormalProvider::estimateNormals(
    const float3* d_points,
    size_t num_points,
    float3* d_normals,
    cudaStream_t stream
) {
    if (!d_points || !d_normals || num_points == 0) {
        std::cerr << "[CAMERA NORMAL PROVIDER] Invalid input parameters" << std::endl;
        return false;
    }
    
    // Configure kernel launch parameters
    dim3 block(256);
    dim3 grid((num_points + block.x - 1) / block.x);
    
    if (CONFIG_BOOL("MESH_LOG_NORMAL_TIMING", 
                    config::NormalProviderConfig::DEFAULT_LOG_NORMAL_TIMING)) {
        std::cout << "[CAMERA NORMAL PROVIDER] Computing normals for " << num_points 
                  << " points" << std::endl;
    }
    
    // Launch kernel
    computeCameraBasedNormalsKernel<<<grid, block, 0, stream>>>(
        d_points, d_normals, num_points, camera_position_
    );
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CAMERA NORMAL PROVIDER] Kernel launch failed: " 
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    return true;
}

} // namespace mesh_service