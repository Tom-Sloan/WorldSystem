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
    float3 camera_position,
    float fallback_distance,
    float default_z,
    int debug_normal_count
) {
    // Handle both 1D and 2D grids
    size_t idx = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
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
    
    if (dist > fallback_distance) {
        // Normalize the direction vector to get the normal
        normals[idx] = make_float3(
            cam_to_point.x / dist,
            cam_to_point.y / dist,
            cam_to_point.z / dist
        );
    } else {
        // Point is too close to camera, use default up vector
        normals[idx] = make_float3(0.0f, 0.0f, default_z);
    }
    
    // Debug output for first few normals
    if (idx < debug_normal_count) {
        printf("[CAMERA NORMAL] Point %lu: pos=[%.3f,%.3f,%.3f], normal=[%.3f,%.3f,%.3f], dist=%.3f\n",
               (unsigned long)idx, point.x, point.y, point.z, 
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
    
    // Check for valid grid dimensions BEFORE creating the grid
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // Calculate 1D grid size
    size_t grid_size_1d = (num_points + block.x - 1) / block.x;
    
    dim3 grid;
    if (grid_size_1d <= prop.maxGridSize[0]) {
        // Use 1D grid
        grid = dim3(grid_size_1d, 1, 1);
    } else {
        // Use 2D grid
        size_t grid_y = (grid_size_1d + prop.maxGridSize[0] - 1) / prop.maxGridSize[0];
        grid = dim3(prop.maxGridSize[0], grid_y, 1);
        std::cout << "[CAMERA NORMAL PROVIDER] Using 2D grid: " << grid.x << " x " << grid.y << std::endl;
    }
    
    if (CONFIG_BOOL("MESH_LOG_NORMAL_TIMING", 
                    config::NormalProviderConfig::DEFAULT_LOG_NORMAL_TIMING)) {
        std::cout << "[CAMERA NORMAL PROVIDER] Computing normals for " << num_points 
                  << " points with grid " << grid.x << " x " << grid.y << " x " << grid.z
                  << " and block " << block.x << std::endl;
    }
    
    // Get configuration values on host side
    float fallback_distance = CONFIG_FLOAT("MESH_CAMERA_FALLBACK_DISTANCE", 
                                          config::NormalProviderConfig::DEFAULT_CAMERA_FALLBACK_DISTANCE);
    float default_z = CONFIG_FLOAT("MESH_CAMERA_DEFAULT_Z",
                                  config::NormalProviderConfig::DEFAULT_CAMERA_DEFAULT_Z);
    int debug_normal_count = CONFIG_INT("MESH_DEBUG_NORMAL_COUNT", 
                                       config::NormalProviderConfig::DEFAULT_DEBUG_NORMAL_COUNT);
    
    // Launch kernel with configuration values as parameters
    computeCameraBasedNormalsKernel<<<grid, block, 0, stream>>>(
        d_points, d_normals, num_points, camera_position_,
        fallback_distance, default_z, debug_normal_count
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CAMERA NORMAL PROVIDER] Kernel launch failed: " 
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Synchronize stream to ensure kernel completion before checking for execution errors
    if (stream != 0) {
        err = cudaStreamSynchronize(stream);
    } else {
        err = cudaDeviceSynchronize();
    }
    
    if (err != cudaSuccess) {
        std::cerr << "[CAMERA NORMAL PROVIDER] Kernel execution failed: " 
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    return true;
}

} // namespace mesh_service