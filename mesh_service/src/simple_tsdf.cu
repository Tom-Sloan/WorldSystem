#include "simple_tsdf.h"
#include "config/configuration_manager.h"
#include "config/mesh_service_config.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <iostream>
#include <chrono>

namespace mesh_service {

// CUDA kernels for TSDF integration
namespace cuda {

__device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// Removed - using Matrix4 struct instead

__device__ float length(const float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

// Simple 4x4 matrix structure
struct Matrix4 {
    float m[16];
    
    __device__ float3 transformPoint(const float3& p) const {
        return make_float3(
            m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12],
            m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
            m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14]
        );
    }
};

__global__ void integrateTSDFKernel(
    float* tsdf_volume,
    float* weight_volume,
    const float3* points,
    const float3* normals,
    size_t num_points,
    int3 volume_dims,
    float3 volume_origin,
    float voxel_size,
    float truncation_distance,
    const Matrix4 world_to_volume,
    const float3 camera_position
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    float3 point = points[idx];
    float3 normal = normals ? normals[idx] : make_float3(0.0f, 0.0f, 1.0f);
    
    // Normalize the normal in case it's not unit length
    float normal_len = length(normal);
    if (normal_len > 0.0f) {
        normal = make_float3(normal.x / normal_len, normal.y / normal_len, normal.z / normal_len);
    }
    
    // Fallback: Use camera-to-point direction as normal if we have invalid normals
    // This creates wrong but usable normals for testing
    if (normal_len < 0.1f || (fabs(normal.x + 1.0f) < 0.01f && fabs(normal.y) < 0.01f && fabs(normal.z) < 0.01f)) {
        // Normal is invalid (zero or the constant [-1,0,0])
        float3 cam_to_point = make_float3(
            point.x - camera_position.x,
            point.y - camera_position.y,
            point.z - camera_position.z
        );
        float cam_dist = length(cam_to_point);
        if (cam_dist > 0.001f) {
            // Use outward direction from camera as fake normal
            normal = make_float3(
                cam_to_point.x / cam_dist,
                cam_to_point.y / cam_dist,
                cam_to_point.z / cam_dist
            );
            if (idx < 5) {
                printf("[TSDF] Using camera-based normal for point %d: normal=[%.3f,%.3f,%.3f]\n",
                       idx, normal.x, normal.y, normal.z);
            }
        }
    }
    
    // Transform point to volume space
    float3 voxel_pos = world_to_volume.transformPoint(point);
    
    // Debug: Print first few transformed points and normals
    if (idx < 5) {
        printf("[TSDF KERNEL] Point %d: world=[%.3f,%.3f,%.3f] -> voxel=[%.3f,%.3f,%.3f], normal=[%.3f,%.3f,%.3f]\n",
               idx, point.x, point.y, point.z, voxel_pos.x, voxel_pos.y, voxel_pos.z,
               normal.x, normal.y, normal.z);
    }
    
    // Get bounding voxels
    int3 voxel_min = make_int3(
        max(0, __float2int_rd(voxel_pos.x - truncation_distance / voxel_size)),
        max(0, __float2int_rd(voxel_pos.y - truncation_distance / voxel_size)),
        max(0, __float2int_rd(voxel_pos.z - truncation_distance / voxel_size))
    );
    
    int3 voxel_max = make_int3(
        min(volume_dims.x - 1, __float2int_ru(voxel_pos.x + truncation_distance / voxel_size)),
        min(volume_dims.y - 1, __float2int_ru(voxel_pos.y + truncation_distance / voxel_size)),
        min(volume_dims.z - 1, __float2int_ru(voxel_pos.z + truncation_distance / voxel_size))
    );
    
    // Debug: Check if voxel is within bounds
    if (idx < 5) {
        printf("[TSDF KERNEL] Point %d voxel bounds: min=[%d,%d,%d], max=[%d,%d,%d], volume_dims=[%d,%d,%d]\n",
               idx, voxel_min.x, voxel_min.y, voxel_min.z, 
               voxel_max.x, voxel_max.y, voxel_max.z,
               volume_dims.x, volume_dims.y, volume_dims.z);
        
        // Check if point is outside volume
        if (voxel_pos.x < 0 || voxel_pos.x >= volume_dims.x ||
            voxel_pos.y < 0 || voxel_pos.y >= volume_dims.y ||
            voxel_pos.z < 0 || voxel_pos.z >= volume_dims.z) {
            printf("[TSDF WARNING] Point %d is outside volume bounds!\n", idx);
        }
    }
    
    // Update TSDF in neighborhood
    for (int z = voxel_min.z; z <= voxel_max.z; z++) {
        for (int y = voxel_min.y; y <= voxel_max.y; y++) {
            for (int x = voxel_min.x; x <= voxel_max.x; x++) {
                // Get voxel center in world space
                float3 voxel_center = make_float3(
                    volume_origin.x + (x + 0.5f) * voxel_size,
                    volume_origin.y + (y + 0.5f) * voxel_size,
                    volume_origin.z + (z + 0.5f) * voxel_size
                );
                
                // Compute distance and direction
                float3 diff = voxel_center - point;
                float distance = length(diff);
                
                // Skip if outside truncation
                if (distance > truncation_distance) continue;
                
                // IMPROVED: Camera carving method for better inside/outside detection
                // Key insight: Space between camera and point should be EMPTY (positive TSDF)
                // Space beyond the point should be UNKNOWN/OCCUPIED (negative TSDF)
                
                float sign = 1.0f;
                
                // Calculate if voxel is between camera and point
                float3 cam_to_point = make_float3(point.x - camera_position.x,
                                                   point.y - camera_position.y,
                                                   point.z - camera_position.z);
                float3 cam_to_voxel = make_float3(voxel_center.x - camera_position.x,
                                                   voxel_center.y - camera_position.y,
                                                   voxel_center.z - camera_position.z);
                
                float cam_to_point_dist = length(cam_to_point);
                float cam_to_voxel_dist = length(cam_to_voxel);
                
                // Check if voxel is along the ray from camera to point
                if (cam_to_point_dist > 0.001f && cam_to_voxel_dist > 0.001f) {
                    float3 ray_dir = make_float3(cam_to_point.x / cam_to_point_dist,
                                                  cam_to_point.y / cam_to_point_dist,
                                                  cam_to_point.z / cam_to_point_dist);
                    float3 voxel_dir = make_float3(cam_to_voxel.x / cam_to_voxel_dist,
                                                    cam_to_voxel.y / cam_to_voxel_dist,
                                                    cam_to_voxel.z / cam_to_voxel_dist);
                    
                    // Check alignment (how close voxel is to the camera-point ray)
                    float alignment = ray_dir.x * voxel_dir.x + ray_dir.y * voxel_dir.y + ray_dir.z * voxel_dir.z;
                    
                    if (alignment > 0.95f) { // Voxel is along the ray
                        if (cam_to_voxel_dist < cam_to_point_dist - voxel_size) {
                            // Voxel is between camera and point: EMPTY space
                            sign = 1.0f;
                        } else if (cam_to_voxel_dist > cam_to_point_dist + voxel_size) {
                            // Voxel is beyond the point: OCCUPIED space
                            sign = -1.0f;
                        } else {
                            // Voxel is near the surface: use normal-based method
                            float3 direction = make_float3(diff.x / distance, diff.y / distance, diff.z / distance);
                            float dot = direction.x * normal.x + direction.y * normal.y + direction.z * normal.z;
                            sign = (dot > 0.0f) ? 1.0f : -1.0f;
                        }
                    } else {
                        // Voxel is not on the ray: use normal-based method
                        if (distance > 0.0f) {
                            float3 direction = make_float3(diff.x / distance, diff.y / distance, diff.z / distance);
                            float dot = direction.x * normal.x + direction.y * normal.y + direction.z * normal.z;
                            sign = (dot > 0.0f) ? 1.0f : -1.0f;
                        }
                    }
                }
                
                // Create a thin band of negative values near the surface
                // This ensures marching cubes finds the zero crossing
                float tsdf_value = sign * distance;
                
                // For very close voxels, ensure we have both positive and negative values
                if (distance < voxel_size * 0.5f) {
                    // Use normal to determine sign more accurately
                    float3 to_voxel = make_float3(voxel_center.x - point.x, 
                                                   voxel_center.y - point.y, 
                                                   voxel_center.z - point.z);
                    float alignment = to_voxel.x * normal.x + to_voxel.y * normal.y + to_voxel.z * normal.z;
                    tsdf_value = alignment > 0.0f ? distance : -distance;
                }
                
                // Update TSDF using weighted average
                int voxel_idx = x + y * volume_dims.x + z * volume_dims.x * volume_dims.y;
                
                // Debug output for carving visualization
                if (idx < 5 && voxel_idx < 1000) {
                    const char* space_type = "UNKNOWN";
                    if (cam_to_point_dist > 0.001f && cam_to_voxel_dist > 0.001f) {
                        float3 ray_dir = make_float3(cam_to_point.x / cam_to_point_dist,
                                                      cam_to_point.y / cam_to_point_dist,
                                                      cam_to_point.z / cam_to_point_dist);
                        float3 voxel_dir = make_float3(cam_to_voxel.x / cam_to_voxel_dist,
                                                        cam_to_voxel.y / cam_to_voxel_dist,
                                                        cam_to_voxel.z / cam_to_voxel_dist);
                        float alignment = ray_dir.x * voxel_dir.x + ray_dir.y * voxel_dir.y + ray_dir.z * voxel_dir.z;
                        
                        if (alignment > 0.95f) {
                            if (cam_to_voxel_dist < cam_to_point_dist - voxel_size) {
                                space_type = "EMPTY";
                            } else if (cam_to_voxel_dist > cam_to_point_dist + voxel_size) {
                                space_type = "OCCUPIED";
                            } else {
                                space_type = "SURFACE";
                            }
                        }
                    }
                    printf("[TSDF CARVING] Point %d: Voxel %d is %s (sign=%.0f, dist=%.3f)\n", 
                           idx, voxel_idx, space_type, sign, distance);
                }
                
                float old_tsdf = tsdf_volume[voxel_idx];
                float old_weight = weight_volume[voxel_idx];
                float new_weight = 1.0f;
                
                float updated_weight = old_weight + new_weight;
                if (updated_weight > 0.0f) {
                    float new_tsdf = (old_tsdf * old_weight + tsdf_value * new_weight) / updated_weight;
                    tsdf_volume[voxel_idx] = new_tsdf;
                    weight_volume[voxel_idx] = min(updated_weight, 100.0f); // Cap weight
                    
                    // Debug: Print first few TSDF updates including sign
                    if (idx < 2 && voxel_idx < 1000) {
                        printf("[TSDF UPDATE] Point %d updated voxel %d: old_tsdf=%.3f, new_tsdf=%.3f, weight=%.1f\n",
                               idx, voxel_idx, old_tsdf, new_tsdf, updated_weight);
                    }
                    
                    // Additional debug: Check for sign changes
                    if (idx < 5 && new_tsdf < 0.0f && voxel_idx < 10000) {
                        printf("[TSDF SIGN] Negative TSDF at voxel %d: %.3f (from point %d)\n",
                               voxel_idx, new_tsdf, idx);
                    }
                }
            }
        }
    }
}

__global__ void resetVolumeKernel(
    float* tsdf_volume,
    float* weight_volume,
    size_t num_voxels,
    float truncation_distance
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_voxels) return;
    
    // Initialize with large positive value (far from any surface)
    // Don't use truncation_distance as that makes everything appear as a surface
    tsdf_volume[idx] = 1.0f;  // 1 meter = very far from surface
    weight_volume[idx] = 0.0f;
}

} // namespace cuda

// SimpleTSDF implementation
class SimpleTSDF::Impl {
public:
    // Volume properties
    float3 volume_min_;
    float3 volume_max_;
    float voxel_size_;
    float truncation_distance_;
    int3 volume_dims_;
    float3 volume_origin_;
    
    // Device memory
    thrust::device_vector<float> d_tsdf_volume_;
    thrust::device_vector<float> d_weight_volume_;
    
    // Transform matrix
    thrust::device_vector<float> d_world_to_volume_;
    
    Impl() : voxel_size_(CONFIG_FLOAT("MESH_VOXEL_SIZE", mesh_service::config::AlgorithmConfig::DEFAULT_VOXEL_SIZE)), 
             truncation_distance_(CONFIG_FLOAT("MESH_TRUNCATION_DISTANCE", 0.1f)) {} // Reduced from 0.15f
    
    void initialize(const float3& volume_min, const float3& volume_max, float voxel_size) {
        volume_min_ = volume_min;
        volume_max_ = volume_max;
        voxel_size_ = voxel_size;
        volume_origin_ = volume_min;
        
        std::cout << "[SIMPLE TSDF INIT] Initializing TSDF volume:" << std::endl;
        std::cout << "  Volume min: [" << volume_min.x << ", " << volume_min.y << ", " << volume_min.z << "]" << std::endl;
        std::cout << "  Volume max: [" << volume_max.x << ", " << volume_max.y << ", " << volume_max.z << "]" << std::endl;
        std::cout << "  Voxel size: " << voxel_size << "m" << std::endl;
        
        // Calculate dimensions
        volume_dims_ = make_int3(
            static_cast<int>((volume_max.x - volume_min.x) / voxel_size + 0.5f),
            static_cast<int>((volume_max.y - volume_min.y) / voxel_size + 0.5f),
            static_cast<int>((volume_max.z - volume_min.z) / voxel_size + 0.5f)
        );
        
        size_t num_voxels = volume_dims_.x * volume_dims_.y * volume_dims_.z;
        std::cout << "  Volume dimensions: [" << volume_dims_.x << ", " << volume_dims_.y << ", " << volume_dims_.z << "]" << std::endl;
        std::cout << "  Total voxels: " << num_voxels << std::endl;
        
        // Allocate memory
        d_tsdf_volume_.resize(num_voxels);
        d_weight_volume_.resize(num_voxels);
        
        // Initialize volumes
        // Use large positive value for empty space, not truncation_distance
        thrust::fill(d_tsdf_volume_.begin(), d_tsdf_volume_.end(), 1.0f);  // 1 meter = far from surface
        thrust::fill(d_weight_volume_.begin(), d_weight_volume_.end(), 0.0f);
        
        // Setup world to volume transform
        d_world_to_volume_.resize(16);
        // Store in column-major order for CUDA kernel
        float h_transform[16] = {
            1.0f / voxel_size,      // m[0] 
            0,                      // m[1]
            0,                      // m[2] 
            0,                      // m[3]
            0,                      // m[4]
            1.0f / voxel_size,      // m[5]
            0,                      // m[6]
            0,                      // m[7]
            0,                      // m[8]
            0,                      // m[9]
            1.0f / voxel_size,      // m[10]
            0,                      // m[11]
            -volume_min.x / voxel_size,  // m[12] translation x
            -volume_min.y / voxel_size,  // m[13] translation y
            -volume_min.z / voxel_size,  // m[14] translation z
            1                            // m[15]
        };
        thrust::copy(h_transform, h_transform + 16, d_world_to_volume_.begin());
        
        std::cout << "SimpleTSDF initialized:" << std::endl;
        std::cout << "  Volume dims: " << volume_dims_.x << "x" << volume_dims_.y << "x" << volume_dims_.z << std::endl;
        std::cout << "  Voxel size: " << voxel_size_ << "m" << std::endl;
        std::cout << "  Memory usage: " << getMemoryUsage() / (1024.0f * 1024.0f) << " MB" << std::endl;
    }
    
    size_t getMemoryUsage() const {
        return (d_tsdf_volume_.size() + d_weight_volume_.size()) * sizeof(float);
    }
};

SimpleTSDF::SimpleTSDF() : pImpl(std::make_unique<Impl>()) {}
SimpleTSDF::~SimpleTSDF() = default;

void SimpleTSDF::initialize(const float3& volume_min, const float3& volume_max, float voxel_size) {
    pImpl->initialize(volume_min, volume_max, voxel_size);
}

void SimpleTSDF::integrate(
    const float3* d_points,
    const float3* d_normals,
    size_t num_points,
    const float* camera_pose,
    cudaStream_t stream
) {
    auto tsdf_start = std::chrono::high_resolution_clock::now();
    
    // Extract camera position from 4x4 pose matrix (translation part)
    float3 camera_position = make_float3(0.0f, 0.0f, 0.0f);
    if (camera_pose) {
        // Camera pose is 4x4 matrix in column-major format
        // Translation is in elements 12, 13, 14
        camera_position = make_float3(
            camera_pose[12],  // X translation
            camera_pose[13],  // Y translation 
            camera_pose[14]   // Z translation
        );
        std::cout << "[TSDF DEBUG] Camera position: [" 
                  << camera_position.x << ", " 
                  << camera_position.y << ", " 
                  << camera_position.z << "]" << std::endl;
    }
    if (num_points == 0) {
        std::cout << "[TSDF DEBUG] integrate() called with 0 points, returning" << std::endl;
        return;
    }
    
    std::cout << "[TSDF DEBUG] integrate() called with " << num_points << " points" << std::endl;
    std::cout << "[TSDF DEBUG] Normal provider: " << (d_normals ? "External normals provided" : "Camera-based fallback (improved carving)") << std::endl;
    std::cout << "[TSDF DEBUG] Integration method: Camera carving with ray-based empty space detection" << std::endl;
    std::cout << "[TSDF DEBUG] Truncation distance: " << pImpl->truncation_distance_ << " m" << std::endl;
    
    // Debug: Check first few points on host
    auto debug_copy_start = std::chrono::high_resolution_clock::now();
    float3 h_points[10];
    size_t debug_count = std::min(num_points, size_t(10));
    cudaMemcpy(h_points, d_points, debug_count * sizeof(float3), cudaMemcpyDeviceToHost);
    auto debug_copy_end = std::chrono::high_resolution_clock::now();
    auto debug_copy_us = std::chrono::duration_cast<std::chrono::microseconds>(debug_copy_end - debug_copy_start).count();
    std::cout << "[TIMING] TSDF debug copy: " << debug_copy_us << " µs" << std::endl;
    
    std::cout << "[TSDF DEBUG] First few points (world space):" << std::endl;
    for (size_t i = 0; i < debug_count; i++) {
        std::cout << "  Point " << i << ": [" << h_points[i].x << ", " 
                  << h_points[i].y << ", " << h_points[i].z << "]" << std::endl;
    }
    
    // Setup transform matrix
    cuda::Matrix4 world_to_volume;
    cudaMemcpy(&world_to_volume.m, pImpl->d_world_to_volume_.data().get(), 
               16 * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "[TSDF DEBUG] World-to-volume transform (column-major):" << std::endl;
    std::cout << "  Scale X: " << world_to_volume.m[0] << ", Translation X: " << world_to_volume.m[12] << std::endl;
    std::cout << "  Scale Y: " << world_to_volume.m[5] << ", Translation Y: " << world_to_volume.m[13] << std::endl;
    std::cout << "  Scale Z: " << world_to_volume.m[10] << ", Translation Z: " << world_to_volume.m[14] << std::endl;
    
    std::cout << "[TSDF DEBUG] TSDF volume params:" << std::endl;
    std::cout << "  Dims: " << pImpl->volume_dims_.x << "x" << pImpl->volume_dims_.y 
              << "x" << pImpl->volume_dims_.z << std::endl;
    std::cout << "  Origin: [" << pImpl->volume_origin_.x << ", " << pImpl->volume_origin_.y 
              << ", " << pImpl->volume_origin_.z << "]" << std::endl;
    std::cout << "  Voxel size: " << pImpl->voxel_size_ << std::endl;
    std::cout << "  Truncation distance: " << pImpl->truncation_distance_ << std::endl;
    
    // Launch integration kernel
    dim3 block(256);
    dim3 grid((num_points + block.x - 1) / block.x);
    
    auto kernel_start = std::chrono::high_resolution_clock::now();
    cuda::integrateTSDFKernel<<<grid, block, 0, stream>>>(
        pImpl->d_tsdf_volume_.data().get(),
        pImpl->d_weight_volume_.data().get(),
        d_points,
        d_normals,
        num_points,
        pImpl->volume_dims_,
        pImpl->volume_origin_,
        pImpl->voxel_size_,
        pImpl->truncation_distance_,
        world_to_volume,
        camera_position
    );
    
    // Synchronize to measure kernel time
    cudaStreamSynchronize(stream);
    auto kernel_end = std::chrono::high_resolution_clock::now();
    auto kernel_ms = std::chrono::duration_cast<std::chrono::milliseconds>(kernel_end - kernel_start).count();
    std::cout << "[TIMING] TSDF integration kernel: " << kernel_ms << " ms" << std::endl;
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[TSDF ERROR] Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    
    auto tsdf_end = std::chrono::high_resolution_clock::now();
    auto tsdf_total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tsdf_end - tsdf_start).count();
    std::cout << "[TIMING] Total TSDF integration: " << tsdf_total_ms << " ms" << std::endl;
}

float* SimpleTSDF::getTSDFVolume() const {
    return pImpl->d_tsdf_volume_.data().get();
}

float* SimpleTSDF::getWeightVolume() const {
    return pImpl->d_weight_volume_.data().get();
}

int3 SimpleTSDF::getVolumeDims() const {
    return pImpl->volume_dims_;
}

float3 SimpleTSDF::getVolumeOrigin() const {
    return pImpl->volume_origin_;
}

float SimpleTSDF::getVoxelSize() const {
    return pImpl->voxel_size_;
}

void SimpleTSDF::reset() {
    size_t num_voxels = pImpl->d_tsdf_volume_.size();
    
    dim3 block(256);
    dim3 grid((num_voxels + block.x - 1) / block.x);
    
    cuda::resetVolumeKernel<<<grid, block>>>(
        pImpl->d_tsdf_volume_.data().get(),
        pImpl->d_weight_volume_.data().get(),
        num_voxels,
        pImpl->truncation_distance_
    );
}

size_t SimpleTSDF::getMemoryUsage() const {
    return pImpl->getMemoryUsage();
}

void SimpleTSDF::setTruncationDistance(float distance) {
    pImpl->truncation_distance_ = distance;
}

} // namespace mesh_service