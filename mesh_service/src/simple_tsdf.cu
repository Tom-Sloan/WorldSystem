#include "simple_tsdf.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <iostream>

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
    size_t num_points,
    int3 volume_dims,
    float3 volume_origin,
    float voxel_size,
    float truncation_distance,
    const Matrix4 world_to_volume
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    float3 point = points[idx];
    
    // Transform point to volume space
    float3 voxel_pos = world_to_volume.transformPoint(point);
    
    // Debug: Print first few transformed points
    if (idx < 5) {
        printf("[TSDF KERNEL] Point %d: world=[%.3f,%.3f,%.3f] -> voxel=[%.3f,%.3f,%.3f]\n",
               idx, point.x, point.y, point.z, voxel_pos.x, voxel_pos.y, voxel_pos.z);
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
                
                // Compute signed distance
                float distance = length(voxel_center - point);
                
                // Skip if outside truncation
                if (distance > truncation_distance) continue;
                
                // Compute TSDF value (negative inside surface)
                float tsdf_value = distance;
                
                // Update TSDF using weighted average
                int voxel_idx = x + y * volume_dims.x + z * volume_dims.x * volume_dims.y;
                float old_tsdf = tsdf_volume[voxel_idx];
                float old_weight = weight_volume[voxel_idx];
                float new_weight = 1.0f;
                
                float updated_weight = old_weight + new_weight;
                if (updated_weight > 0.0f) {
                    float new_tsdf = (old_tsdf * old_weight + tsdf_value * new_weight) / updated_weight;
                    tsdf_volume[voxel_idx] = new_tsdf;
                    weight_volume[voxel_idx] = min(updated_weight, 100.0f); // Cap weight
                    
                    // Debug: Print first few TSDF updates
                    if (idx < 2 && voxel_idx < 1000) {
                        printf("[TSDF UPDATE] Point %d updated voxel %d: old_tsdf=%.3f, new_tsdf=%.3f, weight=%.1f\n",
                               idx, voxel_idx, old_tsdf, new_tsdf, updated_weight);
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
    
    tsdf_volume[idx] = truncation_distance;
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
    
    Impl() : voxel_size_(0.05f), truncation_distance_(0.15f) {}
    
    void initialize(const float3& volume_min, const float3& volume_max, float voxel_size) {
        volume_min_ = volume_min;
        volume_max_ = volume_max;
        voxel_size_ = voxel_size;
        volume_origin_ = volume_min;
        
        // Calculate dimensions
        volume_dims_ = make_int3(
            static_cast<int>((volume_max.x - volume_min.x) / voxel_size + 0.5f),
            static_cast<int>((volume_max.y - volume_min.y) / voxel_size + 0.5f),
            static_cast<int>((volume_max.z - volume_min.z) / voxel_size + 0.5f)
        );
        
        size_t num_voxels = volume_dims_.x * volume_dims_.y * volume_dims_.z;
        
        // Allocate memory
        d_tsdf_volume_.resize(num_voxels);
        d_weight_volume_.resize(num_voxels);
        
        // Initialize volumes
        thrust::fill(d_tsdf_volume_.begin(), d_tsdf_volume_.end(), truncation_distance_);
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
    if (num_points == 0) {
        std::cout << "[TSDF DEBUG] integrate() called with 0 points, returning" << std::endl;
        return;
    }
    
    std::cout << "[TSDF DEBUG] integrate() called with " << num_points << " points" << std::endl;
    
    // Debug: Check first few points on host
    float3 h_points[10];
    size_t debug_count = std::min(num_points, size_t(10));
    cudaMemcpy(h_points, d_points, debug_count * sizeof(float3), cudaMemcpyDeviceToHost);
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
    
    cuda::integrateTSDFKernel<<<grid, block, 0, stream>>>(
        pImpl->d_tsdf_volume_.data().get(),
        pImpl->d_weight_volume_.data().get(),
        d_points,
        num_points,
        pImpl->volume_dims_,
        pImpl->volume_origin_,
        pImpl->voxel_size_,
        pImpl->truncation_distance_,
        world_to_volume
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[TSDF ERROR] Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
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