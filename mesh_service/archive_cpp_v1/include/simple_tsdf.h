#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace mesh_service {

// Simple TSDF implementation for NVIDIA Marching Cubes
class SimpleTSDF {
public:
    SimpleTSDF();
    ~SimpleTSDF();
    
    // Initialize TSDF volume
    void initialize(const float3& volume_min, const float3& volume_max, float voxel_size);
    
    // Integrate point cloud into TSDF
    void integrate(
        const float3* d_points,
        const float3* d_normals,
        size_t num_points,
        const float* camera_pose,
        cudaStream_t stream = 0
    );
    
    // Get TSDF volume data
    float* getTSDFVolume() const;
    float* getWeightVolume() const;
    
    // Get volume dimensions
    int3 getVolumeDims() const;
    float3 getVolumeOrigin() const;
    float getVoxelSize() const;
    
    // Reset TSDF
    void reset();
    
    // Get memory usage
    size_t getMemoryUsage() const;
    
    // Set truncation distance
    void setTruncationDistance(float distance);
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace mesh_service