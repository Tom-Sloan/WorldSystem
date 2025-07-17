#pragma once

#include <cuda_runtime.h>
#include <memory>

namespace mesh_service {

// Abstract interface for normal estimation
class INormalProvider {
public:
    virtual ~INormalProvider() = default;
    
    // Estimate normals for point cloud
    virtual bool estimateNormals(
        const float3* d_points,
        size_t num_points,
        float3* d_normals,
        cudaStream_t stream = 0
    ) = 0;
    
    // Get provider name for logging
    virtual const char* getName() const = 0;
    
    // Check if provider is available
    virtual bool isAvailable() const = 0;
};

// Numeric provider IDs for cleaner configuration
enum NormalProviderType {
    PROVIDER_CAMERA_BASED = 0,    // Fast, low quality (default)
    PROVIDER_OPEN3D = 1,          // High quality, KD-tree based
    PROVIDER_NANOFLANN = 2,       // CPU KD-tree (future)
    PROVIDER_PCL_GPU = 3,         // PCL GPU (future)
    PROVIDER_GPU_CUSTOM = 4       // Custom GPU kernel (future)
};

// Factory for creating normal providers
class NormalProviderFactory {
public:
    static const char* getProviderTypeName(int type) {
        switch(type) {
            case PROVIDER_CAMERA_BASED: return "Camera-based (fast)";
            case PROVIDER_OPEN3D: return "Open3D (quality)";
            case PROVIDER_NANOFLANN: return "Nanoflann (CPU)";
            case PROVIDER_PCL_GPU: return "PCL GPU";
            case PROVIDER_GPU_CUSTOM: return "GPU Custom";
            default: return "Unknown";
        }
    }
    
    static std::unique_ptr<INormalProvider> create(int provider_id);
    static std::unique_ptr<INormalProvider> createFromEnv(); // Create from MESH_NORMAL_PROVIDER env var
};


} // namespace mesh_service