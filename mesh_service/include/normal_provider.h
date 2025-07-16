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

// Factory for creating normal providers
class NormalProviderFactory {
public:
    enum ProviderType {
        CAMERA_BASED,      // Fast but low quality
        NANOFLANN_CPU,     // CPU-based KD-tree
        GPU_OPTIMIZED,     // Custom GPU implementation
        PCL_GPU,           // PCL GPU (if available)
        OPEN3D             // Open3D (if available)
    };
    
    static const char* getProviderTypeName(ProviderType type) {
        switch(type) {
            case CAMERA_BASED: return "CAMERA_BASED";
            case NANOFLANN_CPU: return "NANOFLANN_CPU";
            case GPU_OPTIMIZED: return "GPU_OPTIMIZED";
            case PCL_GPU: return "PCL_GPU";
            case OPEN3D: return "OPEN3D";
            default: return "UNKNOWN";
        }
    }
    
    static std::unique_ptr<INormalProvider> create(ProviderType type);
    static std::unique_ptr<INormalProvider> createBest(); // Auto-select best available
};

// Simple camera-based provider (current fallback)
class CameraBasedNormalProvider : public INormalProvider {
private:
    float3 camera_position_;
    
public:
    explicit CameraBasedNormalProvider(const float3& camera_pos = make_float3(0,0,0));
    
    bool estimateNormals(
        const float3* d_points,
        size_t num_points,
        float3* d_normals,
        cudaStream_t stream = 0
    ) override;
    
    const char* getName() const override { return "CameraBasedNormals"; }
    bool isAvailable() const override { return true; }
    
    void setCameraPosition(const float3& pos) { camera_position_ = pos; }
};

} // namespace mesh_service