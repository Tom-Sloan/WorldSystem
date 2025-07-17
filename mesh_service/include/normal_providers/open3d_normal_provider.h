#pragma once

#ifdef HAS_OPEN3D

#include "normal_provider.h"
#include <cuda_runtime.h>
#include <memory>

namespace mesh_service {

// Open3D-based normal provider - high quality using KD-tree
class Open3DNormalProvider : public INormalProvider {
public:
    Open3DNormalProvider();
    ~Open3DNormalProvider();
    
    bool estimateNormals(
        const float3* d_points,
        size_t num_points,
        float3* d_normals,
        cudaStream_t stream = 0
    ) override;
    
    const char* getName() const override { return "Open3D (KD-tree based)"; }
    bool isAvailable() const override;
    
    // Configuration methods
    void setKNeighbors(int k);
    void setSearchRadius(float radius);
    void setFastNormalComputation(bool fast);
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace mesh_service

#endif // HAS_OPEN3D