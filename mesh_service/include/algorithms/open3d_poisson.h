#pragma once

#include "algorithm_base.h"
#include <memory>

namespace mesh_service {

class Open3DPoisson : public ReconstructionAlgorithm {
public:
    Open3DPoisson();
    ~Open3DPoisson() override;
    
    bool initialize(const AlgorithmParams& params) override;
    
    bool reconstruct(
        const float3* d_points,
        const float3* d_normals,
        size_t num_points,
        const float* camera_pose,  // Ignored for Poisson
        MeshUpdate& output,
        cudaStream_t stream = 0
    ) override;
    
    size_t getMemoryUsage() const override;
    void reset() override;
    ReconstructionMethod getMethod() const override { 
        return ReconstructionMethod::OPEN3D_POISSON; 
    }

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace mesh_service