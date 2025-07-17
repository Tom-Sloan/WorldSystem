#pragma once

#include "normal_provider.h"
#include <cuda_runtime.h>

namespace mesh_service {

// Camera-based normal provider - fast but low quality
// Computes normals as the direction from camera to point
class CameraBasedNormalProvider : public INormalProvider {
private:
    float3 camera_position_;
    
public:
    explicit CameraBasedNormalProvider(const float3& camera_pos = make_float3(0.0f, 0.0f, 0.0f));
    
    bool estimateNormals(
        const float3* d_points,
        size_t num_points,
        float3* d_normals,
        cudaStream_t stream = 0
    ) override;
    
    const char* getName() const override { return "Camera-based (fast)"; }
    bool isAvailable() const override { return true; }
    
    void setCameraPosition(const float3& pos) { camera_position_ = pos; }
    float3 getCameraPosition() const { return camera_position_; }
};

} // namespace mesh_service