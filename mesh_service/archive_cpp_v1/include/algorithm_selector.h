#pragma once

#include "algorithms/algorithm_base.h"
#include <memory>
#include <unordered_map>

namespace mesh_service {

class AlgorithmSelector {
public:
    AlgorithmSelector();
    ~AlgorithmSelector();
    
    // Initialize all algorithms
    bool initialize();
    
    // Select algorithm based on conditions
    ReconstructionMethod selectAlgorithm(
        float camera_velocity,
        size_t point_count,
        float scene_complexity
    );
    
    // Get algorithm instance
    std::shared_ptr<ReconstructionAlgorithm> getAlgorithm(
        ReconstructionMethod method
    );
    
    // Process with automatic selection
    bool processWithAutoSelect(
        const float3* d_points,
        const float3* d_normals,
        size_t num_points,
        const float* camera_pose,
        float camera_velocity,
        MeshUpdate& output,
        cudaStream_t stream = 0
    );

private:
    std::unordered_map<ReconstructionMethod, 
                      std::shared_ptr<ReconstructionAlgorithm>> algorithms_;
    
    // Thresholds for algorithm selection
    struct {
        float velocity_threshold_high = 0.5f;  // m/s
        float velocity_threshold_low = 0.3f;   // m/s
        size_t point_count_threshold = 100000;
        float complexity_threshold = 0.7f;
    } thresholds_;
    
    ReconstructionMethod current_method_;
    
    // Hysteresis to prevent rapid switching
    int method_stable_frames_ = 0;
    static constexpr int SWITCH_STABILITY_FRAMES = 10;
};

} // namespace mesh_service