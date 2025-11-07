#pragma once

namespace mesh_service {
namespace config {

// Poisson-specific configuration that doesn't fit in general config
struct PoissonConfig {
    // Adaptive quality settings for SLAM3R
    struct AdaptiveQuality {
        static constexpr int OCTREE_DEPTH_STATIONARY = 8;      // High quality when still
        static constexpr int OCTREE_DEPTH_SLOW = 7;            // Medium quality
        static constexpr int OCTREE_DEPTH_FAST = 6;            // Low quality when moving
        
        static constexpr float VELOCITY_THRESHOLD_STATIONARY = 0.01f;  // m/s
        static constexpr float VELOCITY_THRESHOLD_SLOW = 0.1f;         // m/s
        static constexpr float VELOCITY_THRESHOLD_FAST = 0.5f;         // m/s
    };
    
    // SLAM3R-specific settings
    struct SLAM3RIntegration {
        static constexpr float CONFIDENCE_THRESHOLD_I2P = 10.0f;   // From SLAM3R config
        static constexpr float CONFIDENCE_THRESHOLD_L2W = 12.0f;   // From SLAM3R config
        static constexpr bool USE_CONFIDENCE_WEIGHTING = true;
        static constexpr bool IGNORE_CAMERA_POSE = true;           // Don't use unreliable poses
    };
    
    // Memory management
    struct Memory {
        static constexpr size_t MAX_POINTS_PER_BATCH = 500000;
        static constexpr size_t GPU_TO_CPU_TRANSFER_BLOCK = 65536;
        static constexpr size_t CPU_TO_GPU_TRANSFER_BLOCK = 65536;
    };
    
    // Solver parameters
    struct Solver {
        static constexpr int MAX_ITERATIONS = 100;
        static constexpr float CONVERGENCE_TOLERANCE = 1e-7f;
        static constexpr float REGULARIZATION_WEIGHT = 0.0001f;
    };
};

} // namespace config
} // namespace mesh_service