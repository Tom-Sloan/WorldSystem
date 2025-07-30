#pragma once

namespace mesh_service {
namespace config {

// ========== NORMAL PROVIDER CONFIGURATION ==========
struct NormalProviderConfig {
    // Default provider selection (0 = camera-based, 1 = open3d, etc.)
    static constexpr int DEFAULT_NORMAL_PROVIDER = 1;  // Changed to Open3D for quality
    
    // Camera-based provider settings
    static constexpr float DEFAULT_CAMERA_FALLBACK_DISTANCE = 0.001f;
    static constexpr float DEFAULT_CAMERA_DEFAULT_Z = 1.0f;
    
    // Open3D provider settings
    static constexpr int DEFAULT_OPEN3D_K_NEIGHBORS = 30;
    static constexpr float DEFAULT_OPEN3D_SEARCH_RADIUS = 0.1f;
    static constexpr bool DEFAULT_OPEN3D_FAST_NORMAL_COMPUTATION = true;
    
    // Common normal estimation parameters
    static constexpr bool DEFAULT_ORIENT_NORMALS_TO_CAMERA = true;
    static constexpr bool DEFAULT_FLIP_NORMALS_TOWARDS_VIEWPOINT = true;
    static constexpr float DEFAULT_NORMAL_CONSISTENCY_ANGLE = 0.7854f; // 45 degrees in radians
    
    // Performance/quality tradeoffs
    static constexpr size_t DEFAULT_MAX_POINTS_FOR_CPU_NORMAL = 100000;
    static constexpr size_t DEFAULT_MIN_NEIGHBORS_FOR_VALID_NORMAL = 3;
    
    // Debug settings
    static constexpr bool DEFAULT_LOG_NORMAL_TIMING = true;
    static constexpr int DEFAULT_DEBUG_NORMAL_COUNT = 10;
};

} // namespace config
} // namespace mesh_service