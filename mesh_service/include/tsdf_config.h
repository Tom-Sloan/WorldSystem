#pragma once

namespace mesh_service {

// TSDF integration configuration
struct TSDFConfig {
    // Basic parameters
    float truncation_distance = 0.1f;   // Reduced from 0.15f for tighter surface
    float max_weight = 100.0f;
    
    // Sign calculation method
    enum SignMethod {
        NORMAL_BASED,       // Use surface normals (current)
        CAMERA_CARVING,     // Carve space between camera and point
        CONFIDENCE_BASED    // Use point confidence if available
    };
    SignMethod sign_method = CAMERA_CARVING;
    
    // Advanced parameters
    float min_confidence = 0.5f;        // Minimum confidence to integrate point
    bool use_color = true;              // Integrate colors
    bool bilateral_filter = false;      // Apply bilateral filtering
    
    // Performance
    int integration_threads = 256;      // CUDA threads per block
};

} // namespace mesh_service