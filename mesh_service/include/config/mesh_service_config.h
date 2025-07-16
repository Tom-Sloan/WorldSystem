#pragma once

namespace mesh_service {
namespace config {

// ========== MEMORY MANAGEMENT CONFIGURATION ==========
struct MemoryConfig {
    // GPU memory pool sizes
    static constexpr size_t DEFAULT_MEMORY_POOL_SIZE = 1024 * 1024 * 1024;      // 1GB main pool
    static constexpr size_t DEFAULT_MEMORY_BLOCK_SIZE = 64 * 1024 * 1024;      // 64MB blocks
    static constexpr size_t DEFAULT_OCTREE_POOL_SIZE = 512 * 1024 * 1024;      // 512MB for octree
    static constexpr size_t DEFAULT_SOLVER_POOL_SIZE = 256 * 1024 * 1024;      // 256MB for solver
    static constexpr size_t DEFAULT_MAX_GPU_MEMORY = 4ULL * 1024 * 1024 * 1024; // 4GB max
    
    // Buffer sizes
    static constexpr int DEFAULT_MAX_VERTICES_PER_BLOCK = 65536;
    static constexpr int DEFAULT_MAX_FACES_PER_BLOCK = 131072;
};

// ========== ALGORITHM PARAMETERS ==========
struct AlgorithmConfig {
    // Normal estimation
    static constexpr int DEFAULT_NORMAL_K_NEIGHBORS = 30;
    static constexpr float DEFAULT_NORMAL_SEARCH_RADIUS = 0.1f;
    
    // TSDF parameters
    static constexpr float DEFAULT_TRUNCATION_DISTANCE = 0.15f;
    static constexpr float DEFAULT_MAX_TSDF_WEIGHT = 100.0f;
    static constexpr float DEFAULT_VOXEL_SIZE = 0.05f;
    
    // Marching cubes
    static constexpr uint DEFAULT_MAX_VERTICES = 5000000;
    static constexpr float DEFAULT_ISO_VALUE = 0.0f;
    
    // Mesh simplification
    static constexpr float DEFAULT_SIMPLIFICATION_RATIO = 0.1f;
    
    // Confidence thresholds
    static constexpr float DEFAULT_CONFIDENCE_THRESHOLD = 0.1f;
    static constexpr float DEFAULT_MIN_CONFIDENCE = 0.5f;
    
    // Spatial processing
    static constexpr float DEFAULT_INFLUENCE_RADIUS = 0.1f;
    static constexpr float DEFAULT_CHUNK_OVERLAP = 0.1f;
    
    // Poisson reconstruction
    static constexpr int DEFAULT_POISSON_OCTREE_DEPTH = 8;
    static constexpr float DEFAULT_POISSON_POINT_WEIGHT = 4.0f;
    static constexpr float DEFAULT_POISSON_SAMPLES_PER_NODE = 1.5f;
    
    // NKSR parameters
    static constexpr float DEFAULT_NKSR_DETAIL_LEVEL = 0.5f;
    static constexpr int DEFAULT_NKSR_CHUNK_SIZE = 500000;
    static constexpr float DEFAULT_NKSR_SUPPORT_RADIUS = 0.05f;
};

// ========== SCENE CONFIGURATION ==========
struct SceneConfig {
    // Scene bounds
    static constexpr float DEFAULT_MAX_SCENE_COORDINATE = 1000.0f;  // 1km max
    
    // Octree configuration
    static constexpr float DEFAULT_OCTREE_SCENE_SIZE = 10.0f;       // 10m scenes
    static constexpr int DEFAULT_OCTREE_MAX_DEPTH = 8;
    static constexpr int DEFAULT_OCTREE_LEAF_SIZE = 64;
    
    // Spatial deduplication
    static constexpr float DEFAULT_OVERLAP_THRESHOLD = 0.9f;         // 90% overlap
    
    // Default TSDF volume (can be overridden by env vars)
    static constexpr float DEFAULT_TSDF_MIN_X = -2.0f;
    static constexpr float DEFAULT_TSDF_MIN_Y = -10.0f;
    static constexpr float DEFAULT_TSDF_MIN_Z = 0.0f;
    static constexpr float DEFAULT_TSDF_MAX_X = 28.0f;
    static constexpr float DEFAULT_TSDF_MAX_Y = 10.0f;
    static constexpr float DEFAULT_TSDF_MAX_Z = 8.0f;
};

// ========== PERFORMANCE THRESHOLDS ==========
struct PerformanceConfig {
    // Camera motion thresholds
    static constexpr float DEFAULT_CAMERA_VELOCITY_THRESHOLD = 0.1f;        // m/s
    static constexpr float DEFAULT_VELOCITY_THRESHOLD_HIGH = 2.0f;          // m/s
    static constexpr float DEFAULT_VELOCITY_THRESHOLD_LOW = 0.5f;           // m/s
    
    // Time thresholds
    static constexpr float DEFAULT_TIME_DELTA_THRESHOLD = 0.001f;           // seconds
    
    // Smoothing factors
    static constexpr float DEFAULT_VELOCITY_SMOOTH_FACTOR = 0.8f;
    static constexpr float DEFAULT_VELOCITY_CURRENT_FACTOR = 0.2f;
    
    // Algorithm switching
    static constexpr int DEFAULT_SWITCH_STABILITY_FRAMES = 5;
    static constexpr size_t DEFAULT_POINT_COUNT_THRESHOLD = 100000;
    static constexpr float DEFAULT_COMPLEXITY_THRESHOLD = 0.7f;
};

// ========== DEBUG CONFIGURATION ==========
struct DebugConfig {
    // Logging intervals
    static constexpr int DEFAULT_FPS_LOG_INTERVAL = 30;             // frames
    static constexpr int DEFAULT_DEBUG_SAVE_INTERVAL = 10;          // frames
    static constexpr int DEFAULT_METRICS_LOG_INTERVAL = 100;        // frames
    
    // Debug limits
    static constexpr int DEFAULT_DEBUG_PRINT_LIMIT = 5;             // items to print
    static constexpr size_t DEFAULT_MAX_POINTS_TO_SCAN = 1000000;   // for bounds checking
    static constexpr int DEFAULT_DEBUG_VOXEL_LIMIT = 10;            // voxels to debug
    static constexpr int DEFAULT_DEBUG_POINT_LIMIT = 10;            // points to debug
};

// ========== CUDA CONFIGURATION (NOT CONFIGURABLE - Hardware specific) ==========
struct CudaConfig {
    static constexpr int THREADS_PER_BLOCK = 256;
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int WARP_SIZE = 32;
    static constexpr size_t VOXEL_BLOCK_SIZE = 256;
};

// ========== MATHEMATICAL CONSTANTS (NOT CONFIGURABLE) ==========
struct MathConstants {
    static constexpr float PI = 3.14159265358979323846f;
    static constexpr float SQRT_3 = 1.732050807568877f;
    static constexpr float EPSILON = 1e-6f;
    static constexpr float EPSILON_STRICT = 1e-10f;
    static constexpr float VOXEL_CENTER_OFFSET = 0.5f;
    
    // Morton code constants
    static constexpr float MORTON_SCALE = 1024.0f;
    static constexpr float MORTON_MAX = 1023.0f;
};

} // namespace config
} // namespace mesh_service