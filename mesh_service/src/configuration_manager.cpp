#include "config/configuration_manager.h"
#include "config/mesh_service_config.h"
#include <cstdlib>
#include <iostream>
#include <iomanip>

namespace mesh_service {

// Static member initialization
ConfigurationManager* ConfigurationManager::instance_ = nullptr;
std::mutex ConfigurationManager::mutex_;

void ConfigurationManager::loadFromEnvironment() {
    // Helper lambda to load environment variable
    auto loadEnv = [](const char* env_key) -> const char* {
        return std::getenv(env_key);
    };
    
    // ========== MEMORY CONFIGURATION ==========
    if (const char* val = loadEnv("MESH_MEMORY_POOL_SIZE")) {
        size_params_["MESH_MEMORY_POOL_SIZE"] = stringToNumber<size_t>(val, config::MemoryConfig::DEFAULT_MEMORY_POOL_SIZE);
    }
    if (const char* val = loadEnv("MESH_MEMORY_BLOCK_SIZE")) {
        size_params_["MESH_MEMORY_BLOCK_SIZE"] = stringToNumber<size_t>(val, config::MemoryConfig::DEFAULT_MEMORY_BLOCK_SIZE);
    }
    if (const char* val = loadEnv("MESH_OCTREE_POOL_SIZE")) {
        size_params_["MESH_OCTREE_POOL_SIZE"] = stringToNumber<size_t>(val, config::MemoryConfig::DEFAULT_OCTREE_POOL_SIZE);
    }
    if (const char* val = loadEnv("MESH_SOLVER_POOL_SIZE")) {
        size_params_["MESH_SOLVER_POOL_SIZE"] = stringToNumber<size_t>(val, config::MemoryConfig::DEFAULT_SOLVER_POOL_SIZE);
    }
    if (const char* val = loadEnv("MESH_MAX_GPU_MEMORY")) {
        size_params_["MESH_MAX_GPU_MEMORY"] = stringToNumber<size_t>(val, config::MemoryConfig::DEFAULT_MAX_GPU_MEMORY);
    }
    
    // ========== ALGORITHM PARAMETERS ==========
    if (const char* val = loadEnv("MESH_NORMAL_K_NEIGHBORS")) {
        int_params_["MESH_NORMAL_K_NEIGHBORS"] = stringToNumber<int>(val, config::AlgorithmConfig::DEFAULT_NORMAL_K_NEIGHBORS);
    }
    if (const char* val = loadEnv("MESH_TRUNCATION_DISTANCE")) {
        float_params_["MESH_TRUNCATION_DISTANCE"] = stringToNumber<float>(val, config::AlgorithmConfig::DEFAULT_TRUNCATION_DISTANCE);
    }
    if (const char* val = loadEnv("MESH_MAX_VERTICES")) {
        int_params_["MESH_MAX_VERTICES"] = stringToNumber<int>(val, config::AlgorithmConfig::DEFAULT_MAX_VERTICES);
    }
    if (const char* val = loadEnv("MESH_SIMPLIFICATION_RATIO")) {
        float_params_["MESH_SIMPLIFICATION_RATIO"] = stringToNumber<float>(val, config::AlgorithmConfig::DEFAULT_SIMPLIFICATION_RATIO);
    }
    if (const char* val = loadEnv("MESH_MAX_TSDF_WEIGHT")) {
        float_params_["MESH_MAX_TSDF_WEIGHT"] = stringToNumber<float>(val, config::AlgorithmConfig::DEFAULT_MAX_TSDF_WEIGHT);
    }
    if (const char* val = loadEnv("MESH_CONFIDENCE_THRESHOLD")) {
        float_params_["MESH_CONFIDENCE_THRESHOLD"] = stringToNumber<float>(val, config::AlgorithmConfig::DEFAULT_CONFIDENCE_THRESHOLD);
    }
    if (const char* val = loadEnv("MESH_INFLUENCE_RADIUS")) {
        float_params_["MESH_INFLUENCE_RADIUS"] = stringToNumber<float>(val, config::AlgorithmConfig::DEFAULT_INFLUENCE_RADIUS);
    }
    if (const char* val = loadEnv("MESH_VOXEL_SIZE")) {
        float_params_["MESH_VOXEL_SIZE"] = stringToNumber<float>(val, config::AlgorithmConfig::DEFAULT_VOXEL_SIZE);
    }
    
    // ========== SCENE CONFIGURATION ==========
    if (const char* val = loadEnv("MESH_MAX_SCENE_COORDINATE")) {
        float_params_["MESH_MAX_SCENE_COORDINATE"] = stringToNumber<float>(val, config::SceneConfig::DEFAULT_MAX_SCENE_COORDINATE);
    }
    if (const char* val = loadEnv("MESH_OCTREE_SCENE_SIZE")) {
        float_params_["MESH_OCTREE_SCENE_SIZE"] = stringToNumber<float>(val, config::SceneConfig::DEFAULT_OCTREE_SCENE_SIZE);
    }
    if (const char* val = loadEnv("MESH_OCTREE_MAX_DEPTH")) {
        int_params_["MESH_OCTREE_MAX_DEPTH"] = stringToNumber<int>(val, config::SceneConfig::DEFAULT_OCTREE_MAX_DEPTH);
    }
    if (const char* val = loadEnv("MESH_OCTREE_LEAF_SIZE")) {
        int_params_["MESH_OCTREE_LEAF_SIZE"] = stringToNumber<int>(val, config::SceneConfig::DEFAULT_OCTREE_LEAF_SIZE);
    }
    if (const char* val = loadEnv("MESH_OVERLAP_THRESHOLD")) {
        float_params_["MESH_OVERLAP_THRESHOLD"] = stringToNumber<float>(val, config::SceneConfig::DEFAULT_OVERLAP_THRESHOLD);
    }
    
    // ========== PERFORMANCE THRESHOLDS ==========
    if (const char* val = loadEnv("MESH_CAMERA_VELOCITY_THRESHOLD")) {
        float_params_["MESH_CAMERA_VELOCITY_THRESHOLD"] = stringToNumber<float>(val, config::PerformanceConfig::DEFAULT_CAMERA_VELOCITY_THRESHOLD);
    }
    if (const char* val = loadEnv("MESH_TIME_DELTA_THRESHOLD")) {
        float_params_["MESH_TIME_DELTA_THRESHOLD"] = stringToNumber<float>(val, config::PerformanceConfig::DEFAULT_TIME_DELTA_THRESHOLD);
    }
    if (const char* val = loadEnv("MESH_VELOCITY_SMOOTH_FACTOR")) {
        float_params_["MESH_VELOCITY_SMOOTH_FACTOR"] = stringToNumber<float>(val, config::PerformanceConfig::DEFAULT_VELOCITY_SMOOTH_FACTOR);
    }
    if (const char* val = loadEnv("MESH_VELOCITY_CURRENT_FACTOR")) {
        float_params_["MESH_VELOCITY_CURRENT_FACTOR"] = stringToNumber<float>(val, config::PerformanceConfig::DEFAULT_VELOCITY_CURRENT_FACTOR);
    }
    
    // ========== DEBUG CONFIGURATION ==========
    if (const char* val = loadEnv("MESH_FPS_LOG_INTERVAL")) {
        int_params_["MESH_FPS_LOG_INTERVAL"] = stringToNumber<int>(val, config::DebugConfig::DEFAULT_FPS_LOG_INTERVAL);
    }
    if (const char* val = loadEnv("MESH_DEBUG_SAVE_INTERVAL")) {
        int_params_["MESH_DEBUG_SAVE_INTERVAL"] = stringToNumber<int>(val, config::DebugConfig::DEFAULT_DEBUG_SAVE_INTERVAL);
    }
    if (const char* val = loadEnv("MESH_DEBUG_PRINT_LIMIT")) {
        int_params_["MESH_DEBUG_PRINT_LIMIT"] = stringToNumber<int>(val, config::DebugConfig::DEFAULT_DEBUG_PRINT_LIMIT);
    }
    if (const char* val = loadEnv("MESH_MAX_POINTS_TO_SCAN")) {
        size_params_["MESH_MAX_POINTS_TO_SCAN"] = stringToNumber<size_t>(val, config::DebugConfig::DEFAULT_MAX_POINTS_TO_SCAN);
    }
    
    // ========== SPECIAL STRING PARAMETERS ==========
    if (const char* val = loadEnv("MESH_DEBUG_OUTPUT_DIR")) {
        string_params_["MESH_DEBUG_OUTPUT_DIR"] = val;
    }
    
    // Log configuration if debug mode is enabled
    if (const char* debug = loadEnv("MESH_DEBUG_CONFIG")) {
        if (std::string(debug) == "true" || std::string(debug) == "1") {
            logConfiguration();
        }
    }
}

void ConfigurationManager::logConfiguration() const {
    std::cout << "\n========== MESH SERVICE CONFIGURATION ==========\n";
    
    std::cout << "\n[Memory Configuration]\n";
    for (const auto& [key, value] : size_params_) {
        if (key.find("MESH_") == 0 && key.find("MEMORY") != std::string::npos) {
            std::cout << "  " << key << " = " << value 
                      << " (" << (value / (1024.0 * 1024.0)) << " MB)" << std::endl;
        }
    }
    
    std::cout << "\n[Algorithm Parameters]\n";
    for (const auto& [key, value] : float_params_) {
        std::cout << "  " << key << " = " << std::fixed << std::setprecision(3) << value << std::endl;
    }
    for (const auto& [key, value] : int_params_) {
        if (key.find("DEBUG") == std::string::npos && key.find("FPS") == std::string::npos) {
            std::cout << "  " << key << " = " << value << std::endl;
        }
    }
    
    std::cout << "\n[Debug Configuration]\n";
    for (const auto& [key, value] : int_params_) {
        if (key.find("DEBUG") != std::string::npos || key.find("FPS") != std::string::npos) {
            std::cout << "  " << key << " = " << value << std::endl;
        }
    }
    for (const auto& [key, value] : string_params_) {
        std::cout << "  " << key << " = " << value << std::endl;
    }
    
    std::cout << "\n===============================================\n" << std::endl;
}

bool ConfigurationManager::validateConfiguration() const {
    bool valid = true;
    
    // Validate memory sizes
    size_t memory_pool = getSize("MESH_MEMORY_POOL_SIZE", config::MemoryConfig::DEFAULT_MEMORY_POOL_SIZE);
    size_t block_size = getSize("MESH_MEMORY_BLOCK_SIZE", config::MemoryConfig::DEFAULT_MEMORY_BLOCK_SIZE);
    
    if (block_size > memory_pool) {
        std::cerr << "[CONFIG ERROR] Memory block size (" << block_size 
                  << ") cannot be larger than memory pool size (" << memory_pool << ")" << std::endl;
        valid = false;
    }
    
    // Validate algorithm parameters
    float truncation = getFloat("MESH_TRUNCATION_DISTANCE", config::AlgorithmConfig::DEFAULT_TRUNCATION_DISTANCE);
    if (truncation <= 0.0f || truncation > 1.0f) {
        std::cerr << "[CONFIG ERROR] Truncation distance must be in range (0, 1], got: " << truncation << std::endl;
        valid = false;
    }
    
    float simplification = getFloat("MESH_SIMPLIFICATION_RATIO", config::AlgorithmConfig::DEFAULT_SIMPLIFICATION_RATIO);
    if (simplification <= 0.0f || simplification > 1.0f) {
        std::cerr << "[CONFIG ERROR] Simplification ratio must be in range (0, 1], got: " << simplification << std::endl;
        valid = false;
    }
    
    // Validate scene parameters
    float overlap = getFloat("MESH_OVERLAP_THRESHOLD", config::SceneConfig::DEFAULT_OVERLAP_THRESHOLD);
    if (overlap < 0.0f || overlap > 1.0f) {
        std::cerr << "[CONFIG ERROR] Overlap threshold must be in range [0, 1], got: " << overlap << std::endl;
        valid = false;
    }
    
    int octree_depth = getInt("MESH_OCTREE_MAX_DEPTH", config::SceneConfig::DEFAULT_OCTREE_MAX_DEPTH);
    if (octree_depth < 1 || octree_depth > 12) {
        std::cerr << "[CONFIG ERROR] Octree depth must be in range [1, 12], got: " << octree_depth << std::endl;
        valid = false;
    }
    
    return valid;
}

} // namespace mesh_service