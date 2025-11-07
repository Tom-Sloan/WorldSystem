#include "algorithm_selector.h"
#include "algorithms/nvidia_marching_cubes.h"
#ifdef HAS_OPEN3D
#include "algorithms/open3d_poisson.h"
#endif
#include "config/configuration_manager.h"
#include "config/mesh_service_config.h"
#include "config/poisson_config.h"
#include <iostream>
#include <string>
#include <cstdlib>

namespace mesh_service {

AlgorithmSelector::AlgorithmSelector() 
    : current_method_(ReconstructionMethod::NVIDIA_MARCHING_CUBES) {
}

AlgorithmSelector::~AlgorithmSelector() = default;

bool AlgorithmSelector::initialize() {
    std::cout << "[DEBUG ALGORITHM_SELECTOR] Initialize started" << std::endl;
    
    // Initialize NVIDIA Marching Cubes
    std::cout << "[DEBUG ALGORITHM_SELECTOR] Creating NvidiaMarchingCubes..." << std::endl;
    auto nvidia_mc = std::make_shared<NvidiaMarchingCubes>();
    std::cout << "[DEBUG ALGORITHM_SELECTOR] NvidiaMarchingCubes created" << std::endl;
    AlgorithmParams mc_params;
    mc_params.marching_cubes.iso_value = CONFIG_FLOAT("MESH_ISO_VALUE", mesh_service::config::AlgorithmConfig::DEFAULT_ISO_VALUE);
    mc_params.marching_cubes.truncation_distance = CONFIG_FLOAT("MESH_TRUNCATION_DISTANCE", mesh_service::config::AlgorithmConfig::DEFAULT_TRUNCATION_DISTANCE);
    mc_params.marching_cubes.max_vertices = CONFIG_INT("MESH_MAX_VERTICES", mesh_service::config::AlgorithmConfig::DEFAULT_MAX_VERTICES);
    
    // Get volume bounds from environment if available
    const char* bounds_min_env = std::getenv("TSDF_VOLUME_MIN");
    const char* bounds_max_env = std::getenv("TSDF_VOLUME_MAX");
    
    if (bounds_min_env && bounds_max_env) {
        float x, y, z;
        if (sscanf(bounds_min_env, "%f,%f,%f", &x, &y, &z) == 3) {
            mc_params.volume_min = make_float3(x, y, z);
            std::cout << "[VOLUME BOUNDS FIX] Using environment TSDF_VOLUME_MIN: [" 
                      << x << ", " << y << ", " << z << "]" << std::endl;
        }
        if (sscanf(bounds_max_env, "%f,%f,%f", &x, &y, &z) == 3) {
            mc_params.volume_max = make_float3(x, y, z);
            std::cout << "[VOLUME BOUNDS FIX] Using environment TSDF_VOLUME_MAX: [" 
                      << x << ", " << y << ", " << z << "]" << std::endl;
        }
    } else {
        // Use default bounds from configuration
        std::cout << "[VOLUME BOUNDS] No environment bounds set, using configuration defaults" << std::endl;
        
        mc_params.volume_min = make_float3(
            mesh_service::config::SceneConfig::DEFAULT_TSDF_MIN_X,
            mesh_service::config::SceneConfig::DEFAULT_TSDF_MIN_Y,
            mesh_service::config::SceneConfig::DEFAULT_TSDF_MIN_Z
        );
        mc_params.volume_max = make_float3(
            mesh_service::config::SceneConfig::DEFAULT_TSDF_MAX_X,
            mesh_service::config::SceneConfig::DEFAULT_TSDF_MAX_Y,
            mesh_service::config::SceneConfig::DEFAULT_TSDF_MAX_Z
        );
        
        std::cout << "[VOLUME BOUNDS FIX] New hallway bounds: min=(-2,-10,0), max=(28,10,8)" << std::endl;
        std::cout << "[VOLUME BOUNDS FIX] This gives 30x20x8 meter volume for hallway+atrium" << std::endl;
        std::cout << "[VOLUME BOUNDS FIX] Supports 25m hallway length + 20m Y-axis range" << std::endl;
    }
    
    // Get voxel size from environment
    const char* voxel_size_env = std::getenv("TSDF_VOXEL_SIZE");
    if (voxel_size_env) {
        mc_params.voxel_size = std::stof(voxel_size_env);
    }
    
    // Get truncation distance from environment
    const char* trunc_dist_env = std::getenv("TSDF_TRUNCATION_DISTANCE");
    if (trunc_dist_env) {
        mc_params.marching_cubes.truncation_distance = std::stof(trunc_dist_env);
    }
    
    std::cout << "Initializing NVIDIA Marching Cubes with:" << std::endl;
    std::cout << "  Volume min: [" << mc_params.volume_min.x << ", " 
              << mc_params.volume_min.y << ", " << mc_params.volume_min.z << "]" << std::endl;
    std::cout << "  Volume max: [" << mc_params.volume_max.x << ", " 
              << mc_params.volume_max.y << ", " << mc_params.volume_max.z << "]" << std::endl;
    std::cout << "  Voxel size: " << mc_params.voxel_size << "m" << std::endl;
    std::cout << "  Truncation distance: " << mc_params.marching_cubes.truncation_distance << "m" << std::endl;
    
    // Calculate and display memory usage
    float3 volume_size = make_float3(
        mc_params.volume_max.x - mc_params.volume_min.x,
        mc_params.volume_max.y - mc_params.volume_min.y,
        mc_params.volume_max.z - mc_params.volume_min.z
    );
    int3 voxel_count = make_int3(
        (int)(volume_size.x / mc_params.voxel_size),
        (int)(volume_size.y / mc_params.voxel_size),
        (int)(volume_size.z / mc_params.voxel_size)
    );
    size_t total_voxels = voxel_count.x * voxel_count.y * voxel_count.z;
    size_t memory_mb = (total_voxels * 2 * sizeof(float)) / (1024 * 1024); // TSDF + weights
    
    std::cout << "  Volume dimensions: " << volume_size.x << " x " << volume_size.y 
              << " x " << volume_size.z << " meters" << std::endl;
    std::cout << "  Voxel grid: " << voxel_count.x << " x " << voxel_count.y 
              << " x " << voxel_count.z << " = " << total_voxels << " voxels" << std::endl;
    std::cout << "  Estimated TSDF memory: " << memory_mb << " MB" << std::endl;
    
    if (!nvidia_mc->initialize(mc_params)) {
        std::cerr << "Failed to initialize NVIDIA Marching Cubes" << std::endl;
        return false;
    }
    algorithms_[ReconstructionMethod::NVIDIA_MARCHING_CUBES] = nvidia_mc;
    
#ifdef HAS_OPEN3D
    // Initialize Open3D Poisson as primary reconstruction method
    std::cout << "[DEBUG] Initializing Open3D Poisson reconstruction" << std::endl;
    auto open3d_poisson = std::make_shared<Open3DPoisson>();
    AlgorithmParams poisson_params;
    poisson_params.poisson.octree_depth = CONFIG_INT("MESH_POISSON_OCTREE_DEPTH", 
        config::AlgorithmConfig::DEFAULT_POISSON_OCTREE_DEPTH);
    poisson_params.poisson.point_weight = CONFIG_FLOAT("MESH_POISSON_POINT_WEIGHT",
        config::AlgorithmConfig::DEFAULT_POISSON_POINT_WEIGHT);
    poisson_params.poisson.solver_iterations = CONFIG_INT("MESH_POISSON_SOLVER_ITERATIONS",
        config::AlgorithmConfig::DEFAULT_POISSON_SOLVER_ITERATIONS);
    
    if (!open3d_poisson->initialize(poisson_params)) {
        std::cerr << "WARNING: Failed to initialize Open3D Poisson - falling back to Marching Cubes" << std::endl;
        // Don't fail completely, just don't add it to available algorithms
    } else {
        algorithms_[ReconstructionMethod::OPEN3D_POISSON] = open3d_poisson;
        
        std::cout << "Initialized Open3D Poisson with:" << std::endl;
        std::cout << "  Octree depth: " << poisson_params.poisson.octree_depth << std::endl;
        std::cout << "  Point weight: " << poisson_params.poisson.point_weight << std::endl;
        std::cout << "  Solver iterations: " << poisson_params.poisson.solver_iterations << std::endl;
    }
#else
    std::cout << "[INFO] Open3D not available - Poisson reconstruction disabled" << std::endl;
#endif
    
    
    return true;
}

ReconstructionMethod AlgorithmSelector::selectAlgorithm(
    float camera_velocity,
    size_t point_count [[maybe_unused]],
    float scene_complexity [[maybe_unused]]
) {
    // For SLAM3R: Prefer Open3D Poisson (doesn't need camera poses)
    // This can be overridden by environment variable MESH_ALGORITHM
    const char* force_algorithm = std::getenv("MESH_ALGORITHM");
    if (force_algorithm) {
        std::string algo_str(force_algorithm);
        if (algo_str == "OPEN3D_POISSON") {
            // Check if it's available
            if (algorithms_.find(ReconstructionMethod::OPEN3D_POISSON) != algorithms_.end()) {
                return ReconstructionMethod::OPEN3D_POISSON;
            } else {
                std::cerr << "WARNING: Open3D Poisson requested but not available, falling back to Marching Cubes" << std::endl;
                return ReconstructionMethod::NVIDIA_MARCHING_CUBES;
            }
        } else if (algo_str == "NVIDIA_MARCHING_CUBES") {
            return ReconstructionMethod::NVIDIA_MARCHING_CUBES;
        }
    }
    
    // Default to Open3D Poisson if available, otherwise fall back to Marching Cubes
    if (algorithms_.find(ReconstructionMethod::OPEN3D_POISSON) != algorithms_.end()) {
        return ReconstructionMethod::OPEN3D_POISSON;
    } else {
        return ReconstructionMethod::NVIDIA_MARCHING_CUBES;
    }
}

std::shared_ptr<ReconstructionAlgorithm> AlgorithmSelector::getAlgorithm(
    ReconstructionMethod method
) {
    auto it = algorithms_.find(method);
    if (it != algorithms_.end()) {
        return it->second;
    }
    return nullptr;
}

bool AlgorithmSelector::processWithAutoSelect(
    const float3* d_points,
    const float3* d_normals,
    size_t num_points,
    const float* camera_pose,
    float camera_velocity,
    MeshUpdate& output,
    cudaStream_t stream
) {
    // Calculate scene complexity (simple heuristic for now)
    float point_threshold = static_cast<float>(CONFIG_SIZE("MESH_POINT_COUNT_THRESHOLD", mesh_service::config::PerformanceConfig::DEFAULT_POINT_COUNT_THRESHOLD));
    float scene_complexity = std::min(1.0f, num_points / point_threshold);
    
    // Select algorithm
    ReconstructionMethod method = selectAlgorithm(
        camera_velocity, num_points, scene_complexity
    );
    
    std::cout << "[ALGORITHM SELECTOR] Selected method: " << static_cast<int>(method)
              << " (0=NVIDIA_MARCHING_CUBES, 1=OPEN3D_POISSON)" << std::endl;
    std::cout << "[ALGORITHM SELECTOR] Camera velocity: " << camera_velocity << " m/s" << std::endl;
    std::cout << "[ALGORITHM SELECTOR] Points: " << num_points << ", Complexity: " << scene_complexity << std::endl;
    
    // Get algorithm instance
    auto algorithm = algorithms_[method];
    if (!algorithm) {
        std::cerr << "Algorithm not available: " << static_cast<int>(method) << std::endl;
        return false;
    }
    
    // Process
    std::cout << "[ALGORITHM SELECTOR] Starting reconstruction..." << std::endl;
    bool result = algorithm->reconstruct(
        d_points, d_normals, num_points, 
        camera_pose, output, stream
    );
    std::cout << "[ALGORITHM SELECTOR] Reconstruction " << (result ? "succeeded" : "failed") << std::endl;
    
    return result;
}

} // namespace mesh_service