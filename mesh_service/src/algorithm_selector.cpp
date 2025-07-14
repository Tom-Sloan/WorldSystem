#include "algorithm_selector.h"
#include "algorithms/nvidia_marching_cubes.h"
#include <iostream>

namespace mesh_service {

AlgorithmSelector::AlgorithmSelector() 
    : current_method_(ReconstructionMethod::NVIDIA_MARCHING_CUBES) {
}

AlgorithmSelector::~AlgorithmSelector() = default;

bool AlgorithmSelector::initialize() {
    // Initialize NVIDIA Marching Cubes
    auto nvidia_mc = std::make_shared<NvidiaMarchingCubes>();
    AlgorithmParams mc_params;
    mc_params.marching_cubes.iso_value = 0.0f;
    mc_params.marching_cubes.truncation_distance = 0.15f;
    mc_params.marching_cubes.max_vertices = 5000000;
    
    // Get volume bounds from environment if available
    const char* bounds_min_env = std::getenv("TSDF_SCENE_BOUNDS_MIN");
    const char* bounds_max_env = std::getenv("TSDF_SCENE_BOUNDS_MAX");
    
    if (bounds_min_env && bounds_max_env) {
        float x, y, z;
        if (sscanf(bounds_min_env, "%f,%f,%f", &x, &y, &z) == 3) {
            mc_params.volume_min = make_float3(x, y, z);
        }
        if (sscanf(bounds_max_env, "%f,%f,%f", &x, &y, &z) == 3) {
            mc_params.volume_max = make_float3(x, y, z);
        }
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
    
    if (!nvidia_mc->initialize(mc_params)) {
        std::cerr << "Failed to initialize NVIDIA Marching Cubes" << std::endl;
        return false;
    }
    algorithms_[ReconstructionMethod::NVIDIA_MARCHING_CUBES] = nvidia_mc;
    
    // TODO: Initialize Open3D Poisson when implemented
    // auto open3d_poisson = std::make_shared<Open3DPoisson>();
    // AlgorithmParams poisson_params;
    // poisson_params.poisson.octree_depth = 8;
    // poisson_params.poisson.point_weight = 4.0f;
    // if (!open3d_poisson->initialize(poisson_params)) {
    //     return false;
    // }
    // algorithms_[ReconstructionMethod::OPEN3D_POISSON] = open3d_poisson;
    
    // TODO: Initialize NKSR Client when implemented
    // auto nksr = std::make_shared<NKSRClient>("localhost:50051");
    // AlgorithmParams nksr_params;
    // nksr_params.nksr.detail_level = 0.5f;
    // nksr_params.nksr.chunk_size = 500000;
    // if (!nksr->initialize(nksr_params)) {
    //     return false;
    // }
    // algorithms_[ReconstructionMethod::NKSR] = nksr;
    
    return true;
}

ReconstructionMethod AlgorithmSelector::selectAlgorithm(
    float camera_velocity,
    size_t point_count,
    float scene_complexity
) {
    ReconstructionMethod desired_method = current_method_;
    
    // High velocity: Always use fast marching cubes
    if (camera_velocity > thresholds_.velocity_threshold_high) {
        desired_method = ReconstructionMethod::NVIDIA_MARCHING_CUBES;
    }
    // Low velocity: Can use higher quality (when available)
    else if (camera_velocity < thresholds_.velocity_threshold_low) {
        // For now, always use NVIDIA MC until other algorithms are implemented
        desired_method = ReconstructionMethod::NVIDIA_MARCHING_CUBES;
        
        // Future logic:
        // Large point clouds: Use NKSR
        // if (point_count > thresholds_.point_count_threshold * 2) {
        //     desired_method = ReconstructionMethod::NKSR;
        // }
        // Medium complexity: Use Poisson
        // else if (scene_complexity > thresholds_.complexity_threshold) {
        //     desired_method = ReconstructionMethod::OPEN3D_POISSON;
        // }
    }
    
    // Implement hysteresis to prevent rapid switching
    if (desired_method != current_method_) {
        method_stable_frames_++;
        if (method_stable_frames_ >= SWITCH_STABILITY_FRAMES) {
            current_method_ = desired_method;
            method_stable_frames_ = 0;
            std::cout << "Switched to algorithm: " 
                      << static_cast<int>(current_method_) << std::endl;
        }
    } else {
        method_stable_frames_ = 0;
    }
    
    return current_method_;
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
    float scene_complexity = std::min(1.0f, num_points / 500000.0f);
    
    // Select algorithm
    ReconstructionMethod method = selectAlgorithm(
        camera_velocity, num_points, scene_complexity
    );
    
    // Get algorithm instance
    auto algorithm = algorithms_[method];
    if (!algorithm) {
        std::cerr << "Algorithm not available: " << static_cast<int>(method) << std::endl;
        return false;
    }
    
    // Process
    return algorithm->reconstruct(
        d_points, d_normals, num_points, 
        camera_pose, output, stream
    );
}

} // namespace mesh_service