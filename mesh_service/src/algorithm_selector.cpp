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
        // CRITICAL FIX: Use bounds for hallway with large atrium
        // Scene: 25m long hallway with 15m x 7m atrium in center
        std::cout << "[VOLUME BOUNDS FIX] No environment bounds set, using hallway+atrium defaults" << std::endl;
        std::cout << "[VOLUME BOUNDS FIX] Old bounds were: min=(-1,-1,2), max=(1,1,5)" << std::endl;
        
        // Assuming hallway runs along X axis, atrium extends in Y
        // X: -2 to 28 meters (25m hallway + margin)
        // Y: -10 to 10 meters (20m total for non-centered atrium)
        // Z: 0 to 8 meters (7m height + margin)
        mc_params.volume_min = make_float3(-2.0f, -10.0f, 0.0f);
        mc_params.volume_max = make_float3(28.0f, 10.0f, 8.0f);
        
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
    size_t point_count [[maybe_unused]],
    float scene_complexity [[maybe_unused]]
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
    
    std::cout << "[ALGORITHM SELECTOR] Selected method: " << static_cast<int>(method)
              << " (0=PoissonRecon, 1=MarchingCubes)" << std::endl;
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