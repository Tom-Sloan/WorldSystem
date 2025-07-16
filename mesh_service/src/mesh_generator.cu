#include "mesh_generator.h"
#include "algorithm_selector.h"
#include "normal_estimation.h"
#include "gpu_octree.h"
#include "config/configuration_manager.h"
#include "config/mesh_service_config.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <unordered_map>
#include <cfloat>

namespace mesh_service {

// Forward declarations
uint64_t computeSpatialHashCPU(const float bbox[6]);

// Constants for RTX 3090 optimization
constexpr size_t VOXEL_BLOCK_SIZE = mesh_service::config::CudaConfig::VOXEL_BLOCK_SIZE;
// constexpr size_t MAX_POINTS_PER_BLOCK = 500000;  // Reserved for future use
// constexpr int THREADS_PER_BLOCK = 128;  // Reserved for future use

// Implementation details
class GPUMeshGenerator::Impl {
public:
    // Algorithm selector and components
    std::unique_ptr<AlgorithmSelector> algorithm_selector;
    std::unique_ptr<NormalEstimation> normal_estimator;
    std::unique_ptr<GPUOctree> gpu_octree;
    
    // Camera tracking
    float3 prev_camera_pos = make_float3(0, 0, 0);
    float camera_velocity = 0.0f;
    std::chrono::steady_clock::time_point last_frame_time;
    
    // CUDA resources
    cudaStream_t stream;
    thrust::device_vector<float3> d_normals;
    
    // Spatial deduplication
    std::unordered_map<uint64_t, uint64_t> processed_regions;
    
    // Memory pool
    void* d_memory_pool = nullptr;
    size_t memory_pool_size = 0;
    
    // State tracking
    MeshMethod method = MeshMethod::TSDF_MARCHING_CUBES;
    bool quality_adaptive = true;
    float simplification_ratio = 0.1f;
    
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
    };
    std::vector<MemoryBlock> memory_blocks;
    
    Impl() {
        // Create CUDA stream
        cudaStreamCreate(&stream);
        
        // Allocate memory pool from configuration
        memory_pool_size = CONFIG_SIZE("MESH_MEMORY_POOL_SIZE", 
                                       mesh_service::config::MemoryConfig::DEFAULT_MEMORY_POOL_SIZE);
        cudaMalloc(&d_memory_pool, memory_pool_size);
        
        // Initialize components
        algorithm_selector = std::make_unique<AlgorithmSelector>();
        normal_estimator = std::make_unique<NormalEstimation>();
        gpu_octree = std::make_unique<GPUOctree>(
            CONFIG_FLOAT("MESH_OCTREE_SCENE_SIZE", mesh_service::config::SceneConfig::DEFAULT_OCTREE_SCENE_SIZE),
            CONFIG_INT("MESH_OCTREE_MAX_DEPTH", mesh_service::config::SceneConfig::DEFAULT_OCTREE_MAX_DEPTH),
            CONFIG_INT("MESH_OCTREE_LEAF_SIZE", mesh_service::config::SceneConfig::DEFAULT_OCTREE_LEAF_SIZE)
        );
        
        if (!algorithm_selector->initialize()) {
            throw std::runtime_error("Failed to initialize algorithm selector");
        }
        
        // Configure normal estimation
        NormalEstimation::Parameters normal_params;
        normal_params.method = NormalEstimation::PCA;
        normal_params.k_neighbors = CONFIG_INT("MESH_NORMAL_K_NEIGHBORS", 
                                               mesh_service::config::AlgorithmConfig::DEFAULT_NORMAL_K_NEIGHBORS);
        normal_estimator->setParameters(normal_params);
        
        last_frame_time = std::chrono::steady_clock::now();
        
        // Initialize memory pool blocks
        size_t memory_block_size = CONFIG_SIZE("MESH_MEMORY_BLOCK_SIZE", 
                                              mesh_service::config::MemoryConfig::DEFAULT_MEMORY_BLOCK_SIZE);
        size_t num_blocks = memory_pool_size / memory_block_size;
        for (size_t i = 0; i < num_blocks; i++) {
            MemoryBlock block;
            block.ptr = (char*)d_memory_pool + i * memory_block_size;
            block.size = memory_block_size;
            block.in_use = false;
            memory_blocks.push_back(block);
        }
        
        std::cout << "GPUMeshGenerator initialized with algorithm selector" << std::endl;
    }
    
    ~Impl() {
        cudaStreamDestroy(stream);
        if (d_memory_pool) {
            cudaFree(d_memory_pool);
        }
    }
    
    void updateCameraVelocity(const float* camera_pose) {
        float3 camera_pos = make_float3(
            camera_pose[12], camera_pose[13], camera_pose[14]
        );
        
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_frame_time).count();
        
        if (dt > CONFIG_FLOAT("MESH_TIME_DELTA_THRESHOLD", mesh_service::config::PerformanceConfig::DEFAULT_TIME_DELTA_THRESHOLD)) {
            float3 diff = make_float3(
                camera_pos.x - prev_camera_pos.x,
                camera_pos.y - prev_camera_pos.y,
                camera_pos.z - prev_camera_pos.z
            );
            float distance = sqrtf(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
            camera_velocity = distance / dt;
            
            // Smooth velocity with exponential moving average
            static float smooth_velocity = 0.0f;
            float smooth_factor = CONFIG_FLOAT("MESH_VELOCITY_SMOOTH_FACTOR", mesh_service::config::PerformanceConfig::DEFAULT_VELOCITY_SMOOTH_FACTOR);
            float current_factor = CONFIG_FLOAT("MESH_VELOCITY_CURRENT_FACTOR", mesh_service::config::PerformanceConfig::DEFAULT_VELOCITY_CURRENT_FACTOR);
            smooth_velocity = smooth_factor * smooth_velocity + current_factor * camera_velocity;
            camera_velocity = smooth_velocity;
        }
        
        prev_camera_pos = camera_pos;
        last_frame_time = now;
    }
    
    // Memory pool allocation
    void* allocateFromPool(size_t size) {
        for (auto& block : memory_blocks) {
            if (!block.in_use && block.size >= size) {
                block.in_use = true;
                return block.ptr;
            }
        }
        return nullptr;
    }
    
    void releaseToPool(void* ptr) {
        for (auto& block : memory_blocks) {
            if (block.ptr == ptr) {
                block.in_use = false;
                break;
            }
        }
    }
};

GPUMeshGenerator::GPUMeshGenerator() : pImpl(std::make_unique<Impl>()) {}
GPUMeshGenerator::~GPUMeshGenerator() = default;

void GPUMeshGenerator::setMethod(MeshMethod method) {
    pImpl->method = method;
    std::cout << "Mesh method set to: " << static_cast<int>(method) << std::endl;
}

void GPUMeshGenerator::setQualityAdaptive(bool adaptive) {
    pImpl->quality_adaptive = adaptive;
}

void GPUMeshGenerator::setSimplificationRatio(float ratio) {
    pImpl->simplification_ratio = ratio;
}

void GPUMeshGenerator::updateCameraVelocity(const float pose_matrix[16]) {
    pImpl->updateCameraVelocity(pose_matrix);
}

void GPUMeshGenerator::generateIncrementalMesh(
    const SharedKeyframe* keyframe,
    MeshUpdate& update
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Check for empty keyframe
    if (keyframe->point_count == 0) {
        std::cout << "[DEBUG] Empty keyframe, skipping mesh generation" << std::endl;
        update.vertices.clear();
        update.faces.clear();
        return;
    }
    
    // Update camera velocity
    updateCameraVelocity(keyframe->pose_matrix);
    
    // Check spatial deduplication with GPU octree
    auto dedup_start = std::chrono::high_resolution_clock::now();
    float overlap_threshold = CONFIG_FLOAT("MESH_OVERLAP_THRESHOLD", mesh_service::config::SceneConfig::DEFAULT_OVERLAP_THRESHOLD);
    bool has_overlap = pImpl->gpu_octree->checkRegionOverlap(keyframe->bbox, overlap_threshold);
    float velocity_threshold = CONFIG_FLOAT("MESH_CAMERA_VELOCITY_THRESHOLD", mesh_service::config::PerformanceConfig::DEFAULT_CAMERA_VELOCITY_THRESHOLD);
    if (has_overlap && pImpl->camera_velocity < velocity_threshold) {
        // Skip if significant overlap and camera is not moving much
        std::cout << "[DEBUG] Region has 90% overlap with existing data, skipping" << std::endl;
        return;
    }
    auto dedup_end = std::chrono::high_resolution_clock::now();
    auto dedup_ms = std::chrono::duration_cast<std::chrono::microseconds>(dedup_end - dedup_start).count();
    std::cout << "[TIMING] Spatial deduplication check: " << dedup_ms << " µs" << std::endl;
    
    uint64_t spatial_hash = computeSpatialHashCPU(keyframe->bbox);
    std::cout << "[DEBUG] Processing keyframe: spatial_hash=" << spatial_hash 
              << ", velocity=" << pImpl->camera_velocity << " m/s"
              << ", points=" << keyframe->point_count << std::endl;
    
    // Get point and color data
    auto data_access_start = std::chrono::high_resolution_clock::now();
    SharedMemoryManager smm;
    float* h_points = smm.get_points(const_cast<SharedKeyframe*>(keyframe));
    uint8_t* h_colors = smm.get_colors(const_cast<SharedKeyframe*>(keyframe));
    auto data_access_end = std::chrono::high_resolution_clock::now();
    auto data_access_us = std::chrono::duration_cast<std::chrono::microseconds>(data_access_end - data_access_start).count();
    std::cout << "[TIMING] Point/color data access: " << data_access_us << " µs" << std::endl;
    
    // Debug: Check point cloud bounds
    std::cout << "[MESH GEN DEBUG] Checking point cloud bounds for " << keyframe->point_count << " points" << std::endl;
    
    // Debug first few points raw values
    int debug_limit = CONFIG_INT("MESH_DEBUG_PRINT_LIMIT", mesh_service::config::DebugConfig::DEFAULT_DEBUG_PRINT_LIMIT);
    std::cout << "[MESH GEN DEBUG] First " << debug_limit << " points raw values:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(debug_limit), size_t(keyframe->point_count)); i++) {
        std::cout << "  Point " << i << ": [" << h_points[i*3] << ", " 
                  << h_points[i*3+1] << ", " << h_points[i*3+2] << "]" << std::endl;
    }
    
    float3 min_bound = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 max_bound = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    
    // Limit scan to prevent reading past valid data
    size_t max_points_to_scan = std::min(size_t(keyframe->point_count), 
                                         CONFIG_SIZE("MESH_MAX_POINTS_TO_SCAN", mesh_service::config::DebugConfig::DEFAULT_MAX_POINTS_TO_SCAN));
    if (keyframe->point_count > max_points_to_scan) {
        std::cout << "[MESH GEN WARNING] Limiting bounds scan to first " << max_points_to_scan 
                  << " points (total: " << keyframe->point_count << ")" << std::endl;
    }
    
    for (size_t i = 0; i < max_points_to_scan; i++) {
        float3 pt = make_float3(h_points[i*3], h_points[i*3+1], h_points[i*3+2]);
        
        // Sanity check for corrupted data
        if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z) &&
            std::abs(pt.x) < 1000000 && std::abs(pt.y) < 1000000 && std::abs(pt.z) < 1000000) {
            min_bound.x = fminf(min_bound.x, pt.x);
            min_bound.y = fminf(min_bound.y, pt.y);
            min_bound.z = fminf(min_bound.z, pt.z);
            max_bound.x = fmaxf(max_bound.x, pt.x);
            max_bound.y = fmaxf(max_bound.y, pt.y);
            max_bound.z = fmaxf(max_bound.z, pt.z);
        } else {
            std::cout << "[MESH GEN ERROR] Corrupted point at index " << i 
                      << ": [" << pt.x << ", " << pt.y << ", " << pt.z << "]" << std::endl;
            break;
        }
    }
    
    std::cout << "[MESH GEN DEBUG] Actual point cloud bounds:" << std::endl;
    std::cout << "  Min: [" << min_bound.x << ", " << min_bound.y << ", " << min_bound.z << "]" << std::endl;
    std::cout << "  Max: [" << max_bound.x << ", " << max_bound.y << ", " << max_bound.z << "]" << std::endl;
    std::cout << "[MESH GEN DEBUG] Stored bbox in keyframe:" << std::endl;
    std::cout << "  Min: [" << keyframe->bbox[0] << ", " << keyframe->bbox[1] << ", " << keyframe->bbox[2] << "]" << std::endl;
    std::cout << "  Max: [" << keyframe->bbox[3] << ", " << keyframe->bbox[4] << ", " << keyframe->bbox[5] << "]" << std::endl;
    
    // Filter corrupted points before processing
    auto filter_start = std::chrono::high_resolution_clock::now();
    std::vector<float3> valid_points;
    std::vector<uint8_t> valid_colors;
    valid_points.reserve(keyframe->point_count);
    if (h_colors) {
        valid_colors.reserve(keyframe->point_count * 3);
    }
    
    size_t corrupted_count = 0;
    const float max_coord = CONFIG_FLOAT("MESH_MAX_SCENE_COORDINATE", mesh_service::config::SceneConfig::DEFAULT_MAX_SCENE_COORDINATE);
    
    for (size_t i = 0; i < keyframe->point_count; i++) {
        float3 pt = make_float3(h_points[i*3], h_points[i*3+1], h_points[i*3+2]);
        
        // Check for NaN, infinity, and extreme values
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z) ||
            std::abs(pt.x) > max_coord || std::abs(pt.y) > max_coord || std::abs(pt.z) > max_coord) {
            corrupted_count++;
            if (corrupted_count <= CONFIG_INT("MESH_DEBUG_PRINT_LIMIT", mesh_service::config::DebugConfig::DEFAULT_DEBUG_PRINT_LIMIT)) {
                std::cout << "[MESH GEN] Filtering corrupted point " << i 
                          << ": [" << pt.x << ", " << pt.y << ", " << pt.z << "]" << std::endl;
            }
            continue;
        }
        
        valid_points.push_back(pt);
        if (h_colors) {
            valid_colors.push_back(h_colors[i*3]);
            valid_colors.push_back(h_colors[i*3+1]);
            valid_colors.push_back(h_colors[i*3+2]);
        }
    }
    auto filter_end = std::chrono::high_resolution_clock::now();
    auto filter_ms = std::chrono::duration_cast<std::chrono::milliseconds>(filter_end - filter_start).count();
    std::cout << "[TIMING] Point filtering: " << filter_ms << " ms" << std::endl;
    
    size_t valid_point_count = valid_points.size();
    
    if (corrupted_count > 0) {
        std::cout << "[MESH GEN] Filtered " << corrupted_count 
                  << " corrupted points, " << valid_point_count << " valid points remain" << std::endl;
    }
    
    if (valid_point_count == 0) {
        std::cout << "[MESH GEN] No valid points after filtering, skipping mesh generation" << std::endl;
        update.vertices.clear();
        update.faces.clear();
        return;
    }
    
    // Allocate device memory for valid points only
    auto gpu_alloc_start = std::chrono::high_resolution_clock::now();
    float3* d_points = (float3*)pImpl->allocateFromPool(valid_point_count * sizeof(float3));
    if (!d_points) {
        // Fallback to regular allocation
        cudaMallocAsync(&d_points, valid_point_count * sizeof(float3), pImpl->stream);
    }
    auto gpu_alloc_end = std::chrono::high_resolution_clock::now();
    auto gpu_alloc_us = std::chrono::duration_cast<std::chrono::microseconds>(gpu_alloc_end - gpu_alloc_start).count();
    std::cout << "[TIMING] GPU memory allocation: " << gpu_alloc_us << " µs" << std::endl;
    
    // Copy valid points to device
    auto h2d_start = std::chrono::high_resolution_clock::now();
    cudaMemcpyAsync(d_points, valid_points.data(), 
                    valid_point_count * sizeof(float3), 
                    cudaMemcpyHostToDevice, pImpl->stream);
    cudaStreamSynchronize(pImpl->stream);  // Sync to measure actual transfer time
    auto h2d_end = std::chrono::high_resolution_clock::now();
    auto h2d_ms = std::chrono::duration_cast<std::chrono::milliseconds>(h2d_end - h2d_start).count();
    std::cout << "[TIMING] Host to device copy: " << h2d_ms << " ms (" << (valid_point_count * sizeof(float3) / 1024.0 / 1024.0) << " MB)" << std::endl;
    
    // Allocate device memory for normals
    pImpl->d_normals.resize(valid_point_count);
    
    // Estimate normals
    // Update octree with new points for neighbor queries
    auto octree_start = std::chrono::high_resolution_clock::now();
    pImpl->gpu_octree->incrementalUpdate(d_points, valid_point_count, 
                                        keyframe->pose_matrix, pImpl->stream);
    cudaStreamSynchronize(pImpl->stream);
    auto octree_end = std::chrono::high_resolution_clock::now();
    auto octree_ms = std::chrono::duration_cast<std::chrono::milliseconds>(octree_end - octree_start).count();
    std::cout << "[TIMING] Octree update: " << octree_ms << " ms" << std::endl;
    
    // Estimate normals using octree for efficient neighbor search
    auto normal_start = std::chrono::high_resolution_clock::now();
    pImpl->normal_estimator->estimateNormals(
        d_points, 
        valid_point_count,
        pImpl->d_normals.data().get(),
        pImpl->stream
    );
    auto normal_end = std::chrono::high_resolution_clock::now();
    auto normal_ms = std::chrono::duration_cast<std::chrono::milliseconds>(normal_end - normal_start).count();
    std::cout << "[TIMING] Normal estimation breakdown: " << normal_ms << " ms" << std::endl;
    
    // Process with algorithm selector
    auto algo_start = std::chrono::high_resolution_clock::now();
    bool success = pImpl->algorithm_selector->processWithAutoSelect(
        d_points,
        pImpl->d_normals.data().get(),
        valid_point_count,
        keyframe->pose_matrix,
        pImpl->camera_velocity,
        update,
        pImpl->stream
    );
    auto algo_end = std::chrono::high_resolution_clock::now();
    auto algo_ms = std::chrono::duration_cast<std::chrono::milliseconds>(algo_end - algo_start).count();
    std::cout << "[TIMING] Algorithm processing (TSDF + MC): " << algo_ms << " ms" << std::endl;
    
    if (!success) {
        std::cerr << "Mesh generation failed" << std::endl;
        update.vertices.clear();
        update.faces.clear();
    }
    
    // Copy colors if available
    auto color_start = std::chrono::high_resolution_clock::now();
    if (!valid_colors.empty() && update.vertices.size() > 0) {
        update.vertex_colors.resize(update.vertices.size());
        // Simple color assignment for now - could be improved
        size_t num_vertices = update.vertices.size() / 3;
        for (size_t i = 0; i < num_vertices && i < valid_point_count; i++) {
            update.vertex_colors[i*3] = valid_colors[i*3];
            update.vertex_colors[i*3+1] = valid_colors[i*3+1];
            update.vertex_colors[i*3+2] = valid_colors[i*3+2];
        }
    }
    auto color_end = std::chrono::high_resolution_clock::now();
    auto color_ms = std::chrono::duration_cast<std::chrono::milliseconds>(color_end - color_start).count();
    if (!valid_colors.empty()) {
        std::cout << "[TIMING] Color assignment: " << color_ms << " ms" << std::endl;
    }
    
    // Periodic debug saves - moved here from NvidiaMarchingCubes::reconstruct
    static int debug_frame_count = 0;
    debug_frame_count++;
    
    if (debug_frame_count % CONFIG_INT("MESH_DEBUG_SAVE_INTERVAL", mesh_service::config::DebugConfig::DEFAULT_DEBUG_SAVE_INTERVAL) == 0) {
        std::cout << "[MESH GEN DEBUG SAVE] Saving debug data for frame " << debug_frame_count << std::endl;
        
        // Save point cloud to PLY file
        char filename[256];
        snprintf(filename, sizeof(filename), "/debug_output/pointcloud_%06d.ply", debug_frame_count);
        
        FILE* fp = fopen(filename, "w");
        if (fp) {
            fprintf(fp, "ply\n");
            fprintf(fp, "format ascii 1.0\n");
            fprintf(fp, "element vertex %zu\n", valid_point_count);
            fprintf(fp, "property float x\n");
            fprintf(fp, "property float y\n");
            fprintf(fp, "property float z\n");
            if (!valid_colors.empty()) {
                fprintf(fp, "property uchar red\n");
                fprintf(fp, "property uchar green\n");
                fprintf(fp, "property uchar blue\n");
            }
            fprintf(fp, "end_header\n");
            
            for (size_t i = 0; i < valid_point_count; i++) {
                fprintf(fp, "%.6f %.6f %.6f", valid_points[i].x, valid_points[i].y, valid_points[i].z);
                if (!valid_colors.empty()) {
                    fprintf(fp, " %d %d %d", valid_colors[i*3], valid_colors[i*3+1], valid_colors[i*3+2]);
                }
                fprintf(fp, "\n");
            }
            
            fclose(fp);
            std::cout << "[MESH GEN DEBUG SAVE] Saved point cloud to " << filename 
                      << " (" << valid_point_count << " points)" << std::endl;
        } else {
            std::cerr << "[MESH GEN DEBUG SAVE] Failed to open " << filename << " for writing" << std::endl;
        }
        
        // Save mesh if we have one
        if (update.vertices.size() > 0) {
            snprintf(filename, sizeof(filename), "/debug_output/mesh_%06d.ply", debug_frame_count);
            fp = fopen(filename, "w");
            if (fp) {
                size_t num_verts = update.vertices.size() / 3;
                size_t num_faces = update.faces.size() / 3;
                
                fprintf(fp, "ply\n");
                fprintf(fp, "format ascii 1.0\n");
                fprintf(fp, "element vertex %zu\n", num_verts);
                fprintf(fp, "property float x\n");
                fprintf(fp, "property float y\n");
                fprintf(fp, "property float z\n");
                fprintf(fp, "element face %zu\n", num_faces);
                fprintf(fp, "property list uchar int vertex_indices\n");
                fprintf(fp, "end_header\n");
                
                for (size_t i = 0; i < num_verts; i++) {
                    fprintf(fp, "%.6f %.6f %.6f\n", 
                            update.vertices[i*3], update.vertices[i*3+1], update.vertices[i*3+2]);
                }
                
                for (size_t i = 0; i < num_faces; i++) {
                    fprintf(fp, "3 %d %d %d\n", 
                            update.faces[i*3], update.faces[i*3+1], update.faces[i*3+2]);
                }
                
                fclose(fp);
                std::cout << "[MESH GEN DEBUG SAVE] Saved mesh to " << filename 
                          << " (" << num_verts << " vertices, " << num_faces << " faces)" << std::endl;
            }
        } else {
            std::cout << "[MESH GEN DEBUG SAVE] No mesh generated yet for frame " << debug_frame_count << std::endl;
        }
    }
    
    // Update metadata
    update.keyframe_id = std::to_string(keyframe->timestamp_ns);
    update.timestamp_ns = keyframe->timestamp_ns;
    
    // Mark region as processed in octree
    pImpl->gpu_octree->markRegionProcessed(keyframe->bbox);
    pImpl->processed_regions[spatial_hash] = keyframe->timestamp_ns;
    
    // Clean up
    auto cleanup_start = std::chrono::high_resolution_clock::now();
    if (pImpl->allocateFromPool(0)) {  // Check if we used pool
        pImpl->releaseToPool(d_points);
    } else {
        cudaFreeAsync(d_points, pImpl->stream);
    }
    
    // Synchronize
    cudaStreamSynchronize(pImpl->stream);
    auto cleanup_end = std::chrono::high_resolution_clock::now();
    auto cleanup_us = std::chrono::duration_cast<std::chrono::microseconds>(cleanup_end - cleanup_start).count();
    std::cout << "[TIMING] GPU cleanup: " << cleanup_us << " µs" << std::endl;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    std::cout << "Mesh generation completed in " << duration << "ms: "
              << update.vertices.size()/3 << " vertices, "
              << update.faces.size()/3 << " faces" << std::endl;
    
    // Print timing summary
    std::cout << "\n[TIMING SUMMARY]" << std::endl;
    std::cout << "  Total mesh generation: " << duration << " ms" << std::endl;
    std::cout << "  Major components:" << std::endl;
    std::cout << "    - Point filtering: ~" << filter_ms << " ms" << std::endl;
    std::cout << "    - Normal estimation: ~" << normal_ms << " ms (largest component)" << std::endl;
    std::cout << "    - TSDF + Marching Cubes: ~" << algo_ms << " ms" << std::endl;
    std::cout << "  Performance metrics:" << std::endl;
    std::cout << "    - Points processed: " << valid_point_count << std::endl;
    std::cout << "    - Points/sec: " << (valid_point_count * 1000.0 / duration) << std::endl;
    std::cout << "    - Vertices generated: " << update.vertices.size()/3 << std::endl;
    std::cout << "    - Faces generated: " << update.faces.size()/3 << std::endl;
}

void GPUMeshGenerator::generateMesh(
    float3* d_points,
    size_t num_points,
    const float* camera_pose,
    MeshUpdate& update
) {
    // Simplified version for direct point cloud input
    
    // Allocate normals if needed
    pImpl->d_normals.resize(num_points);
    
    // Estimate normals
    pImpl->normal_estimator->estimateNormals(
        d_points, 
        num_points,
        pImpl->d_normals.data().get(),
        pImpl->stream
    );
    
    // Process with algorithm selector
    bool success = pImpl->algorithm_selector->processWithAutoSelect(
        d_points,
        pImpl->d_normals.data().get(),
        num_points,
        camera_pose,
        0.0f,  // Unknown velocity
        update,
        pImpl->stream
    );
    
    if (!success) {
        std::cerr << "Mesh generation failed" << std::endl;
        update.vertices.clear();
        update.faces.clear();
    }
    
    cudaStreamSynchronize(pImpl->stream);
}

// Utility function
uint64_t computeSpatialHashCPU(const float bbox[6]) {
    // Simple spatial hash based on bounding box center
    float center_x = (bbox[0] + bbox[3]) / 2.0f;
    float center_y = (bbox[1] + bbox[4]) / 2.0f;
    float center_z = (bbox[2] + bbox[5]) / 2.0f;
    
    // Quantize to voxel grid
    int vx = static_cast<int>(center_x / VOXEL_BLOCK_SIZE);
    int vy = static_cast<int>(center_y / VOXEL_BLOCK_SIZE);
    int vz = static_cast<int>(center_z / VOXEL_BLOCK_SIZE);
    
    // Simple hash combining
    uint64_t hash = 0;
    hash ^= std::hash<int>{}(vx) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int>{}(vy) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int>{}(vz) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    
    return hash;
}

} // namespace mesh_service