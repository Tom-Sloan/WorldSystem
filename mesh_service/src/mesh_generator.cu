#include "mesh_generator.h"
#include "algorithm_selector.h"
#include "normal_estimation.h"
#include "gpu_octree.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <unordered_map>

namespace mesh_service {

// Forward declarations
uint64_t computeSpatialHashCPU(const float bbox[6]);

// Constants for RTX 3090 optimization
constexpr size_t VOXEL_BLOCK_SIZE = 256;
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
        
        // Allocate memory pool (1GB for RTX 3090)
        memory_pool_size = 1024 * 1024 * 1024;
        cudaMalloc(&d_memory_pool, memory_pool_size);
        
        // Initialize components
        algorithm_selector = std::make_unique<AlgorithmSelector>();
        normal_estimator = std::make_unique<NormalEstimation>();
        gpu_octree = std::make_unique<GPUOctree>(10.0f, 8, 64);  // 10m scene, depth 8
        
        if (!algorithm_selector->initialize()) {
            throw std::runtime_error("Failed to initialize algorithm selector");
        }
        
        // Configure normal estimation
        NormalEstimation::Parameters normal_params;
        normal_params.method = NormalEstimation::PCA;
        normal_params.k_neighbors = 30;
        normal_estimator->setParameters(normal_params);
        
        last_frame_time = std::chrono::steady_clock::now();
        
        // Initialize memory pool blocks
        size_t memory_block_size = 64 * 1024 * 1024;  // 64MB blocks
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
        
        if (dt > 0.001f) {  // Avoid division by zero
            float3 diff = make_float3(
                camera_pos.x - prev_camera_pos.x,
                camera_pos.y - prev_camera_pos.y,
                camera_pos.z - prev_camera_pos.z
            );
            float distance = sqrtf(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
            camera_velocity = distance / dt;
            
            // Smooth velocity with exponential moving average
            static float smooth_velocity = 0.0f;
            smooth_velocity = 0.8f * smooth_velocity + 0.2f * camera_velocity;
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
    
    // Update camera velocity
    updateCameraVelocity(keyframe->pose_matrix);
    
    // Check spatial deduplication with GPU octree (handle 90% overlap)
    bool has_overlap = pImpl->gpu_octree->checkRegionOverlap(keyframe->bbox, 0.9f);
    if (has_overlap && pImpl->camera_velocity < 0.1f) {
        // Skip if significant overlap and camera is not moving much
        std::cout << "[DEBUG] Region has 90% overlap with existing data, skipping" << std::endl;
        return;
    }
    
    uint64_t spatial_hash = computeSpatialHashCPU(keyframe->bbox);
    std::cout << "[DEBUG] Processing keyframe: spatial_hash=" << spatial_hash 
              << ", velocity=" << pImpl->camera_velocity << " m/s"
              << ", points=" << keyframe->point_count << std::endl;
    
    // Get point and color data
    SharedMemoryManager smm;
    float* h_points = smm.get_points(const_cast<SharedKeyframe*>(keyframe));
    uint8_t* h_colors = smm.get_colors(const_cast<SharedKeyframe*>(keyframe));
    
    // Allocate device memory using memory pool
    float3* d_points = (float3*)pImpl->allocateFromPool(keyframe->point_count * sizeof(float3));
    if (!d_points) {
        // Fallback to regular allocation
        cudaMallocAsync(&d_points, keyframe->point_count * sizeof(float3), pImpl->stream);
    }
    
    // Copy points to device
    cudaMemcpyAsync(d_points, h_points, 
                    keyframe->point_count * sizeof(float3), 
                    cudaMemcpyHostToDevice, pImpl->stream);
    
    // Allocate device memory for normals
    pImpl->d_normals.resize(keyframe->point_count);
    
    // Estimate normals
    // Update octree with new points for neighbor queries
    pImpl->gpu_octree->incrementalUpdate(d_points, keyframe->point_count, 
                                        keyframe->pose_matrix, pImpl->stream);
    
    // Estimate normals using octree for efficient neighbor search
    pImpl->normal_estimator->estimateNormals(
        d_points, 
        keyframe->point_count,
        pImpl->d_normals.data().get(),
        pImpl->stream
    );
    
    // Process with algorithm selector
    bool success = pImpl->algorithm_selector->processWithAutoSelect(
        d_points,
        pImpl->d_normals.data().get(),
        keyframe->point_count,
        keyframe->pose_matrix,
        pImpl->camera_velocity,
        update,
        pImpl->stream
    );
    
    if (!success) {
        std::cerr << "Mesh generation failed" << std::endl;
        update.vertices.clear();
        update.faces.clear();
    }
    
    // Copy colors if available
    if (h_colors && update.vertices.size() > 0) {
        update.vertex_colors.resize(update.vertices.size());
        // Simple color assignment for now - could be improved
        size_t num_vertices = update.vertices.size() / 3;
        for (size_t i = 0; i < num_vertices && i < keyframe->point_count; i++) {
            update.vertex_colors[i*3] = h_colors[i*3];
            update.vertex_colors[i*3+1] = h_colors[i*3+1];
            update.vertex_colors[i*3+2] = h_colors[i*3+2];
        }
    }
    
    // Update metadata
    update.keyframe_id = std::to_string(keyframe->timestamp_ns);
    update.timestamp_ns = keyframe->timestamp_ns;
    
    // Mark region as processed in octree
    pImpl->gpu_octree->markRegionProcessed(keyframe->bbox);
    pImpl->processed_regions[spatial_hash] = keyframe->timestamp_ns;
    
    // Clean up
    if (pImpl->allocateFromPool(0)) {  // Check if we used pool
        pImpl->releaseToPool(d_points);
    } else {
        cudaFreeAsync(d_points, pImpl->stream);
    }
    
    // Synchronize
    cudaStreamSynchronize(pImpl->stream);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    std::cout << "Mesh generation completed in " << duration << "ms: "
              << update.vertices.size()/3 << " vertices, "
              << update.faces.size()/3 << " faces" << std::endl;
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