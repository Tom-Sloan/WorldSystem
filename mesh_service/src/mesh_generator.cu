// TSDF-Only Mode
// This file has been simplified to use only TSDF with Marching Cubes
// for optimal real-time performance in indoor drone mapping.
// To re-enable Poisson/NKSR algorithms, see FUTURE_ALGORITHMS.md

#include "mesh_generator.h"
// FUTURE: Uncomment these includes for multi-algorithm support
// #include "poisson_reconstruction.h"
// #include "gpu_poisson_reconstruction.h"
// #include "nksr_reconstruction.h"
#include "marching_cubes.h"
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
constexpr size_t MAX_POINTS_PER_BLOCK = 500000;
constexpr int THREADS_PER_BLOCK = 128;  // Optimal for RTX 3090

// Implementation details
class GPUMeshGenerator::Impl {
public:
    MeshMethod method = MeshMethod::INCREMENTAL_POISSON;
    bool quality_adaptive = true;
    float simplification_ratio = 0.1f;
    float camera_velocity = 0.0f;
    
    // CUDA streams for parallel processing
    cudaStream_t streams[5];
    
    // Memory pool
    void* d_memory_pool = nullptr;
    size_t memory_pool_size = 0;
    
    // Spatial deduplication
    std::unordered_map<uint64_t, uint64_t> processed_regions;
    
    // Previous camera position for velocity calculation
    float prev_camera_pos[3] = {0, 0, 0};
    
    // Mesh generation components
    // FUTURE: Uncomment these for Poisson/NKSR support
    // std::unique_ptr<PoissonReconstruction> poisson;
    // std::unique_ptr<IncrementalPoissonReconstruction> incremental_poisson;
    // std::unique_ptr<GPUPoissonReconstruction> gpu_poisson;  // New GPU implementation
    // std::unique_ptr<NKSRReconstruction> nksr;  // Neural Kernel Surface Reconstruction
    std::unique_ptr<MarchingCubesGPU> marching_cubes;
    std::unique_ptr<IncrementalTSDFFusion> tsdf_fusion;
    std::unique_ptr<NormalEstimation> normal_estimator;
    std::unique_ptr<GPUOctree> gpu_octree;  // GPU octree for spatial indexing
    
    // Device memory for normals
    thrust::device_vector<float3> d_normals;
    
    // Memory pool management
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
    };
    std::vector<MemoryBlock> memory_blocks;
    
    // Multi-stream synchronization
    cudaEvent_t stream_events[5];
    
    // Helper functions for environment variables
    static float getEnvFloat(const char* name, float default_val) {
        const char* val = std::getenv(name);
        return val ? std::stof(val) : default_val;
    }
    
    static int getEnvInt(const char* name, int default_val) {
        const char* val = std::getenv(name);
        return val ? std::stoi(val) : default_val;
    }
    
    static float3 getEnvFloat3(const char* name, float3 default_val) {
        const char* val = std::getenv(name);
        if (!val) return default_val;
        
        float x, y, z;
        if (sscanf(val, "%f,%f,%f", &x, &y, &z) == 3) {
            return make_float3(x, y, z);
        }
        return default_val;
    }
    
    Impl() {
        // Create CUDA streams
        for (int i = 0; i < 5; i++) {
            cudaStreamCreate(&streams[i]);
        }
        
        // Allocate memory pool (1GB for RTX 3090)
        memory_pool_size = 1024 * 1024 * 1024;
        cudaMalloc(&d_memory_pool, memory_pool_size);
        
        // Initialize mesh generation components
        // FUTURE: Uncomment these for Poisson/NKSR support
        // poisson = std::make_unique<PoissonReconstruction>();
        // incremental_poisson = std::make_unique<IncrementalPoissonReconstruction>();
        // gpu_poisson = std::make_unique<GPUPoissonReconstruction>();  // GPU Poisson
        // nksr = std::make_unique<NKSRReconstruction>();  // NKSR
        marching_cubes = std::make_unique<MarchingCubesGPU>();
        tsdf_fusion = std::make_unique<IncrementalTSDFFusion>();
        normal_estimator = std::make_unique<NormalEstimation>();
        gpu_octree = std::make_unique<GPUOctree>(10.0f, 8, 64);  // 10m scene, depth 8
        
        // Create stream events for synchronization
        for (int i = 0; i < 5; i++) {
            cudaEventCreate(&stream_events[i]);
        }
        
        // Configure components
        // FUTURE: Uncomment for Poisson support
        // PoissonReconstruction::Parameters poisson_params;
        // poisson_params.octree_depth = 8;
        // poisson_params.point_weight = 4.0f;
        // poisson->setParameters(poisson_params);
        
        MarchingCubesGPU::Parameters mc_params;
        mc_params.voxel_size = 0.05f;
        mc_params.truncation_distance = 0.1f;
        marching_cubes->setParameters(mc_params);
        
        NormalEstimation::Parameters normal_params;
        normal_params.method = NormalEstimation::PCA;
        normal_params.k_neighbors = 30;
        normal_estimator->setParameters(normal_params);
        
        // Configure TSDF from environment variables
        float voxel_size = getEnvFloat("TSDF_VOXEL_SIZE", 0.04f);
        float truncation_dist = getEnvFloat("TSDF_TRUNCATION_DISTANCE", 0.12f);
        float max_weight = getEnvFloat("TSDF_MAX_WEIGHT", 100.0f);
        int block_size = getEnvInt("TSDF_BLOCK_SIZE", 8);
        
        tsdf_fusion->setVoxelSize(voxel_size);
        
        std::cout << "TSDF Configuration from environment:" << std::endl;
        std::cout << "  Voxel size: " << voxel_size << "m" << std::endl;
        std::cout << "  Truncation distance: " << truncation_dist << "m" << std::endl;
        std::cout << "  Max weight: " << max_weight << std::endl;
        std::cout << "  Block size: " << block_size << std::endl;
        
        // FUTURE: Uncomment for Poisson/NKSR configuration
        // incremental_poisson->initialize(10.0f, 8);
        // gpu_poisson->initialize(10.0f, 8);
        // GPUPoissonReconstruction::Parameters gpu_poisson_params;
        // ...
        // NKSRReconstruction::Parameters nksr_params;
        // ...
        
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
    }
    
    ~Impl() {
        // Destroy streams and events
        for (int i = 0; i < 5; i++) {
            cudaStreamDestroy(streams[i]);
            cudaEventDestroy(stream_events[i]);
        }
        
        // Free memory pool
        if (d_memory_pool) {
            cudaFree(d_memory_pool);
        }
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
    // TSDF-only mode - ignore method changes
    pImpl->method = MeshMethod::TSDF_MARCHING_CUBES;
    if (method != MeshMethod::TSDF_MARCHING_CUBES) {
        std::cout << "[INFO] Mesh service is configured for TSDF-only mode. Ignoring method change request." << std::endl;
    }
}

void GPUMeshGenerator::setQualityAdaptive(bool adaptive) {
    pImpl->quality_adaptive = adaptive;
}

void GPUMeshGenerator::setSimplificationRatio(float ratio) {
    pImpl->simplification_ratio = ratio;
}

void GPUMeshGenerator::updateCameraVelocity(const float pose_matrix[16]) {
    // Extract camera position from pose matrix
    float curr_pos[3] = {pose_matrix[12], pose_matrix[13], pose_matrix[14]};
    
    // Calculate velocity
    float dx = curr_pos[0] - pImpl->prev_camera_pos[0];
    float dy = curr_pos[1] - pImpl->prev_camera_pos[1];
    float dz = curr_pos[2] - pImpl->prev_camera_pos[2];
    
    pImpl->camera_velocity = std::sqrt(dx*dx + dy*dy + dz*dz);
    
    // Update previous position
    std::copy(curr_pos, curr_pos + 3, pImpl->prev_camera_pos);
}

void GPUMeshGenerator::generateIncrementalMesh(
    const SharedKeyframe* keyframe,
    MeshUpdate& update
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Update camera velocity
    updateCameraVelocity(keyframe->pose_matrix);
    
    // TSDF-only mode - no adaptive switching
    pImpl->method = MeshMethod::TSDF_MARCHING_CUBES;
    
    // Check spatial deduplication with GPU octree (handle 90% overlap)
    bool has_overlap = pImpl->gpu_octree->checkRegionOverlap(keyframe->bbox, 0.9f);
    if (has_overlap && pImpl->camera_velocity < 0.1f) {
        // Skip if significant overlap and camera is not moving much
        std::cout << "[DEBUG] Region has 90% overlap with existing data, skipping" << std::endl;
        return;
    }
    
    uint64_t spatial_hash = computeSpatialHashCPU(keyframe->bbox);
    std::cout << "[DEBUG] Spatial hash: " << spatial_hash 
              << ", bbox: [" << keyframe->bbox[0] << "," << keyframe->bbox[1] 
              << "," << keyframe->bbox[2] << " - " << keyframe->bbox[3] 
              << "," << keyframe->bbox[4] << "," << keyframe->bbox[5] << "]" << std::endl;
    
    // Get point and color data
    SharedMemoryManager smm;
    float* h_points = smm.get_points(const_cast<SharedKeyframe*>(keyframe));
    uint8_t* h_colors = smm.get_colors(const_cast<SharedKeyframe*>(keyframe));
    
    // Allocate device memory using memory pool
    float3* d_points = (float3*)pImpl->allocateFromPool(keyframe->point_count * sizeof(float3));
    if (!d_points) {
        // Fallback to regular allocation
        cudaMallocAsync(&d_points, keyframe->point_count * sizeof(float3), pImpl->streams[0]);
    }
    
    // Copy points to device using multiple streams for large point clouds
    size_t points_per_stream = keyframe->point_count / 5;
    for (int i = 0; i < 5; i++) {
        size_t offset = i * points_per_stream;
        size_t count = (i == 4) ? keyframe->point_count - offset : points_per_stream;
        
        cudaMemcpyAsync(d_points + offset, h_points + offset * 3, 
                        count * sizeof(float3), 
                        cudaMemcpyHostToDevice, pImpl->streams[i]);
    }
    
    // Allocate device memory for normals
    pImpl->d_normals.resize(keyframe->point_count);
    
    // Estimate normals first (required for Poisson)
    // Update octree with new points for neighbor queries
    pImpl->gpu_octree->incrementalUpdate(d_points, keyframe->point_count, 
                                        keyframe->pose_matrix, pImpl->streams[0]);
    
    // Estimate normals using octree for efficient neighbor search
    pImpl->normal_estimator->estimateNormals(
        d_points, 
        keyframe->point_count,
        pImpl->d_normals.data().get(),
        pImpl->streams[0]
    );
    
    // Synchronize normal estimation across streams
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(pImpl->stream_events[i], pImpl->streams[i]);
    }
    
    // TSDF-only mesh generation
    generateMarchingCubesMesh(keyframe, d_points, pImpl->d_normals.data().get(), update);
    
    // FUTURE: Uncomment for multi-algorithm support
    // switch (pImpl->method) {
    //     case MeshMethod::INCREMENTAL_POISSON:
    //         generateIncrementalPoissonMesh(keyframe, d_points, pImpl->d_normals.data().get(), update);
    //         break;
    //     case MeshMethod::NKSR:
    //         generateNKSRMesh(keyframe, d_points, pImpl->d_normals.data().get(), update);
    //         break;
    //     case MeshMethod::TSDF_MARCHING_CUBES:
    //         generateMarchingCubesMesh(keyframe, d_points, pImpl->d_normals.data().get(), update);
    //         break;
    // }
    
    // Copy colors
    update.vertex_colors.resize(update.vertices.size());
    // Simple color assignment for now
    for (size_t i = 0; i < update.vertices.size() / 3; i++) {
        if (i < keyframe->point_count) {
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
        cudaFreeAsync(d_points, pImpl->streams[0]);
    }
    
    // Synchronize all streams
    for (int i = 0; i < 5; i++) {
        cudaStreamSynchronize(pImpl->streams[i]);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    std::cout << "Mesh generation completed in " << duration << "ms: "
              << update.vertices.size()/3 << " vertices, "
              << update.faces.size()/3 << " faces" << std::endl;
}

// FUTURE: Uncomment this method to enable GPU Poisson reconstruction
/*
void GPUMeshGenerator::generateIncrementalPoissonMesh(
    const SharedKeyframe* keyframe,
    float3* d_points, 
    float3* d_normals,
    MeshUpdate& update
) {
    // Use GPU Poisson reconstruction
    SharedMemoryManager smm;
    uint8_t* h_colors = smm.get_colors(const_cast<SharedKeyframe*>(keyframe));
    
    // Copy colors to device
    uint8_t* d_colors;
    cudaMalloc(&d_colors, keyframe->point_count * 3);
    cudaMemcpy(d_colors, h_colors, keyframe->point_count * 3, cudaMemcpyHostToDevice);
    
    // Add points to GPU Poisson reconstruction
    pImpl->gpu_poisson->addPoints(d_points, d_normals, d_colors, 
                                 keyframe->point_count, 
                                 pImpl->gpu_octree.get(), 
                                 pImpl->streams[0]);
    
    // Get dirty blocks that need mesh extraction
    std::vector<int> dirty_blocks;
    pImpl->gpu_poisson->getDirtyBlocks(dirty_blocks);
    
    // Extract mesh only for dirty blocks
    pImpl->gpu_poisson->extractMesh(update.vertices, update.faces, 
                                   update.vertex_colors, dirty_blocks,
                                   pImpl->streams[0]);
    
    // Clear processed blocks
    pImpl->gpu_poisson->clearDirtyBlocks(dirty_blocks);
    
    // Store which regions were updated
    update.updated_regions = dirty_blocks;
    
    cudaFree(d_colors);
}
*/

void GPUMeshGenerator::generateMarchingCubesMesh(
    const SharedKeyframe* keyframe,
    float3* d_points,
    float3* d_normals, 
    MeshUpdate& update
) {
    // Initialize grid if needed
    if (pImpl->tsdf_fusion->getMemoryUsage() == 0) {
        // Get bounds from environment or use keyframe bounds
        float3 min_bounds = pImpl->getEnvFloat3("TSDF_SCENE_BOUNDS_MIN", 
            make_float3(keyframe->bbox[0] - 1.0f, keyframe->bbox[1] - 1.0f, keyframe->bbox[2] - 1.0f));
        float3 max_bounds = pImpl->getEnvFloat3("TSDF_SCENE_BOUNDS_MAX",
            make_float3(keyframe->bbox[3] + 1.0f, keyframe->bbox[4] + 1.0f, keyframe->bbox[5] + 1.0f));
        
        std::cout << "Initializing TSDF volume:" << std::endl;
        std::cout << "  Min bounds: [" << min_bounds.x << ", " << min_bounds.y << ", " << min_bounds.z << "]" << std::endl;
        std::cout << "  Max bounds: [" << max_bounds.x << ", " << max_bounds.y << ", " << max_bounds.z << "]" << std::endl;
        
        pImpl->tsdf_fusion->initialize(min_bounds, max_bounds);
    }
    
    // Get colors
    SharedMemoryManager smm;
    uint8_t* h_colors = smm.get_colors(const_cast<SharedKeyframe*>(keyframe));
    uint8_t* d_colors = nullptr;
    
    if (h_colors) {
        d_colors = (uint8_t*)pImpl->allocateFromPool(keyframe->point_count * 3);
        if (!d_colors) {
            cudaMalloc(&d_colors, keyframe->point_count * 3);
        }
        cudaMemcpyAsync(d_colors, h_colors, keyframe->point_count * 3, 
                        cudaMemcpyHostToDevice, pImpl->streams[0]);
    }
    
    // Integrate points into TSDF using incremental fusion
    pImpl->tsdf_fusion->integratePoints(
        d_points,
        nullptr,  // normals will be computed if needed
        d_colors,
        keyframe->point_count,
        keyframe->pose_matrix,
        pImpl->streams[0]
    );
    
    // Extract mesh with enhanced marching cubes
    pImpl->tsdf_fusion->extractMesh(update, pImpl->streams[0]);
    
    // Clean up colors
    if (d_colors) {
        if (pImpl->allocateFromPool(0)) {
            pImpl->releaseToPool(d_colors);
        } else {
            cudaFreeAsync(d_colors, pImpl->streams[0]);
        }
    }
}

// FUTURE: Uncomment this method to enable NKSR (Neural Kernel Surface Reconstruction)
/*
void GPUMeshGenerator::generateNKSRMesh(
    const SharedKeyframe* keyframe,
    float3* d_points,
    float3* d_normals, 
    MeshUpdate& update
) {
    // Get colors
    SharedMemoryManager smm;
    uint8_t* h_colors = smm.get_colors(const_cast<SharedKeyframe*>(keyframe));
    
    // Estimate point confidences based on local density
    thrust::device_vector<float> d_confidences(keyframe->point_count);
    
    // Simple confidence estimation - could be improved
    thrust::fill(d_confidences.begin(), d_confidences.end(), 1.0f);
    
    // Add points to NKSR
    pImpl->nksr->addPointStream(
        d_points,
        d_normals,
        d_confidences.data().get(),
        h_colors,  // Pass host colors directly
        keyframe->point_count,
        pImpl->streams[0]
    );
    
    // Process chunks
    pImpl->nksr->processChunks();
    
    // Extract mesh
    std::vector<float> confidence_map;
    pImpl->nksr->extractMesh(
        update.vertices,
        update.faces,
        update.vertex_colors,
        confidence_map,
        keyframe->bbox  // Use keyframe bounds for extraction
    );
    
    // Log progress
    float progress = pImpl->nksr->getProgress();
    size_t total_points = pImpl->nksr->getNumProcessedPoints();
    
    std::cout << "NKSR Progress: " << (progress * 100.0f) << "%, "
              << "Total points processed: " << total_points << std::endl;
}
*/

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

// CUDA kernels
namespace kernels {

__global__ void updateOctreeRegions(
    float3* new_points,
    int num_points,
    void* octree_nodes,
    int* dirty_flags
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    // Placeholder implementation
    // In production, this would update octree spatial index
}

__global__ void processIPSRBlock(
    float3* points,
    uint32_t num_points,
    void* block,
    float* d_implicit_function
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    // Placeholder implementation
    // In production, this would compute Poisson implicit function
}

__device__ uint64_t computeSpatialHash(float3 min, float3 max) {
    // Quantize to voxel grid
    int3 min_voxel = make_int3(
        __float2int_rd(min.x / VOXEL_BLOCK_SIZE),
        __float2int_rd(min.y / VOXEL_BLOCK_SIZE),
        __float2int_rd(min.z / VOXEL_BLOCK_SIZE)
    );
    
    // Morton encoding for spatial locality
    uint64_t hash = 0;
    for (int i = 0; i < 21; i++) {
        hash |= (min_voxel.x & (1 << i)) << (2 * i);
        hash |= (min_voxel.y & (1 << i)) << (2 * i + 1);
        hash |= (min_voxel.z & (1 << i)) << (2 * i + 2);
    }
    return hash;
}

} // namespace kernels
} // namespace mesh_service