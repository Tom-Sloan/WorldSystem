#include "mesh_generator.h"
#include "poisson_reconstruction.h"
#include "marching_cubes.h"
#include "normal_estimation.h"
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
    std::unique_ptr<PoissonReconstruction> poisson;
    std::unique_ptr<IncrementalPoissonReconstruction> incremental_poisson;
    std::unique_ptr<MarchingCubesGPU> marching_cubes;
    std::unique_ptr<IncrementalTSDFFusion> tsdf_fusion;
    std::unique_ptr<NormalEstimation> normal_estimator;
    
    // Device memory for normals
    thrust::device_vector<float3> d_normals;
    
    Impl() {
        // Create CUDA streams
        for (int i = 0; i < 5; i++) {
            cudaStreamCreate(&streams[i]);
        }
        
        // Allocate memory pool (1GB for RTX 3090)
        memory_pool_size = 1024 * 1024 * 1024;
        cudaMalloc(&d_memory_pool, memory_pool_size);
        
        // Initialize mesh generation components
        poisson = std::make_unique<PoissonReconstruction>();
        incremental_poisson = std::make_unique<IncrementalPoissonReconstruction>();
        marching_cubes = std::make_unique<MarchingCubesGPU>();
        tsdf_fusion = std::make_unique<IncrementalTSDFFusion>();
        normal_estimator = std::make_unique<NormalEstimation>();
        
        // Configure components
        PoissonReconstruction::Parameters poisson_params;
        poisson_params.octree_depth = 8;
        poisson_params.point_weight = 4.0f;
        poisson->setParameters(poisson_params);
        
        MarchingCubesGPU::Parameters mc_params;
        mc_params.voxel_size = 0.05f;
        mc_params.truncation_distance = 0.1f;
        marching_cubes->setParameters(mc_params);
        
        NormalEstimation::Parameters normal_params;
        normal_params.method = NormalEstimation::PCA;
        normal_params.k_neighbors = 30;
        normal_estimator->setParameters(normal_params);
        
        // Initialize incremental systems
        incremental_poisson->initialize(10.0f, 8);  // 10m scene, 8x8x8 grid
        tsdf_fusion->setVoxelSize(0.05f);
    }
    
    ~Impl() {
        // Destroy streams
        for (int i = 0; i < 5; i++) {
            cudaStreamDestroy(streams[i]);
        }
        
        // Free memory pool
        if (d_memory_pool) {
            cudaFree(d_memory_pool);
        }
    }
};

GPUMeshGenerator::GPUMeshGenerator() : pImpl(std::make_unique<Impl>()) {}
GPUMeshGenerator::~GPUMeshGenerator() = default;

void GPUMeshGenerator::setMethod(MeshMethod method) {
    pImpl->method = method;
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
    
    // Adaptive method selection based on camera motion
    if (pImpl->quality_adaptive) {
        if (pImpl->camera_velocity > 0.5f) {
            pImpl->method = MeshMethod::TSDF_MARCHING_CUBES;  // Fast for motion
        } else {
            pImpl->method = MeshMethod::INCREMENTAL_POISSON;  // Quality when static
        }
    }
    
    // Check spatial deduplication (handle 90% overlap)
    uint64_t spatial_hash = computeSpatialHashCPU(keyframe->bbox);
    std::cout << "[DEBUG] Spatial hash: " << spatial_hash 
              << ", bbox: [" << keyframe->bbox[0] << "," << keyframe->bbox[1] 
              << "," << keyframe->bbox[2] << " - " << keyframe->bbox[3] 
              << "," << keyframe->bbox[4] << "," << keyframe->bbox[5] << "]" << std::endl;
    
    auto it = pImpl->processed_regions.find(spatial_hash);
    if (it != pImpl->processed_regions.end() && 
        it->second == keyframe->timestamp_ns) {
        // Already processed this region at this timestamp
        std::cout << "[DEBUG] Region already processed, skipping" << std::endl;
        return;
    }
    
    // Also check if we've processed this exact region before (different timestamp)
    if (it != pImpl->processed_regions.end()) {
        std::cout << "[DEBUG] Region processed before with different timestamp" << std::endl;
        // For now, let's process it anyway since it's a different timestamp
        // return;
    }
    
    // Get point and color data
    SharedMemoryManager smm;
    float* h_points = smm.get_points(const_cast<SharedKeyframe*>(keyframe));
    uint8_t* h_colors = smm.get_colors(const_cast<SharedKeyframe*>(keyframe));
    
    // Allocate device memory
    float3* d_points;
    size_t points_size = keyframe->point_count * sizeof(float3);
    cudaMallocAsync(&d_points, points_size, pImpl->streams[0]);
    
    // Copy points to device
    cudaMemcpyAsync(d_points, h_points, points_size, 
                    cudaMemcpyHostToDevice, pImpl->streams[0]);
    
    // Allocate device memory for normals
    pImpl->d_normals.resize(keyframe->point_count);
    
    // Estimate normals first (required for Poisson)
    pImpl->normal_estimator->estimateNormals(
        d_points, 
        keyframe->point_count,
        pImpl->d_normals.data().get(),
        pImpl->streams[0]
    );
    
    // Generate mesh based on selected method
    switch (pImpl->method) {
        case MeshMethod::INCREMENTAL_POISSON:
            generateIncrementalPoissonMesh(keyframe, d_points, pImpl->d_normals.data().get(), update);
            break;
        case MeshMethod::TSDF_MARCHING_CUBES:
            generateMarchingCubesMesh(keyframe, d_points, pImpl->d_normals.data().get(), update);
            break;
        default:
            generateIncrementalPoissonMesh(keyframe, d_points, pImpl->d_normals.data().get(), update);
    }
    
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
    
    // Mark region as processed
    pImpl->processed_regions[spatial_hash] = keyframe->timestamp_ns;
    
    // Clean up
    cudaFreeAsync(d_points, pImpl->streams[0]);
    cudaStreamSynchronize(pImpl->streams[0]);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    std::cout << "Mesh generation completed in " << duration << "ms: "
              << update.vertices.size()/3 << " vertices, "
              << update.faces.size()/3 << " faces" << std::endl;
}

void GPUMeshGenerator::generateIncrementalPoissonMesh(
    const SharedKeyframe* keyframe,
    float3* d_points, 
    float3* d_normals,
    MeshUpdate& update
) {
    // Copy points and normals to host for CGAL processing
    std::vector<float> h_points(keyframe->point_count * 3);
    std::vector<float> h_normals(keyframe->point_count * 3);
    
    cudaMemcpy(h_points.data(), d_points, keyframe->point_count * sizeof(float3),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_normals.data(), d_normals, keyframe->point_count * sizeof(float3),
               cudaMemcpyDeviceToHost);
    
    // Add points to incremental reconstruction
    pImpl->incremental_poisson->addPoints(
        h_points.data(),
        h_normals.data(),
        keyframe->point_count,
        keyframe->timestamp_ns
    );
    
    // Update only dirty blocks
    pImpl->incremental_poisson->updateDirtyBlocks(update);
}

void GPUMeshGenerator::generateMarchingCubesMesh(
    const SharedKeyframe* keyframe,
    float3* d_points,
    float3* d_normals, 
    MeshUpdate& update
) {
    // Initialize grid if needed
    if (pImpl->tsdf_fusion->getMemoryUsage() == 0) {
        float3 min_bounds = make_float3(keyframe->bbox[0], keyframe->bbox[1], keyframe->bbox[2]);
        float3 max_bounds = make_float3(keyframe->bbox[3], keyframe->bbox[4], keyframe->bbox[5]);
        
        // Expand bounds for room
        min_bounds.x -= 1.0f;
        min_bounds.y -= 1.0f;
        min_bounds.z -= 1.0f;
        max_bounds.x += 1.0f;
        max_bounds.y += 1.0f;
        max_bounds.z += 1.0f;
        
        pImpl->marching_cubes->initializeGrid(min_bounds, max_bounds);
    }
    
    // Integrate points into TSDF
    pImpl->marching_cubes->integrateTSDF(
        d_points,
        d_normals,
        keyframe->point_count,
        keyframe->pose_matrix,
        pImpl->streams[0]
    );
    
    // Extract mesh
    pImpl->marching_cubes->extractMesh(update, pImpl->streams[0]);
}

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