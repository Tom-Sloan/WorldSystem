#include "mesh_generator.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <unordered_map>

namespace mesh_service {

// Forward declaration
uint64_t computeSpatialHashCPU(const float bbox[6]);

// Simple triangle mesh generation from point cloud
__global__ void generateTriangleMesh(
    float3* points,
    uint32_t num_points,
    uint32_t* triangles,
    uint32_t* triangle_count,
    uint32_t max_triangles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points - 2) return;
    
    // Simple triangulation: connect every 3 consecutive points
    if (idx % 3 == 0 && idx + 2 < num_points) {
        uint32_t tri_idx = atomicAdd(triangle_count, 1);
        if (tri_idx < max_triangles) {
            triangles[tri_idx * 3] = idx;
            triangles[tri_idx * 3 + 1] = idx + 1;
            triangles[tri_idx * 3 + 2] = idx + 2;
        }
    }
}

// Constants
constexpr size_t VOXEL_BLOCK_SIZE = 256;
constexpr int THREADS_PER_BLOCK = 128;

// Implementation
class GPUMeshGenerator::Impl {
public:
    MeshMethod method = MeshMethod::INCREMENTAL_POISSON;
    bool quality_adaptive = true;
    float simplification_ratio = 0.1f;
    float camera_velocity = 0.0f;
    
    cudaStream_t streams[5];
    void* d_memory_pool = nullptr;
    size_t memory_pool_size = 0;
    
    std::unordered_map<uint64_t, uint64_t> processed_regions;
    float prev_camera_pos[3] = {0, 0, 0};
    
    // Device memory
    thrust::device_vector<uint32_t> d_triangles;
    thrust::device_vector<uint32_t> d_triangle_count;
    
    Impl() {
        for (int i = 0; i < 5; i++) {
            cudaStreamCreate(&streams[i]);
        }
        
        memory_pool_size = 512 * 1024 * 1024; // 512MB
        cudaMalloc(&d_memory_pool, memory_pool_size);
        
        // Allocate triangle buffer
        d_triangles.resize(1000000 * 3); // Max 1M triangles
        d_triangle_count.resize(1);
    }
    
    ~Impl() {
        for (int i = 0; i < 5; i++) {
            cudaStreamDestroy(streams[i]);
        }
        
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
    float curr_pos[3] = {pose_matrix[12], pose_matrix[13], pose_matrix[14]};
    
    float dx = curr_pos[0] - pImpl->prev_camera_pos[0];
    float dy = curr_pos[1] - pImpl->prev_camera_pos[1];
    float dz = curr_pos[2] - pImpl->prev_camera_pos[2];
    
    pImpl->camera_velocity = std::sqrt(dx*dx + dy*dy + dz*dz);
    std::copy(curr_pos, curr_pos + 3, pImpl->prev_camera_pos);
}

void GPUMeshGenerator::generateIncrementalMesh(
    const SharedKeyframe* keyframe,
    MeshUpdate& update
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    updateCameraVelocity(keyframe->pose_matrix);
    
    // Check spatial deduplication
    uint64_t spatial_hash = computeSpatialHashCPU(keyframe->bbox);
    std::cout << "[DEBUG] SimpleMeshGen - Spatial hash: " << spatial_hash 
              << ", bbox: [" << keyframe->bbox[0] << "," << keyframe->bbox[1] 
              << "," << keyframe->bbox[2] << " - " << keyframe->bbox[3] 
              << "," << keyframe->bbox[4] << "," << keyframe->bbox[5] << "]" << std::endl;
    
    auto it = pImpl->processed_regions.find(spatial_hash);
    if (it != pImpl->processed_regions.end()) {
        std::cout << "[DEBUG] SimpleMeshGen - Region seen before, but processing anyway for testing" << std::endl;
        // For testing, don't skip duplicate regions
        // return;
    }
    
    // Get point data
    SharedMemoryManager smm;
    float* h_points = smm.get_points(const_cast<SharedKeyframe*>(keyframe));
    uint8_t* h_colors = smm.get_colors(const_cast<SharedKeyframe*>(keyframe));
    
    // Allocate device memory
    float3* d_points;
    size_t points_size = keyframe->point_count * sizeof(float3);
    cudaMallocAsync(&d_points, points_size, pImpl->streams[0]);
    
    // Copy to device
    cudaMemcpyAsync(d_points, h_points, points_size, 
                    cudaMemcpyHostToDevice, pImpl->streams[0]);
    
    // Reset triangle count
    thrust::fill(pImpl->d_triangle_count.begin(), pImpl->d_triangle_count.end(), 0);
    
    // Generate simple triangle mesh
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((keyframe->point_count + block.x - 1) / block.x);
    
    generateTriangleMesh<<<grid, block, 0, pImpl->streams[0]>>>(
        d_points,
        keyframe->point_count,
        pImpl->d_triangles.data().get(),
        pImpl->d_triangle_count.data().get(),
        pImpl->d_triangles.size() / 3
    );
    
    cudaStreamSynchronize(pImpl->streams[0]);
    
    // Get triangle count
    uint32_t triangle_count = pImpl->d_triangle_count[0];
    
    // Copy vertices
    update.vertices.resize(keyframe->point_count * 3);
    cudaMemcpy(update.vertices.data(), d_points, points_size,
               cudaMemcpyDeviceToHost);
    
    // Copy triangles
    if (triangle_count > 0) {
        update.faces.resize(triangle_count * 3);
        cudaMemcpy(update.faces.data(), pImpl->d_triangles.data().get(),
                   triangle_count * 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    
    // Copy colors
    update.vertex_colors.resize(keyframe->point_count * 3);
    std::copy(h_colors, h_colors + keyframe->point_count * 3, 
              update.vertex_colors.begin());
    
    // Update metadata
    update.keyframe_id = std::to_string(keyframe->timestamp_ns);
    update.timestamp_ns = keyframe->timestamp_ns;
    
    pImpl->processed_regions[spatial_hash] = keyframe->timestamp_ns;
    
    cudaFreeAsync(d_points, pImpl->streams[0]);
    cudaStreamSynchronize(pImpl->streams[0]);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    std::cout << "Simple mesh generation completed in " << duration << "ms: "
              << update.vertices.size()/3 << " vertices, "
              << update.faces.size()/3 << " faces" << std::endl;
}

void GPUMeshGenerator::generateIncrementalPoissonMesh(
    const SharedKeyframe* keyframe,
    float3* d_points,
    float3* d_normals,
    MeshUpdate& update
) {
    // Simplified version - just copy points as vertices
    update.vertices.resize(keyframe->point_count * 3);
    cudaMemcpy(update.vertices.data(), d_points, 
               keyframe->point_count * sizeof(float3),
               cudaMemcpyDeviceToHost);
    
    // Create a simple triangulation
    update.faces.clear();
    for (uint32_t i = 0; i + 2 < keyframe->point_count; i += 3) {
        update.faces.push_back(i);
        update.faces.push_back(i + 1);
        update.faces.push_back(i + 2);
    }
}

void GPUMeshGenerator::generateMarchingCubesMesh(
    const SharedKeyframe* keyframe,
    float3* d_points,
    float3* d_normals,
    MeshUpdate& update
) {
    // Simplified version - same as Poisson for now
    generateIncrementalPoissonMesh(keyframe, d_points, d_normals, update);
}

uint64_t computeSpatialHashCPU(const float bbox[6]) {
    float center_x = (bbox[0] + bbox[3]) / 2.0f;
    float center_y = (bbox[1] + bbox[4]) / 2.0f;
    float center_z = (bbox[2] + bbox[5]) / 2.0f;
    
    int vx = static_cast<int>(center_x / VOXEL_BLOCK_SIZE);
    int vy = static_cast<int>(center_y / VOXEL_BLOCK_SIZE);
    int vz = static_cast<int>(center_z / VOXEL_BLOCK_SIZE);
    
    uint64_t hash = 0;
    hash ^= std::hash<int>{}(vx) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int>{}(vy) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int>{}(vz) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    
    return hash;
}

// Empty kernel implementations for linking
namespace kernels {

__global__ void updateOctreeRegions(
    float3* new_points,
    int num_points,
    void* octree_nodes,
    int* dirty_flags
) {
    // Placeholder
}

__global__ void processIPSRBlock(
    float3* points,
    uint32_t num_points,
    void* block,
    float* d_implicit_function
) {
    // Placeholder
}

__device__ uint64_t computeSpatialHash(float3 min, float3 max) {
    return 0;
}

} // namespace kernels
} // namespace mesh_service