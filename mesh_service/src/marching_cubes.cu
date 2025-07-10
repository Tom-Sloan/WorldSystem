#include "marching_cubes.h"
#include "mesh_generator.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include <iostream>
#include <unordered_map>
#include <chrono>

namespace mesh_service {

// Marching cubes lookup tables (subset shown for brevity)
__constant__ int edge_table_device[256];
__constant__ int tri_table_device[256][16];
__constant__ int vertex_offset_device[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
};

// Edge vertex positions
__constant__ float edge_vertex_positions[12][3] = {
    {0.5f, 0.0f, 0.0f}, {1.0f, 0.5f, 0.0f}, {0.5f, 1.0f, 0.0f}, {0.0f, 0.5f, 0.0f},
    {0.5f, 0.0f, 1.0f}, {1.0f, 0.5f, 1.0f}, {0.5f, 1.0f, 1.0f}, {0.0f, 0.5f, 1.0f},
    {0.0f, 0.0f, 0.5f}, {1.0f, 0.0f, 0.5f}, {1.0f, 1.0f, 0.5f}, {0.0f, 1.0f, 0.5f}
};

namespace cuda {

__device__ float3 interpolateVertex(
    float3 p1, float3 p2,
    float val1, float val2,
    float iso_value
) {
    float t = (iso_value - val1) / (val2 - val1);
    return make_float3(
        p1.x + t * (p2.x - p1.x),
        p1.y + t * (p2.y - p1.y),
        p1.z + t * (p2.z - p1.z)
    );
}

__device__ float length(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator/(float3 v, float s) {
    return make_float3(v.x / s, v.y / s, v.z / s);
}

__global__ void integrateTSDFKernel(
    float* tsdf_volume,
    float* weight_volume,
    int3 volume_size,
    float voxel_size,
    float truncation_distance,
    float3* points,
    int num_points,
    float* world_to_volume_mat
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    float3 point = points[idx];
    
    // Transform point to volume coordinates
    float3 voxel_pos = make_float3(
        world_to_volume_mat[0] * point.x + world_to_volume_mat[4] * point.y + 
        world_to_volume_mat[8] * point.z + world_to_volume_mat[12],
        world_to_volume_mat[1] * point.x + world_to_volume_mat[5] * point.y + 
        world_to_volume_mat[9] * point.z + world_to_volume_mat[13],
        world_to_volume_mat[2] * point.x + world_to_volume_mat[6] * point.y + 
        world_to_volume_mat[10] * point.z + world_to_volume_mat[14]
    );
    
    // Get voxel indices
    int3 voxel_idx = make_int3(
        __float2int_rn(voxel_pos.x),
        __float2int_rn(voxel_pos.y),
        __float2int_rn(voxel_pos.z)
    );
    
    // Check bounds
    if (voxel_idx.x < 1 || voxel_idx.x >= volume_size.x - 1 ||
        voxel_idx.y < 1 || voxel_idx.y >= volume_size.y - 1 ||
        voxel_idx.z < 1 || voxel_idx.z >= volume_size.z - 1) {
        return;
    }
    
    // Update TSDF in neighborhood
    for (int dz = -2; dz <= 2; dz++) {
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                int3 neighbor = make_int3(
                    voxel_idx.x + dx,
                    voxel_idx.y + dy,
                    voxel_idx.z + dz
                );
                
                if (neighbor.x < 0 || neighbor.x >= volume_size.x ||
                    neighbor.y < 0 || neighbor.y >= volume_size.y ||
                    neighbor.z < 0 || neighbor.z >= volume_size.z) {
                    continue;
                }
                
                // Compute distance from voxel center to point
                float3 voxel_center = make_float3(
                    neighbor.x * voxel_size,
                    neighbor.y * voxel_size,
                    neighbor.z * voxel_size
                );
                
                float3 diff = make_float3(
                    voxel_center.x - point.x,
                    voxel_center.y - point.y,
                    voxel_center.z - point.z
                );
                float dist = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
                
                if (dist < truncation_distance) {
                    int linear_idx = neighbor.x + 
                                   neighbor.y * volume_size.x + 
                                   neighbor.z * volume_size.x * volume_size.y;
                    
                    // Weighted average update
                    float weight = 1.0f;
                    float old_weight = weight_volume[linear_idx];
                    float new_weight = old_weight + weight;
                    
                    float old_tsdf = tsdf_volume[linear_idx];
                    float new_tsdf = (old_tsdf * old_weight + dist * weight) / new_weight;
                    
                    tsdf_volume[linear_idx] = new_tsdf;
                    weight_volume[linear_idx] = new_weight;
                }
            }
        }
    }
}

__global__ void marchingCubesKernel(
    float* tsdf_volume,
    int3 volume_size,
    float iso_value,
    float voxel_size,
    float3 volume_origin,
    float* vertices,
    int* triangles,
    int* vertex_count,
    int* triangle_count,
    int max_vertices
) {
    int3 voxel_pos;
    voxel_pos.x = blockIdx.x * blockDim.x + threadIdx.x;
    voxel_pos.y = blockIdx.y * blockDim.y + threadIdx.y;
    voxel_pos.z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (voxel_pos.x >= volume_size.x - 1 ||
        voxel_pos.y >= volume_size.y - 1 ||
        voxel_pos.z >= volume_size.z - 1) {
        return;
    }
    
    // Sample TSDF at cube vertices
    float cube_tsdf[8];
    int cube_index = 0;
    
    for (int i = 0; i < 8; i++) {
        int3 vertex_pos = make_int3(
            voxel_pos.x + vertex_offset_device[i][0],
            voxel_pos.y + vertex_offset_device[i][1],
            voxel_pos.z + vertex_offset_device[i][2]
        );
        
        int idx = vertex_pos.x + 
                 vertex_pos.y * volume_size.x + 
                 vertex_pos.z * volume_size.x * volume_size.y;
        
        cube_tsdf[i] = tsdf_volume[idx];
        
        if (cube_tsdf[i] < iso_value) {
            cube_index |= (1 << i);
        }
    }
    
    // Skip if cube is entirely inside or outside
    if (edge_table_device[cube_index] == 0) {
        return;
    }
    
    // Get vertices positions
    float3 cube_vertices[8];
    for (int i = 0; i < 8; i++) {
        cube_vertices[i] = make_float3(
            volume_origin.x + (voxel_pos.x + vertex_offset_device[i][0]) * voxel_size,
            volume_origin.y + (voxel_pos.y + vertex_offset_device[i][1]) * voxel_size,
            volume_origin.z + (voxel_pos.z + vertex_offset_device[i][2]) * voxel_size
        );
    }
    
    // Generate vertices on edges
    float3 edge_vertices[12];
    if (edge_table_device[cube_index] & 1) {
        edge_vertices[0] = interpolateVertex(cube_vertices[0], cube_vertices[1],
                                            cube_tsdf[0], cube_tsdf[1], iso_value);
    }
    if (edge_table_device[cube_index] & 2) {
        edge_vertices[1] = interpolateVertex(cube_vertices[1], cube_vertices[2],
                                            cube_tsdf[1], cube_tsdf[2], iso_value);
    }
    // ... continue for all 12 edges
    
    // Generate triangles
    for (int i = 0; tri_table_device[cube_index][i] != -1; i += 3) {
        int vertex_idx = atomicAdd(vertex_count, 3);
        if (vertex_idx + 2 >= max_vertices) break;
        
        int triangle_idx = atomicAdd(triangle_count, 1) * 3;
        
        for (int j = 0; j < 3; j++) {
            int edge_idx = tri_table_device[cube_index][i + j];
            vertices[vertex_idx + j * 3] = edge_vertices[edge_idx].x;
            vertices[vertex_idx + j * 3 + 1] = edge_vertices[edge_idx].y;
            vertices[vertex_idx + j * 3 + 2] = edge_vertices[edge_idx].z;
            
            triangles[triangle_idx + j] = vertex_idx + j;
        }
    }
}

__global__ void computeNormalsKernel(
    float* tsdf_volume,
    int3 volume_size,
    float voxel_size,
    float* vertices,
    float* normals,
    int num_vertices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vertices) return;
    
    float3 vertex = make_float3(
        vertices[idx * 3],
        vertices[idx * 3 + 1],
        vertices[idx * 3 + 2]
    );
    
    // Compute gradient using finite differences
    float3 gradient;
    float h = voxel_size * 0.5f;
    
    // Sample TSDF at neighboring points
    // This is simplified - in practice you'd transform back to voxel space
    gradient.x = 0.0f;  // Placeholder
    gradient.y = 0.0f;
    gradient.z = 1.0f;
    
    // Normalize
    float len = sqrtf(gradient.x * gradient.x + gradient.y * gradient.y + gradient.z * gradient.z);
    if (len > 0.0f) {
        gradient.x /= len;
        gradient.y /= len;
        gradient.z /= len;
    }
    
    normals[idx * 3] = gradient.x;
    normals[idx * 3 + 1] = gradient.y;
    normals[idx * 3 + 2] = gradient.z;
}

} // namespace cuda

// MarchingCubesGPU implementation
class MarchingCubesGPU::Impl {
public:
    Parameters params;
    int3 volume_size;
    float3 volume_origin;
    thrust::device_vector<float> tsdf_volume;
    thrust::device_vector<float> weight_volume;
    thrust::device_vector<float> d_vertices;
    thrust::device_vector<int> d_triangles;
    thrust::device_vector<int> d_counts;
    
    void allocateVolume(float3 min_bounds, float3 max_bounds) {
        volume_origin = min_bounds;
        
        volume_size.x = static_cast<int>((max_bounds.x - min_bounds.x) / params.voxel_size) + 1;
        volume_size.y = static_cast<int>((max_bounds.y - min_bounds.y) / params.voxel_size) + 1;
        volume_size.z = static_cast<int>((max_bounds.z - min_bounds.z) / params.voxel_size) + 1;
        
        // Clamp to maximum size
        volume_size.x = min(volume_size.x, params.max_voxels_per_dim);
        volume_size.y = min(volume_size.y, params.max_voxels_per_dim);
        volume_size.z = min(volume_size.z, params.max_voxels_per_dim);
        
        size_t total_voxels = volume_size.x * volume_size.y * volume_size.z;
        tsdf_volume.resize(total_voxels);
        weight_volume.resize(total_voxels);
        
        // Initialize with truncation distance
        thrust::fill(tsdf_volume.begin(), tsdf_volume.end(), params.truncation_distance);
        thrust::fill(weight_volume.begin(), weight_volume.end(), 0.0f);
        
        // Allocate output buffers
        size_t max_vertices = total_voxels * 3;  // Conservative estimate
        d_vertices.resize(max_vertices * 3);
        d_triangles.resize(max_vertices);
        d_counts.resize(2);  // vertex count, triangle count
    }
};

MarchingCubesGPU::MarchingCubesGPU() : pImpl(std::make_unique<Impl>()) {}
MarchingCubesGPU::~MarchingCubesGPU() = default;

void MarchingCubesGPU::setParameters(const Parameters& params) {
    pImpl->params = params;
}

void MarchingCubesGPU::initializeGrid(float3 min_bounds, float3 max_bounds) {
    pImpl->allocateVolume(min_bounds, max_bounds);
    std::cout << "Marching cubes grid: " << pImpl->volume_size.x << "x" 
              << pImpl->volume_size.y << "x" << pImpl->volume_size.z 
              << " voxels" << std::endl;
}

void MarchingCubesGPU::integrateTSDF(
    float3* d_points,
    float3* d_normals,
    size_t num_points,
    const float* camera_pose,
    cudaStream_t stream
) {
    // Create world to volume transformation
    thrust::device_vector<float> d_transform(16);
    // Simplified - just identity for now
    float h_transform[16] = {
        1.0f/pImpl->params.voxel_size, 0, 0, -pImpl->volume_origin.x/pImpl->params.voxel_size,
        0, 1.0f/pImpl->params.voxel_size, 0, -pImpl->volume_origin.y/pImpl->params.voxel_size,
        0, 0, 1.0f/pImpl->params.voxel_size, -pImpl->volume_origin.z/pImpl->params.voxel_size,
        0, 0, 0, 1
    };
    cudaMemcpyAsync(d_transform.data().get(), h_transform, 16 * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    
    // Launch integration kernel
    dim3 block(256);
    dim3 grid((num_points + block.x - 1) / block.x);
    
    cuda::integrateTSDFKernel<<<grid, block, 0, stream>>>(
        pImpl->tsdf_volume.data().get(),
        pImpl->weight_volume.data().get(),
        pImpl->volume_size,
        pImpl->params.voxel_size,
        pImpl->params.truncation_distance,
        d_points,
        num_points,
        d_transform.data().get()
    );
}

void MarchingCubesGPU::extractMesh(MeshUpdate& output, cudaStream_t stream) {
    // Reset counts
    thrust::fill(pImpl->d_counts.begin(), pImpl->d_counts.end(), 0);
    
    // Launch marching cubes kernel
    dim3 block(8, 8, 8);
    dim3 grid(
        (pImpl->volume_size.x + block.x - 1) / block.x,
        (pImpl->volume_size.y + block.y - 1) / block.y,
        (pImpl->volume_size.z + block.z - 1) / block.z
    );
    
    cuda::marchingCubesKernel<<<grid, block, 0, stream>>>(
        pImpl->tsdf_volume.data().get(),
        pImpl->volume_size,
        pImpl->params.iso_value,
        pImpl->params.voxel_size,
        pImpl->volume_origin,
        pImpl->d_vertices.data().get(),
        pImpl->d_triangles.data().get(),
        pImpl->d_counts.data().get(),
        pImpl->d_counts.data().get() + 1,
        pImpl->d_vertices.size() / 3
    );
    
    cudaStreamSynchronize(stream);
    
    // Copy counts back
    thrust::host_vector<int> h_counts = pImpl->d_counts;
    int vertex_count = h_counts[0];
    int triangle_count = h_counts[1];
    
    // Copy mesh data
    output.vertices.resize(vertex_count * 3);
    output.faces.resize(triangle_count * 3);
    
    if (vertex_count > 0) {
        cudaMemcpy(output.vertices.data(), pImpl->d_vertices.data().get(),
                   vertex_count * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    if (triangle_count > 0) {
        cudaMemcpy(output.faces.data(), pImpl->d_triangles.data().get(),
                   triangle_count * 3 * sizeof(int), cudaMemcpyDeviceToHost);
    }
    
    std::cout << "Marching cubes extracted: " << vertex_count << " vertices, "
              << triangle_count << " triangles" << std::endl;
}

void MarchingCubesGPU::reset() {
    thrust::fill(pImpl->tsdf_volume.begin(), pImpl->tsdf_volume.end(), 
                 pImpl->params.truncation_distance);
    thrust::fill(pImpl->weight_volume.begin(), pImpl->weight_volume.end(), 0.0f);
}

// IncrementalTSDFFusion implementation
class IncrementalTSDFFusion::Impl {
public:
    float voxel_size = 0.05f;
    std::unordered_map<uint64_t, std::unique_ptr<VoxelBlock>> blocks;
    thrust::device_vector<float> d_temp_vertices;
    thrust::device_vector<int> d_temp_triangles;
    
    uint64_t hashBlock(int3 pos) {
        // Simple spatial hash
        uint64_t h = 0;
        h ^= std::hash<int>{}(pos.x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(pos.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(pos.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

IncrementalTSDFFusion::IncrementalTSDFFusion() : pImpl(std::make_unique<Impl>()) {}
IncrementalTSDFFusion::~IncrementalTSDFFusion() = default;

void IncrementalTSDFFusion::setVoxelSize(float size) {
    pImpl->voxel_size = size;
}

void IncrementalTSDFFusion::allocateBlock(int3 block_pos, cudaStream_t stream) {
    uint64_t hash = pImpl->hashBlock(block_pos);
    
    if (pImpl->blocks.find(hash) == pImpl->blocks.end()) {
        auto block = std::make_unique<VoxelBlock>();
        block->block_pos = block_pos;
        
        size_t block_voxels = VoxelBlock::BLOCK_SIZE * 
                             VoxelBlock::BLOCK_SIZE * 
                             VoxelBlock::BLOCK_SIZE;
        
        cudaMallocAsync(&block->tsdf_values, block_voxels * sizeof(float), stream);
        cudaMallocAsync(&block->weights, block_voxels * sizeof(float), stream);
        cudaMallocAsync(&block->colors, block_voxels * 3 * sizeof(uint8_t), stream);
        
        // Initialize
        cudaMemsetAsync(block->tsdf_values, 0, block_voxels * sizeof(float), stream);
        cudaMemsetAsync(block->weights, 0, block_voxels * sizeof(float), stream);
        
        block->allocated = true;
        pImpl->blocks[hash] = std::move(block);
    }
}

void IncrementalTSDFFusion::integratePointCloud(
    float3* d_points,
    size_t num_points,
    const float* pose,
    cudaStream_t stream
) {
    // Determine which blocks need allocation
    // This would require downloading points to CPU or a GPU kernel
    // For now, simplified implementation
    
    // Mark blocks as needing update
    for (auto& [hash, block] : pImpl->blocks) {
        block->needs_update = true;
        block->last_update = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
}

void IncrementalTSDFFusion::extractMeshIncremental(
    MeshUpdate& output,
    cudaStream_t stream
) {
    output.vertices.clear();
    output.faces.clear();
    
    // Process each dirty block
    for (auto& [hash, block] : pImpl->blocks) {
        if (!block->needs_update) continue;
        
        // Extract mesh from this block using marching cubes
        // This would use a kernel similar to marchingCubesKernel but per-block
        
        block->needs_update = false;
    }
}

size_t IncrementalTSDFFusion::getMemoryUsage() const {
    size_t total = 0;
    size_t block_size = VoxelBlock::BLOCK_SIZE * 
                       VoxelBlock::BLOCK_SIZE * 
                       VoxelBlock::BLOCK_SIZE;
    
    for (const auto& [hash, block] : pImpl->blocks) {
        if (block->allocated) {
            total += block_size * sizeof(float) * 2;  // TSDF + weights
            total += block_size * 3;                  // Colors
        }
    }
    
    return total;
}

} // namespace mesh_service