#include "marching_cubes.h"
#include "marching_cubes_tables.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include <cub/cub.cuh>
#include <iostream>

namespace mesh_service {

// Constants for voxel block management
constexpr int VOXEL_BLOCK_SIZE = 8;
constexpr int THREADS_PER_BLOCK = 64;
constexpr float EPSILON = 1e-6f;

// Voxel block structure for incremental updates
struct VoxelBlock {
    int3 block_coord;
    float3 min_bound;
    float3 max_bound;
    int voxel_offset;  // Offset into global voxel array
    bool is_allocated;
    bool is_dirty;
    uint32_t last_update;
};

namespace cuda {

// Complete edge table for all 12 edges
__device__ const int edge_connections[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},  // Bottom face edges
    {4, 5}, {5, 6}, {6, 7}, {7, 4},  // Top face edges
    {0, 4}, {1, 5}, {2, 6}, {3, 7}   // Vertical edges
};

// Optimized vertex interpolation with gradient-based normal estimation
__device__ float3 interpolateVertexWithNormal(
    float3 p1, float3 p2,
    float val1, float val2,
    float iso_value,
    float3* normal_out,
    const float3& grad1,
    const float3& grad2
) {
    // Ensure we don't divide by zero
    float denominator = val2 - val1;
    if (fabsf(denominator) < EPSILON) {
        *normal_out = normalize(grad1);
        return p1;
    }
    
    float t = (iso_value - val1) / denominator;
    t = fmaxf(0.0f, fminf(1.0f, t));  // Clamp to [0,1]
    
    // Interpolate position
    float3 result = make_float3(
        p1.x + t * (p2.x - p1.x),
        p1.y + t * (p2.y - p1.y),
        p1.z + t * (p2.z - p1.z)
    );
    
    // Interpolate normal from gradients
    *normal_out = make_float3(
        grad1.x + t * (grad2.x - grad1.x),
        grad1.y + t * (grad2.y - grad1.y),
        grad1.z + t * (grad2.z - grad1.z)
    );
    *normal_out = normalize(*normal_out);
    
    return result;
}

// Compute gradient at a voxel using central differences
__device__ float3 computeGradient(
    const float* tsdf_volume,
    int3 voxel_pos,
    int3 volume_size,
    float voxel_size
) {
    float3 gradient = make_float3(0, 0, 0);
    
    // Central differences for each axis
    if (voxel_pos.x > 0 && voxel_pos.x < volume_size.x - 1) {
        int idx_neg = (voxel_pos.x - 1) + voxel_pos.y * volume_size.x + voxel_pos.z * volume_size.x * volume_size.y;
        int idx_pos = (voxel_pos.x + 1) + voxel_pos.y * volume_size.x + voxel_pos.z * volume_size.x * volume_size.y;
        gradient.x = (tsdf_volume[idx_pos] - tsdf_volume[idx_neg]) / (2.0f * voxel_size);
    }
    
    if (voxel_pos.y > 0 && voxel_pos.y < volume_size.y - 1) {
        int idx_neg = voxel_pos.x + (voxel_pos.y - 1) * volume_size.x + voxel_pos.z * volume_size.x * volume_size.y;
        int idx_pos = voxel_pos.x + (voxel_pos.y + 1) * volume_size.x + voxel_pos.z * volume_size.x * volume_size.y;
        gradient.y = (tsdf_volume[idx_pos] - tsdf_volume[idx_neg]) / (2.0f * voxel_size);
    }
    
    if (voxel_pos.z > 0 && voxel_pos.z < volume_size.z - 1) {
        int idx_neg = voxel_pos.x + voxel_pos.y * volume_size.x + (voxel_pos.z - 1) * volume_size.x * volume_size.y;
        int idx_pos = voxel_pos.x + voxel_pos.y * volume_size.x + (voxel_pos.z + 1) * volume_size.x * volume_size.y;
        gradient.z = (tsdf_volume[idx_pos] - tsdf_volume[idx_neg]) / (2.0f * voxel_size);
    }
    
    return gradient;
}

// Incremental TSDF integration with weighted averaging
__global__ void incrementalTSDFIntegration(
    float* tsdf_volume,
    float* weight_volume,
    uint8_t* color_volume,  // RGB colors
    const VoxelBlock* voxel_blocks,
    const int* dirty_block_indices,
    int num_dirty_blocks,
    const float3* points,
    const float3* normals,
    const uint8_t* colors,
    int num_points,
    int3 volume_size,
    float voxel_size,
    float truncation_distance,
    const float* camera_pose
) {
    int block_idx = blockIdx.x;
    if (block_idx >= num_dirty_blocks) return;
    
    int block_id = dirty_block_indices[block_idx];
    const VoxelBlock& block = voxel_blocks[block_id];
    
    if (!block.is_allocated) return;
    
    // Each thread processes one voxel in the block
    int local_idx = threadIdx.x;
    int voxels_per_block = VOXEL_BLOCK_SIZE * VOXEL_BLOCK_SIZE * VOXEL_BLOCK_SIZE;
    
    for (int v = local_idx; v < voxels_per_block; v += blockDim.x) {
        // Convert local index to 3D coordinates
        int local_x = v % VOXEL_BLOCK_SIZE;
        int local_y = (v / VOXEL_BLOCK_SIZE) % VOXEL_BLOCK_SIZE;
        int local_z = v / (VOXEL_BLOCK_SIZE * VOXEL_BLOCK_SIZE);
        
        // Global voxel position
        int3 global_voxel = make_int3(
            block.block_coord.x * VOXEL_BLOCK_SIZE + local_x,
            block.block_coord.y * VOXEL_BLOCK_SIZE + local_y,
            block.block_coord.z * VOXEL_BLOCK_SIZE + local_z
        );
        
        if (global_voxel.x >= volume_size.x || 
            global_voxel.y >= volume_size.y || 
            global_voxel.z >= volume_size.z) continue;
        
        // World position of voxel center
        float3 voxel_center = make_float3(
            block.min_bound.x + (local_x + 0.5f) * voxel_size,
            block.min_bound.y + (local_y + 0.5f) * voxel_size,
            block.min_bound.z + (local_z + 0.5f) * voxel_size
        );
        
        // Camera position from pose matrix
        float3 camera_pos = make_float3(camera_pose[12], camera_pose[13], camera_pose[14]);
        
        // Ray from camera to voxel
        float3 ray = voxel_center - camera_pos;
        float ray_length = length(ray);
        ray = ray / ray_length;
        
        // Find closest point
        float min_dist = truncation_distance;
        int closest_point = -1;
        
        for (int p = 0; p < num_points; p++) {
            float3 diff = points[p] - voxel_center;
            float dist = length(diff);
            
            if (dist < min_dist) {
                // Check if point is visible from this voxel
                float3 point_ray = points[p] - camera_pos;
                float dot_product = dot(normalize(point_ray), ray);
                
                if (dot_product > 0.8f) {  // Within ~37 degree cone
                    min_dist = dist;
                    closest_point = p;
                }
            }
        }
        
        if (closest_point >= 0) {
            // Compute signed distance
            float3 point_to_voxel = voxel_center - points[closest_point];
            float sdf = dot(point_to_voxel, normals[closest_point]);
            
            // Truncate SDF
            sdf = fmaxf(-truncation_distance, fminf(truncation_distance, sdf));
            
            // Update TSDF with weighted averaging
            int voxel_idx = global_voxel.x + 
                           global_voxel.y * volume_size.x + 
                           global_voxel.z * volume_size.x * volume_size.y;
            
            float weight = 1.0f / (1.0f + min_dist * min_dist);  // Distance-based weight
            float old_weight = weight_volume[voxel_idx];
            float new_weight = old_weight + weight;
            
            // Update TSDF
            float old_tsdf = tsdf_volume[voxel_idx];
            tsdf_volume[voxel_idx] = (old_tsdf * old_weight + sdf * weight) / new_weight;
            weight_volume[voxel_idx] = new_weight;
            
            // Update color
            if (colors != nullptr) {
                int color_idx = voxel_idx * 3;
                const uint8_t* point_color = &colors[closest_point * 3];
                
                for (int c = 0; c < 3; c++) {
                    float old_color = color_volume[color_idx + c];
                    color_volume[color_idx + c] = (uint8_t)((old_color * old_weight + point_color[c] * weight) / new_weight);
                }
            }
        }
    }
}

// Enhanced marching cubes with complete edge interpolation
__global__ void enhancedMarchingCubes(
    const float* tsdf_volume,
    const float* weight_volume,
    const uint8_t* color_volume,
    const VoxelBlock* voxel_blocks,
    const int* blocks_to_extract,
    int num_blocks,
    int3 volume_size,
    float voxel_size,
    float iso_value,
    float3* vertices,
    float3* normals,
    uint8_t* vertex_colors,
    uint32_t* faces,
    int* vertex_counter,
    int* face_counter,
    int max_vertices
) {
    int block_idx = blockIdx.x;
    if (block_idx >= num_blocks) return;
    
    int block_id = blocks_to_extract[block_idx];
    const VoxelBlock& block = voxel_blocks[block_id];
    
    if (!block.is_allocated || !block.is_dirty) return;
    
    // Shared memory for vertex caching
    extern __shared__ float shared_mem[];
    float* shared_vertices = shared_mem;
    
    int tid = threadIdx.x;
    int voxels_per_block = (VOXEL_BLOCK_SIZE - 1) * (VOXEL_BLOCK_SIZE - 1) * (VOXEL_BLOCK_SIZE - 1);
    
    for (int v = tid; v < voxels_per_block; v += blockDim.x) {
        // Convert to 3D local coordinates
        int local_x = v % (VOXEL_BLOCK_SIZE - 1);
        int local_y = (v / (VOXEL_BLOCK_SIZE - 1)) % (VOXEL_BLOCK_SIZE - 1);
        int local_z = v / ((VOXEL_BLOCK_SIZE - 1) * (VOXEL_BLOCK_SIZE - 1));
        
        // Global voxel position
        int3 voxel_pos = make_int3(
            block.block_coord.x * VOXEL_BLOCK_SIZE + local_x,
            block.block_coord.y * VOXEL_BLOCK_SIZE + local_y,
            block.block_coord.z * VOXEL_BLOCK_SIZE + local_z
        );
        
        if (voxel_pos.x >= volume_size.x - 1 || 
            voxel_pos.y >= volume_size.y - 1 || 
            voxel_pos.z >= volume_size.z - 1) continue;
        
        // Sample TSDF at cube vertices
        float cube_tsdf[8];
        float3 cube_gradients[8];
        float3 cube_positions[8];
        uint8_t cube_colors[8][3];
        int cube_index = 0;
        
        // Sample all 8 corners
        for (int corner = 0; corner < 8; corner++) {
            int dx = corner & 1;
            int dy = (corner >> 1) & 1;
            int dz = (corner >> 2) & 1;
            
            int3 corner_voxel = make_int3(
                voxel_pos.x + dx,
                voxel_pos.y + dy,
                voxel_pos.z + dz
            );
            
            int idx = corner_voxel.x + 
                     corner_voxel.y * volume_size.x + 
                     corner_voxel.z * volume_size.x * volume_size.y;
            
            cube_tsdf[corner] = tsdf_volume[idx];
            cube_gradients[corner] = computeGradient(tsdf_volume, corner_voxel, volume_size, voxel_size);
            
            cube_positions[corner] = make_float3(
                block.min_bound.x + (local_x + dx) * voxel_size,
                block.min_bound.y + (local_y + dy) * voxel_size,
                block.min_bound.z + (local_z + dz) * voxel_size
            );
            
            // Sample colors
            if (color_volume != nullptr) {
                int color_idx = idx * 3;
                cube_colors[corner][0] = color_volume[color_idx];
                cube_colors[corner][1] = color_volume[color_idx + 1];
                cube_colors[corner][2] = color_volume[color_idx + 2];
            }
            
            if (cube_tsdf[corner] < iso_value) {
                cube_index |= (1 << corner);
            }
        }
        
        // Skip if no intersection
        if (edgeTable[cube_index] == 0) continue;
        
        // Interpolate vertices on all 12 edges
        float3 edge_vertices[12];
        float3 edge_normals[12];
        uint8_t edge_colors[12][3];
        
        for (int edge = 0; edge < 12; edge++) {
            if (edgeTable[cube_index] & (1 << edge)) {
                int v1 = edge_connections[edge][0];
                int v2 = edge_connections[edge][1];
                
                edge_vertices[edge] = interpolateVertexWithNormal(
                    cube_positions[v1], cube_positions[v2],
                    cube_tsdf[v1], cube_tsdf[v2],
                    iso_value,
                    &edge_normals[edge],
                    cube_gradients[v1], cube_gradients[v2]
                );
                
                // Interpolate colors
                if (color_volume != nullptr) {
                    float t = (iso_value - cube_tsdf[v1]) / (cube_tsdf[v2] - cube_tsdf[v1]);
                    t = fmaxf(0.0f, fminf(1.0f, t));
                    
                    for (int c = 0; c < 3; c++) {
                        edge_colors[edge][c] = (uint8_t)(cube_colors[v1][c] * (1.0f - t) + cube_colors[v2][c] * t);
                    }
                }
            }
        }
        
        // Generate triangles from the table
        for (int i = 0; triTable[cube_index][i] != -1; i += 3) {
            int base_vertex = atomicAdd(vertex_counter, 3);
            if (base_vertex + 2 >= max_vertices) break;
            
            int face_idx = atomicAdd(face_counter, 1);
            
            // Add three vertices of the triangle
            for (int j = 0; j < 3; j++) {
                int edge = triTable[cube_index][i + j];
                int vertex_idx = base_vertex + j;
                
                vertices[vertex_idx] = edge_vertices[edge];
                normals[vertex_idx] = edge_normals[edge];
                
                if (vertex_colors != nullptr && color_volume != nullptr) {
                    vertex_colors[vertex_idx * 3] = edge_colors[edge][0];
                    vertex_colors[vertex_idx * 3 + 1] = edge_colors[edge][1];
                    vertex_colors[vertex_idx * 3 + 2] = edge_colors[edge][2];
                }
                
                faces[face_idx * 3 + j] = vertex_idx;
            }
        }
    }
}

// Kernel to mark voxel blocks as dirty based on new points
__global__ void markDirtyVoxelBlocks(
    VoxelBlock* voxel_blocks,
    int num_blocks,
    const float3* new_points,
    int num_points,
    float influence_radius
) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;
    
    VoxelBlock& block = voxel_blocks[block_idx];
    if (!block.is_allocated) return;
    
    // Expand bounds by influence radius
    float3 expanded_min = make_float3(
        block.min_bound.x - influence_radius,
        block.min_bound.y - influence_radius,
        block.min_bound.z - influence_radius
    );
    float3 expanded_max = make_float3(
        block.max_bound.x + influence_radius,
        block.max_bound.y + influence_radius,
        block.max_bound.z + influence_radius
    );
    
    // Check if any point affects this block
    for (int p = 0; p < num_points; p++) {
        float3 point = new_points[p];
        
        if (point.x >= expanded_min.x && point.x <= expanded_max.x &&
            point.y >= expanded_min.y && point.y <= expanded_max.y &&
            point.z >= expanded_min.z && point.z <= expanded_max.z) {
            block.is_dirty = true;
            break;
        }
    }
}

} // namespace cuda

// IncrementalTSDFFusion implementation
class IncrementalTSDFFusion::Impl {
public:
    float voxel_size = 0.05f;
    float truncation_distance = 0.15f;
    int3 volume_size;
    float3 volume_bounds_min;
    float3 volume_bounds_max;
    
    // Voxel data
    thrust::device_vector<float> d_tsdf_volume;
    thrust::device_vector<float> d_weight_volume;
    thrust::device_vector<uint8_t> d_color_volume;
    
    // Block management
    thrust::device_vector<VoxelBlock> d_voxel_blocks;
    thrust::device_vector<int> d_dirty_blocks;
    int num_blocks;
    int3 block_grid_size;
    
    // Output buffers
    thrust::device_vector<float3> d_vertices;
    thrust::device_vector<float3> d_normals;
    thrust::device_vector<uint8_t> d_vertex_colors;
    thrust::device_vector<uint32_t> d_faces;
    thrust::device_vector<int> d_counters;  // vertex count, face count
    
    void initializeBlockGrid(float3 min_bounds, float3 max_bounds) {
        volume_bounds_min = min_bounds;
        volume_bounds_max = max_bounds;
        
        // Calculate volume size in voxels
        volume_size.x = (int)((max_bounds.x - min_bounds.x) / voxel_size) + 1;
        volume_size.y = (int)((max_bounds.y - min_bounds.y) / voxel_size) + 1;
        volume_size.z = (int)((max_bounds.z - min_bounds.z) / voxel_size) + 1;
        
        // Calculate block grid
        block_grid_size.x = (volume_size.x + VOXEL_BLOCK_SIZE - 1) / VOXEL_BLOCK_SIZE;
        block_grid_size.y = (volume_size.y + VOXEL_BLOCK_SIZE - 1) / VOXEL_BLOCK_SIZE;
        block_grid_size.z = (volume_size.z + VOXEL_BLOCK_SIZE - 1) / VOXEL_BLOCK_SIZE;
        
        num_blocks = block_grid_size.x * block_grid_size.y * block_grid_size.z;
        
        // Allocate block array
        std::vector<VoxelBlock> h_blocks(num_blocks);
        int allocated_blocks = 0;
        
        for (int z = 0; z < block_grid_size.z; z++) {
            for (int y = 0; y < block_grid_size.y; y++) {
                for (int x = 0; x < block_grid_size.x; x++) {
                    int idx = x + y * block_grid_size.x + z * block_grid_size.x * block_grid_size.y;
                    VoxelBlock& block = h_blocks[idx];
                    
                    block.block_coord = make_int3(x, y, z);
                    block.min_bound = make_float3(
                        min_bounds.x + x * VOXEL_BLOCK_SIZE * voxel_size,
                        min_bounds.y + y * VOXEL_BLOCK_SIZE * voxel_size,
                        min_bounds.z + z * VOXEL_BLOCK_SIZE * voxel_size
                    );
                    block.max_bound = make_float3(
                        block.min_bound.x + VOXEL_BLOCK_SIZE * voxel_size,
                        block.min_bound.y + VOXEL_BLOCK_SIZE * voxel_size,
                        block.min_bound.z + VOXEL_BLOCK_SIZE * voxel_size
                    );
                    
                    // Allocate blocks near the center initially
                    float3 center = make_float3(
                        (min_bounds.x + max_bounds.x) / 2,
                        (min_bounds.y + max_bounds.y) / 2,
                        (min_bounds.z + max_bounds.z) / 2
                    );
                    float3 block_center = make_float3(
                        (block.min_bound.x + block.max_bound.x) / 2,
                        (block.min_bound.y + block.max_bound.y) / 2,
                        (block.min_bound.z + block.max_bound.z) / 2
                    );
                    
                    float dist = length(block_center - center);
                    block.is_allocated = (dist < 5.0f);  // Allocate within 5m of center
                    block.is_dirty = false;
                    block.last_update = 0;
                    block.voxel_offset = allocated_blocks * VOXEL_BLOCK_SIZE * VOXEL_BLOCK_SIZE * VOXEL_BLOCK_SIZE;
                    
                    if (block.is_allocated) {
                        allocated_blocks++;
                    }
                }
            }
        }
        
        d_voxel_blocks = h_blocks;
        
        // Allocate voxel data only for allocated blocks
        size_t total_voxels = allocated_blocks * VOXEL_BLOCK_SIZE * VOXEL_BLOCK_SIZE * VOXEL_BLOCK_SIZE;
        d_tsdf_volume.resize(volume_size.x * volume_size.y * volume_size.z);
        d_weight_volume.resize(volume_size.x * volume_size.y * volume_size.z);
        d_color_volume.resize(volume_size.x * volume_size.y * volume_size.z * 3);
        
        // Initialize TSDF with truncation distance
        thrust::fill(d_tsdf_volume.begin(), d_tsdf_volume.end(), truncation_distance);
        thrust::fill(d_weight_volume.begin(), d_weight_volume.end(), 0.0f);
        thrust::fill(d_color_volume.begin(), d_color_volume.end(), 0);
        
        // Allocate output buffers
        size_t max_vertices = total_voxels * 3;
        d_vertices.resize(max_vertices);
        d_normals.resize(max_vertices);
        d_vertex_colors.resize(max_vertices * 3);
        d_faces.resize(max_vertices);
        d_counters.resize(2);
        
        d_dirty_blocks.resize(num_blocks);
        
        std::cout << "TSDF Volume initialized: " << volume_size.x << "x" << volume_size.y << "x" << volume_size.z 
                  << " voxels in " << allocated_blocks << " blocks" << std::endl;
    }
};

IncrementalTSDFFusion::IncrementalTSDFFusion() : pImpl(std::make_unique<Impl>()) {}
IncrementalTSDFFusion::~IncrementalTSDFFusion() = default;

void IncrementalTSDFFusion::setVoxelSize(float size) {
    pImpl->voxel_size = size;
    pImpl->truncation_distance = size * 3.0f;  // 3 voxels truncation
}

void IncrementalTSDFFusion::initialize(float3 min_bounds, float3 max_bounds) {
    pImpl->initializeBlockGrid(min_bounds, max_bounds);
}

void IncrementalTSDFFusion::integratePoints(
    const float3* d_points,
    const float3* d_normals,
    const uint8_t* d_colors,
    size_t num_points,
    const float* camera_pose,
    cudaStream_t stream
) {
    if (num_points == 0) return;
    
    // Mark dirty blocks based on new points
    dim3 block(256);
    dim3 grid((pImpl->num_blocks + 255) / 256);
    
    cuda::markDirtyVoxelBlocks<<<grid, block, 0, stream>>>(
        pImpl->d_voxel_blocks.data().get(),
        pImpl->num_blocks,
        d_points,
        num_points,
        pImpl->truncation_distance
    );
    
    // Get list of dirty blocks
    VoxelBlock* d_blocks = pImpl->d_voxel_blocks.data().get();
    int* d_dirty = pImpl->d_dirty_blocks.data().get();
    
    // Use CUB to compact dirty block indices
    size_t temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes,
                               thrust::make_counting_iterator(0),
                               thrust::make_transform_iterator(d_blocks,
                                   [] __device__ (const VoxelBlock& b) { return b.is_dirty ? 1 : 0; }),
                               d_dirty,
                               pImpl->d_counters.data().get(),
                               pImpl->num_blocks,
                               stream);
    
    thrust::device_vector<char> temp_storage(temp_storage_bytes);
    cub::DeviceSelect::Flagged(temp_storage.data().get(), temp_storage_bytes,
                               thrust::make_counting_iterator(0),
                               thrust::make_transform_iterator(d_blocks,
                                   [] __device__ (const VoxelBlock& b) { return b.is_dirty ? 1 : 0; }),
                               d_dirty,
                               pImpl->d_counters.data().get(),
                               pImpl->num_blocks,
                               stream);
    
    int num_dirty_blocks;
    cudaMemcpyAsync(&num_dirty_blocks, pImpl->d_counters.data().get(), sizeof(int), 
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    if (num_dirty_blocks == 0) return;
    
    // Integrate TSDF for dirty blocks
    dim3 integrate_grid(num_dirty_blocks);
    dim3 integrate_block(THREADS_PER_BLOCK);
    
    cuda::incrementalTSDFIntegration<<<integrate_grid, integrate_block, 0, stream>>>(
        pImpl->d_tsdf_volume.data().get(),
        pImpl->d_weight_volume.data().get(),
        d_colors ? pImpl->d_color_volume.data().get() : nullptr,
        d_blocks,
        d_dirty,
        num_dirty_blocks,
        d_points,
        d_normals,
        d_colors,
        num_points,
        pImpl->volume_size,
        pImpl->voxel_size,
        pImpl->truncation_distance,
        camera_pose
    );
}

void IncrementalTSDFFusion::extractMesh(
    MeshUpdate& update,
    cudaStream_t stream
) {
    // Reset counters
    thrust::fill(pImpl->d_counters.begin(), pImpl->d_counters.end(), 0);
    
    // Get dirty blocks for extraction
    VoxelBlock* d_blocks = pImpl->d_voxel_blocks.data().get();
    int* d_dirty = pImpl->d_dirty_blocks.data().get();
    
    int num_dirty_blocks;
    cudaMemcpyAsync(&num_dirty_blocks, pImpl->d_counters.data().get(), sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    if (num_dirty_blocks == 0) return;
    
    // Extract mesh from dirty blocks
    dim3 grid(num_dirty_blocks);
    dim3 block(THREADS_PER_BLOCK);
    size_t shared_mem = THREADS_PER_BLOCK * 3 * sizeof(float);
    
    cuda::enhancedMarchingCubes<<<grid, block, shared_mem, stream>>>(
        pImpl->d_tsdf_volume.data().get(),
        pImpl->d_weight_volume.data().get(),
        pImpl->d_color_volume.data().get(),
        d_blocks,
        d_dirty,
        num_dirty_blocks,
        pImpl->volume_size,
        pImpl->voxel_size,
        0.0f,  // iso_value
        pImpl->d_vertices.data().get(),
        pImpl->d_normals.data().get(),
        pImpl->d_vertex_colors.data().get(),
        pImpl->d_faces.data().get(),
        &pImpl->d_counters[0],
        &pImpl->d_counters[1],
        pImpl->d_vertices.size()
    );
    
    // Copy results to host
    int num_vertices, num_faces;
    cudaMemcpy(&num_vertices, &pImpl->d_counters[0], sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&num_faces, &pImpl->d_counters[1], sizeof(int), cudaMemcpyDeviceToHost);
    
    // Resize output arrays
    update.vertices.resize(num_vertices * 3);
    update.faces.resize(num_faces * 3);
    update.vertex_colors.resize(num_vertices * 3);
    
    // Copy vertex data
    std::vector<float3> temp_vertices(num_vertices);
    std::vector<float3> temp_normals(num_vertices);
    
    cudaMemcpy(temp_vertices.data(), pImpl->d_vertices.data().get(),
               num_vertices * sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_normals.data(), pImpl->d_normals.data().get(),
               num_vertices * sizeof(float3), cudaMemcpyDeviceToHost);
    
    // Convert to output format
    for (int i = 0; i < num_vertices; i++) {
        update.vertices[i * 3] = temp_vertices[i].x;
        update.vertices[i * 3 + 1] = temp_vertices[i].y;
        update.vertices[i * 3 + 2] = temp_vertices[i].z;
    }
    
    // Copy faces and colors
    cudaMemcpy(update.faces.data(), pImpl->d_faces.data().get(),
               num_faces * 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(update.vertex_colors.data(), pImpl->d_vertex_colors.data().get(),
               num_vertices * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    
    // Clear dirty flags
    cudaMemsetAsync(d_blocks, 0, pImpl->num_blocks * sizeof(VoxelBlock), stream);
}

size_t IncrementalTSDFFusion::getMemoryUsage() const {
    return pImpl->d_tsdf_volume.size() * sizeof(float) +
           pImpl->d_weight_volume.size() * sizeof(float) +
           pImpl->d_color_volume.size() * sizeof(uint8_t) +
           pImpl->d_voxel_blocks.size() * sizeof(VoxelBlock);
}

void IncrementalTSDFFusion::reset() {
    thrust::fill(pImpl->d_tsdf_volume.begin(), pImpl->d_tsdf_volume.end(), pImpl->truncation_distance);
    thrust::fill(pImpl->d_weight_volume.begin(), pImpl->d_weight_volume.end(), 0.0f);
    thrust::fill(pImpl->d_color_volume.begin(), pImpl->d_color_volume.end(), 0);
}

} // namespace mesh_service