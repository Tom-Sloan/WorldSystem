#include "gpu_octree.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <iostream>
#include <cuda_runtime.h>

namespace mesh_service {

// Constants for octree construction
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;

// Move comparator here for CUDA compatibility
struct MortonComparator {
    __host__ __device__ bool operator()(const SpatialPoint& a, const SpatialPoint& b) const {
        return a.morton_code < b.morton_code;
    }
};

// Helper functions
namespace octree_kernels {
__device__ inline uint32_t expandBits(uint32_t v) {
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v <<  8)) & 0x0300F00F;
    v = (v | (v <<  4)) & 0x030C30C3;
    v = (v | (v <<  2)) & 0x09249249;
    return v;
}

__device__ inline uint32_t morton3D(float x, float y, float z) {
    x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    
    uint32_t xx = expandBits((uint32_t)x);
    uint32_t yy = expandBits((uint32_t)y);
    uint32_t zz = expandBits((uint32_t)z);
    
    return (zz << 2) | (yy << 1) | xx;
}

__device__ inline int findOctant(const float3& point, const float3& center) {
    int octant = 0;
    if (point.x > center.x) octant |= 1;
    if (point.y > center.y) octant |= 2;
    if (point.z > center.z) octant |= 4;
    return octant;
}

__device__ inline bool intersectAABB(
    const float3& min1, const float3& max1,
    const float3& min2, const float3& max2) {
    return (min1.x <= max2.x && max1.x >= min2.x) &&
           (min1.y <= max2.y && max1.y >= min2.y) &&
           (min1.z <= max2.z && max1.z >= min2.z);
}

// Kernel to compute Morton codes for points
__global__ void computeMortonCodes(
    const float3* points,
    SpatialPoint* spatial_points,
    int num_points,
    float scene_min,
    float scene_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    float3 p = points[idx];
    
    // Normalize to [0,1]
    float nx = (p.x - scene_min) / scene_size;
    float ny = (p.y - scene_min) / scene_size;
    float nz = (p.z - scene_min) / scene_size;
    
    // Compute Morton code
    uint32_t code = morton3D(nx, ny, nz);
    
    // Store spatial point
    spatial_points[idx].position = p;
    spatial_points[idx].morton_code = code;
    spatial_points[idx].original_index = idx;
    spatial_points[idx].normal = make_float3(0, 0, 0);  // Will be computed later
}

// Kernel to build octree nodes from sorted points
__global__ void buildOctreeNodes(
    SpatialPoint* points,
    int num_points,
    OctreeNode* nodes,
    int* node_counter,
    int max_depth,
    int max_points_per_node) {
    
    // This is a simplified version - in practice we'd use a more sophisticated
    // parallel construction algorithm
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Initialize root node
        nodes[0].center = make_float3(0, 0, 0);
        nodes[0].half_size = 5.0f;  // Assuming 10m scene
        nodes[0].parent = -1;
        nodes[0].point_start = 0;
        nodes[0].point_count = num_points;
        nodes[0].level = 0;
        nodes[0].morton_code = 0;
        nodes[0].dirty_flag = 1;
        nodes[0].last_update = 0;
        
        for (int i = 0; i < 8; i++) {
            nodes[0].children[i] = -1;
        }
        
        *node_counter = 1;
    }
}

// Kernel to mark dirty nodes based on new points
__global__ void markDirtyNodes(
    const float3* new_points,
    int num_new_points,
    OctreeNode* nodes,
    int num_nodes,
    float influence_radius) {
    
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= num_new_points) return;
    
    float3 point = new_points[point_idx];
    
    // Check each node to see if this point affects it
    for (int node_idx = 0; node_idx < num_nodes; node_idx++) {
        OctreeNode& node = nodes[node_idx];
        
        // Calculate distance from point to node center
        float3 diff = make_float3(
            point.x - node.center.x,
            point.y - node.center.y,
            point.z - node.center.z
        );
        
        float dist_sq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
        float node_radius = node.half_size * 1.732f;  // sqrt(3) for diagonal
        
        // Mark as dirty if point is within influence radius of node
        if (dist_sq <= (node_radius + influence_radius) * (node_radius + influence_radius)) {
            atomicOr(&node.dirty_flag, 1);
        }
    }
}

} // namespace octree_kernels

// Kernel to find k-nearest neighbors using octree
__global__ void findKNearestNeighbors(
    const float3* query_points,
    int num_queries,
    const SpatialPoint* octree_points,
    const OctreeNode* nodes,
    int num_nodes,
    int* neighbor_indices,
    int* neighbor_counts,
    float search_radius,
    int max_neighbors) {
    
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= num_queries) return;
    
    float3 query = query_points[query_idx];
    float radius_sq = search_radius * search_radius;
    
    // Output arrays for this query
    int* my_neighbors = neighbor_indices + query_idx * max_neighbors;
    int count = 0;
    
    // Simple brute force within leaf nodes that intersect search sphere
    // In practice, we'd traverse the octree more efficiently
    for (int node_idx = 0; node_idx < num_nodes; node_idx++) {
        const OctreeNode& node = nodes[node_idx];
        
        // Skip non-leaf nodes
        bool is_leaf = true;
        for (int i = 0; i < 8; i++) {
            if (node.children[i] != -1) {
                is_leaf = false;
                break;
            }
        }
        if (!is_leaf) continue;
        
        // Check if search sphere intersects node
        float3 closest = make_float3(
            fmaxf(node.center.x - node.half_size, fminf(query.x, node.center.x + node.half_size)),
            fmaxf(node.center.y - node.half_size, fminf(query.y, node.center.y + node.half_size)),
            fmaxf(node.center.z - node.half_size, fminf(query.z, node.center.z + node.half_size))
        );
        
        float3 diff = make_float3(query.x - closest.x, query.y - closest.y, query.z - closest.z);
        float dist_sq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
        
        if (dist_sq > radius_sq) continue;
        
        // Check points in this node
        for (int i = 0; i < node.point_count && count < max_neighbors; i++) {
            int point_idx = node.point_start + i;
            float3 p = octree_points[point_idx].position;
            
            float3 d = make_float3(query.x - p.x, query.y - p.y, query.z - p.z);
            float pd_sq = d.x * d.x + d.y * d.y + d.z * d.z;
            
            if (pd_sq <= radius_sq) {
                my_neighbors[count++] = octree_points[point_idx].original_index;
            }
        }
    }
    
    neighbor_counts[query_idx] = count;
}

// Kernel to check spatial overlap with existing regions
__global__ void octree_kernels::checkSpatialOverlap(
    const float* bbox,
    const OctreeNode* nodes,
    int num_nodes,
    int* overlap_count,
    float overlap_threshold) {
    
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_idx >= num_nodes) return;
    
    const OctreeNode& node = nodes[node_idx];
    
    // Skip nodes without points
    if (node.point_count == 0) return;
    
    // Calculate node bounds
    float3 node_min = make_float3(
        node.center.x - node.half_size,
        node.center.y - node.half_size,
        node.center.z - node.half_size
    );
    float3 node_max = make_float3(
        node.center.x + node.half_size,
        node.center.y + node.half_size,
        node.center.z + node.half_size
    );
    
    // Input bbox
    float3 bbox_min = make_float3(bbox[0], bbox[1], bbox[2]);
    float3 bbox_max = make_float3(bbox[3], bbox[4], bbox[5]);
    
    // Check intersection
    if (intersectAABB(node_min, node_max, bbox_min, bbox_max)) {
        // Calculate overlap volume
        float3 int_min = make_float3(
            fmaxf(node_min.x, bbox_min.x),
            fmaxf(node_min.y, bbox_min.y),
            fmaxf(node_min.z, bbox_min.z)
        );
        float3 int_max = make_float3(
            fminf(node_max.x, bbox_max.x),
            fminf(node_max.y, bbox_max.y),
            fminf(node_max.z, bbox_max.z)
        );
        
        float int_volume = (int_max.x - int_min.x) * 
                          (int_max.y - int_min.y) * 
                          (int_max.z - int_min.z);
        
        float bbox_volume = (bbox_max.x - bbox_min.x) * 
                           (bbox_max.y - bbox_min.y) * 
                           (bbox_max.z - bbox_min.z);
        
        if (int_volume / bbox_volume >= overlap_threshold) {
            atomicAdd(overlap_count, 1);
        }
    }
}

// GPUOctree implementation
GPUOctree::GPUOctree(float scene_size, int max_depth, int max_points_per_node)
    : scene_size(scene_size), max_depth(max_depth), 
      max_points_per_node(max_points_per_node),
      node_count(0), point_count(0) {
    
    // Calculate maximum possible nodes (complete octree)
    max_nodes = 0;
    for (int i = 0; i <= max_depth; i++) {
        max_nodes += (1 << (3 * i));  // 8^i nodes at level i
    }
    
    max_points = 10000000;  // 10M points max
    pool_size = 512 * 1024 * 1024;  // 512MB pool
    
    allocateMemory();
}

GPUOctree::~GPUOctree() {
    freeMemory();
}

void GPUOctree::allocateMemory() {
    // Allocate device memory
    cudaMalloc(&d_nodes, max_nodes * sizeof(OctreeNode));
    cudaMalloc(&d_points, max_points * sizeof(SpatialPoint));
    cudaMalloc(&d_node_pool, max_nodes * sizeof(int));
    cudaMalloc(&d_node_counter, sizeof(int));
    cudaMalloc(&d_point_counter, sizeof(int));
    cudaMalloc(&memory_pool, pool_size);
    
    // Initialize counters
    cudaMemset(d_node_counter, 0, sizeof(int));
    cudaMemset(d_point_counter, 0, sizeof(int));
    
    pool_offset = 0;
}

void GPUOctree::freeMemory() {
    if (d_nodes) cudaFree(d_nodes);
    if (d_points) cudaFree(d_points);
    if (d_node_pool) cudaFree(d_node_pool);
    if (d_node_counter) cudaFree(d_node_counter);
    if (d_point_counter) cudaFree(d_point_counter);
    if (memory_pool) cudaFree(memory_pool);
}

void GPUOctree::build(const float3* points, int num_points, cudaStream_t stream) {
    if (num_points == 0 || num_points > max_points) return;
    
    point_count = num_points;
    
    // Compute Morton codes
    dim3 block(BLOCK_SIZE);
    dim3 grid((num_points + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    float scene_min = -scene_size / 2.0f;
    octree_kernels::computeMortonCodes<<<grid, block, 0, stream>>>(
        points, d_points, num_points, scene_min, scene_size
    );
    
    // Sort points by Morton code
    sortPointsByMortonCode(stream);
    
    // Build tree structure
    buildTreeStructure(stream);
    
    // Update node bounds and point assignments
    updateNodeBounds(stream);
    
    cudaStreamSynchronize(stream);
}

void GPUOctree::sortPointsByMortonCode(cudaStream_t stream) {
    // Use Thrust to sort by Morton code
    thrust::device_ptr<SpatialPoint> d_points_ptr(d_points);
    
    thrust::sort(thrust::cuda::par.on(stream),
                 d_points_ptr, d_points_ptr + point_count,
                 MortonComparator());
}

void GPUOctree::buildTreeStructure(cudaStream_t stream) {
    // Simplified: just create root node for now
    // In practice, we'd build the full tree based on Morton codes
    octree_kernels::buildOctreeNodes<<<1, 1, 0, stream>>>(
        d_points, point_count, d_nodes, d_node_counter,
        max_depth, max_points_per_node
    );
    
    cudaStreamSynchronize(stream);
    cudaMemcpy(&node_count, d_node_counter, sizeof(int), cudaMemcpyDeviceToHost);
}

void GPUOctree::updateNodeBounds(cudaStream_t stream) {
    // Update bounds based on actual points
    // This would be done in parallel in practice
}

void GPUOctree::incrementalUpdate(const float3* new_points, int num_new_points,
                                 const float* camera_pose, cudaStream_t stream) {
    if (num_new_points == 0) return;
    
    // Mark nodes that are affected by new points
    dim3 block(BLOCK_SIZE);
    dim3 grid((num_new_points + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    float influence_radius = 0.1f;  // 10cm influence
    octree_kernels::markDirtyNodes<<<grid, block, 0, stream>>>(
        new_points, num_new_points, d_nodes, node_count, influence_radius
    );
    
    // Insert new points into octree
    // This would involve:
    // 1. Computing Morton codes for new points
    // 2. Finding appropriate leaf nodes
    // 3. Splitting nodes if they exceed max_points_per_node
    // 4. Updating the tree structure
    
    cudaStreamSynchronize(stream);
}

void GPUOctree::findNeighbors(const float3* query_points, int num_queries,
                             int* neighbor_indices, int* neighbor_counts,
                             float search_radius, int max_neighbors,
                             cudaStream_t stream) {
    if (num_queries == 0) return;
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((num_queries + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    octree_kernels::findKNearestNeighbors<<<grid, block, 0, stream>>>(
        query_points, num_queries, d_points, d_nodes, node_count,
        neighbor_indices, neighbor_counts, search_radius, max_neighbors
    );
}

void GPUOctree::getDirtyNodes(std::vector<int>& dirty_node_indices) {
    // Copy nodes to host and check dirty flags
    h_nodes.resize(node_count);
    cudaMemcpy(h_nodes.data(), d_nodes, node_count * sizeof(OctreeNode), 
               cudaMemcpyDeviceToHost);
    
    dirty_node_indices.clear();
    for (int i = 0; i < node_count; i++) {
        if (h_nodes[i].dirty_flag) {
            dirty_node_indices.push_back(i);
        }
    }
}

void GPUOctree::clearDirtyFlags(cudaStream_t stream) {
    // Clear all dirty flags
    cudaMemsetAsync(d_nodes, 0, node_count * sizeof(OctreeNode), stream);
}

bool GPUOctree::checkRegionOverlap(const float* bbox, float overlap_threshold) {
    float* d_bbox;
    int* d_overlap_count;
    
    cudaMalloc(&d_bbox, 6 * sizeof(float));
    cudaMalloc(&d_overlap_count, sizeof(int));
    
    cudaMemcpy(d_bbox, bbox, 6 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_overlap_count, 0, sizeof(int));
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((node_count + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    octree_kernels::checkSpatialOverlap<<<grid, block>>>(
        d_bbox, d_nodes, node_count, d_overlap_count, overlap_threshold
    );
    
    int overlap_count;
    cudaMemcpy(&overlap_count, d_overlap_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_bbox);
    cudaFree(d_overlap_count);
    
    return overlap_count > 0;
}

void GPUOctree::markRegionProcessed(const float* bbox) {
    // Mark all nodes within bbox as processed
    // This would update a processed flag in the nodes
}

} // namespace mesh_service