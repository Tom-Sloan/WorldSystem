#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <atomic>

namespace mesh_service {

// GPU Octree node structure - aligned for efficient GPU access
struct alignas(32) OctreeNode {
    float3 center;          // Center of the node
    float half_size;        // Half the size of the node
    
    int children[8];        // Indices to child nodes (-1 if no child)
    int parent;             // Index to parent node (-1 if root)
    
    int point_start;        // Start index in point array
    int point_count;        // Number of points in this node
    
    int level;              // Depth level in the tree
    uint32_t morton_code;   // Morton code for spatial ordering
    
    // Flags for incremental updates
    uint32_t dirty_flag;    // 1 if node needs update, 0 otherwise
    uint32_t last_update;   // Timestamp of last update
};

// GPU-friendly point structure with spatial info
struct alignas(16) SpatialPoint {
    float3 position;
    uint32_t morton_code;
    float3 normal;
    uint32_t original_index;
    uint8_t color[4];  // RGBA
};

class GPUOctree {
public:
    GPUOctree(float scene_size = 10.0f, int max_depth = 8, int max_points_per_node = 64);
    ~GPUOctree();
    
    // Build octree from points
    void build(const float3* points, int num_points, cudaStream_t stream = 0);
    
    // Incremental update with new points
    void incrementalUpdate(const float3* new_points, int num_new_points, 
                          const float* camera_pose, cudaStream_t stream = 0);
    
    // Query operations
    void findNeighbors(const float3* query_points, int num_queries,
                      int* neighbor_indices, int* neighbor_counts,
                      float search_radius, int max_neighbors = 32,
                      cudaStream_t stream = 0);
    
    // Get dirty nodes for incremental mesh generation
    void getDirtyNodes(std::vector<int>& dirty_node_indices);
    void clearDirtyFlags(cudaStream_t stream = 0);
    
    // Access octree data
    OctreeNode* getDeviceNodes() { return d_nodes; }
    SpatialPoint* getDevicePoints() { return d_points; }
    int getNodeCount() const { return node_count; }
    int getPointCount() const { return point_count; }
    
    // Spatial deduplication
    bool checkRegionOverlap(const float* bbox, float overlap_threshold = 0.9f);
    void markRegionProcessed(const float* bbox);
    
private:
    // Device memory
    OctreeNode* d_nodes;
    SpatialPoint* d_points;
    int* d_node_pool;  // Pool for dynamic node allocation
    
    // Atomic counters
    int* d_node_counter;
    int* d_point_counter;
    
    // Host data
    std::vector<OctreeNode> h_nodes;
    int node_count;
    int point_count;
    int max_nodes;
    int max_points;
    
    // Parameters
    float scene_size;
    int max_depth;
    int max_points_per_node;
    
    // Memory pools
    void* memory_pool;
    size_t pool_size;
    size_t pool_offset;
    
    // Helper methods
    void allocateMemory();
    void freeMemory();
    void sortPointsByMortonCode(cudaStream_t stream);
    void buildTreeStructure(cudaStream_t stream);
    void updateNodeBounds(cudaStream_t stream);
};

// CUDA kernels for octree operations
namespace octree_kernels {

__global__ void computeMortonCodes(
    const float3* points,
    SpatialPoint* spatial_points,
    int num_points,
    float scene_min,
    float scene_size
);

__global__ void buildOctreeNodes(
    SpatialPoint* points,
    int num_points,
    OctreeNode* nodes,
    int* node_counter,
    int max_depth,
    int max_points_per_node
);

__global__ void markDirtyNodes(
    const float3* new_points,
    int num_new_points,
    OctreeNode* nodes,
    int num_nodes,
    float influence_radius
);

__global__ void findKNearestNeighbors(
    const float3* query_points,
    int num_queries,
    const SpatialPoint* octree_points,
    const OctreeNode* nodes,
    int num_nodes,
    int* neighbor_indices,
    int* neighbor_counts,
    float search_radius,
    int max_neighbors
);

__global__ void checkSpatialOverlap(
    const float* bbox,
    const OctreeNode* nodes,
    int num_nodes,
    int* overlap_count,
    float overlap_threshold
);

__device__ uint32_t expandBits(uint32_t v);
__device__ uint32_t morton3D(float x, float y, float z);
__device__ int findOctant(const float3& point, const float3& center);
__device__ bool intersectAABB(const float3& min1, const float3& max1,
                             const float3& min2, const float3& max2);

} // namespace octree_kernels

} // namespace mesh_service