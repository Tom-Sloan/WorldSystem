#include "algorithms/nvidia_marching_cubes.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cfloat>

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(error) << std::endl; \
        throw std::runtime_error("CUDA error"); \
    } \
} while(0)

namespace mesh_service {

// Import NVIDIA tables directly
extern "C" {
#include "../../external/nvidia_mc/tables.h"
}

// Device copies of the tables
__constant__ uint d_numVertsTable[256];
__constant__ int d_triTable[256][16];

// Functors for thrust operations (must be defined at namespace level)
struct CheckModified {
    float truncation;
    CheckModified(float t) : truncation(t) {}
    __device__ bool operator()(float val) const {
        return fabsf(val - truncation) > 0.001f;
    }
};

struct CheckWeighted {
    __device__ bool operator()(float val) const {
        return val > 0.0f;
    }
};

// Helper function to save PLY point cloud
void savePLYPointCloud(const char* filename, const std::vector<float3>& points) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        std::cerr << "Failed to open " << filename << " for writing" << std::endl;
        return;
    }
    
    fprintf(fp, "ply\n");
    fprintf(fp, "format ascii 1.0\n");
    fprintf(fp, "element vertex %zu\n", points.size());
    fprintf(fp, "property float x\n");
    fprintf(fp, "property float y\n");
    fprintf(fp, "property float z\n");
    fprintf(fp, "end_header\n");
    
    for (const auto& p : points) {
        fprintf(fp, "%.6f %.6f %.6f\n", p.x, p.y, p.z);
    }
    
    fclose(fp);
}

// Helper function to save TSDF slice
void saveTSDFSlice(const char* filename, const float* tsdf_volume, 
                   int3 dims, int z_slice, float voxel_size, float3 origin) {
    FILE* fp = fopen(filename, "w");
    if (!fp) return;
    
    fprintf(fp, "# TSDF slice at Z=%d (%.2f meters)\n", z_slice, origin.z + z_slice * voxel_size);
    fprintf(fp, "# Format: X Y TSDF_value Weight\n");
    
    for (int y = 0; y < dims.y; y++) {
        for (int x = 0; x < dims.x; x++) {
            int idx = x + y * dims.x + z_slice * dims.x * dims.y;
            float world_x = origin.x + x * voxel_size;
            float world_y = origin.y + y * voxel_size;
            fprintf(fp, "%.3f %.3f %.6f\n", world_x, world_y, tsdf_volume[idx]);
        }
        fprintf(fp, "\n"); // Empty line between rows for gnuplot
    }
    
    fclose(fp);
}

// Helper CUDA kernels
namespace cuda {

// Edge table for marching cubes - defines vertex pairs for each edge
__device__ const int edge_vertex[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},  // Bottom face edges
    {4, 5}, {5, 6}, {6, 7}, {7, 4},  // Top face edges
    {0, 4}, {1, 5}, {2, 6}, {3, 7}   // Vertical edges
};

// Find voxels that have been updated (weight > 0)
__global__ void findActiveVoxelsKernel(
    const float* weight_volume,
    uint* active_voxels,
    uint* active_count,
    uint num_voxels,
    uint buffer_size  // Add buffer size parameter
) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_voxels) return;
    
    if (weight_volume[idx] > 0.0f) {
        uint pos = atomicAdd(active_count, 1);
        
        // Debug: First few active voxels
        if (pos < 10) {
            printf("[FIND ACTIVE DEBUG] Found active voxel at idx %u, weight=%.3f, pos=%u\n", 
                   idx, weight_volume[idx], pos);
        }
        
        // Critical bounds check to prevent buffer overflow
        if (pos < buffer_size) {
            active_voxels[pos] = idx;
        } else if (pos == buffer_size) {
            // Log once when we hit the limit
            printf("[FIND ACTIVE WARNING] Active voxel buffer full at %u entries!\n", buffer_size);
        }
    }
}

// Conversion kernels
__global__ void convertFloat4ToFloat3(
    const float4* src_verts,
    const float4* src_normals,
    float3* dst_verts,
    float3* dst_normals,
    uint num_vertices
) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vertices) return;
    
    float4 v = src_verts[idx];
    float4 n = src_normals[idx];
    
    dst_verts[idx] = make_float3(v.x, v.y, v.z);
    dst_normals[idx] = make_float3(n.x, n.y, n.z);
}

__global__ void generateFaceIndices(
    int3* faces,
    uint num_faces
) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_faces) return;
    
    // Simple triangle soup indexing
    faces[idx] = make_int3(idx * 3, idx * 3 + 1, idx * 3 + 2);
}

// NVIDIA-style voxel classification
__global__ void classifyVoxelKernel(
    const float* tsdf,
    int3 dims,
    float iso_value,
    uint* voxel_verts,
    uint* voxel_occupied
) {
    uint blockId = blockIdx.y * gridDim.x + blockIdx.x;
    uint threadId = blockId * blockDim.x + threadIdx.x;
    uint num_voxels = dims.x * dims.y * dims.z;
    
    if (threadId >= num_voxels) return;
    
    // This function is currently not used - keeping for potential future use
    // The active voxel version below is used instead
}

// Classify only active voxels
__global__ void classifyActiveVoxelsKernel(
    const float* tsdf,
    const uint* active_voxels,
    uint num_active,
    int3 dims,
    float iso_value,
    uint* voxel_verts,
    uint* voxel_occupied
) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_active) return;
    
    uint voxel_idx = active_voxels[idx];
    
    // Convert to 3D coordinates
    uint z = voxel_idx / (dims.x * dims.y);
    uint y = (voxel_idx % (dims.x * dims.y)) / dims.x;
    uint x = voxel_idx % dims.x;
    
    // Debug first few active voxels
    if (idx < 5) {
        printf("[MC CLASSIFY ACTIVE] Active voxel %u (global idx %u): coords=[%u,%u,%u]\n", 
               idx, voxel_idx, x, y, z);
    }
    
    // Skip boundary voxels
    if (x >= dims.x - 1 || y >= dims.y - 1 || z >= dims.z - 1) {
        voxel_verts[idx] = 0;
        voxel_occupied[idx] = 0;
        return;
    }
    
    // Sample 8 corners of the voxel
    float field[8];
    field[0] = tsdf[voxel_idx];
    field[1] = tsdf[voxel_idx + 1];
    field[2] = tsdf[voxel_idx + 1 + dims.x];
    field[3] = tsdf[voxel_idx + dims.x];
    field[4] = tsdf[voxel_idx + dims.x * dims.y];
    field[5] = tsdf[voxel_idx + 1 + dims.x * dims.y];
    field[6] = tsdf[voxel_idx + 1 + dims.x + dims.x * dims.y];
    field[7] = tsdf[voxel_idx + dims.x + dims.x * dims.y];
    
    // Debug TSDF values for first few active voxels
    if (idx < 5) {
        printf("[MC CLASSIFY ACTIVE] Voxel %u TSDF values: [%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f]\n",
               idx, field[0], field[1], field[2], field[3], 
               field[4], field[5], field[6], field[7]);
    }
    
    // Calculate cube index
    uint cube_index = 0;
    for (int i = 0; i < 8; i++) {
        if (field[i] < iso_value) {
            cube_index |= (1 << i);
        }
    }
    
    // Use lookup table
    uint num_verts = d_numVertsTable[cube_index];
    voxel_verts[idx] = num_verts;
    voxel_occupied[idx] = (num_verts > 0) ? 1 : 0;
    
    if (idx < 5) {
        printf("[MC CLASSIFY ACTIVE] Voxel %u: cube_index=%u, num_verts=%u\n", 
               idx, cube_index, num_verts);
    }
}

// Compact voxels
__global__ void compactVoxelsKernel(
    const uint* voxel_occupied,
    const uint* voxel_occupied_scan,
    uint* compressed_voxels,
    uint num_voxels
) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_voxels) return;
    
    if (voxel_occupied[idx]) {
        compressed_voxels[voxel_occupied_scan[idx]] = idx;
    }
}

// Simplified triangle generation
__global__ void generateTrianglesKernel(
    const float* tsdf,
    int3 dims,
    float3 origin,
    float voxel_size,
    float iso_value,
    const uint* compressed_voxels,
    const uint* num_verts_scan,
    uint num_active_voxels,
    float4* vertices,
    float4* normals,
    uint* face_count
) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_active_voxels) return;
    
    uint voxel_idx = compressed_voxels[idx];
    uint vertex_offset = num_verts_scan[voxel_idx];
    
    // Convert voxel index to 3D coordinates
    uint z = voxel_idx / (dims.x * dims.y);
    uint y = (voxel_idx % (dims.x * dims.y)) / dims.x;
    uint x = voxel_idx % dims.x;
    
    // Calculate world position of voxel corner
    float3 pos = make_float3(
        origin.x + x * voxel_size,
        origin.y + y * voxel_size,
        origin.z + z * voxel_size
    );
    
    // Sample field values at 8 corners
    float field[8];
    uint base_idx = x + y * dims.x + z * dims.x * dims.y;
    field[0] = tsdf[base_idx];
    field[1] = tsdf[base_idx + 1];
    field[2] = tsdf[base_idx + 1 + dims.x];
    field[3] = tsdf[base_idx + dims.x];
    field[4] = tsdf[base_idx + dims.x * dims.y];
    field[5] = tsdf[base_idx + 1 + dims.x * dims.y];
    field[6] = tsdf[base_idx + 1 + dims.x + dims.x * dims.y];
    field[7] = tsdf[base_idx + dims.x + dims.x * dims.y];
    
    // Calculate cube index
    uint cube_index = 0;
    for (int i = 0; i < 8; i++) {
        if (field[i] < iso_value) {
            cube_index |= (1 << i);
        }
    }
    
    // Get vertex positions for the 8 corners
    float3 vertex_pos[8];
    vertex_pos[0] = pos;
    vertex_pos[1] = make_float3(pos.x + voxel_size, pos.y, pos.z);
    vertex_pos[2] = make_float3(pos.x + voxel_size, pos.y + voxel_size, pos.z);
    vertex_pos[3] = make_float3(pos.x, pos.y + voxel_size, pos.z);
    vertex_pos[4] = make_float3(pos.x, pos.y, pos.z + voxel_size);
    vertex_pos[5] = make_float3(pos.x + voxel_size, pos.y, pos.z + voxel_size);
    vertex_pos[6] = make_float3(pos.x + voxel_size, pos.y + voxel_size, pos.z + voxel_size);
    vertex_pos[7] = make_float3(pos.x, pos.y + voxel_size, pos.z + voxel_size);
    
    // Use the device constant edge_vertex table defined above
    
    // Generate triangles
    uint num_triangles = 0;
    for (int i = 0; d_triTable[cube_index][i] != -1; i += 3) {
        for (int j = 0; j < 3; j++) {
            int edge = d_triTable[cube_index][i + j];
            int v0 = edge_vertex[edge][0];
            int v1 = edge_vertex[edge][1];
            
            // Linear interpolation
            float t = (iso_value - field[v0]) / (field[v1] - field[v0]);
            t = fmaxf(0.0f, fminf(1.0f, t));
            
            float3 p = make_float3(
                vertex_pos[v0].x + t * (vertex_pos[v1].x - vertex_pos[v0].x),
                vertex_pos[v0].y + t * (vertex_pos[v1].y - vertex_pos[v0].y),
                vertex_pos[v0].z + t * (vertex_pos[v1].z - vertex_pos[v0].z)
            );
            
            // Store vertex
            vertices[vertex_offset + num_triangles * 3 + j] = make_float4(p.x, p.y, p.z, 1.0f);
            
            // Compute normal using TSDF gradient (finite differences)
            float3 gradient;
            
            // Sample TSDF at neighboring voxels for gradient computation
            float dx_pos = (x + 1 < dims.x) ? tsdf[voxel_idx + 1] : field[0];
            float dx_neg = (x > 0) ? tsdf[voxel_idx - 1] : field[0];
            float dy_pos = (y + 1 < dims.y) ? tsdf[voxel_idx + dims.x] : field[0];
            float dy_neg = (y > 0) ? tsdf[voxel_idx - dims.x] : field[0];
            float dz_pos = (z + 1 < dims.z) ? tsdf[voxel_idx + dims.x * dims.y] : field[0];
            float dz_neg = (z > 0) ? tsdf[voxel_idx - dims.x * dims.y] : field[0];
            
            // Central differences for gradient
            gradient.x = (dx_pos - dx_neg) * 0.5f;
            gradient.y = (dy_pos - dy_neg) * 0.5f;
            gradient.z = (dz_pos - dz_neg) * 0.5f;
            
            // Normalize the gradient to get the normal
            float len = sqrtf(gradient.x * gradient.x + gradient.y * gradient.y + gradient.z * gradient.z);
            if (len > 0.0001f) {
                gradient.x /= len;
                gradient.y /= len;
                gradient.z /= len;
            } else {
                gradient = make_float3(0.0f, 0.0f, 1.0f); // Default if gradient is too small
            }
            
            normals[vertex_offset + num_triangles * 3 + j] = make_float4(gradient.x, gradient.y, gradient.z, 0.0f);
        }
        num_triangles++;
    }
    
    // Update face count
    if (idx == 0) {
        atomicAdd(face_count, num_triangles);
    }
}

} // namespace cuda

NvidiaMarchingCubes::NvidiaMarchingCubes() {
    tsdf_ = std::make_unique<SimpleTSDF>();
}

NvidiaMarchingCubes::~NvidiaMarchingCubes() {
    freeBuffers();
}

bool NvidiaMarchingCubes::initialize(const AlgorithmParams& params) {
    params_ = params;
    
    std::cout << "[NVIDIA MC INIT] Initializing with bounds:" << std::endl;
    std::cout << "  Volume min: [" << params.volume_min.x << ", " 
              << params.volume_min.y << ", " << params.volume_min.z << "]" << std::endl;
    std::cout << "  Volume max: [" << params.volume_max.x << ", " 
              << params.volume_max.y << ", " << params.volume_max.z << "]" << std::endl;
    std::cout << "  Voxel size: " << params.voxel_size << "m" << std::endl;
    std::cout << "  Truncation distance: " << params.marching_cubes.truncation_distance << "m" << std::endl;
    
    // Initialize TSDF volume
    tsdf_->initialize(
        params.volume_min,
        params.volume_max,
        params.voxel_size
    );
    tsdf_->setTruncationDistance(params.marching_cubes.truncation_distance);
    
    // Upload marching cubes tables to GPU
    uploadTables();
    
    // Pre-allocate buffers based on volume size
    allocateBuffers(tsdf_->getVolumeDims());
    
    std::cout << "[NVIDIA MC INIT] NvidiaMarchingCubes initialized successfully" << std::endl;
    return true;
}

void NvidiaMarchingCubes::uploadTables() {
    // Copy tables to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_numVertsTable, numVertsTable, 256 * sizeof(uint)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_triTable, triTable, 256 * 16 * sizeof(int)));
}

void NvidiaMarchingCubes::allocateBuffers(const int3& volume_dims) {
    size_t num_voxels = volume_dims.x * volume_dims.y * volume_dims.z;
    
    // Allocate voxel classification buffers
    CUDA_CHECK(cudaMalloc(&buffers_.d_voxel_verts_scan, num_voxels * sizeof(uint)));
    CUDA_CHECK(cudaMalloc(&buffers_.d_voxel_occupied_scan, num_voxels * sizeof(uint)));
    CUDA_CHECK(cudaMalloc(&buffers_.d_compressed_voxel_array, num_voxels * sizeof(uint)));
    
    // Allocate output buffers (generous size)
    buffers_.allocated_vertices = params_.marching_cubes.max_vertices;
    CUDA_CHECK(cudaMalloc(&buffers_.d_vertex_buffer, buffers_.allocated_vertices * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&buffers_.d_normal_buffer, buffers_.allocated_vertices * sizeof(float4)));
    
    buffers_.allocated_voxels = num_voxels;
}

void NvidiaMarchingCubes::freeBuffers() {
    if (buffers_.d_voxel_verts_scan) cudaFree(buffers_.d_voxel_verts_scan);
    if (buffers_.d_voxel_occupied_scan) cudaFree(buffers_.d_voxel_occupied_scan);
    if (buffers_.d_compressed_voxel_array) cudaFree(buffers_.d_compressed_voxel_array);
    if (buffers_.d_vertex_buffer) cudaFree(buffers_.d_vertex_buffer);
    if (buffers_.d_normal_buffer) cudaFree(buffers_.d_normal_buffer);
}

bool NvidiaMarchingCubes::reconstruct(
    const float3* d_points,
    const float3* d_normals,
    size_t num_points,
    const float* camera_pose,
    MeshUpdate& output,
    cudaStream_t stream
) {
    std::cout << "[MC DEBUG] reconstruct() called with " << num_points << " points" << std::endl;
    
    // Step 1: Integrate points into TSDF
    std::cout << "[MC DEBUG] Integrating points into TSDF..." << std::endl;
    tsdf_->integrate(d_points, d_normals, num_points, camera_pose, stream);
    cudaStreamSynchronize(stream);
    std::cout << "[MC DEBUG] TSDF integration complete" << std::endl;
    
    // Periodic debugging: Save point cloud every 10 frames
    static int frame_count = 0;
    frame_count++;
    
    if (frame_count % 5 == 0) {  // Save every 5 frames for better debugging
        std::cout << "[DEBUG SAVE] Saving debug data for frame " << frame_count << std::endl;
        
        // Save point cloud to PLY file
        std::vector<float3> h_points(num_points);
        cudaMemcpy(h_points.data(), d_points, num_points * sizeof(float3), cudaMemcpyDeviceToHost);
        
        char filename[256];
        snprintf(filename, sizeof(filename), "/debug_output/pointcloud_%06d.ply", frame_count);
        savePLYPointCloud(filename, h_points);
        std::cout << "[DEBUG SAVE] Saved point cloud to " << filename 
                  << " (" << num_points << " points)" << std::endl;
        
        // Sample first few points for orientation check
        std::cout << "[DEBUG ORIENTATION] First 5 points:" << std::endl;
        for (int i = 0; i < std::min(5, (int)num_points); i++) {
            std::cout << "  Point " << i << ": [" << h_points[i].x 
                      << ", " << h_points[i].y << ", " << h_points[i].z << "]" << std::endl;
        }
        
        // Calculate point cloud bounds
        float3 min_pt = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        float3 max_pt = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        for (const auto& p : h_points) {
            min_pt.x = fminf(min_pt.x, p.x);
            min_pt.y = fminf(min_pt.y, p.y);
            min_pt.z = fminf(min_pt.z, p.z);
            max_pt.x = fmaxf(max_pt.x, p.x);
            max_pt.y = fmaxf(max_pt.y, p.y);
            max_pt.z = fmaxf(max_pt.z, p.z);
        }
        std::cout << "[DEBUG BOUNDS] Point cloud bounds: min=[" << min_pt.x << ", " << min_pt.y 
                  << ", " << min_pt.z << "], max=[" << max_pt.x << ", " << max_pt.y 
                  << ", " << max_pt.z << "]" << std::endl;
        
        // Get camera position from pose matrix
        float cam_x = camera_pose[12];
        float cam_y = camera_pose[13];
        float cam_z = camera_pose[14];
        std::cout << "[DEBUG CAMERA] Camera position: [" << cam_x 
                  << ", " << cam_y << ", " << cam_z << "]" << std::endl;
        
        // Save TSDF slice at camera height
        int3 dims = tsdf_->getVolumeDims();
        float3 origin = tsdf_->getVolumeOrigin();
        float voxel_size = tsdf_->getVoxelSize();
        int z_slice = (int)((cam_z - origin.z) / voxel_size);
        z_slice = std::max(0, std::min(dims.z - 1, z_slice));
        
        // Get TSDF data for slice
        size_t slice_size = dims.x * dims.y * dims.z;
        std::vector<float> h_tsdf(slice_size);
        cudaMemcpy(h_tsdf.data(), tsdf_->getTSDFVolume(), slice_size * sizeof(float), cudaMemcpyDeviceToHost);
        
        snprintf(filename, sizeof(filename), "/debug_output/tsdf_slice_%06d.txt", frame_count);
        saveTSDFSlice(filename, h_tsdf.data(), dims, z_slice, voxel_size, origin);
        std::cout << "[DEBUG SAVE] Saved TSDF slice at Z=" << z_slice 
                  << " to " << filename << std::endl;
    }
    
    // Debug: Check if TSDF has been modified
    float* d_tsdf_debug = tsdf_->getTSDFVolume();
    float* d_weight_debug = tsdf_->getWeightVolume();
    int3 dims_debug = tsdf_->getVolumeDims();
    size_t total_voxels = dims_debug.x * dims_debug.y * dims_debug.z;
    
    // Count non-default TSDF values
    thrust::device_ptr<float> d_tsdf_ptr(d_tsdf_debug);
    thrust::device_ptr<float> d_weight_ptr(d_weight_debug);
    
    // Count non-default values using thrust
    CheckModified check_mod(params_.marching_cubes.truncation_distance);
    auto count_modified = thrust::count_if(d_tsdf_ptr, d_tsdf_ptr + total_voxels, check_mod);
    auto count_weighted = thrust::count_if(d_weight_ptr, d_weight_ptr + total_voxels, CheckWeighted());
    
    std::cout << "[MC DEBUG] TSDF volume stats:" << std::endl;
    std::cout << "  Total voxels: " << total_voxels << std::endl;
    std::cout << "  Modified voxels: " << count_modified << std::endl;
    std::cout << "  Weighted voxels: " << count_weighted << std::endl;
    
    // Get TSDF data
    float* d_tsdf = tsdf_->getTSDFVolume();
    int3 dims = tsdf_->getVolumeDims();
    float3 origin = tsdf_->getVolumeOrigin();
    float voxel_size = tsdf_->getVoxelSize();
    
    std::cout << "[MC DEBUG] TSDF volume info:" << std::endl;
    std::cout << "  Dims: " << dims.x << "x" << dims.y << "x" << dims.z << std::endl;
    std::cout << "  Origin: [" << origin.x << ", " << origin.y << ", " << origin.z << "]" << std::endl;
    std::cout << "  Voxel size: " << voxel_size << std::endl;
    
    // Step 2: Find active voxels (those with weight > 0)
    float* d_weights = tsdf_->getWeightVolume();
    
    // Allocate with safety margin to prevent overflow
    size_t active_buffer_size = std::min(total_voxels, std::max(size_t(count_weighted * 2), size_t(1000)));
    thrust::device_vector<uint> d_active_voxels(active_buffer_size); 
    thrust::device_vector<uint> d_active_count(1);
    
    // CRITICAL FIX: Explicitly set count to 0 to prevent garbage values
    cudaMemset(d_active_count.data().get(), 0, sizeof(uint));
    CUDA_CHECK(cudaGetLastError());
    
    std::cout << "[MC DEBUG FIX] Initialized d_active_count to 0" << std::endl;
    std::cout << "[MC DEBUG FIX] Allocated buffer for " << active_buffer_size << " active voxels (safety margin from " << count_weighted << " weighted)" << std::endl;
    
    dim3 find_block(256);
    dim3 find_grid((total_voxels + find_block.x - 1) / find_block.x);
    
    std::cout << "[MC DEBUG] Finding active voxels from " << count_weighted << " weighted voxels..." << std::endl;
    cuda::findActiveVoxelsKernel<<<find_grid, find_block, 0, stream>>>(
        d_weights,
        d_active_voxels.data().get(),
        d_active_count.data().get(),
        total_voxels,
        active_buffer_size  // Pass buffer size to prevent overflow
    );
    CUDA_CHECK(cudaGetLastError());
    cudaStreamSynchronize(stream);
    
    // Get active count
    uint h_active_count;
    cudaMemcpy(&h_active_count, d_active_count.data().get(), sizeof(uint), cudaMemcpyDeviceToHost);
    std::cout << "[MC DEBUG] Found " << h_active_count << " active voxels to process" << std::endl;
    
    // CRITICAL DEBUG: Verify active count is reasonable
    if (h_active_count > active_buffer_size) {
        std::cout << "[MC ERROR] Active count (" << h_active_count 
                  << ") exceeds buffer size (" << active_buffer_size << ")!" << std::endl;
        std::cout << "[MC ERROR] This indicates memory corruption or uninitialized memory" << std::endl;
        h_active_count = active_buffer_size;  // Clamp to prevent crash
    }
    
    if (h_active_count == 0) {
        std::cout << "[MC WARNING] No active voxels found! Check TSDF integration." << std::endl;
        std::cout << "[MC DEBUG] Total voxels: " << total_voxels 
                  << ", Weighted voxels: " << count_weighted << std::endl;
        output.vertices.clear();
        output.faces.clear();
        return true;
    }
    
    // Additional debug: Sample some weights to verify they're being set
    if (count_weighted > 0) {
        std::vector<float> sample_weights(std::min(size_t(10), total_voxels));
        cudaMemcpy(sample_weights.data(), d_weights, sample_weights.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "[MC DEBUG] First few weight values: ";
        for (size_t i = 0; i < sample_weights.size(); i++) {
            std::cout << sample_weights[i] << " ";
        }
        std::cout << std::endl;
    }
    
    // Step 3: Classify only active voxels
    std::cout << "[MC DEBUG] Classifying " << h_active_count << " active voxels..." << std::endl;
    
    // Allocate smaller arrays for active voxel processing
    thrust::device_vector<uint> d_active_voxel_verts(h_active_count);
    thrust::device_vector<uint> d_active_voxel_occupied(h_active_count);
    
    dim3 classify_block(256);
    dim3 classify_grid((h_active_count + classify_block.x - 1) / classify_block.x);
    
    cuda::classifyActiveVoxelsKernel<<<classify_grid, classify_block, 0, stream>>>(
        d_tsdf,
        d_active_voxels.data().get(),
        h_active_count,
        dims,
        params_.marching_cubes.iso_value,
        d_active_voxel_verts.data().get(),
        d_active_voxel_occupied.data().get()
    );
    cudaStreamSynchronize(stream);
    std::cout << "[MC DEBUG] Active voxel classification complete" << std::endl;
    
    // Step 4: Scan to get total vertices and compaction offsets for active voxels
    size_t temp_storage_bytes = 0;
    
    // Get scan storage requirements for active voxels
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes,
                                   d_active_voxel_verts.data().get(),
                                   d_active_voxel_verts.data().get(),
                                   h_active_count, stream);
    
    // Allocate temporary storage
    thrust::device_vector<uint8_t> d_temp_storage(temp_storage_bytes);
    
    // Create scan arrays for active voxels
    thrust::device_vector<uint> d_active_voxel_verts_scan(h_active_count);
    thrust::device_vector<uint> d_active_voxel_occupied_scan(h_active_count);
    
    // Perform scans on active voxels only
    cub::DeviceScan::ExclusiveSum(d_temp_storage.data().get(), temp_storage_bytes,
                                   d_active_voxel_verts.data().get(),
                                   d_active_voxel_verts_scan.data().get(),
                                   h_active_count, stream);
    
    cub::DeviceScan::ExclusiveSum(d_temp_storage.data().get(), temp_storage_bytes,
                                   d_active_voxel_occupied.data().get(),
                                   d_active_voxel_occupied_scan.data().get(),
                                   h_active_count, stream);
    
    cudaStreamSynchronize(stream);
    
    // Get totals from active voxel arrays
    uint h_last_vert_scan, h_last_vert_orig;
    uint h_last_occupied_scan, h_last_occupied_orig;
    
    if (h_active_count > 0) {
        // Copy last elements
        cudaMemcpy(&h_last_vert_scan, 
                   d_active_voxel_verts_scan.data().get() + h_active_count - 1, 
                   sizeof(uint), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_last_vert_orig, 
                   d_active_voxel_verts.data().get() + h_active_count - 1, 
                   sizeof(uint), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_last_occupied_scan, 
                   d_active_voxel_occupied_scan.data().get() + h_active_count - 1, 
                   sizeof(uint), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_last_occupied_orig, 
                   d_active_voxel_occupied.data().get() + h_active_count - 1, 
                   sizeof(uint), cudaMemcpyDeviceToHost);
    } else {
        h_last_vert_scan = h_last_vert_orig = 0;
        h_last_occupied_scan = h_last_occupied_orig = 0;
    }
    
    uint total_vertices = h_last_vert_scan + h_last_vert_orig;
    uint active_voxels_with_triangles = h_last_occupied_scan + h_last_occupied_orig;
    
    std::cout << "[MC DEBUG] Scan results:" << std::endl;
    std::cout << "  Active voxels processed: " << h_active_count << std::endl;
    std::cout << "  Active voxels with triangles: " << active_voxels_with_triangles << std::endl;
    std::cout << "  Total vertices to generate: " << total_vertices << std::endl;
    std::cout << "  Last vert scan: " << h_last_vert_scan << ", orig: " << h_last_vert_orig << std::endl;
    std::cout << "  Last occupied scan: " << h_last_occupied_scan << ", orig: " << h_last_occupied_orig << std::endl;
    
    if (total_vertices == 0) {
        std::cout << "[MC WARNING] No vertices to generate! Returning empty mesh." << std::endl;
        output.vertices.clear();
        output.faces.clear();
        return true;
    }
    
    // Check if we have enough buffer space
    if (total_vertices > buffers_.allocated_vertices) {
        std::cerr << "Warning: Total vertices (" << total_vertices 
                  << ") exceeds allocated buffer (" << buffers_.allocated_vertices << ")" << std::endl;
        total_vertices = buffers_.allocated_vertices;  // Clamp to allocated size
    }
    
    // Step 5: Compact active voxels (only those that have triangles)
    thrust::device_vector<uint> d_compressed_active_voxels(active_voxels_with_triangles);
    
    dim3 compact_block(256);
    dim3 compact_grid((h_active_count + compact_block.x - 1) / compact_block.x);
    
    cuda::compactVoxelsKernel<<<compact_grid, compact_block, 0, stream>>>(
        d_active_voxel_occupied.data().get(),
        d_active_voxel_occupied_scan.data().get(),
        d_compressed_active_voxels.data().get(),
        h_active_count
    );
    
    // Step 6: Generate triangles for active voxels only
    // Need to modify generateTriangles to work with active voxel indices
    // For now, let's see if we get any vertices at all
    std::cout << "[MC DEBUG] Would generate triangles for " << active_voxels_with_triangles 
              << " active voxels with " << total_vertices << " vertices" << std::endl;
    
    // Step 6: Copy results to output
    uint num_triangles = total_vertices / 3;
    output.vertices.resize(total_vertices * 3);
    output.faces.resize(num_triangles * 3);
    
    // Convert float4 to float3 format
    thrust::device_vector<float3> d_vertices_temp(total_vertices);
    thrust::device_vector<float3> d_normals_temp(total_vertices);
    
    dim3 block(256);
    dim3 grid((total_vertices + block.x - 1) / block.x);
    
    cuda::convertFloat4ToFloat3<<<grid, block, 0, stream>>>(
        buffers_.d_vertex_buffer,
        buffers_.d_normal_buffer,
        d_vertices_temp.data().get(),
        d_normals_temp.data().get(),
        total_vertices
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Copy to host
    std::vector<float3> h_vertices(total_vertices);
    thrust::copy(d_vertices_temp.begin(), d_vertices_temp.end(), h_vertices.begin());
    
    // Convert to output format
    for (size_t i = 0; i < total_vertices; i++) {
        output.vertices[i * 3] = h_vertices[i].x;
        output.vertices[i * 3 + 1] = h_vertices[i].y;
        output.vertices[i * 3 + 2] = h_vertices[i].z;
    }
    
    // Generate face indices
    thrust::device_vector<int3> d_faces(num_triangles);
    grid = dim3((num_triangles + block.x - 1) / block.x);
    
    cuda::generateFaceIndices<<<grid, block, 0, stream>>>(
        d_faces.data().get(),
        num_triangles
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Copy faces to host
    std::vector<int3> h_faces(num_triangles);
    thrust::copy(d_faces.begin(), d_faces.end(), h_faces.begin());
    
    for (size_t i = 0; i < num_triangles; i++) {
        output.faces[i * 3] = static_cast<uint32_t>(h_faces[i].x);
        output.faces[i * 3 + 1] = static_cast<uint32_t>(h_faces[i].y);
        output.faces[i * 3 + 2] = static_cast<uint32_t>(h_faces[i].z);
    }
    
    // Final synchronization to ensure all operations complete
    cudaStreamSynchronize(stream);
    
    // Save generated mesh periodically for debugging
    if (frame_count % 5 == 0 && output.vertices.size() > 0) {  // Match point cloud saving frequency
        char mesh_filename[256];
        snprintf(mesh_filename, sizeof(mesh_filename), "/debug_output/mesh_%06d.ply", frame_count);
        
        FILE* fp = fopen(mesh_filename, "w");
        if (fp) {
            size_t num_verts = output.vertices.size() / 3;
            size_t num_faces = output.faces.size() / 3;
            
            fprintf(fp, "ply\n");
            fprintf(fp, "format ascii 1.0\n");
            fprintf(fp, "element vertex %zu\n", num_verts);
            fprintf(fp, "property float x\n");
            fprintf(fp, "property float y\n");
            fprintf(fp, "property float z\n");
            fprintf(fp, "element face %zu\n", num_faces);
            fprintf(fp, "property list uchar int vertex_indices\n");
            fprintf(fp, "end_header\n");
            
            // Write vertices
            for (size_t i = 0; i < output.vertices.size(); i += 3) {
                fprintf(fp, "%.6f %.6f %.6f\n", 
                        output.vertices[i], output.vertices[i+1], output.vertices[i+2]);
            }
            
            // Write faces
            for (size_t i = 0; i < output.faces.size(); i += 3) {
                fprintf(fp, "3 %u %u %u\n", 
                        output.faces[i], output.faces[i+1], output.faces[i+2]);
            }
            
            fclose(fp);
            std::cout << "[DEBUG SAVE] Saved mesh to " << mesh_filename 
                      << " (" << num_verts << " vertices, " << num_faces << " faces)" << std::endl;
        }
    }
    
    return true;
}

void NvidiaMarchingCubes::classifyVoxels(
    const float* d_tsdf,
    const int3& dims,
    uint* d_voxel_verts,
    uint* d_voxel_occupied,
    cudaStream_t stream
) {
    uint num_voxels = dims.x * dims.y * dims.z;
    dim3 grid((num_voxels + 127) / 128);
    dim3 block(128);
    
    cuda::classifyVoxelKernel<<<grid, block, 0, stream>>>(
        d_tsdf, dims,
        params_.marching_cubes.iso_value,
        d_voxel_verts,
        d_voxel_occupied
    );
    CUDA_CHECK(cudaGetLastError());
}

void NvidiaMarchingCubes::compactVoxels(
    const uint* d_voxel_occupied,
    const uint* d_voxel_occupied_scan,
    uint* d_compressed_voxels,
    uint num_voxels,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((num_voxels + block.x - 1) / block.x);
    
    cuda::compactVoxelsKernel<<<grid, block, 0, stream>>>(
        d_voxel_occupied,
        d_voxel_occupied_scan,
        d_compressed_voxels,
        num_voxels
    );
    CUDA_CHECK(cudaGetLastError());
}

void NvidiaMarchingCubes::generateTriangles(
    const float* d_tsdf,
    const int3& dims,
    const float3& origin,
    const uint* d_compressed_voxels,
    const uint* d_num_verts_scan,
    uint num_active_voxels,
    float4* d_vertices,
    float4* d_normals,
    cudaStream_t stream
) {
    thrust::device_vector<uint> d_face_count(1, 0);
    
    dim3 block(64);
    dim3 grid((num_active_voxels + block.x - 1) / block.x);
    
    cuda::generateTrianglesKernel<<<grid, block, 0, stream>>>(
        d_tsdf, dims, origin, tsdf_->getVoxelSize(),
        params_.marching_cubes.iso_value,
        d_compressed_voxels,
        d_num_verts_scan,
        num_active_voxels,
        d_vertices,
        d_normals,
        d_face_count.data().get()
    );
    CUDA_CHECK(cudaGetLastError());
}

size_t NvidiaMarchingCubes::getMemoryUsage() const {
    size_t usage = 0;
    usage += tsdf_->getMemoryUsage();
    usage += buffers_.allocated_voxels * sizeof(uint) * 3; // voxel buffers
    usage += buffers_.allocated_vertices * sizeof(float4) * 2; // vertex/normal buffers
    return usage;
}

void NvidiaMarchingCubes::reset() {
    tsdf_->reset();
}

} // namespace mesh_service