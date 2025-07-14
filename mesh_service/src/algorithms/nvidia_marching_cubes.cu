#include "algorithms/nvidia_marching_cubes.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <iostream>

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

// Helper CUDA kernels
namespace cuda {

// Edge table for marching cubes - defines vertex pairs for each edge
__device__ const int edge_vertex[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},  // Bottom face edges
    {4, 5}, {5, 6}, {6, 7}, {7, 4},  // Top face edges
    {0, 4}, {1, 5}, {2, 6}, {3, 7}   // Vertical edges
};

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
    
    // Convert to 3D coordinates
    uint z = threadId / (dims.x * dims.y);
    uint y = (threadId % (dims.x * dims.y)) / dims.x;
    uint x = threadId % dims.x;
    
    // Skip boundary voxels
    if (x >= dims.x - 1 || y >= dims.y - 1 || z >= dims.z - 1) {
        voxel_verts[threadId] = 0;
        voxel_occupied[threadId] = 0;
        return;
    }
    
    // Sample 8 corners of the voxel
    float field[8];
    uint idx = x + y * dims.x + z * dims.x * dims.y;
    field[0] = tsdf[idx];
    field[1] = tsdf[idx + 1];
    field[2] = tsdf[idx + 1 + dims.x];
    field[3] = tsdf[idx + dims.x];
    field[4] = tsdf[idx + dims.x * dims.y];
    field[5] = tsdf[idx + 1 + dims.x * dims.y];
    field[6] = tsdf[idx + 1 + dims.x + dims.x * dims.y];
    field[7] = tsdf[idx + dims.x + dims.x * dims.y];
    
    // Calculate cube index
    uint cube_index = 0;
    for (int i = 0; i < 8; i++) {
        if (field[i] < iso_value) {
            cube_index |= (1 << i);
        }
    }
    
    // Use lookup table
    uint num_verts = d_numVertsTable[cube_index];
    voxel_verts[threadId] = num_verts;
    voxel_occupied[threadId] = (num_verts > 0) ? 1 : 0;
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
    // Step 1: Integrate points into TSDF
    tsdf_->integrate(d_points, d_normals, num_points, camera_pose, stream);
    
    // Get TSDF data
    float* d_tsdf = tsdf_->getTSDFVolume();
    int3 dims = tsdf_->getVolumeDims();
    float3 origin = tsdf_->getVolumeOrigin();
    float voxel_size = tsdf_->getVoxelSize();
    
    // Step 2: Classify voxels
    classifyVoxels(d_tsdf, dims, 
                   buffers_.d_voxel_verts_scan,
                   buffers_.d_voxel_occupied_scan,
                   stream);
    
    // Step 3: Scan to get total vertices and compaction offsets
    size_t temp_storage_bytes = 0;
    uint num_voxels = dims.x * dims.y * dims.z;
    
    // Get scan storage requirements
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes,
                                   buffers_.d_voxel_verts_scan,
                                   buffers_.d_voxel_verts_scan,
                                   num_voxels, stream);
    
    // Allocate temporary storage
    thrust::device_vector<uint8_t> d_temp_storage(temp_storage_bytes);
    
    // Perform scans
    cub::DeviceScan::ExclusiveSum(d_temp_storage.data().get(), temp_storage_bytes,
                                   buffers_.d_voxel_verts_scan,
                                   buffers_.d_voxel_verts_scan,
                                   num_voxels, stream);
    
    cub::DeviceScan::ExclusiveSum(d_temp_storage.data().get(), temp_storage_bytes,
                                   buffers_.d_voxel_occupied_scan,
                                   buffers_.d_voxel_occupied_scan,
                                   num_voxels, stream);
    
    // Get total counts by using thrust to get the last element + sum
    thrust::device_ptr<uint> d_verts_ptr(buffers_.d_voxel_verts_scan);
    thrust::device_ptr<uint> d_occupied_ptr(buffers_.d_voxel_occupied_scan);
    
    // The total is the last scan value + the last original value
    // Since we need the original values, let's save them before scan
    thrust::device_vector<uint> d_voxel_verts_orig(buffers_.d_voxel_verts_scan, 
                                                   buffers_.d_voxel_verts_scan + num_voxels);
    thrust::device_vector<uint> d_voxel_occupied_orig(buffers_.d_voxel_occupied_scan,
                                                      buffers_.d_voxel_occupied_scan + num_voxels);
    
    // Now do the exclusive scan (this modifies the buffers in place)
    thrust::exclusive_scan(thrust::cuda::par.on(stream),
                          d_voxel_verts_orig.begin(), d_voxel_verts_orig.end(),
                          d_verts_ptr);
    thrust::exclusive_scan(thrust::cuda::par.on(stream),
                          d_voxel_occupied_orig.begin(), d_voxel_occupied_orig.end(),
                          d_occupied_ptr);
    
    cudaStreamSynchronize(stream);
    
    // Get totals
    uint h_last_vert_scan = d_verts_ptr[num_voxels - 1];
    uint h_last_vert_orig = d_voxel_verts_orig[num_voxels - 1];
    uint h_last_occupied_scan = d_occupied_ptr[num_voxels - 1];
    uint h_last_occupied_orig = d_voxel_occupied_orig[num_voxels - 1];
    
    uint total_vertices = h_last_vert_scan + h_last_vert_orig;
    uint active_voxels = h_last_occupied_scan + h_last_occupied_orig;
    
    if (total_vertices == 0) {
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
    
    // Step 4: Compact active voxels
    compactVoxels(d_voxel_occupied_orig.data().get(),  // Need the original occupied flags, not scan
                  buffers_.d_voxel_occupied_scan,
                  buffers_.d_compressed_voxel_array,
                  num_voxels,
                  stream);
    
    // Step 5: Generate triangles
    generateTriangles(d_tsdf, dims, origin,
                      buffers_.d_compressed_voxel_array,
                      buffers_.d_voxel_verts_scan,
                      active_voxels,
                      buffers_.d_vertex_buffer,
                      buffers_.d_normal_buffer,
                      stream);
    
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