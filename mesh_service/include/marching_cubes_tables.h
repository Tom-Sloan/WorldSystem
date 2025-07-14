#pragma once

#include <cuda_runtime.h>

namespace mesh_service {
namespace cuda {

// Declare the marching cubes lookup tables as extern CUDA constants
// These are defined in marching_cubes_tables.cu
extern __constant__ int edge_table[256];
extern __constant__ int tri_table[256][16];

// Edge vertex mapping for marching cubes
// Maps edge index (0-11) to the two vertices that form the edge
constexpr int edge_vertices[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},  // Bottom face edges
    {4, 5}, {5, 6}, {6, 7}, {7, 4},  // Top face edges
    {0, 4}, {1, 5}, {2, 6}, {3, 7}   // Vertical edges
};

// Vertex offsets for a voxel cube
// Maps vertex index (0-7) to (x,y,z) offsets
constexpr int vertex_offsets[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},  // Bottom face
    {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}   // Top face
};

} // namespace cuda
} // namespace mesh_service