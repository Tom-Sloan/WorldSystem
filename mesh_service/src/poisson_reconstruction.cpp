#include "poisson_reconstruction.h"
#include "mesh_generator.h"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Poisson_reconstruction_function.h>
#include <CGAL/Point_with_normal_3.h>
#include <CGAL/property_map.h>
#include <CGAL/compute_average_spacing.h>
#include <CGAL/jet_estimate_normals.h>
#include <CGAL/mst_orient_normals.h>
#include <CGAL/poisson_surface_reconstruction.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <chrono>

namespace mesh_service {

// CGAL types
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::FT FT;
typedef Kernel::Point_3 Point;
typedef Kernel::Vector_3 Vector;
typedef std::pair<Point, Vector> Pwn;
typedef CGAL::Surface_mesh<Point> Surface_mesh;
typedef CGAL::Polyhedron_3<Kernel> Polyhedron;

// Implementation for PoissonReconstruction
class PoissonReconstruction::Impl {
public:
    Parameters params;
    
    void reconstructCGAL(
        const std::vector<Pwn>& points,
        Surface_mesh& output_mesh
    ) {
        if (points.empty()) return;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Poisson reconstruction
        double average_spacing = CGAL::compute_average_spacing<CGAL::Sequential_tag>(
            points, 6,
            CGAL::parameters::point_map(CGAL::First_of_pair_property_map<Pwn>())
        );
        
        // Reconstruct surface
        CGAL::poisson_surface_reconstruction_delaunay(
            points.begin(), points.end(),
            CGAL::First_of_pair_property_map<Pwn>(),
            CGAL::Second_of_pair_property_map<Pwn>(),
            output_mesh,
            average_spacing
        );
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        std::cout << "Poisson reconstruction: " << output_mesh.number_of_vertices() 
                  << " vertices, " << output_mesh.number_of_faces() 
                  << " faces in " << duration << "ms" << std::endl;
    }
};

PoissonReconstruction::PoissonReconstruction() : pImpl(std::make_unique<Impl>()) {}
PoissonReconstruction::~PoissonReconstruction() = default;

void PoissonReconstruction::setParameters(const Parameters& params) {
    pImpl->params = params;
}

void PoissonReconstruction::reconstruct(
    const float* points,
    const float* normals,
    size_t num_points,
    MeshUpdate& output
) {
    // Convert to CGAL format
    std::vector<Pwn> cgal_points;
    cgal_points.reserve(num_points);
    
    for (size_t i = 0; i < num_points; ++i) {
        Point p(points[i*3], points[i*3+1], points[i*3+2]);
        Vector n(normals[i*3], normals[i*3+1], normals[i*3+2]);
        cgal_points.emplace_back(p, n);
    }
    
    // Reconstruct
    Surface_mesh mesh;
    pImpl->reconstructCGAL(cgal_points, mesh);
    
    // Convert back to our format
    output.vertices.clear();
    output.faces.clear();
    
    // Map vertices
    std::unordered_map<Surface_mesh::Vertex_index, uint32_t> vertex_map;
    uint32_t vertex_id = 0;
    
    for (auto v : mesh.vertices()) {
        const Point& p = mesh.point(v);
        output.vertices.push_back(p.x());
        output.vertices.push_back(p.y());
        output.vertices.push_back(p.z());
        vertex_map[v] = vertex_id++;
    }
    
    // Convert faces
    for (auto f : mesh.faces()) {
        auto h = mesh.halfedge(f);
        auto v0 = mesh.source(h);
        auto v1 = mesh.target(h);
        auto v2 = mesh.target(mesh.next(h));
        
        output.faces.push_back(vertex_map[v0]);
        output.faces.push_back(vertex_map[v1]);
        output.faces.push_back(vertex_map[v2]);
    }
}

void PoissonReconstruction::reconstructGPU(
    float3* d_points,
    float3* d_normals,
    size_t num_points,
    cudaStream_t stream,
    MeshUpdate& output
) {
    // Copy points from GPU to CPU
    std::vector<float> h_points(num_points * 3);
    std::vector<float> h_normals(num_points * 3);
    
    cudaMemcpyAsync(h_points.data(), d_points, num_points * sizeof(float3),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_normals.data(), d_normals, num_points * sizeof(float3),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Run CPU reconstruction
    reconstruct(h_points.data(), h_normals.data(), num_points, output);
}

// Implementation for IncrementalPoissonReconstruction
class IncrementalPoissonReconstruction::Impl {
public:
    PoissonReconstruction::Parameters base_params;
    std::vector<Block> blocks;
    std::vector<float> all_points;
    std::vector<float> all_normals;
    std::unordered_map<uint64_t, std::vector<size_t>> block_points;
    float scene_size = 10.0f;
    int grid_resolution = 8;
    
    uint64_t getBlockIndex(const float* point) {
        int x = static_cast<int>((point[0] + scene_size/2) / scene_size * grid_resolution);
        int y = static_cast<int>((point[1] + scene_size/2) / scene_size * grid_resolution);
        int z = static_cast<int>((point[2] + scene_size/2) / scene_size * grid_resolution);
        
        x = std::max(0, std::min(x, grid_resolution - 1));
        y = std::max(0, std::min(y, grid_resolution - 1));
        z = std::max(0, std::min(z, grid_resolution - 1));
        
        return x + y * grid_resolution + z * grid_resolution * grid_resolution;
    }
    
    void initializeBlocks() {
        blocks.resize(grid_resolution * grid_resolution * grid_resolution);
        float block_size = scene_size / grid_resolution;
        
        for (int z = 0; z < grid_resolution; ++z) {
            for (int y = 0; y < grid_resolution; ++y) {
                for (int x = 0; x < grid_resolution; ++x) {
                    size_t idx = x + y * grid_resolution + z * grid_resolution * grid_resolution;
                    blocks[idx].center[0] = (x + 0.5f) * block_size - scene_size/2;
                    blocks[idx].center[1] = (y + 0.5f) * block_size - scene_size/2;
                    blocks[idx].center[2] = (z + 0.5f) * block_size - scene_size/2;
                    blocks[idx].size = block_size;
                }
            }
        }
    }
};

IncrementalPoissonReconstruction::IncrementalPoissonReconstruction() 
    : pImpl(std::make_unique<Impl>()) {}
IncrementalPoissonReconstruction::~IncrementalPoissonReconstruction() = default;

void IncrementalPoissonReconstruction::initialize(float scene_size, int grid_resolution) {
    pImpl->scene_size = scene_size;
    pImpl->grid_resolution = grid_resolution;
    pImpl->initializeBlocks();
}

void IncrementalPoissonReconstruction::addPoints(
    const float* points,
    const float* normals,
    size_t num_points,
    uint64_t timestamp
) {
    size_t base_idx = pImpl->all_points.size() / 3;
    
    // Append new points
    pImpl->all_points.insert(pImpl->all_points.end(), 
                            points, points + num_points * 3);
    pImpl->all_normals.insert(pImpl->all_normals.end(),
                             normals, normals + num_points * 3);
    
    // Assign points to blocks
    for (size_t i = 0; i < num_points; ++i) {
        const float* p = points + i * 3;
        uint64_t block_idx = pImpl->getBlockIndex(p);
        
        if (block_idx < pImpl->blocks.size()) {
            pImpl->blocks[block_idx].point_indices.push_back(base_idx + i);
            pImpl->blocks[block_idx].is_dirty = true;
            pImpl->blocks[block_idx].last_update = timestamp;
        }
    }
}

void IncrementalPoissonReconstruction::updateDirtyBlocks(MeshUpdate& output) {
    output.vertices.clear();
    output.faces.clear();
    
    PoissonReconstruction poisson;
    poisson.setParameters(pImpl->base_params);
    
    // Process each dirty block
    for (auto& block : pImpl->blocks) {
        if (!block.is_dirty || block.point_indices.empty()) continue;
        
        // Gather points for this block (including overlap from neighbors)
        std::vector<float> block_points;
        std::vector<float> block_normals;
        
        for (size_t idx : block.point_indices) {
            block_points.push_back(pImpl->all_points[idx * 3]);
            block_points.push_back(pImpl->all_points[idx * 3 + 1]);
            block_points.push_back(pImpl->all_points[idx * 3 + 2]);
            
            block_normals.push_back(pImpl->all_normals[idx * 3]);
            block_normals.push_back(pImpl->all_normals[idx * 3 + 1]);
            block_normals.push_back(pImpl->all_normals[idx * 3 + 2]);
        }
        
        // Reconstruct this block
        MeshUpdate block_mesh;
        poisson.reconstruct(block_points.data(), block_normals.data(),
                          block_points.size() / 3, block_mesh);
        
        // Merge into output (with vertex offset)
        size_t vertex_offset = output.vertices.size() / 3;
        output.vertices.insert(output.vertices.end(),
                             block_mesh.vertices.begin(),
                             block_mesh.vertices.end());
        
        for (uint32_t face_idx : block_mesh.faces) {
            output.faces.push_back(face_idx + vertex_offset);
        }
        
        block.is_dirty = false;
    }
}

void IncrementalPoissonReconstruction::forceFullUpdate(MeshUpdate& output) {
    if (pImpl->all_points.empty()) return;
    
    PoissonReconstruction poisson;
    poisson.setParameters(pImpl->base_params);
    poisson.reconstruct(pImpl->all_points.data(), pImpl->all_normals.data(),
                       pImpl->all_points.size() / 3, output);
}

} // namespace mesh_service