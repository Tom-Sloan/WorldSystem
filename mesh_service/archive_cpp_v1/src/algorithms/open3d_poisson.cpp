#include "algorithms/open3d_poisson.h"
#include "config/mesh_service_config.h"
#include "config/poisson_config.h"
#include "config/configuration_manager.h"
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <algorithm>

#ifdef HAS_OPEN3D
#include <open3d/Open3D.h>

namespace mesh_service {

class Open3DPoisson::Impl {
public:
    AlgorithmParams params;
    
    // Open3D Poisson parameters
    int depth = 8;
    float width = 0.0f;
    float scale = 1.1f;
    bool linear_fit = false;
    int n_threads = 0;  // 0 = automatic
    
    // Confidence filtering
    bool use_confidence = true;
    float confidence_threshold = 12.0f;
    
    size_t memory_usage = 0;
};

Open3DPoisson::Open3DPoisson() : pImpl(std::make_unique<Impl>()) {}
Open3DPoisson::~Open3DPoisson() = default;

bool Open3DPoisson::initialize(const AlgorithmParams& params) {
    pImpl->params = params;
    
    // Set Open3D parameters from config
    pImpl->depth = params.poisson.octree_depth;
    pImpl->width = CONFIG_FLOAT("MESH_POISSON_WIDTH", 0.0f);
    pImpl->scale = CONFIG_FLOAT("MESH_POISSON_SCALE", 1.1f);
    pImpl->linear_fit = CONFIG_BOOL("MESH_POISSON_LINEAR_FIT", false);
    pImpl->n_threads = CONFIG_INT("MESH_POISSON_THREADS", 0);
    
    // SLAM3R-specific settings
    pImpl->use_confidence = CONFIG_BOOL("MESH_POISSON_USE_CONFIDENCE", 
        config::PoissonConfig::SLAM3RIntegration::USE_CONFIDENCE_WEIGHTING);
    pImpl->confidence_threshold = CONFIG_FLOAT("MESH_POISSON_CONFIDENCE_THRESHOLD",
        config::PoissonConfig::SLAM3RIntegration::CONFIDENCE_THRESHOLD_L2W);
    
    return true;
}

bool Open3DPoisson::reconstruct(
    const float3* d_points,
    const float3* d_normals,
    size_t num_points,
    const float* camera_pose,  // Ignored - Poisson doesn't need camera poses!
    MeshUpdate& output,
    cudaStream_t stream
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Validate inputs
    if (!d_points || !d_normals || num_points == 0) {
        std::cerr << "[Open3D Poisson] Invalid input: null pointers or zero points" << std::endl;
        return false;
    }
    
    // Step 1: Copy GPU data to CPU
    // Note: float3 is a CUDA vector type with x, y, z members
    std::vector<float> h_points_flat(num_points * 3);
    std::vector<float> h_normals_flat(num_points * 3);
    
    cudaError_t err = cudaMemcpyAsync(h_points_flat.data(), d_points, 
                                      num_points * sizeof(float3), 
                                      cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        std::cerr << "[Open3D Poisson] Failed to copy points: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = cudaMemcpyAsync(h_normals_flat.data(), d_normals, 
                          num_points * sizeof(float3), 
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        std::cerr << "[Open3D Poisson] Failed to copy normals: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << "[Open3D Poisson] Failed to synchronize stream: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Step 2: Create Open3D point cloud
    auto pcd = std::make_shared<open3d::geometry::PointCloud>();
    pcd->points_.reserve(num_points);
    pcd->normals_.reserve(num_points);
    
    // Convert flat arrays to Open3D format
    // float3 is stored as 3 consecutive floats in memory
    for (size_t i = 0; i < num_points; ++i) {
        size_t idx = i * 3;
        pcd->points_.emplace_back(
            h_points_flat[idx], h_points_flat[idx + 1], h_points_flat[idx + 2]
        );
        pcd->normals_.emplace_back(
            h_normals_flat[idx], h_normals_flat[idx + 1], h_normals_flat[idx + 2]
        );
    }
    
    std::cout << "[Open3D Poisson] Processing " << pcd->points_.size() 
              << " points with octree depth " << pImpl->depth << std::endl;
    
    // Step 3: Run Poisson reconstruction using Open3D
    std::shared_ptr<open3d::geometry::TriangleMesh> mesh;
    std::vector<double> densities;
    
    try {
        std::tie(mesh, densities) = open3d::geometry::TriangleMesh::CreateFromPointCloudPoisson(
            *pcd, 
            pImpl->depth,
            pImpl->width,
            pImpl->scale,
            pImpl->linear_fit,
            pImpl->n_threads
        );
    } catch (const std::exception& e) {
        std::cerr << "[Open3D Poisson] Reconstruction failed: " << e.what() << std::endl;
        return false;
    }
    
    // Step 4: Filter by density to remove spurious geometry
    if (!densities.empty() && CONFIG_BOOL("MESH_POISSON_DENSITY_FILTER", true)) {
        // Calculate density threshold
        std::vector<double> sorted_densities = densities;
        std::sort(sorted_densities.begin(), sorted_densities.end());
        
        float percentile = CONFIG_FLOAT("MESH_POISSON_DENSITY_PERCENTILE",
            config::AlgorithmConfig::DEFAULT_POISSON_DENSITY_PERCENTILE);
        size_t threshold_idx = static_cast<size_t>(sorted_densities.size() * percentile);
        double density_threshold = sorted_densities[threshold_idx];
        
        // Remove low-density vertices
        std::vector<bool> vertices_to_remove(mesh->vertices_.size(), false);
        for (size_t i = 0; i < densities.size(); ++i) {
            if (densities[i] < density_threshold) {
                vertices_to_remove[i] = true;
            }
        }
        
        mesh->RemoveVerticesByMask(vertices_to_remove);
    }
    
    // Step 5: Ensure mesh has vertex normals
    if (!mesh->HasVertexNormals()) {
        mesh->ComputeVertexNormals();
    }
    
    // Step 6: Convert to MeshUpdate format (CPU vectors)
    size_t num_vertices = mesh->vertices_.size();
    size_t num_faces = mesh->triangles_.size();
    
    // Update memory usage estimate
    pImpl->memory_usage = num_vertices * sizeof(float) * 3 + num_faces * sizeof(uint32_t) * 3;
    
    // Clear and resize output vectors
    output.vertices.clear();
    output.faces.clear();
    output.vertex_colors.clear();
    
    output.vertices.reserve(num_vertices * 3);
    output.faces.reserve(num_faces * 3);
    
    // Copy vertices as flat array (x,y,z triplets)
    for (size_t i = 0; i < num_vertices; ++i) {
        output.vertices.push_back(static_cast<float>(mesh->vertices_[i].x()));
        output.vertices.push_back(static_cast<float>(mesh->vertices_[i].y()));
        output.vertices.push_back(static_cast<float>(mesh->vertices_[i].z()));
    }
    
    // Copy faces as flat array (triangle indices)
    for (size_t i = 0; i < num_faces; ++i) {
        output.faces.push_back(static_cast<uint32_t>(mesh->triangles_[i](0)));
        output.faces.push_back(static_cast<uint32_t>(mesh->triangles_[i](1)));
        output.faces.push_back(static_cast<uint32_t>(mesh->triangles_[i](2)));
    }
    
    // If mesh has colors, copy them
    if (mesh->HasVertexColors()) {
        output.vertex_colors.reserve(num_vertices * 3);
        for (size_t i = 0; i < num_vertices; ++i) {
            output.vertex_colors.push_back(static_cast<uint8_t>(mesh->vertex_colors_[i].x() * 255));
            output.vertex_colors.push_back(static_cast<uint8_t>(mesh->vertex_colors_[i].y() * 255));
            output.vertex_colors.push_back(static_cast<uint8_t>(mesh->vertex_colors_[i].z() * 255));
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "[Open3D Poisson] Generated mesh with " << num_vertices 
              << " vertices, " << num_faces << " faces in " << duration << "ms" << std::endl;
    
    return true;
}

size_t Open3DPoisson::getMemoryUsage() const {
    return pImpl->memory_usage;
}

void Open3DPoisson::reset() {
    pImpl->memory_usage = 0;
}

} // namespace mesh_service

#else // !HAS_OPEN3D

// Stub implementation when Open3D is not available
namespace mesh_service {

class Open3DPoisson::Impl {
public:
    AlgorithmParams params;
    size_t memory_usage = 0;
};

Open3DPoisson::Open3DPoisson() : pImpl(std::make_unique<Impl>()) {}
Open3DPoisson::~Open3DPoisson() = default;

bool Open3DPoisson::initialize(const AlgorithmParams& params) {
    std::cerr << "[Open3D Poisson] Open3D not available - this algorithm cannot be used" << std::endl;
    return false;
}

bool Open3DPoisson::reconstruct(
    const float3* d_points,
    const float3* d_normals,
    size_t num_points,
    const float* camera_pose,
    MeshUpdate& output,
    cudaStream_t stream
) {
    std::cerr << "[Open3D Poisson] Open3D not available - reconstruction failed" << std::endl;
    return false;
}

size_t Open3DPoisson::getMemoryUsage() const {
    return 0;
}

void Open3DPoisson::reset() {
}

} // namespace mesh_service

#endif // HAS_OPEN3D