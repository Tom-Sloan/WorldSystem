#include "rerun_publisher.h"
#include <rerun.hpp>
#include <rerun/archetypes/mesh3d.hpp>
#include <rerun/archetypes/transform3d.hpp>
#include <rerun/archetypes/clear.hpp>
#include <rerun/components/position3d.hpp>
#include <rerun/components/vector3d.hpp>
#include <rerun/components/color.hpp>
// #include <rerun/components/material.hpp> // Not available in this version
#include <rerun/datatypes/mat4x4.hpp>
#include <iostream>
#include <chrono>

namespace mesh_service {

class RerunPublisher::Impl {
public:
    std::string app_id;
    std::string address;
    bool enabled;
    bool connected = false;
    std::unique_ptr<rerun::RecordingStream> stream;
    
    Impl(const std::string& id, const std::string& addr, bool en)
        : app_id(id), address(addr), enabled(en) {}
    
    bool connect() {
        if (!enabled) {
            std::cout << "Rerun publishing disabled" << std::endl;
            return true;
        }
        
        try {
            // Create recording stream
            stream = std::make_unique<rerun::RecordingStream>(app_id);
            
            // Connect to existing viewer at specified address
            // Use connect_grpc to connect to existing viewer on host
            // Format: rerun+http://address:port/proxy
            std::string grpc_url = "rerun+http://" + address + "/proxy";
            auto result = stream->connect_grpc(grpc_url.c_str());
            if (result.is_err()) {
                std::cerr << "Failed to connect to Rerun at " << grpc_url << std::endl;
                std::cerr << "Make sure Rerun viewer is running on host with: rerun --port 9876" << std::endl;
                return false;
            }
            
            connected = true;
            std::cout << "Connected to Rerun viewer at " << address << std::endl;
            
            // Log initial setup with the new API
            stream->set_time_timestamp("time", std::chrono::steady_clock::now());
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Error connecting to Rerun: " << e.what() << std::endl;
            return false;
        }
    }
    
    void disconnect() {
        if (stream) {
            stream.reset();
        }
        connected = false;
    }
    
    void publishMesh(const MeshUpdate& update, const std::string& entity_path) {
        if (!enabled || !connected || !stream) return;
        
        try {
            // Set current time
            stream->set_time_timestamp("time", std::chrono::steady_clock::now());
            
            // Convert vertices to positions
            std::vector<rerun::Position3D> positions;
            positions.reserve(update.vertices.size() / 3);
            
            for (size_t i = 0; i < update.vertices.size(); i += 3) {
                positions.push_back({update.vertices[i], 
                                   update.vertices[i+1], 
                                   update.vertices[i+2]});
            }
            
            // Create mesh archetype
            auto mesh = rerun::archetypes::Mesh3D(positions);
            
            // Add triangle indices if available
            if (!update.faces.empty()) {
                // Convert face indices to rerun format
                std::vector<std::array<uint32_t, 3>> triangles;
                triangles.reserve(update.faces.size() / 3);
                
                for (size_t i = 0; i < update.faces.size(); i += 3) {
                    triangles.push_back({update.faces[i], 
                                       update.faces[i+1], 
                                       update.faces[i+2]});
                }
                
                mesh = std::move(mesh).with_triangle_indices(triangles);
            }
            
            // Add vertex colors if available
            if (!update.vertex_colors.empty()) {
                std::vector<rerun::Color> colors;
                colors.reserve(update.vertex_colors.size() / 3);
                
                for (size_t i = 0; i < update.vertex_colors.size(); i += 3) {
                    colors.push_back(rerun::Color(update.vertex_colors[i], 
                                                 update.vertex_colors[i+1], 
                                                 update.vertex_colors[i+2]));
                }
                
                mesh = std::move(mesh).with_vertex_colors(colors);
            }
            
            // Note: Material/albedo_factor component may not be available in all versions
            // The mesh will use default shading
            
            // Log the mesh
            stream->log(entity_path, mesh);
            
        } catch (const std::exception& e) {
            std::cerr << "Error publishing mesh to Rerun: " << e.what() << std::endl;
        }
    }
    
    void publishColoredMesh(const std::vector<float>& vertices,
                           const std::vector<uint32_t>& faces,
                           const std::vector<uint8_t>& colors,
                           const std::string& entity_path) {
        if (!enabled || !connected || !stream) return;
        
        try {
            // Set current time
            stream->set_time_timestamp("time", std::chrono::steady_clock::now());
            
            // Convert vertices to positions
            std::vector<rerun::Position3D> positions;
            positions.reserve(vertices.size() / 3);
            
            for (size_t i = 0; i < vertices.size(); i += 3) {
                positions.push_back({vertices[i], vertices[i+1], vertices[i+2]});
            }
            
            // Create mesh with positions
            auto mesh = rerun::archetypes::Mesh3D(positions);
            
            // Add triangle indices
            if (!faces.empty()) {
                std::vector<std::array<uint32_t, 3>> triangles;
                triangles.reserve(faces.size() / 3);
                
                for (size_t i = 0; i < faces.size(); i += 3) {
                    triangles.push_back({faces[i], faces[i+1], faces[i+2]});
                }
                
                mesh = std::move(mesh).with_triangle_indices(triangles);
            }
            
            // Add vertex colors
            if (!colors.empty()) {
                std::vector<rerun::Color> vertex_colors;
                vertex_colors.reserve(colors.size() / 3);
                
                for (size_t i = 0; i < colors.size(); i += 3) {
                    vertex_colors.push_back(rerun::Color(colors[i], colors[i+1], colors[i+2]));
                }
                
                mesh = std::move(mesh).with_vertex_colors(vertex_colors);
            }
            
            // Material component not available in all versions
            // The mesh will use default shading
            
            // Log the mesh
            stream->log(entity_path, mesh);
            
        } catch (const std::exception& e) {
            std::cerr << "Error publishing colored mesh to Rerun: " << e.what() << std::endl;
        }
    }
    
    void logCameraPose(const float pose[16], const std::string& entity_path) {
        if (!enabled || !connected || !stream) return;
        
        try {
            // Extract translation and rotation from 4x4 pose matrix
            rerun::components::Translation3D translation{
                pose[12], pose[13], pose[14]  // Last column is translation
            };
            
            // For now, use translation only (full matrix transform may not be supported)
            auto transform3d = rerun::archetypes::Transform3D(translation);
            
            // Log the transform
            stream->log(entity_path, transform3d);
            
        } catch (const std::exception& e) {
            std::cerr << "Error logging camera pose to Rerun: " << e.what() << std::endl;
        }
    }
    
    void clearEntity(const std::string& entity_path) {
        if (!enabled || !connected || !stream) return;
        
        try {
            stream->log(entity_path, rerun::archetypes::Clear());
        } catch (const std::exception& e) {
            std::cerr << "Error clearing entity in Rerun: " << e.what() << std::endl;
        }
    }
};

// Public interface implementation

RerunPublisher::RerunPublisher(const std::string& app_id,
                             const std::string& address, 
                             bool enabled)
    : pImpl(std::make_unique<Impl>(app_id, address, enabled)) {}

RerunPublisher::~RerunPublisher() = default;

bool RerunPublisher::connect() {
    return pImpl->connect();
}

void RerunPublisher::disconnect() {
    pImpl->disconnect();
}

bool RerunPublisher::isConnected() const {
    return pImpl->connected;
}

void RerunPublisher::publishMesh(const MeshUpdate& update, const std::string& entity_path) {
    pImpl->publishMesh(update, entity_path);
}

void RerunPublisher::publishColoredMesh(const std::vector<float>& vertices,
                                       const std::vector<uint32_t>& faces,
                                       const std::vector<uint8_t>& colors,
                                       const std::string& entity_path) {
    pImpl->publishColoredMesh(vertices, faces, colors, entity_path);
}

void RerunPublisher::logCameraPose(const float pose[16], const std::string& entity_path) {
    pImpl->logCameraPose(pose, entity_path);
}

void RerunPublisher::clearEntity(const std::string& entity_path) {
    pImpl->clearEntity(entity_path);
}

void RerunPublisher::setEnabled(bool enabled) {
    pImpl->enabled = enabled;
}

} // namespace mesh_service