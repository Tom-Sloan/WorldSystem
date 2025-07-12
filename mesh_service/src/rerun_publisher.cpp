#include "rerun_publisher.h"
#include <rerun.hpp>
#include <rerun/archetypes/mesh3d.hpp>
#include <rerun/archetypes/transform3d.hpp>
#include <rerun/archetypes/clear.hpp>
#include <rerun/components/position3d.hpp>
#include <rerun/components/vector3d.hpp>
#include <rerun/components/color.hpp>
#include <rerun/components/material.hpp>
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
            
            // Connect to viewer at specified address
            auto result = stream->connect(address);
            if (result.is_err()) {
                std::cerr << "Failed to connect to Rerun at " << address 
                         << ": " << result.error().description << std::endl;
                return false;
            }
            
            connected = true;
            std::cout << "Connected to Rerun viewer at " << address << std::endl;
            
            // Log initial setup
            stream->set_time_seconds("time", std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
            
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
            stream->set_time_seconds("time", std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
            
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
            if (!update.colors.empty()) {
                std::vector<rerun::Color> colors;
                colors.reserve(update.colors.size() / 3);
                
                for (size_t i = 0; i < update.colors.size(); i += 3) {
                    colors.push_back(rerun::Color(update.colors[i], 
                                                 update.colors[i+1], 
                                                 update.colors[i+2]));
                }
                
                mesh = std::move(mesh).with_vertex_colors(colors);
            }
            
            // Add material for better visualization
            auto material = rerun::components::Material();
            material.albedo_factor = {0.8f, 0.8f, 0.8f, 1.0f};
            mesh = std::move(mesh).with_mesh_material(material);
            
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
            stream->set_time_seconds("time", std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
            
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
            
            // Add material
            auto material = rerun::components::Material();
            material.albedo_factor = {1.0f, 1.0f, 1.0f, 1.0f};
            mesh = std::move(mesh).with_mesh_material(material);
            
            // Log the mesh
            stream->log(entity_path, mesh);
            
        } catch (const std::exception& e) {
            std::cerr << "Error publishing colored mesh to Rerun: " << e.what() << std::endl;
        }
    }
    
    void logCameraPose(const float pose[16], const std::string& entity_path) {
        if (!enabled || !connected || !stream) return;
        
        try {
            // Convert pose to Rerun mat4x4
            rerun::datatypes::Mat4x4 transform;
            for (int i = 0; i < 16; ++i) {
                transform.coefficients[i] = pose[i];
            }
            
            // Create transform archetype
            auto transform3d = rerun::archetypes::Transform3D(transform);
            
            // Log the transform
            stream->log(entity_path, transform3d);
            
        } catch (const std::exception& e) {
            std::cerr << "Error logging camera pose to Rerun: " << e.what() << std::endl;
        }
    }
    
    void clearEntity(const std::string& entity_path) {
        if (!enabled || !connected || !stream) return;
        
        try {
            stream->log(entity_path, rerun::archetypes::Clear::recursive());
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