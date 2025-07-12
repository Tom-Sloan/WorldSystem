#pragma once

#include <memory>
#include <string>
#include <vector>
#include "mesh_generator.h"

namespace mesh_service {

/**
 * Publishes mesh data to Rerun viewer for real-time visualization.
 * Uses the Rerun C++ SDK to stream triangle meshes.
 */
class RerunPublisher {
public:
    /**
     * Initialize connection to Rerun viewer.
     * @param app_id Application identifier for Rerun
     * @param address Rerun viewer address (e.g., "127.0.0.1:9090")
     * @param enabled Whether Rerun publishing is enabled
     */
    RerunPublisher(const std::string& app_id = "mesh_service",
                   const std::string& address = "127.0.0.1:9876", 
                   bool enabled = true);
    ~RerunPublisher();
    
    /**
     * Connect to Rerun viewer.
     * @return true if connection successful
     */
    bool connect();
    
    /**
     * Disconnect from Rerun viewer.
     */
    void disconnect();
    
    /**
     * Check if connected to Rerun.
     */
    bool isConnected() const;
    
    /**
     * Publish mesh update to Rerun.
     * @param update Mesh data to publish
     * @param entity_path Entity path in Rerun (default: "/mesh_service/reconstruction")
     */
    void publishMesh(const MeshUpdate& update, 
                     const std::string& entity_path = "/mesh_service/reconstruction");
    
    /**
     * Publish colored mesh to Rerun.
     * @param vertices Vertex positions (x,y,z triplets)
     * @param faces Triangle indices (i,j,k triplets)
     * @param colors Vertex colors (r,g,b triplets, 0-255)
     * @param entity_path Entity path in Rerun
     */
    void publishColoredMesh(const std::vector<float>& vertices,
                           const std::vector<uint32_t>& faces,
                           const std::vector<uint8_t>& colors,
                           const std::string& entity_path = "/mesh_service/reconstruction");
    
    /**
     * Log camera pose to Rerun.
     * @param pose 4x4 transformation matrix
     * @param entity_path Entity path for camera
     */
    void logCameraPose(const float pose[16], 
                       const std::string& entity_path = "/mesh_service/camera");
    
    /**
     * Clear entity data in Rerun.
     * @param entity_path Entity path to clear
     */
    void clearEntity(const std::string& entity_path);
    
    /**
     * Set whether publishing is enabled.
     */
    void setEnabled(bool enabled);
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace mesh_service