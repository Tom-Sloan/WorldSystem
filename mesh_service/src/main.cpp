#include <iostream>
#include <memory>
#include <thread>
#include <csignal>
#include <atomic>
#include <chrono>

#include "shared_memory.h"
#include "mesh_generator.h"

std::atomic<bool> g_running{true};

void signal_handler(int signal) {
    std::cout << "Received signal " << signal << ", shutting down..." << std::endl;
    g_running = false;
}

int main(int argc, char* argv[]) {
    // Set up signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    std::cout << "Mesh Service starting (minimal version)..." << std::endl;
    
    try {
        // Initialize components
        auto shared_memory = std::make_shared<mesh_service::SharedMemoryManager>();
        auto mesh_generator = std::make_shared<mesh_service::GPUMeshGenerator>();
        
        // Configure mesh generator
        mesh_generator->setMethod(mesh_service::MeshMethod::INCREMENTAL_POISSON);
        mesh_generator->setQualityAdaptive(true);
        mesh_generator->setSimplificationRatio(0.1f);
        
        std::cout << "Mesh Service running..." << std::endl;
        std::cout << "Waiting for keyframes from SLAM3R..." << std::endl;
        
        int frame_count = 0;
        
        // Main loop - poll for shared memory segments
        while (g_running) {
            // Check for test keyframe
            std::string test_shm_key = "/slam3r_keyframe_test";
            auto* keyframe = shared_memory->open_keyframe(test_shm_key);
            
            if (keyframe) {
                frame_count++;
                std::cout << "Found keyframe " << frame_count 
                         << ": " << keyframe->point_count << " points" << std::endl;
                
                // Generate mesh (placeholder)
                mesh_service::MeshUpdate update;
                mesh_generator->generateIncrementalMesh(keyframe, update);
                
                std::cout << "Generated mesh with " 
                         << update.vertices.size() / 3 << " vertices, "
                         << update.faces.size() / 3 << " faces" << std::endl;
                
                // Close shared memory
                shared_memory->close_keyframe(keyframe);
            }
            
            // Sleep briefly
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // Shutdown
        std::cout << "Shutting down..." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Mesh Service stopped." << std::endl;
    return 0;
}