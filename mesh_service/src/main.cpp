#include <iostream>
#include <memory>
#include <thread>
#include <csignal>
#include <atomic>
#include <chrono>
#include <cstdlib>

#include "shared_memory.h"
#include "mesh_generator.h"
#include "rabbitmq_consumer.h"
#include "metrics.h"
#include "rerun_publisher.h"

std::atomic<bool> g_running{true};

void signal_handler(int signal) {
    std::cout << "Received signal " << signal << ", shutting down..." << std::endl;
    g_running = false;
}

int main(int argc, char* argv[]) {
    // Set up signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    std::cout << "Mesh Service starting..." << std::endl;
    
    try {
        // Get configuration from environment
        const char* rabbitmq_url_env = std::getenv("RABBITMQ_URL");
        std::string rabbitmq_url = rabbitmq_url_env ? rabbitmq_url_env : "amqp://127.0.0.1:5672";
        
        const char* rerun_address_env = std::getenv("RERUN_VIEWER_ADDRESS");
        std::string rerun_address = rerun_address_env ? rerun_address_env : "127.0.0.1:9876";
        
        const char* rerun_enabled_env = std::getenv("RERUN_ENABLED");
        bool rerun_enabled = rerun_enabled_env ? (std::string(rerun_enabled_env) == "true") : true;
        
        const char* metrics_port_env = std::getenv("METRICS_PORT");
        int metrics_port = metrics_port_env ? std::atoi(metrics_port_env) : 9091;
        
        const char* unlink_shm_env = std::getenv("MESH_SERVICE_UNLINK_SHM");
        bool unlink_shm = unlink_shm_env ? (std::string(unlink_shm_env) == "true") : false;
        
        // Initialize components
        auto shared_memory = std::make_shared<mesh_service::SharedMemoryManager>();
        auto mesh_generator = std::make_shared<mesh_service::GPUMeshGenerator>();
        auto rabbitmq_consumer = std::make_shared<mesh_service::RabbitMQConsumer>(rabbitmq_url);
        auto rerun_publisher = std::make_shared<mesh_service::RerunPublisher>("mesh_service", rerun_address, rerun_enabled);
        
        // Configure mesh generator
        mesh_generator->setMethod(mesh_service::MeshMethod::INCREMENTAL_POISSON);
        mesh_generator->setQualityAdaptive(true);
        mesh_generator->setSimplificationRatio(0.1f);
        
        // Connect to Rerun
        if (rerun_enabled) {
            if (rerun_publisher->connect()) {
                std::cout << "Connected to Rerun viewer at " << rerun_address << std::endl;
            } else {
                std::cerr << "Failed to connect to Rerun viewer" << std::endl;
                rerun_enabled = false;
                rerun_publisher->setEnabled(false);
            }
        }
        
        // Statistics
        int frame_count = 0;
        auto start_time = std::chrono::steady_clock::now();
        
        // Set up RabbitMQ keyframe handler
        rabbitmq_consumer->setKeyframeHandler([&](const mesh_service::KeyframeMessage& msg) {
            try {
                frame_count++;
                auto process_start = std::chrono::steady_clock::now();
                
                std::cout << "\nReceived keyframe " << msg.keyframe_id 
                         << " via RabbitMQ, shm_key: " << msg.shm_key << std::endl;
                
                // Open shared memory segment
                auto* keyframe = shared_memory->open_keyframe(msg.shm_key);
                
                if (keyframe) {
                    std::cout << "Processing keyframe with " << keyframe->point_count << " points" << std::endl;
                    
                    // Generate mesh
                    mesh_service::MeshUpdate update;
                    mesh_generator->generateIncrementalMesh(keyframe, update);
                    
                    auto process_end = std::chrono::steady_clock::now();
                    auto process_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        process_end - process_start).count();
                    
                    std::cout << "Generated mesh with " 
                             << update.vertices.size() / 3 << " vertices, "
                             << update.faces.size() / 3 << " faces"
                             << " in " << process_time << "ms" << std::endl;
                    
                    // Record metrics
                    mesh_service::Metrics::instance().recordMeshGeneration(
                        update.vertices.size() / 3, 
                        update.faces.size() / 3
                    );
                    mesh_service::Metrics::instance().recordProcessingTime(process_time / 1000.0);
                    
                    // Send mesh to Rerun
                    if (rerun_enabled && rerun_publisher->isConnected()) {
                        // Extract vertex colors from the keyframe if available
                        if (keyframe->colors && update.vertices.size() > 0) {
                            // Colors are in the keyframe data
                            std::vector<uint8_t> colors;
                            colors.reserve(keyframe->point_count * 3);
                            for (uint32_t i = 0; i < keyframe->point_count * 3; i++) {
                                colors.push_back(keyframe->colors[i]);
                            }
                            
                            // Publish colored mesh
                            rerun_publisher->publishColoredMesh(
                                update.vertices, 
                                update.faces, 
                                colors,
                                "/mesh_service/reconstruction"
                            );
                        } else {
                            // Publish mesh without colors
                            rerun_publisher->publishMesh(update, "/mesh_service/reconstruction");
                        }
                        
                        // Also log camera pose if available
                        if (keyframe->pose_matrix[0] != 0.0f) {  // Check if pose is valid
                            rerun_publisher->logCameraPose(keyframe->pose_matrix, "/mesh_service/camera");
                        }
                    }
                    
                    // Close shared memory
                    shared_memory->close_keyframe(keyframe);
                    
                    // Optionally unlink the shared memory segment
                    if (unlink_shm) {
                        shared_memory->unlink_keyframe(msg.shm_key);
                    }
                    
                    // Log FPS periodically
                    if (frame_count % 10 == 0) {
                        auto now = std::chrono::steady_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
                        if (elapsed > 0) {
                            float fps = static_cast<float>(frame_count) / elapsed;
                            std::cout << "Average FPS: " << fps << std::endl;
                        }
                    }
                } else {
                    std::cerr << "Failed to open shared memory segment: " << msg.shm_key << std::endl;
                    mesh_service::Metrics::instance().recordError();
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Error processing keyframe: " << e.what() << std::endl;
                mesh_service::Metrics::instance().recordError();
            }
        });
        
        // Start metrics server
        mesh_service::MetricsServer metrics_server(metrics_port);
        metrics_server.run();
        std::cout << "Metrics server started on port " << metrics_port << std::endl;
        
        // Connect to RabbitMQ
        std::cout << "Connecting to RabbitMQ at " << rabbitmq_url << "..." << std::endl;
        rabbitmq_consumer->connect();
        rabbitmq_consumer->start();
        
        std::cout << "Mesh Service running. Waiting for keyframes from SLAM3R..." << std::endl;
        
        // Main loop - just keep running
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // Shutdown
        std::cout << "Shutting down..." << std::endl;
        rabbitmq_consumer->stop();
        metrics_server.stop();
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Mesh Service stopped." << std::endl;
    return 0;
}