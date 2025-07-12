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

std::atomic<bool> g_running{true};

void signal_handler(int signal) {
    std::cout << "Received signal " << signal << ", shutting down..." << std::endl;
    g_running = false;
}

int main(int argc, char* argv[]) {
    // Set up signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    std::cout << "Mesh Service TEST VERSION starting..." << std::endl;
    
    try {
        // Get configuration from environment
        const char* rabbitmq_url_env = std::getenv("RABBITMQ_URL");
        std::string rabbitmq_url = rabbitmq_url_env ? rabbitmq_url_env : "amqp://127.0.0.1:5672";
        
        const char* metrics_port_env = std::getenv("METRICS_PORT");
        int metrics_port = metrics_port_env ? std::atoi(metrics_port_env) : 9091;
        
        // Initialize components
        auto shared_memory = std::make_shared<mesh_service::SharedMemoryManager>();
        auto mesh_generator = std::make_shared<mesh_service::GPUMeshGenerator>();
        auto rabbitmq_consumer = std::make_shared<mesh_service::RabbitMQConsumer>(rabbitmq_url);
        
        // Configure mesh generator
        mesh_generator->setMethod(mesh_service::MeshMethod::INCREMENTAL_POISSON);
        mesh_generator->setQualityAdaptive(true);
        mesh_generator->setSimplificationRatio(0.1f);
        
        // Statistics
        int frame_count = 0;
        
        // Set up RabbitMQ keyframe handler
        rabbitmq_consumer->setKeyframeHandler([&](const mesh_service::KeyframeMessage& msg) {
            try {
                frame_count++;
                
                std::cout << "\nReceived keyframe " << msg.keyframe_id 
                         << " via RabbitMQ, shm_key: " << msg.shm_key 
                         << ", point_count: " << msg.point_count << std::endl;
                
                // Open shared memory segment
                auto* keyframe = shared_memory->open_keyframe(msg.shm_key);
                
                if (keyframe) {
                    std::cout << "Successfully opened shared memory segment" << std::endl;
                    std::cout << "  Actual point count: " << keyframe->point_count << std::endl;
                    std::cout << "  Timestamp: " << keyframe->timestamp_ns << std::endl;
                    std::cout << "  Color channels: " << keyframe->color_channels << std::endl;
                    
                    // Generate mesh
                    mesh_service::MeshUpdate update;
                    mesh_generator->generateIncrementalMesh(keyframe, update);
                    
                    std::cout << "Generated mesh with " 
                             << update.vertices.size() / 3 << " vertices, "
                             << update.faces.size() / 3 << " faces" << std::endl;
                    
                    // Record metrics
                    mesh_service::Metrics::instance().recordMeshGeneration(
                        update.vertices.size() / 3, 
                        update.faces.size() / 3
                    );
                    
                    // Close shared memory
                    shared_memory->close_keyframe(keyframe);
                    
                    std::cout << "Processed keyframe successfully!" << std::endl;
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
        
        std::cout << "Mesh Service TEST VERSION running. Waiting for keyframes..." << std::endl;
        
        // Main loop
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