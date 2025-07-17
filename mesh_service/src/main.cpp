#include <iostream>
#include <memory>
#include <thread>
#include <csignal>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <exception>

#include "shared_memory.h"
#include "mesh_generator.h"
#include "rabbitmq_consumer.h"
#include "metrics.h"
#include "rerun_publisher.h"
#include "normal_provider.h"
#include "config/configuration_manager.h"
#include "config/mesh_service_config.h"
#include "config/normal_provider_config.h"

std::atomic<bool> g_running{true};

void signal_handler(int signal) {
    std::cout << "Received signal " << signal << ", shutting down..." << std::endl;
    g_running = false;
}

int main(int argc, char* argv[]) {
    // Suppress unused parameter warnings
    (void)argc;
    (void)argv;
    // Set up signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    std::cout << "\n============================================" << std::endl;
    std::cout << "        MESH SERVICE v2.1 STARTING          " << std::endl;
    std::cout << "============================================" << std::endl;
    
    // Get normal provider configuration
    int normal_provider = CONFIG_INT("MESH_NORMAL_PROVIDER", 
                                    mesh_service::config::NormalProviderConfig::DEFAULT_NORMAL_PROVIDER);
    std::cout << "[CONFIG] Normal Provider: " << normal_provider 
              << " (" << mesh_service::NormalProviderFactory::getProviderTypeName(normal_provider) << ")" << std::endl;
    
    // Check Open3D availability
#ifdef HAS_OPEN3D
    std::cout << "[CONFIG] Open3D Support: ENABLED" << std::endl;
#else
    std::cout << "[CONFIG] Open3D Support: DISABLED" << std::endl;
#endif
    
    std::cout << "[CONFIG] TSDF Method: Improved camera carving" << std::endl;
    std::cout << "[CONFIG] Performance Mode: " 
              << (normal_provider == 0 ? "REAL-TIME (~50 FPS)" : "QUALITY") << std::endl;
    std::cout << "============================================\n" << std::endl;
    
    // Initialize configuration manager
    auto& config = mesh_service::ConfigurationManager::getInstance();
    
    // Validate configuration
    if (!config.validateConfiguration()) {
        std::cerr << "Invalid configuration detected. Please check environment variables." << std::endl;
        return 1;
    }
    
    // Log configuration if debug mode is enabled
    if (std::getenv("MESH_DEBUG_CONFIG")) {
        config.logConfiguration();
    }
    
    // Set up terminate handler for better debugging
    std::set_terminate([]() {
        std::cerr << "Uncaught exception or termination!" << std::endl;
        try {
            auto ex = std::current_exception();
            if (ex) {
                std::rethrow_exception(ex);
            }
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown exception" << std::endl;
        }
        std::abort();
    });
    
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
        std::cout << "[DEBUG MAIN] Creating SharedMemoryManager..." << std::endl;
        auto shared_memory = std::make_shared<mesh_service::SharedMemoryManager>();
        std::cout << "[DEBUG MAIN] SharedMemoryManager created" << std::endl;
        
        std::cout << "[DEBUG MAIN] Creating GPUMeshGenerator..." << std::endl;
        auto mesh_generator = std::make_shared<mesh_service::GPUMeshGenerator>();
        std::cout << "[DEBUG MAIN] GPUMeshGenerator created" << std::endl;
        
        std::cout << "[DEBUG MAIN] Creating RabbitMQConsumer..." << std::endl;
        auto rabbitmq_consumer = std::make_shared<mesh_service::RabbitMQConsumer>(rabbitmq_url);
        std::cout << "[DEBUG MAIN] RabbitMQConsumer created" << std::endl;
        
        std::cout << "[DEBUG MAIN] Creating RerunPublisher..." << std::endl;
        auto rerun_publisher = std::make_shared<mesh_service::RerunPublisher>("mesh_service", rerun_address, rerun_enabled);
        std::cout << "[DEBUG MAIN] RerunPublisher created" << std::endl;
        
        // Configure mesh generator using configuration manager
        mesh_generator->setMethod(mesh_service::MeshMethod::TSDF_MARCHING_CUBES);
        mesh_generator->setQualityAdaptive(true);  // Enable adaptive quality based on camera velocity
        mesh_generator->setSimplificationRatio(
            CONFIG_FLOAT("MESH_SIMPLIFICATION_RATIO", mesh_service::config::AlgorithmConfig::DEFAULT_SIMPLIFICATION_RATIO)
        );
        
        std::cout << "Mesh service configured with new algorithm selector:" << std::endl;
        std::cout << "  - NVIDIA Marching Cubes with optimized TSDF" << std::endl;
        std::cout << "  - Velocity-based algorithm switching (future-ready)" << std::endl;
        std::cout << "  - GPU Octree for spatial indexing" << std::endl;
        std::cout << "  - Memory pool allocation (1GB)" << std::endl;
        std::cout << "  - 90% spatial overlap deduplication" << std::endl;
        std::cout << "  - Environment-based configuration" << std::endl;
        
        // Display TSDF configuration from environment
        const char* voxel_size = std::getenv("TSDF_VOXEL_SIZE");
        const char* truncation = std::getenv("TSDF_TRUNCATION_DISTANCE");
        const char* max_weight = std::getenv("TSDF_MAX_WEIGHT");
        const char* bounds_min = std::getenv("TSDF_VOLUME_MIN");
        const char* bounds_max = std::getenv("TSDF_VOLUME_MAX");
        
        if (voxel_size || truncation || max_weight || bounds_min || bounds_max) {
            std::cout << "\nTSDF Environment Configuration:" << std::endl;
            if (voxel_size) std::cout << "  TSDF_VOXEL_SIZE: " << voxel_size << "m" << std::endl;
            if (truncation) std::cout << "  TSDF_TRUNCATION_DISTANCE: " << truncation << "m" << std::endl;
            if (max_weight) std::cout << "  TSDF_MAX_WEIGHT: " << max_weight << std::endl;
            if (bounds_min) std::cout << "  TSDF_VOLUME_MIN: " << bounds_min << std::endl;
            if (bounds_max) std::cout << "  TSDF_VOLUME_MAX: " << bounds_max << std::endl;
        }
        
        // Connect to Rerun
        if (rerun_enabled) {
            std::cout << "[DEBUG MAIN] About to connect to Rerun..." << std::endl;
            std::cout << "[DEBUG MAIN] rerun_publisher pointer: " << rerun_publisher.get() << std::endl;
            
            // TEMPORARILY DISABLE RERUN to isolate crash
            std::cout << "[DEBUG MAIN] TEMPORARILY DISABLING RERUN CONNECTION TO ISOLATE CRASH" << std::endl;
            rerun_enabled = false;
            rerun_publisher->setEnabled(false);
            
            /*
            if (rerun_publisher->connect()) {
                std::cout << "Connected to Rerun viewer at " << rerun_address << std::endl;
                std::cout << "[DEBUG MAIN] Rerun connection successful" << std::endl;
                std::cout << "[DEBUG MAIN] About to test rerun functionality..." << std::endl;
                
                // Set up a pinhole camera view for better visualization
                // This helps Rerun understand the 3D space better
                // Camera matrix - will be used when we add camera visualization
                // float camera_matrix[9] = {
                //     500.0f, 0.0f, 320.0f,    // fx, 0, cx
                //     0.0f, 500.0f, 240.0f,    // 0, fy, cy
                //     0.0f, 0.0f, 1.0f         // 0, 0, 1
                // };
                
                // Log initial camera setup
                std::cout << "[DEBUG MAIN] Creating initial camera pose array..." << std::endl;
                float initial_pose[16] = {
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, -5,  // Position camera 5 units back
                    0, 0, 0, 1
                };
                std::cout << "[DEBUG MAIN] About to log camera pose to Rerun..." << std::endl;
                rerun_publisher->logCameraPose(initial_pose, "/mesh_service/camera");
                std::cout << "[DEBUG MAIN] Camera pose logged successfully" << std::endl;
                std::cout << "[DEBUG MAIN] Exiting Rerun setup block successfully" << std::endl;
            } else {
                std::cerr << "Failed to connect to Rerun viewer" << std::endl;
                rerun_enabled = false;
                rerun_publisher->setEnabled(false);
            }
            */
        }
        std::cout << "[DEBUG MAIN] After Rerun connection block" << std::endl;
        
        // Statistics
        int frame_count = 0;
        auto start_time = std::chrono::steady_clock::now();
        
        std::cout << "[DEBUG MAIN] About to set up RabbitMQ keyframe handler" << std::endl;
        
        // Set up RabbitMQ keyframe handler
        rabbitmq_consumer->setKeyframeHandler([&](const mesh_service::KeyframeMessage& msg) {
            try {
                frame_count++;
                auto process_start = std::chrono::steady_clock::now();
                
                std::cout << "\nReceived keyframe " << msg.keyframe_id 
                         << " via RabbitMQ, shm_key: " << msg.shm_key << std::endl;
                
                // Open shared memory segment
                std::cout << "[DEBUG] Opening shared memory segment: " << msg.shm_key << std::endl;
                auto* keyframe = shared_memory->open_keyframe(msg.shm_key);
                
                if (keyframe) {
                    std::cout << "[DEBUG] Successfully opened shared memory" << std::endl;
                    std::cout << "Processing keyframe with " << keyframe->point_count << " points" << std::endl;
                    
                    // Generate mesh
                    std::cout << "[DEBUG] Creating MeshUpdate object" << std::endl;
                    mesh_service::MeshUpdate update;
                    std::cout << "[DEBUG] Calling generateIncrementalMesh" << std::endl;
                    mesh_generator->generateIncrementalMesh(keyframe, update);
                    std::cout << "[DEBUG] Mesh generation complete" << std::endl;
                    
                    auto process_end = std::chrono::steady_clock::now();
                    auto process_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        process_end - process_start).count();
                    
                    std::cout << "Generated mesh with " 
                             << update.vertices.size() / 3 << " vertices, "
                             << update.faces.size() / 3 << " faces"
                             << " in " << process_time << "ms" << std::endl;
                    
                    // Store individual component times for summary
                    auto mesh_gen_time = process_time;
                    
                    // Record metrics
                    std::cout << "[DEBUG] Recording metrics" << std::endl;
                    mesh_service::Metrics::instance().recordMeshGeneration(
                        update.vertices.size() / 3, 
                        update.faces.size() / 3
                    );
                    mesh_service::Metrics::instance().recordProcessingTime(process_time / 1000.0);
                    
                    // Send mesh to Rerun
                    auto rerun_start = std::chrono::steady_clock::now();
                    long rerun_total_ms = 0;  // Initialize at outer scope
                    std::cout << "[DEBUG] Checking Rerun: enabled=" << rerun_enabled 
                             << ", connected=" << (rerun_publisher ? rerun_publisher->isConnected() : false) << std::endl;
                    if (rerun_enabled && rerun_publisher->isConnected()) {
                        // Extract vertex colors from the keyframe if available
                        uint8_t* keyframe_colors = shared_memory->get_colors(keyframe);
                        std::cout << "[DEBUG] Keyframe has colors: " << (keyframe_colors != nullptr) 
                                 << ", vertices count: " << update.vertices.size() << std::endl;
                        if (keyframe_colors && update.vertices.size() > 0) {
                            // Colors are in the keyframe data
                            std::cout << "[DEBUG] Extracting colors for " << keyframe->point_count << " points" << std::endl;
                            auto color_extract_start = std::chrono::steady_clock::now();
                            
                            // Get colors pointer from shared memory
                            uint8_t* shm_colors = shared_memory->get_colors(keyframe);
                            if (!shm_colors) {
                                std::cerr << "[DEBUG] Failed to get colors pointer from shared memory" << std::endl;
                            } else {
                                std::vector<uint8_t> colors;
                                colors.reserve(keyframe->point_count * 3);
                                for (uint32_t i = 0; i < keyframe->point_count * 3; i++) {
                                    colors.push_back(shm_colors[i]);
                                }
                                auto color_extract_end = std::chrono::steady_clock::now();
                                auto color_extract_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    color_extract_end - color_extract_start).count();
                                std::cout << "[DEBUG] Color extraction complete, size: " << colors.size() << std::endl;
                                std::cout << "[TIMING] Color extraction: " << color_extract_ms << " ms" << std::endl;
                                
                                // Publish colored mesh
                                std::cout << "[DEBUG] Publishing colored mesh to Rerun" << std::endl;
                                auto publish_start = std::chrono::steady_clock::now();
                                rerun_publisher->publishColoredMesh(
                                    update.vertices, 
                                    update.faces, 
                                    colors,
                                    "/mesh_service/reconstruction"
                                );
                                auto publish_end = std::chrono::steady_clock::now();
                                auto publish_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    publish_end - publish_start).count();
                                std::cout << "[DEBUG] Colored mesh published" << std::endl;
                                std::cout << "[TIMING] Rerun publish (colored): " << publish_ms << " ms" << std::endl;
                            }
                        } else {
                            // Publish mesh without colors
                            std::cout << "[DEBUG] Publishing mesh without colors to Rerun" << std::endl;
                            auto publish_start = std::chrono::steady_clock::now();
                            rerun_publisher->publishMesh(update, "/mesh_service/reconstruction");
                            auto publish_end = std::chrono::steady_clock::now();
                            auto publish_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                publish_end - publish_start).count();
                            std::cout << "[DEBUG] Mesh published" << std::endl;
                            std::cout << "[TIMING] Rerun publish (no color): " << publish_ms << " ms" << std::endl;
                        }
                        
                        // Also log camera pose if available
                        // TODO: Fix camera pose logging - currently causing segfault
                        /*
                        if (keyframe->pose_matrix[0] != 0.0f) {  // Check if pose is valid
                            std::cout << "[DEBUG] Logging camera pose" << std::endl;
                            auto pose_start = std::chrono::steady_clock::now();
                            rerun_publisher->logCameraPose(keyframe->pose_matrix, "/mesh_service/camera");
                            auto pose_end = std::chrono::steady_clock::now();
                            auto pose_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                pose_end - pose_start).count();
                            std::cout << "[DEBUG] Camera pose logged" << std::endl;
                            std::cout << "[TIMING] Camera pose log: " << pose_us << " µs" << std::endl;
                        }
                        */
                    }
                    
                    auto rerun_end = std::chrono::steady_clock::now();
                    rerun_total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        rerun_end - rerun_start).count();
                    if (rerun_enabled && rerun_publisher->isConnected()) {
                        std::cout << "[TIMING] Total Rerun publishing: " << rerun_total_ms << " ms" << std::endl;
                    }
                    
                    // Close shared memory
                    auto cleanup_start = std::chrono::steady_clock::now();
                    std::cout << "[DEBUG] Closing shared memory" << std::endl;
                    shared_memory->close_keyframe(keyframe);
                    std::cout << "[DEBUG] Shared memory closed" << std::endl;
                    
                    // Optionally unlink the shared memory segment
                    if (unlink_shm) {
                        std::cout << "[DEBUG] Unlinking shared memory: " << msg.shm_key << std::endl;
                        shared_memory->unlink_keyframe(msg.shm_key);
                        std::cout << "[DEBUG] Shared memory unlinked" << std::endl;
                    }
                    auto cleanup_end = std::chrono::steady_clock::now();
                    auto cleanup_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        cleanup_end - cleanup_start).count();
                    std::cout << "[TIMING] Cleanup operations: " << cleanup_ms << " ms" << std::endl;
                    
                    // Calculate total frame processing time
                    auto frame_end = std::chrono::steady_clock::now();
                    auto frame_total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        frame_end - process_start).count();
                    
                    // Print comprehensive timing summary
                    std::cout << "\n========== FRAME PROCESSING TIMING SUMMARY ==========" << std::endl;
                    std::cout << "[NORMAL PROVIDER STATUS] " << mesh_service::NormalProviderFactory::getProviderTypeName(
                                  CONFIG_INT("MESH_NORMAL_PROVIDER", mesh_service::config::NormalProviderConfig::DEFAULT_NORMAL_PROVIDER))
                             << std::endl;
                    std::cout << "Frame " << frame_count << " - Total Time: " << frame_total_ms << " ms" << std::endl;
                    std::cout << "Breakdown:" << std::endl;
                    std::cout << "  1. Mesh Generation: " << mesh_gen_time << " ms (" 
                             << (mesh_gen_time * 100.0 / frame_total_ms) << "%)" << std::endl;
                    std::cout << "     - Normal provider: " << mesh_service::NormalProviderFactory::getProviderTypeName(
                                  CONFIG_INT("MESH_NORMAL_PROVIDER", mesh_service::config::NormalProviderConfig::DEFAULT_NORMAL_PROVIDER))
                             << std::endl;
                    std::cout << "     - TSDF + MC: ~" << (mesh_gen_time * 0.80) << " ms (estimate)" << std::endl;
                    std::cout << "     - Other: ~" << (mesh_gen_time * 0.20) << " ms (estimate)" << std::endl;
                    std::cout << "  2. Rerun Publishing: " << rerun_total_ms << " ms (" 
                             << (rerun_total_ms * 100.0 / frame_total_ms) << "%)" << std::endl;
                    std::cout << "  3. Cleanup: " << cleanup_ms << " ms (" 
                             << (cleanup_ms * 100.0 / frame_total_ms) << "%)" << std::endl;
                    std::cout << "  4. Other (metrics, etc): " << (frame_total_ms - mesh_gen_time - rerun_total_ms - cleanup_ms) 
                             << " ms" << std::endl;
                    std::cout << "Performance: " << (1000.0 / frame_total_ms) << " FPS potential" << std::endl;
                    std::cout << "===================================================\n" << std::endl;
                    
                    // Log FPS periodically
                    if (frame_count % CONFIG_INT("MESH_FPS_LOG_INTERVAL", mesh_service::config::DebugConfig::DEFAULT_FPS_LOG_INTERVAL) == 0) {
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
        
        std::cout << "[DEBUG MAIN] RabbitMQ handler set up successfully" << std::endl;
        
        // Start metrics server
        std::cout << "[DEBUG MAIN] Creating metrics server on port " << metrics_port << std::endl;
        mesh_service::MetricsServer metrics_server(metrics_port);
        std::cout << "[DEBUG MAIN] Starting metrics server..." << std::endl;
        metrics_server.run();
        std::cout << "Metrics server started on port " << metrics_port << std::endl;
        
        // Connect to RabbitMQ
        std::cout << "[DEBUG MAIN] About to connect to RabbitMQ at " << rabbitmq_url << std::endl;
        std::cout << "Connecting to RabbitMQ at " << rabbitmq_url << "..." << std::endl;
        rabbitmq_consumer->connect();
        std::cout << "[DEBUG MAIN] RabbitMQ connected, starting consumer..." << std::endl;
        rabbitmq_consumer->start();
        std::cout << "[DEBUG MAIN] RabbitMQ consumer started" << std::endl;
        
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