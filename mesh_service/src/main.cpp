#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <array>


#include <rabbitmq-c/amqp.h>
#include <rabbitmq-c/tcp_socket.h>
#include <rabbitmq-c/framing.h>

#include <rerun.hpp>  // Rerun SDK C++

#include <msgpack.hpp>  // msgpack-c

#include "shared_memory.h"  // Your provided header

// Namespace alias
using namespace mesh_service;

// Global Rerun recording stream
std::shared_ptr<rerun::RecordingStream> rec;

// Batch buffers (flush every BATCH_SIZE)
constexpr int BATCH_SIZE = 5;
struct BatchedData {
    std::vector<rerun::Position3D> positions;
    std::vector<rerun::Color> colors;
    std::array<float, 16> transform_matrix;  // 4x4 matrix
    std::string entity_path;
};
std::vector<BatchedData> batch;

// SharedMemoryManager instance (global for simplicity)
SharedMemoryManager shm_manager;

// Function to connect to Rerun
bool connect_to_rerun(const std::string& addr) {
    rec = std::make_shared<rerun::RecordingStream>("slam3r_viz_stream");
    
    // Use gRPC connection (default localhost:9876)
    auto result = rec->connect_grpc();
    if (result.is_err()) {
        std::cerr << "Failed to connect to Rerun via gRPC" << std::endl;
        return false;
    }
    
    std::cout << "Connected to Rerun via gRPC" << std::endl;
    return true;
}

// Function to flush batch to Rerun
void flush_batch_to_rerun() {
    if (batch.empty()) return;

    for (const auto& data : batch) {
        // Log points with colors
        rec->log(data.entity_path,
                rerun::Points3D(data.positions)
                    .with_colors(data.colors));
        
        // Log transform (for camera pose) - extract from 4x4 matrix
        std::array<float, 9> mat3_flat = {
            data.transform_matrix[0], data.transform_matrix[4], data.transform_matrix[8],  // col 0
            data.transform_matrix[1], data.transform_matrix[5], data.transform_matrix[9],  // col 1
            data.transform_matrix[2], data.transform_matrix[6], data.transform_matrix[10]  // col 2
        };
        rec->log(data.entity_path + "/camera",
                rerun::Transform3D(
                    rerun::components::Translation3D(data.transform_matrix[12], data.transform_matrix[13], data.transform_matrix[14]),
                    rerun::components::TransformMat3x3(mat3_flat),
                    true
                ));
    }

    batch.clear();
}

// Function to process a keyframe and add to batch
void process_keyframe(const std::string& shm_key, uint32_t point_count, const std::string& keyframe_id) {
    // Open keyframe using manager
    SharedKeyframe* keyframe = shm_manager.open_keyframe(shm_key);
    if (!keyframe) {
        std::cerr << "Failed to open keyframe: " << shm_key << std::endl;
        return;
    }
    std::cout << "Successfully opened shared memory: " << shm_key << std::endl;

    // Get data pointers
    float* points = shm_manager.get_points(keyframe);
    uint8_t* colors = shm_manager.get_colors(keyframe);

    // Prepare batch data
    BatchedData data;
    data.entity_path = "world/points/" + keyframe_id;
    
    // Convert points and colors
    data.positions.reserve(point_count);
    data.colors.reserve(point_count);
    
    for (uint32_t i = 0; i < point_count; ++i) {
        data.positions.emplace_back(
            points[i * 3], 
            points[i * 3 + 1], 
            points[i * 3 + 2]
        );
        
        data.colors.emplace_back(
            colors[i * 3], 
            colors[i * 3 + 1], 
            colors[i * 3 + 2]
        );
    }

    // Copy transform matrix
    std::copy(keyframe->pose_matrix, keyframe->pose_matrix + 16, data.transform_matrix.begin());

    // Add to batch
    batch.push_back(std::move(data));

    // Flush if batch full
    if (batch.size() >= BATCH_SIZE) {
        flush_batch_to_rerun();
    }

    // Close and optionally unlink
    shm_manager.close_keyframe(keyframe);
    const char* unlink_env = getenv("MESH_SERVICE_UNLINK_SHM");
    if (unlink_env && std::string(unlink_env) == "true") {
        shm_manager.unlink_keyframe(shm_key);
        std::cout << "Unlinked shm: " << shm_key << std::endl;
    }
}

// RabbitMQ consumer loop
void rabbitmq_consumer_loop() {
    amqp_connection_state_t conn = amqp_new_connection();
    amqp_socket_t* socket = amqp_tcp_socket_new(conn);
    if (amqp_socket_open(socket, "127.0.0.1", 5672) != 0) {
        std::cerr << "Failed to connect to RabbitMQ" << std::endl;
        return;
    }

    amqp_rpc_reply_t login_reply = amqp_login(conn, "/", 0, 131072, 0, AMQP_SASL_METHOD_PLAIN, "guest", "guest");
    if (login_reply.reply_type != AMQP_RESPONSE_NORMAL) {
        std::cerr << "Failed to login to RabbitMQ" << std::endl;
        amqp_destroy_connection(conn);
        return;
    }
    std::cout << "Connected and logged in to RabbitMQ successfully" << std::endl;

    amqp_channel_open(conn, 1);

    amqp_exchange_declare(conn, 1, amqp_cstring_bytes("slam3r_keyframe_exchange"), amqp_cstring_bytes("topic"), 
                           0, 1, 0, 0, amqp_empty_table);
    amqp_rpc_reply_t declare_reply = amqp_get_rpc_reply(conn);
    if (declare_reply.reply_type != AMQP_RESPONSE_NORMAL) {
        std::cerr << "Failed to declare exchange" << std::endl;
        return;
    }

    amqp_queue_declare_ok_t* q = amqp_queue_declare(conn, 1, amqp_empty_bytes, 0, 0, 0, 1, amqp_empty_table);
    amqp_queue_bind(conn, 1, q->queue, amqp_cstring_bytes("slam3r_keyframe_exchange"), amqp_cstring_bytes("#"), amqp_empty_table);
    amqp_rpc_reply_t bind_reply = amqp_get_rpc_reply(conn);
    if (bind_reply.reply_type != AMQP_RESPONSE_NORMAL) {
        std::cerr << "Failed to bind queue" << std::endl;
        return;
    }

    amqp_basic_consume(conn, 1, q->queue, amqp_empty_bytes, 0, 1, 0, amqp_empty_table);
    amqp_rpc_reply_t consume_reply = amqp_get_rpc_reply(conn);
    if (consume_reply.reply_type != AMQP_RESPONSE_NORMAL) {
        std::cerr << "Failed to consume" << std::endl;
        return;
    }

    while (true) {
        amqp_maybe_release_buffers(conn);
        amqp_envelope_t envelope;
        amqp_consume_message(conn, &envelope, NULL, 0);

        // Real msgpack parsing
        msgpack::object_handle oh = msgpack::unpack(static_cast<const char*>(envelope.message.body.bytes), envelope.message.body.len);
        msgpack::object obj = oh.get();

        // Convert to map
        std::unordered_map<std::string, msgpack::object> msg_map;
        obj.convert(msg_map);

        // Extract required fields (verified types from data log)
        std::string type = msg_map["type"].as<std::string>();
        if (type != "keyframe.new") {
            amqp_destroy_envelope(&envelope);
            continue;
        }

        std::string shm_key = msg_map["shm_key"].as<std::string>();
        uint32_t point_count = msg_map["point_count"].as<uint32_t>();
        std::string keyframe_id = msg_map["keyframe_id"].as<std::string>();

        // Process the keyframe
        process_keyframe(shm_key, point_count, keyframe_id);

        amqp_destroy_envelope(&envelope);
    }

    amqp_channel_close(conn, 1, AMQP_REPLY_SUCCESS);
    amqp_connection_close(conn, AMQP_REPLY_SUCCESS);
    amqp_destroy_connection(conn);
}

int main() {
    // Connect to Rerun (gRPC uses default port 9876)
    if (!connect_to_rerun("")) {
        return 1;
    }

    // Test logging to verify Rerun connection
    std::cout << "Logging test points to Rerun..." << std::endl;
    std::vector<rerun::Position3D> test_points = {rerun::Position3D(0.0f, 0.0f, 0.0f), rerun::Position3D(1.0f, 1.0f, 1.0f)};
    std::vector<rerun::Color> test_colors = {rerun::Color(255, 0, 0, 255), rerun::Color(0, 255, 0, 255)};
    rec->log("test_points", rerun::Points3D(test_points).with_colors(test_colors).with_radii({0.1f}));
    std::cout << "Test points logged to Rerun." << std::endl;

    // Start consumer loop
    rabbitmq_consumer_loop();

    // Cleanup on exit (flush remaining batch)
    flush_batch_to_rerun();
    return 0;
}