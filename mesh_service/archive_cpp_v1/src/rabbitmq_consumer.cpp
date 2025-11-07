#include "rabbitmq_consumer.h"
#include <amqpcpp.h>
#include <amqpcpp/libev.h>
#include <ev.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <sstream>
#include <map>

// msgpack-c for message deserialization
#include <msgpack.hpp>

namespace mesh_service {

class RabbitMQConsumer::Impl {
public:
    std::string url;
    std::string exchange_name;
    std::string queue_name;
    std::function<void(const KeyframeMessage&)> keyframe_handler;
    
    // AMQP-CPP objects
    std::unique_ptr<AMQP::LibEvHandler> handler;
    std::unique_ptr<AMQP::TcpConnection> connection;
    std::unique_ptr<AMQP::TcpChannel> channel;
    struct ev_loop* loop = nullptr;
    std::thread consumer_thread;
    
    bool connected = false;
    bool running = false;
    bool should_stop = false;
    
    Impl(const std::string& u) : url(u) {
        // Parse exchange name from environment or use default
        const char* exchange_env = std::getenv("SLAM3R_KEYFRAME_EXCHANGE");
        exchange_name = exchange_env ? exchange_env : "slam3r_keyframe_exchange";
        queue_name = "mesh_service_keyframes";
    }
    
    ~Impl() {
        stop();
    }
    
    void connect() {
        int retry_count = 0;
        const int max_retries = 5;
        const int retry_delay_ms = 2000;
        
        while (retry_count < max_retries && !should_stop) {
            try {
                // Create event loop
                loop = ev_loop_new(0);
                if (!loop) {
                    throw std::runtime_error("Failed to create event loop");
                }
                
                // Create handler
                handler = std::make_unique<AMQP::LibEvHandler>(loop);
                
                // Parse URL and create connection
                AMQP::Address address(url);
                connection = std::make_unique<AMQP::TcpConnection>(handler.get(), address);
                
                // Create channel
                channel = std::make_unique<AMQP::TcpChannel>(connection.get());
                
                // Set up error handlers with reconnection
                channel->onError([this](const char* message) {
                    std::cerr << "RabbitMQ channel error: " << message << std::endl;
                    connected = false;
                    // Schedule reconnection
                    if (!should_stop) {
                        std::thread([this]() {
                            std::this_thread::sleep_for(std::chrono::seconds(5));
                            if (!should_stop) {
                                reconnect();
                            }
                        }).detach();
                    }
                });
                
                channel->onReady([this]() {
                    std::cout << "RabbitMQ channel ready" << std::endl;
                    connected = true;
                    setupConsumer();
                });
                
                // Start event loop in separate thread
                consumer_thread = std::thread([this]() {
                    while (!should_stop) {
                        ev_run(loop, EVRUN_NOWAIT);
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                });
                
                // Wait for connection
                std::this_thread::sleep_for(std::chrono::seconds(2));
                
                if (connected) {
                    std::cout << "Successfully connected to RabbitMQ" << std::endl;
                    return;
                }
                
            } catch (const std::exception& e) {
                std::cerr << "RabbitMQ connection attempt " << (retry_count + 1) 
                         << " failed: " << e.what() << std::endl;
            }
            
            retry_count++;
            if (retry_count < max_retries) {
                std::cout << "Retrying connection in " << retry_delay_ms << "ms..." << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(retry_delay_ms));
            }
        }
        
        throw std::runtime_error("Failed to connect to RabbitMQ after " + 
                                std::to_string(max_retries) + " attempts");
    }
    
    void reconnect() {
        std::cout << "Attempting to reconnect to RabbitMQ..." << std::endl;
        
        // Clean up existing connection
        if (consumer_thread.joinable()) {
            consumer_thread.join();
        }
        if (channel) {
            channel.reset();
        }
        if (connection) {
            connection.reset();
        }
        if (handler) {
            handler.reset();
        }
        if (loop) {
            ev_loop_destroy(loop);
            loop = nullptr;
        }
        
        connected = false;
        
        // Try to reconnect
        try {
            connect();
        } catch (const std::exception& e) {
            std::cerr << "Reconnection failed: " << e.what() << std::endl;
        }
    }
    
    void setupConsumer() {
        // Declare queue
        channel->declareQueue(queue_name, AMQP::durable)
            .onSuccess([this](const std::string& name, uint32_t messageCount, uint32_t consumerCount) {
                std::cout << "Queue " << name << " declared successfully. "
                         << "Messages: " << messageCount << ", Consumers: " << consumerCount << std::endl;
                
                // Bind to exchange with routing key pattern
                // Using "#" to match all routing keys, or "keyframe.*" for specific pattern
                channel->bindQueue(exchange_name, queue_name, "#")
                    .onSuccess([this]() {
                        std::cout << "Queue bound to exchange " << exchange_name << " with routing key '#'" << std::endl;
                        startConsuming();
                    })
                    .onError([](const char* message) {
                        std::cerr << "Failed to bind queue: " << message << std::endl;
                    });
            })
            .onError([](const char* message) {
                std::cerr << "Failed to declare queue: " << message << std::endl;
            });
    }
    
    void startConsuming() {
        // Set prefetch count for flow control
        channel->setQos(1);
        
        // Start consuming (without auto-ack so we can manually acknowledge)
        channel->consume(queue_name)
            .onReceived([this](const AMQP::Message& message, uint64_t deliveryTag, bool /*redelivered*/) {
                handleMessage(message, deliveryTag);
            })
            .onSuccess([](const std::string& consumertag) {
                std::cout << "Started consuming with tag: " << consumertag << std::endl;
            })
            .onError([](const char* message) {
                std::cerr << "Failed to start consuming: " << message << std::endl;
            });
        
        running = true;
    }
    
    void handleMessage(const AMQP::Message& message, uint64_t deliveryTag) {
        auto msg_start = std::chrono::high_resolution_clock::now();
        try {
            // Get message body
            std::string body(message.body(), message.bodySize());
            
            // Debug: Print message size
            std::cout << "Received message, size: " << body.size() << " bytes" << std::endl;
            
            // Deserialize with msgpack
            auto unpack_start = std::chrono::high_resolution_clock::now();
            msgpack::object_handle oh = msgpack::unpack(body.data(), body.size());
            msgpack::object obj = oh.get();
            auto unpack_end = std::chrono::high_resolution_clock::now();
            auto unpack_ms = std::chrono::duration_cast<std::chrono::microseconds>(unpack_end - unpack_start).count();
            std::cout << "[TIMING] msgpack unpacking: " << unpack_ms << " µs" << std::endl;
            
            // Debug: Print object type
            std::cout << "Message type: " << obj.type << std::endl;
            
            // Parse message into KeyframeMessage
            KeyframeMessage msg;
            
            // Extract fields from msgpack map
            if (obj.type == msgpack::type::MAP) {
                try {
                    auto parse_start = std::chrono::high_resolution_clock::now();
                    // Try to iterate through the map directly without converting
                    std::cout << "Iterating through msgpack map:" << std::endl;
                    
                    if (obj.via.map.size > 0) {
                        for (uint32_t i = 0; i < obj.via.map.size; ++i) {
                            // Get key and value
                            auto& kv = obj.via.map.ptr[i];
                            
                            // Try to get key as string
                            if (kv.key.type == msgpack::type::STR) {
                                std::string key(kv.key.via.str.ptr, kv.key.via.str.size);
                                std::cout << "  Key: " << key << ", Value type: " << kv.val.type << std::endl;
                                
                                // Handle each field
                                if (key == "shm_key" && kv.val.type == msgpack::type::STR) {
                                    msg.shm_key = std::string(kv.val.via.str.ptr, kv.val.via.str.size);
                                }
                                else if (key == "keyframe_id" && kv.val.type == msgpack::type::STR) {
                                    msg.keyframe_id = std::string(kv.val.via.str.ptr, kv.val.via.str.size);
                                }
                                else if (key == "timestamp") {
                                    // Handle both float and int types
                                    // msgpack object types: FLOAT (4), POSITIVE_INTEGER (1), NEGATIVE_INTEGER (2)
                                    if (kv.val.type == msgpack::type::FLOAT) {
                                        double timestamp_ms = kv.val.via.f64;
                                        msg.timestamp_ns = static_cast<uint64_t>(timestamp_ms * 1000000);
                                    } else if (kv.val.type == msgpack::type::POSITIVE_INTEGER) {
                                        uint64_t timestamp_ms = kv.val.via.u64;
                                        msg.timestamp_ns = timestamp_ms * 1000000;
                                    }
                                    std::cout << "    Timestamp type: " << kv.val.type << ", value: " << msg.timestamp_ns << std::endl;
                                }
                                else if (key == "type" && kv.val.type == msgpack::type::STR) {
                                    msg.type = std::string(kv.val.via.str.ptr, kv.val.via.str.size);
                                }
                                else if (key == "point_count" && kv.val.type == msgpack::type::POSITIVE_INTEGER) {
                                    msg.point_count = kv.val.via.u64;
                                }
                                else if (key == "pose_matrix" && kv.val.type == msgpack::type::ARRAY) {
                                    if (kv.val.via.array.size == 16) {
                                        for (uint32_t i = 0; i < 16; i++) {
                                            auto& elem = kv.val.via.array.ptr[i];
                                            if (elem.type == msgpack::type::FLOAT) {
                                                msg.pose_matrix[i] = elem.via.f64;
                                            } else if (elem.type == msgpack::type::POSITIVE_INTEGER) {
                                                msg.pose_matrix[i] = static_cast<float>(elem.via.u64);
                                            } else if (elem.type == msgpack::type::NEGATIVE_INTEGER) {
                                                msg.pose_matrix[i] = static_cast<float>(elem.via.i64);
                                            }
                                        }
                                        std::cout << "    Parsed pose_matrix, camera position: [" 
                                                  << msg.pose_matrix[12] << ", " 
                                                  << msg.pose_matrix[13] << ", "
                                                  << msg.pose_matrix[14] << "]" << std::endl;
                                    }
                                }
                                else if (key == "bbox" && kv.val.type == msgpack::type::ARRAY) {
                                    if (kv.val.via.array.size == 6) {
                                        for (uint32_t i = 0; i < 6; i++) {
                                            auto& elem = kv.val.via.array.ptr[i];
                                            if (elem.type == msgpack::type::FLOAT) {
                                                msg.bbox[i] = elem.via.f64;
                                            } else if (elem.type == msgpack::type::POSITIVE_INTEGER) {
                                                msg.bbox[i] = static_cast<float>(elem.via.u64);
                                            } else if (elem.type == msgpack::type::NEGATIVE_INTEGER) {
                                                msg.bbox[i] = static_cast<float>(elem.via.i64);
                                            }
                                        }
                                    }
                                }
                            } else {
                                std::cout << "  Key type: " << kv.key.type << " (not string)" << std::endl;
                            }
                        }
                    }
                    
                    // Set default type if not provided
                    if (msg.type.empty()) {
                        std::cout << "[MESSAGE TYPE FIX] Message has NO 'type' field, defaulting to 'keyframe.new'" << std::endl;
                        std::cout << "[MESSAGE TYPE FIX] Message size: " << body.size() << " bytes" << std::endl;
                        std::cout << "[MESSAGE TYPE FIX] Message fields present: timestamp=" << (msg.timestamp_ns > 0 ? "yes" : "no")
                                  << ", keyframe_id=" << (!msg.keyframe_id.empty() ? "yes" : "no")
                                  << ", shm_key=" << (!msg.shm_key.empty() ? "yes" : "no")
                                  << ", point_count=" << msg.point_count << std::endl;
                        // CRITICAL FIX: Default to keyframe.new to process messages without type field
                        msg.type = "keyframe.new";
                    } else {
                        std::cout << "[DEBUG] Message has 'type' field: " << msg.type << std::endl;
                    }
                    
                    std::cout << "Parsed message - shm_key: " << msg.shm_key 
                             << ", keyframe_id: " << msg.keyframe_id
                             << ", type: " << msg.type
                             << ", point_count: " << msg.point_count << std::endl;
                    
                    auto parse_end = std::chrono::high_resolution_clock::now();
                    auto parse_ms = std::chrono::duration_cast<std::chrono::microseconds>(parse_end - parse_start).count();
                    std::cout << "[TIMING] Message parsing: " << parse_ms << " µs" << std::endl;
                    
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing msgpack fields: " << e.what() << std::endl;
                    throw;
                }
                
                // CRITICAL FIX: Accept both keyframe.new and keyframe_update messages
                // Previously only processed keyframe.new, causing most messages to be ignored
                if (keyframe_handler && (msg.type == "keyframe.new" || msg.type == "keyframe_update")) {
                    // CRITICAL: Skip messages without shm_key - these are direct publishes without pose data
                    if (msg.shm_key.empty()) {
                        std::cout << "[MESSAGE VALIDATION] Skipping message without shm_key (no pose data)" << std::endl;
                        // Acknowledge the message but don't process it
                        channel->ack(deliveryTag);
                        auto msg_end = std::chrono::high_resolution_clock::now();
                        auto msg_total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(msg_end - msg_start).count();
                        std::cout << "[TIMING] Total message handling: " << msg_total_ms << " ms" << std::endl;
                        return;
                    }
                    std::cout << "[MESSAGE TYPE FIX] Processing message with type: " << msg.type << std::endl;
                    auto handler_start = std::chrono::high_resolution_clock::now();
                    keyframe_handler(msg);
                    auto handler_end = std::chrono::high_resolution_clock::now();
                    auto handler_ms = std::chrono::duration_cast<std::chrono::milliseconds>(handler_end - handler_start).count();
                    std::cout << "[TIMING] Keyframe handler total: " << handler_ms << " ms" << std::endl;
                } else if (keyframe_handler) {
                    std::cout << "[MESSAGE TYPE FIX] Ignoring unknown message type: " << msg.type << std::endl;
                }
            }
            
            // Acknowledge message
            channel->ack(deliveryTag);
            
            auto msg_end = std::chrono::high_resolution_clock::now();
            auto msg_total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(msg_end - msg_start).count();
            std::cout << "[TIMING] Total message handling: " << msg_total_ms << " ms" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "Error handling message: " << e.what() << std::endl;
            // Reject message without requeue
            channel->reject(deliveryTag, false);
        }
    }
    
    void stop() {
        if (running) {
            running = false;
            should_stop = true;
            
            if (channel) {
                channel->close();
            }
            if (connection) {
                connection->close();
            }
            
            if (consumer_thread.joinable()) {
                consumer_thread.join();
            }
            
            if (loop) {
                ev_loop_destroy(loop);
                loop = nullptr;
            }
        }
    }
};

RabbitMQConsumer::RabbitMQConsumer(const std::string& url) 
    : pImpl(std::make_unique<Impl>(url)) {}

RabbitMQConsumer::~RabbitMQConsumer() = default;

void RabbitMQConsumer::connect() {
    pImpl->connect();
}

void RabbitMQConsumer::start() {
    if (!pImpl->connected) {
        throw std::runtime_error("Not connected to RabbitMQ");
    }
    // Already started in connect()
}

void RabbitMQConsumer::stop() {
    pImpl->stop();
}

void RabbitMQConsumer::setKeyframeHandler(std::function<void(const KeyframeMessage&)> handler) {
    pImpl->keyframe_handler = handler;
}

} // namespace mesh_service