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
                
                // Bind to exchange
                channel->bindQueue(exchange_name, queue_name, "")
                    .onSuccess([this]() {
                        std::cout << "Queue bound to exchange " << exchange_name << std::endl;
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
        
        // Start consuming
        channel->consume(queue_name, AMQP::noack)
            .onReceived([this](const AMQP::Message& message, uint64_t deliveryTag, bool redelivered) {
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
        try {
            // Get message body
            std::string body(message.body(), message.bodySize());
            
            // Deserialize with msgpack
            msgpack::object_handle oh = msgpack::unpack(body.data(), body.size());
            msgpack::object obj = oh.get();
            
            // Parse message into KeyframeMessage
            KeyframeMessage msg;
            
            // Extract fields from msgpack map
            if (obj.type == msgpack::type::MAP) {
                std::map<std::string, msgpack::object> map;
                obj.convert(map);
                
                // Extract required fields
                if (map.find("shm_key") != map.end()) {
                    map["shm_key"].convert(msg.shm_key);
                }
                if (map.find("keyframe_id") != map.end()) {
                    map["keyframe_id"].convert(msg.keyframe_id);
                }
                if (map.find("timestamp") != map.end()) {
                    uint64_t timestamp_ms;
                    map["timestamp"].convert(timestamp_ms);
                    msg.timestamp_ns = timestamp_ms * 1000000; // Convert ms to ns
                }
                if (map.find("type") != map.end()) {
                    map["type"].convert(msg.type);
                } else {
                    msg.type = "keyframe_update";
                }
                
                // Optional fields
                if (map.find("point_count") != map.end()) {
                    map["point_count"].convert(msg.point_count);
                }
                
                // Notify handler
                if (keyframe_handler) {
                    keyframe_handler(msg);
                }
            }
            
            // Acknowledge message
            channel->ack(deliveryTag);
            
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