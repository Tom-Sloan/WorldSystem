#include "rabbitmq_consumer.h"
#include <iostream>
#include <thread>

namespace mesh_service {

class RabbitMQConsumer::Impl {
public:
    std::string url;
    std::function<void(const KeyframeMessage&)> keyframe_handler;
    bool connected = false;
    bool running = false;
    
    Impl(const std::string& u) : url(u) {}
};

RabbitMQConsumer::RabbitMQConsumer(const std::string& url) 
    : pImpl(std::make_unique<Impl>(url)) {}
RabbitMQConsumer::~RabbitMQConsumer() = default;

void RabbitMQConsumer::connect() {
    // TODO: Implement RabbitMQ connection
    pImpl->connected = true;
    std::cout << "Connected to RabbitMQ at " << pImpl->url << std::endl;
}

void RabbitMQConsumer::start() {
    pImpl->running = true;
    // TODO: Start consuming messages
    std::cout << "RabbitMQ consumer started" << std::endl;
}

void RabbitMQConsumer::stop() {
    pImpl->running = false;
}

void RabbitMQConsumer::setKeyframeHandler(std::function<void(const KeyframeMessage&)> handler) {
    pImpl->keyframe_handler = handler;
}

} // namespace mesh_service