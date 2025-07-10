#pragma once

#include <memory>
#include <string>
#include <functional>

namespace mesh_service {

struct KeyframeMessage {
    std::string type;
    std::string keyframe_id;
    uint64_t timestamp_ns;
    float pose_matrix[16];
    std::string shm_key;
    uint32_t point_count;
    float bbox[6];
};

class RabbitMQConsumer {
public:
    RabbitMQConsumer(const std::string& url);
    ~RabbitMQConsumer();
    
    void connect();
    void start();
    void stop();
    
    void setKeyframeHandler(std::function<void(const KeyframeMessage&)> handler);
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace mesh_service