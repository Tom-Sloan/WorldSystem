#pragma once

#include <memory>

namespace mesh_service {

class Metrics {
public:
    static Metrics& instance();
    
    void recordMeshGeneration(size_t vertex_count, size_t face_count);
    void recordProcessingTime(double seconds);
    void recordMemoryUsage(size_t bytes);
    
private:
    Metrics();
    ~Metrics();
    Metrics(const Metrics&) = delete;
    Metrics& operator=(const Metrics&) = delete;
    
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

class MetricsServer {
public:
    MetricsServer(int port);
    ~MetricsServer();
    
    void run();
    void stop();
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace mesh_service