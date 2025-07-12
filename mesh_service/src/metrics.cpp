#include "metrics.h"
#include <iostream>
#include <atomic>
#include <chrono>
#include <mutex>
#include <deque>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <thread>
#include <algorithm>

#ifdef __linux__
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#endif

namespace mesh_service {

// Metrics implementation
class Metrics::Impl {
public:
    // Counters
    std::atomic<size_t> total_vertices{0};
    std::atomic<size_t> total_faces{0};
    std::atomic<size_t> mesh_count{0};
    std::atomic<size_t> keyframes_processed{0};
    std::atomic<size_t> errors{0};
    
    // Timing metrics (keep last N samples)
    std::mutex timing_mutex;
    std::deque<double> processing_times;
    static constexpr size_t MAX_SAMPLES = 100;
    
    // Memory metrics
    std::atomic<size_t> current_memory_usage{0};
    std::atomic<size_t> peak_memory_usage{0};
    
    // Start time for uptime calculation
    std::chrono::steady_clock::time_point start_time;
    
    Impl() : start_time(std::chrono::steady_clock::now()) {}
    
    void addProcessingTime(double seconds) {
        std::lock_guard<std::mutex> lock(timing_mutex);
        processing_times.push_back(seconds);
        if (processing_times.size() > MAX_SAMPLES) {
            processing_times.pop_front();
        }
    }
    
    double getAverageProcessingTime() const {
        std::lock_guard<std::mutex> lock(timing_mutex);
        if (processing_times.empty()) return 0.0;
        return std::accumulate(processing_times.begin(), processing_times.end(), 0.0) / processing_times.size();
    }
    
    double getMinProcessingTime() const {
        std::lock_guard<std::mutex> lock(timing_mutex);
        if (processing_times.empty()) return 0.0;
        return *std::min_element(processing_times.begin(), processing_times.end());
    }
    
    double getMaxProcessingTime() const {
        std::lock_guard<std::mutex> lock(timing_mutex);
        if (processing_times.empty()) return 0.0;
        return *std::max_element(processing_times.begin(), processing_times.end());
    }
    
    double getUptime() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - start_time).count();
    }
    
    std::string getPrometheusMetrics() const {
        std::ostringstream oss;
        
        // Add HELP and TYPE annotations
        oss << "# HELP mesh_service_uptime_seconds Time since service started\n";
        oss << "# TYPE mesh_service_uptime_seconds gauge\n";
        oss << "mesh_service_uptime_seconds " << std::fixed << std::setprecision(3) << getUptime() << "\n\n";
        
        oss << "# HELP mesh_service_meshes_total Total number of meshes generated\n";
        oss << "# TYPE mesh_service_meshes_total counter\n";
        oss << "mesh_service_meshes_total " << mesh_count.load() << "\n\n";
        
        oss << "# HELP mesh_service_vertices_total Total number of vertices generated\n";
        oss << "# TYPE mesh_service_vertices_total counter\n";
        oss << "mesh_service_vertices_total " << total_vertices.load() << "\n\n";
        
        oss << "# HELP mesh_service_faces_total Total number of faces generated\n";
        oss << "# TYPE mesh_service_faces_total counter\n";
        oss << "mesh_service_faces_total " << total_faces.load() << "\n\n";
        
        oss << "# HELP mesh_service_keyframes_processed_total Total keyframes processed\n";
        oss << "# TYPE mesh_service_keyframes_processed_total counter\n";
        oss << "mesh_service_keyframes_processed_total " << keyframes_processed.load() << "\n\n";
        
        oss << "# HELP mesh_service_errors_total Total number of errors\n";
        oss << "# TYPE mesh_service_errors_total counter\n";
        oss << "mesh_service_errors_total " << errors.load() << "\n\n";
        
        oss << "# HELP mesh_service_processing_time_seconds Mesh generation processing time\n";
        oss << "# TYPE mesh_service_processing_time_seconds summary\n";
        oss << "mesh_service_processing_time_seconds{quantile=\"0.0\"} " << getMinProcessingTime() << "\n";
        oss << "mesh_service_processing_time_seconds{quantile=\"0.5\"} " << getAverageProcessingTime() << "\n";
        oss << "mesh_service_processing_time_seconds{quantile=\"1.0\"} " << getMaxProcessingTime() << "\n";
        oss << "mesh_service_processing_time_seconds_count " << processing_times.size() << "\n\n";
        
        oss << "# HELP mesh_service_memory_bytes Current memory usage\n";
        oss << "# TYPE mesh_service_memory_bytes gauge\n";
        oss << "mesh_service_memory_bytes " << current_memory_usage.load() << "\n\n";
        
        oss << "# HELP mesh_service_memory_peak_bytes Peak memory usage\n";
        oss << "# TYPE mesh_service_memory_peak_bytes gauge\n";
        oss << "mesh_service_memory_peak_bytes " << peak_memory_usage.load() << "\n";
        
        return oss.str();
    }
};

Metrics::Metrics() : pImpl(std::make_unique<Impl>()) {}
Metrics::~Metrics() = default;

Metrics& Metrics::instance() {
    static Metrics instance;
    return instance;
}

void Metrics::recordMeshGeneration(size_t vertex_count, size_t face_count) {
    pImpl->total_vertices += vertex_count;
    pImpl->total_faces += face_count;
    pImpl->mesh_count++;
    pImpl->keyframes_processed++;
}

void Metrics::recordProcessingTime(double seconds) {
    pImpl->addProcessingTime(seconds);
}

void Metrics::recordMemoryUsage(size_t bytes) {
    pImpl->current_memory_usage = bytes;
    
    // Update peak if necessary
    size_t current_peak = pImpl->peak_memory_usage.load();
    while (bytes > current_peak && 
           !pImpl->peak_memory_usage.compare_exchange_weak(current_peak, bytes)) {
        // Loop until successful
    }
}

// Add method to get Prometheus metrics string
std::string Metrics::getPrometheusMetrics() const {
    return pImpl->getPrometheusMetrics();
}

void Metrics::recordError() {
    pImpl->errors++;
}

// MetricsServer implementation
class MetricsServer::Impl {
public:
    int port;
    bool running = false;
    std::thread server_thread;
    int server_socket = -1;
    
    Impl(int p) : port(p) {}
    
    ~Impl() {
        stop();
    }
    
    void run() {
        running = true;
        server_thread = std::thread([this]() {
            runServer();
        });
    }
    
    void stop() {
        running = false;
        if (server_socket >= 0) {
            close(server_socket);
            server_socket = -1;
        }
        if (server_thread.joinable()) {
            server_thread.join();
        }
    }
    
private:
    void runServer() {
        #ifdef __linux__
        
        // Create socket
        server_socket = socket(AF_INET, SOCK_STREAM, 0);
        if (server_socket < 0) {
            std::cerr << "Failed to create socket for metrics server" << std::endl;
            return;
        }
        
        // Allow socket reuse
        int opt = 1;
        if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
            std::cerr << "Failed to set socket options" << std::endl;
            close(server_socket);
            return;
        }
        
        // Bind to port
        struct sockaddr_in address;
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(port);
        
        if (bind(server_socket, (struct sockaddr*)&address, sizeof(address)) < 0) {
            std::cerr << "Failed to bind to port " << port << std::endl;
            close(server_socket);
            return;
        }
        
        // Listen for connections
        if (listen(server_socket, 3) < 0) {
            std::cerr << "Failed to listen on socket" << std::endl;
            close(server_socket);
            return;
        }
        
        std::cout << "Metrics server listening on port " << port << std::endl;
        
        while (running) {
            // Accept connection
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            int client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_len);
            
            if (client_socket < 0) {
                if (running) {
                    std::cerr << "Failed to accept connection" << std::endl;
                }
                continue;
            }
            
            // Handle request
            handleRequest(client_socket);
            close(client_socket);
        }
        #else
        std::cerr << "Metrics server not implemented for this platform" << std::endl;
        #endif
    }
    
    void handleRequest(int client_socket) {
        // Read request (we don't parse it, just send metrics for any request)
        char buffer[1024] = {0};
        read(client_socket, buffer, sizeof(buffer) - 1);
        
        // Get metrics
        std::string metrics = Metrics::instance().getPrometheusMetrics();
        
        // Send HTTP response
        std::ostringstream response;
        response << "HTTP/1.1 200 OK\r\n";
        response << "Content-Type: text/plain; version=0.0.4\r\n";
        response << "Content-Length: " << metrics.length() << "\r\n";
        response << "\r\n";
        response << metrics;
        
        send(client_socket, response.str().c_str(), response.str().length(), 0);
    }
};

MetricsServer::MetricsServer(int port) : pImpl(std::make_unique<Impl>(port)) {}
MetricsServer::~MetricsServer() = default;

void MetricsServer::run() {
    pImpl->run();
}

void MetricsServer::stop() {
    pImpl->stop();
}

} // namespace mesh_service