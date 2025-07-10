#include "websocket_server.h"
#include <iostream>

namespace mesh_service {

class WebSocketServer::Impl {
public:
    int port;
    bool running = false;
    
    Impl(int p) : port(p) {}
};

WebSocketServer::WebSocketServer(int port) : pImpl(std::make_unique<Impl>(port)) {}
WebSocketServer::~WebSocketServer() = default;

void WebSocketServer::run() {
    pImpl->running = true;
    std::cout << "WebSocket server running on port " << pImpl->port << std::endl;
    // TODO: Implement WebSocket server
}

void WebSocketServer::stop() {
    pImpl->running = false;
}

void WebSocketServer::streamMeshUpdate(const MeshUpdate& update) {
    // TODO: Implement mesh streaming
    std::cout << "Streaming mesh update: " << update.vertices.size()/3 
              << " vertices, " << update.faces.size()/3 << " faces" << std::endl;
}

} // namespace mesh_service