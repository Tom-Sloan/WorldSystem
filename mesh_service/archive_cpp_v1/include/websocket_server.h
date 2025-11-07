#pragma once

#include <memory>
#include <string>
#include "mesh_generator.h"

namespace mesh_service {

class WebSocketServer {
public:
    WebSocketServer(int port);
    ~WebSocketServer();
    
    void run();
    void stop();
    void streamMeshUpdate(const MeshUpdate& update);
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace mesh_service