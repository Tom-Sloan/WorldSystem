#include "normal_provider.h"
#include "normal_providers/camera_based_normal_provider.h"
#ifdef HAS_OPEN3D
#include "normal_providers/open3d_normal_provider.h"
#endif
#include "config/configuration_manager.h"
#include "config/normal_provider_config.h"
#include <iostream>
#include <cstdlib>

namespace mesh_service {

std::unique_ptr<INormalProvider> NormalProviderFactory::create(int provider_id) {
    switch(provider_id) {
        case PROVIDER_CAMERA_BASED:
            std::cout << "[NORMAL PROVIDER] Creating Camera-based provider (fast)" << std::endl;
            return std::make_unique<CameraBasedNormalProvider>();
        
        case PROVIDER_OPEN3D:
#ifdef HAS_OPEN3D
            std::cout << "[NORMAL PROVIDER] Creating Open3D provider (quality)" << std::endl;
            return std::make_unique<Open3DNormalProvider>();
#else
            std::cerr << "[WARNING] Open3D not available, falling back to camera-based" << std::endl;
            return std::make_unique<CameraBasedNormalProvider>();
#endif
        
        case PROVIDER_NANOFLANN:
            std::cerr << "[WARNING] Nanoflann provider not yet implemented, falling back to camera-based" << std::endl;
            return std::make_unique<CameraBasedNormalProvider>();
            
        case PROVIDER_PCL_GPU:
            std::cerr << "[WARNING] PCL GPU provider not yet implemented, falling back to camera-based" << std::endl;
            return std::make_unique<CameraBasedNormalProvider>();
            
        case PROVIDER_GPU_CUSTOM:
            std::cerr << "[WARNING] Custom GPU provider not yet implemented, falling back to camera-based" << std::endl;
            return std::make_unique<CameraBasedNormalProvider>();
        
        default:
            std::cerr << "[WARNING] Unknown provider " << provider_id << ", using default camera-based" << std::endl;
            return std::make_unique<CameraBasedNormalProvider>();
    }
}

std::unique_ptr<INormalProvider> NormalProviderFactory::createFromEnv() {
    int provider_id = CONFIG_INT("MESH_NORMAL_PROVIDER", 
                                 config::NormalProviderConfig::DEFAULT_NORMAL_PROVIDER);
    
    std::cout << "[NORMAL PROVIDER] Creating provider from environment: " 
              << provider_id << " (" << getProviderTypeName(provider_id) << ")" << std::endl;
    
    return create(provider_id);
}

} // namespace mesh_service