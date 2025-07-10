#include "shared_memory.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace mesh_service {

struct SharedMemoryManager::Impl {
    struct MappedRegion {
        void* ptr;
        size_t size;
        int fd;
    };
    
    std::unordered_map<void*, MappedRegion> mappings;
};

SharedMemoryManager::SharedMemoryManager() : pImpl(std::make_unique<Impl>()) {}

SharedMemoryManager::~SharedMemoryManager() {
    // Clean up any remaining mappings
    for (auto& [ptr, region] : pImpl->mappings) {
        munmap(region.ptr, region.size);
        if (region.fd >= 0) {
            close(region.fd);
        }
    }
}

SharedKeyframe* SharedMemoryManager::open_keyframe(const std::string& shm_name) {
    // Open shared memory object
    int fd = shm_open(shm_name.c_str(), O_RDONLY, 0);
    if (fd < 0) {
        // Return nullptr if shared memory doesn't exist yet
        return nullptr;
    }
    
    // First, map just the header to read the size
    void* header_ptr = mmap(nullptr, sizeof(SharedKeyframe), 
                           PROT_READ, MAP_SHARED, fd, 0);
    if (header_ptr == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("Failed to map shared memory header");
    }
    
    auto* header = static_cast<SharedKeyframe*>(header_ptr);
    
    // Calculate total size including point data
    size_t total_size = calculate_size(header->point_count, 
                                      header->color_format == 1 ? 4 : 3);
    
    // Remap with full size
    munmap(header_ptr, sizeof(SharedKeyframe));
    
    void* full_ptr = mmap(nullptr, total_size, PROT_READ, MAP_SHARED, fd, 0);
    if (full_ptr == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("Failed to map full shared memory");
    }
    
    // Store mapping info
    pImpl->mappings[full_ptr] = {full_ptr, total_size, fd};
    
    return static_cast<SharedKeyframe*>(full_ptr);
}

void SharedMemoryManager::close_keyframe(SharedKeyframe* keyframe) {
    auto it = pImpl->mappings.find(keyframe);
    if (it != pImpl->mappings.end()) {
        munmap(it->second.ptr, it->second.size);
        close(it->second.fd);
        pImpl->mappings.erase(it);
    }
}

float* SharedMemoryManager::get_points(SharedKeyframe* keyframe) {
    // Points data starts immediately after the header
    return reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(keyframe) + sizeof(SharedKeyframe)
    );
}

uint8_t* SharedMemoryManager::get_colors(SharedKeyframe* keyframe) {
    // Colors data starts after points
    size_t points_size = keyframe->point_count * 3 * sizeof(float);
    return reinterpret_cast<uint8_t*>(
        reinterpret_cast<uint8_t*>(keyframe) + sizeof(SharedKeyframe) + points_size
    );
}

size_t SharedMemoryManager::calculate_size(uint32_t point_count, uint32_t color_channels) {
    return sizeof(SharedKeyframe) + 
           (point_count * 3 * sizeof(float)) +     // Points
           (point_count * color_channels);         // Colors
}

} // namespace mesh_service