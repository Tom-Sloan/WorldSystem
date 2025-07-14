#include "shared_memory.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <cerrno>
#include <iostream>
#include <cstddef>

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
    std::cout << "[SHM DEBUG] Opening shared memory: " << shm_name << std::endl;
    
    // Open shared memory object
    int fd = shm_open(shm_name.c_str(), O_RDONLY, 0);
    if (fd < 0) {
        std::cerr << "[SHM DEBUG] shm_open failed, errno: " << errno << " (" << strerror(errno) << ")" << std::endl;
        // Return nullptr if shared memory doesn't exist yet
        return nullptr;
    }
    std::cout << "[SHM DEBUG] shm_open succeeded, fd: " << fd << std::endl;
    
    // First, map just the header to read the size
    std::cout << "[SHM DEBUG] Mapping header, size: " << sizeof(SharedKeyframe) << std::endl;
    
    // Debug struct offsets
    std::cout << "[SHM DEBUG] SharedKeyframe struct layout:" << std::endl;
    std::cout << "  offset of timestamp_ns: " << offsetof(SharedKeyframe, timestamp_ns) << std::endl;
    std::cout << "  offset of point_count: " << offsetof(SharedKeyframe, point_count) << std::endl;
    std::cout << "  offset of color_channels: " << offsetof(SharedKeyframe, color_channels) << std::endl;
    std::cout << "  offset of pose_matrix: " << offsetof(SharedKeyframe, pose_matrix) << std::endl;
    std::cout << "  offset of bbox: " << offsetof(SharedKeyframe, bbox) << std::endl;
    void* header_ptr = mmap(nullptr, sizeof(SharedKeyframe), 
                           PROT_READ, MAP_SHARED, fd, 0);
    if (header_ptr == MAP_FAILED) {
        std::cerr << "[SHM DEBUG] mmap header failed, errno: " << errno << std::endl;
        close(fd);
        throw std::runtime_error("Failed to map shared memory header");
    }
    
    auto* header = static_cast<SharedKeyframe*>(header_ptr);
    std::cout << "[SHM DEBUG] Header mapped, point_count: " << header->point_count 
              << ", color_channels: " << header->color_channels << std::endl;
    std::cout << "[SHM DEBUG] Header timestamp: " << header->timestamp_ns << std::endl;
    std::cout << "[SHM DEBUG] Header bbox: [" 
              << header->bbox[0] << ", " << header->bbox[1] << ", " << header->bbox[2] << "] to ["
              << header->bbox[3] << ", " << header->bbox[4] << ", " << header->bbox[5] << "]" << std::endl;
    
    // Calculate total size including point data
    size_t total_size = calculate_size(header->point_count, 
                                      header->color_channels);
    std::cout << "[SHM DEBUG] Calculated total size: " << total_size << std::endl;
    
    // Remap with full size
    munmap(header_ptr, sizeof(SharedKeyframe));
    
    std::cout << "[SHM DEBUG] Remapping with full size" << std::endl;
    void* full_ptr = mmap(nullptr, total_size, PROT_READ, MAP_SHARED, fd, 0);
    if (full_ptr == MAP_FAILED) {
        std::cerr << "[SHM DEBUG] mmap full failed, errno: " << errno << std::endl;
        close(fd);
        throw std::runtime_error("Failed to map full shared memory");
    }
    std::cout << "[SHM DEBUG] Full mapping successful at " << full_ptr << std::endl;
    
    // Store mapping info
    pImpl->mappings[full_ptr] = {full_ptr, total_size, fd};
    
    SharedKeyframe* keyframe = static_cast<SharedKeyframe*>(full_ptr);
    // Note: Can't modify the struct since it's mapped read-only
    // Users should use get_points() and get_colors() methods
    
    return keyframe;
}

void SharedMemoryManager::close_keyframe(SharedKeyframe* keyframe) {
    auto it = pImpl->mappings.find(keyframe);
    if (it != pImpl->mappings.end()) {
        munmap(it->second.ptr, it->second.size);
        close(it->second.fd);
        pImpl->mappings.erase(it);
    }
}

void SharedMemoryManager::unlink_keyframe(const std::string& shm_name) {
    // Unlink the shared memory segment to remove it from the system
    shm_unlink(shm_name.c_str());
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