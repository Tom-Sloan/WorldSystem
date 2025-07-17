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
#include <chrono>
#include <iomanip>
#include <cmath>

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
    auto shm_start = std::chrono::high_resolution_clock::now();
    std::cout << "[SHM DEBUG] Opening shared memory: " << shm_name << std::endl;
    
    // Open shared memory object
    auto open_start = std::chrono::high_resolution_clock::now();
    int fd = shm_open(shm_name.c_str(), O_RDONLY, 0);
    if (fd < 0) {
        std::cerr << "[SHM DEBUG] shm_open failed, errno: " << errno << " (" << strerror(errno) << ")" << std::endl;
        // Return nullptr if shared memory doesn't exist yet
        return nullptr;
    }
    auto open_end = std::chrono::high_resolution_clock::now();
    auto open_us = std::chrono::duration_cast<std::chrono::microseconds>(open_end - open_start).count();
    std::cout << "[SHM DEBUG] shm_open succeeded, fd: " << fd << std::endl;
    std::cout << "[TIMING] shm_open: " << open_us << " µs" << std::endl;
    
    // First, map just the header to read the size
    std::cout << "[SHM DEBUG] Mapping header, size: " << sizeof(SharedKeyframe) << std::endl;
    
    // Debug struct offsets
    std::cout << "[SHM DEBUG] SharedKeyframe struct layout:" << std::endl;
    std::cout << "  offset of timestamp_ns: " << offsetof(SharedKeyframe, timestamp_ns) << std::endl;
    std::cout << "  offset of point_count: " << offsetof(SharedKeyframe, point_count) << std::endl;
    std::cout << "  offset of color_channels: " << offsetof(SharedKeyframe, color_channels) << std::endl;
    std::cout << "  offset of pose_matrix: " << offsetof(SharedKeyframe, pose_matrix) << std::endl;
    std::cout << "  offset of bbox: " << offsetof(SharedKeyframe, bbox) << std::endl;
    auto mmap_header_start = std::chrono::high_resolution_clock::now();
    void* header_ptr = mmap(nullptr, sizeof(SharedKeyframe), 
                           PROT_READ, MAP_SHARED, fd, 0);
    if (header_ptr == MAP_FAILED) {
        std::cerr << "[SHM DEBUG] mmap header failed, errno: " << errno << std::endl;
        close(fd);
        throw std::runtime_error("Failed to map shared memory header");
    }
    auto mmap_header_end = std::chrono::high_resolution_clock::now();
    auto mmap_header_us = std::chrono::duration_cast<std::chrono::microseconds>(mmap_header_end - mmap_header_start).count();
    std::cout << "[TIMING] mmap header: " << mmap_header_us << " µs" << std::endl;
    
    auto* header = static_cast<SharedKeyframe*>(header_ptr);
    std::cout << "[SHM DEBUG] Header mapped, point_count: " << header->point_count 
              << ", color_channels: " << header->color_channels << std::endl;
    std::cout << "[SHM DEBUG] Header timestamp: " << header->timestamp_ns << std::endl;
    
    // DEBUG: Print full pose matrix to trace the issue
    std::cout << "[SHM DEBUG] Full pose matrix from shared memory (row-major):" << std::endl;
    for (int row = 0; row < 4; row++) {
        std::cout << "  [";
        for (int col = 0; col < 4; col++) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) 
                      << header->pose_matrix[row * 4 + col];
            if (col < 3) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "[SHM DEBUG] Camera position (translation) from pose: [" 
              << header->pose_matrix[12] << ", " 
              << header->pose_matrix[13] << ", "
              << header->pose_matrix[14] << "]" << std::endl;
    
    // Check if pose is identity or zero
    bool is_identity = true;
    bool is_zero = true;
    for (int i = 0; i < 16; i++) {
        if (i == 0 || i == 5 || i == 10 || i == 15) {
            if (std::abs(header->pose_matrix[i] - 1.0f) > 1e-6) is_identity = false;
        } else {
            if (std::abs(header->pose_matrix[i]) > 1e-6) is_identity = false;
        }
        if (std::abs(header->pose_matrix[i]) > 1e-6) is_zero = false;
    }
    
    if (is_identity) {
        std::cout << "[SHM WARNING] Pose matrix is identity - no camera transform!" << std::endl;
    } else if (is_zero) {
        std::cout << "[SHM WARNING] Pose matrix is all zeros - invalid!" << std::endl;
    }
    
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
    auto mmap_full_start = std::chrono::high_resolution_clock::now();
    void* full_ptr = mmap(nullptr, total_size, PROT_READ, MAP_SHARED, fd, 0);
    if (full_ptr == MAP_FAILED) {
        std::cerr << "[SHM DEBUG] mmap full failed, errno: " << errno << std::endl;
        close(fd);
        throw std::runtime_error("Failed to map full shared memory");
    }
    auto mmap_full_end = std::chrono::high_resolution_clock::now();
    auto mmap_full_us = std::chrono::duration_cast<std::chrono::microseconds>(mmap_full_end - mmap_full_start).count();
    std::cout << "[SHM DEBUG] Full mapping successful at " << full_ptr << std::endl;
    std::cout << "[TIMING] mmap full: " << mmap_full_us << " µs" << std::endl;
    
    // Store mapping info
    pImpl->mappings[full_ptr] = {full_ptr, total_size, fd};
    
    SharedKeyframe* keyframe = static_cast<SharedKeyframe*>(full_ptr);
    // Note: Can't modify the struct since it's mapped read-only
    // Users should use get_points() and get_colors() methods
    
    auto shm_end = std::chrono::high_resolution_clock::now();
    auto shm_total_us = std::chrono::duration_cast<std::chrono::microseconds>(shm_end - shm_start).count();
    std::cout << "[TIMING] Total SharedMemory open: " << shm_total_us << " µs (" << shm_total_us/1000.0 << " ms)" << std::endl;
    
    return keyframe;
}

void SharedMemoryManager::close_keyframe(SharedKeyframe* keyframe) {
    auto close_start = std::chrono::high_resolution_clock::now();
    auto it = pImpl->mappings.find(keyframe);
    if (it != pImpl->mappings.end()) {
        munmap(it->second.ptr, it->second.size);
        close(it->second.fd);
        pImpl->mappings.erase(it);
    }
    auto close_end = std::chrono::high_resolution_clock::now();
    auto close_us = std::chrono::duration_cast<std::chrono::microseconds>(close_end - close_start).count();
    std::cout << "[TIMING] SharedMemory close: " << close_us << " µs" << std::endl;
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