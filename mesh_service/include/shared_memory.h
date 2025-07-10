#pragma once

#include <cstdint>
#include <string>
#include <memory>

namespace mesh_service {

// Shared memory structure for keyframe data
struct SharedKeyframe {
    uint64_t timestamp_ns;
    uint32_t point_count;
    uint32_t color_format;  // 0=RGB, 1=RGBA
    float pose_matrix[16];  // Row-major 4x4
    float bbox[6];          // min_x, min_y, min_z, max_x, max_y, max_z
    
    // Variable length data follows in memory:
    // float points[point_count * 3];
    // uint8_t colors[point_count * 3 or 4];
};

class SharedMemoryManager {
public:
    SharedMemoryManager();
    ~SharedMemoryManager();
    
    // Open existing shared memory segment for reading
    SharedKeyframe* open_keyframe(const std::string& shm_name);
    
    // Close shared memory segment
    void close_keyframe(SharedKeyframe* keyframe);
    
    // Get points data pointer (after header)
    float* get_points(SharedKeyframe* keyframe);
    
    // Get colors data pointer (after points)
    uint8_t* get_colors(SharedKeyframe* keyframe);
    
    // Calculate total size needed for a keyframe
    static size_t calculate_size(uint32_t point_count, uint32_t color_channels);
    
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace mesh_service