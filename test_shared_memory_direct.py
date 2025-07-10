#!/usr/bin/env python3
"""Test shared memory directly to verify IPC works."""

import posix_ipc
import numpy as np
import struct
import mmap
import time

def write_test_keyframe():
    """Write a test keyframe to shared memory."""
    # Create test data
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    pose = np.eye(4, dtype=np.float32)
    bbox = np.array([1.0, 2.0, 3.0, 7.0, 8.0, 9.0], dtype=np.float32)
    
    # Create shared memory name
    shm_name = "/slam3r_keyframe_test"
    
    # Calculate sizes
    header_format = "QII" + "f" * 16 + "f" * 6
    header_size = struct.calcsize(header_format)
    points_size = points.nbytes
    colors_size = colors.nbytes
    total_size = header_size + points_size + colors_size
    
    print(f"Creating shared memory segment: {shm_name}")
    print(f"Total size: {total_size} bytes")
    print(f"Points: {len(points)} x 3")
    
    # Try to unlink if exists
    try:
        posix_ipc.unlink_shared_memory(shm_name)
    except:
        pass
    
    # Create shared memory
    shm = posix_ipc.SharedMemory(shm_name, posix_ipc.O_CREAT | posix_ipc.O_EXCL, size=total_size)
    
    # Map to memory
    mapfile = mmap.mmap(shm.fd, total_size)
    
    # Write header
    header_data = struct.pack(header_format,
        int(time.time() * 1e9),  # timestamp_ns
        len(points),              # point_count
        0,                        # color_format (0=RGB)
        *pose.flatten(),          # pose_matrix
        *bbox                     # bbox
    )
    
    # Write data
    offset = 0
    mapfile[offset:offset+header_size] = header_data
    offset += header_size
    
    mapfile[offset:offset+points_size] = points.tobytes()
    offset += points_size
    
    mapfile[offset:offset+colors_size] = colors.tobytes()
    
    print(f"Wrote keyframe to shared memory")
    print(f"Waiting 30 seconds for mesh service to read it...")
    
    # Keep alive for mesh service to read
    time.sleep(30)
    
    # Cleanup
    mapfile.close()
    shm.close_fd()
    shm.unlink()
    print("Cleaned up shared memory")

if __name__ == "__main__":
    write_test_keyframe()