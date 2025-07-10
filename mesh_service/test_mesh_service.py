#!/usr/bin/env python3
"""Test script to verify mesh_service functionality"""

import posix_ipc
import numpy as np
import struct
import time
import sys

def create_test_keyframe():
    """Create a test shared memory keyframe for mesh_service to process"""
    
    shm_name = "/slam3r_keyframe_test"
    
    # Clean up if exists
    try:
        shm = posix_ipc.SharedMemory(shm_name)
        shm.close_fd()
        shm.unlink()
    except:
        pass
    
    # Create test data
    num_points = 1000
    points = np.random.randn(num_points, 3).astype(np.float32) * 5.0  # Random points in 5m cube
    colors = np.random.randint(0, 255, (num_points, 3), dtype=np.uint8)
    
    # Create pose matrix (identity)
    pose = np.eye(4, dtype=np.float32)
    
    # Calculate bounding box
    bbox = np.array([
        points[:, 0].min(), points[:, 1].min(), points[:, 2].min(),
        points[:, 0].max(), points[:, 1].max(), points[:, 2].max()
    ], dtype=np.float32)
    
    # Calculate total size
    header_format = "QII16f6f"  # timestamp, count, color_format, pose[16], bbox[6]
    header_size = struct.calcsize(header_format)
    points_size = points.nbytes
    colors_size = colors.nbytes
    total_size = header_size + points_size + colors_size
    
    # Create shared memory
    shm = posix_ipc.SharedMemory(shm_name, posix_ipc.O_CREAT | posix_ipc.O_EXCL, size=total_size)
    mapfile = shm.map_file()
    
    # Write header
    header_data = struct.pack(header_format,
        int(time.time() * 1e9),  # timestamp_ns
        num_points,              # point_count
        0,                       # color_format (0=RGB)
        *pose.flatten(),         # pose_matrix (16 floats)
        *bbox                    # bbox (6 floats)
    )
    
    # Write data
    offset = 0
    mapfile[offset:offset+header_size] = header_data
    offset += header_size
    
    mapfile[offset:offset+points_size] = points.tobytes()
    offset += points_size
    
    mapfile[offset:offset+colors_size] = colors.tobytes()
    
    print(f"Created test keyframe: {shm_name}")
    print(f"  - {num_points} points")
    print(f"  - Total size: {total_size} bytes")
    print(f"  - Bounding box: [{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}] to [{bbox[3]:.2f}, {bbox[4]:.2f}, {bbox[5]:.2f}]")
    
    return shm, mapfile

def main():
    print("Creating test keyframe for mesh_service...")
    shm, mapfile = create_test_keyframe()
    
    print("\nTest keyframe created. Mesh service should detect and process it.")
    print("Keep this script running to maintain the shared memory.")
    print("Press Ctrl+C to clean up and exit.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nCleaning up...")
        mapfile.close()
        shm.close_fd()
        shm.unlink()
        print("Done.")

if __name__ == "__main__":
    main()