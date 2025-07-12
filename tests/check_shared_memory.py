#!/usr/bin/env python3
"""Check if SLAM3R is writing to shared memory."""

import posix_ipc
import time
import struct
import mmap

def check_shared_memory():
    print("Checking for SLAM3R keyframes in shared memory...")
    
    found_any = False
    for i in range(20):
        shm_name = f"/slam3r_keyframe_{i}"
        try:
            # Try to open existing shared memory
            shm = posix_ipc.SharedMemory(shm_name)
            print(f"\nFound shared memory segment: {shm_name}")
            
            # Get the header to read point count
            header_format = "QII" + "f" * 16 + "f" * 6
            header_size = struct.calcsize(header_format)
            
            # Map just the header
            mapfile = mmap.mmap(shm.fd, header_size, access=mmap.ACCESS_READ)
            header_data = mapfile[:header_size]
            
            # Unpack header
            unpacked = struct.unpack(header_format, header_data)
            timestamp_ns = unpacked[0]
            point_count = unpacked[1]
            color_format = unpacked[2]
            
            print(f"  - Timestamp: {timestamp_ns}")
            print(f"  - Point count: {point_count}")
            print(f"  - Color format: {color_format}")
            
            mapfile.close()
            shm.close_fd()
            found_any = True
            
        except posix_ipc.ExistentialError:
            # Segment doesn't exist
            pass
        except Exception as e:
            print(f"Error reading {shm_name}: {e}")
    
    if not found_any:
        print("\nNo shared memory segments found!")
        print("Possible issues:")
        print("1. SLAM3R hasn't published any keyframes yet")
        print("2. SharedMemoryPublisher is not properly initialized")
        print("3. Keyframes were already cleaned up")
    
    # Also check for bootstrap keyframes
    print("\nChecking for bootstrap keyframes...")
    for i in range(10):
        shm_name = f"/slam3r_keyframe_bootstrap_{i}"
        try:
            shm = posix_ipc.SharedMemory(shm_name)
            print(f"Found bootstrap keyframe: {shm_name}")
            shm.close_fd()
        except:
            pass

if __name__ == "__main__":
    check_shared_memory()