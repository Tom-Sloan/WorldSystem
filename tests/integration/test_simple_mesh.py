#!/usr/bin/env python3
"""Simple test for mesh service with fixed keyframe name."""

import numpy as np
import time
import sys
import os

# Add SLAM3R to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../slam3r/SLAM3R_engine'))

from shared_memory import SharedMemoryManager

def test_simple_mesh():
    """Test mesh service with a simple fixed keyframe."""
    
    print("Testing simple mesh generation...")
    
    # Create shared memory manager
    smm = SharedMemoryManager()
    
    # Create test keyframe data
    num_points = 1000
    points = np.random.randn(num_points, 3).astype(np.float32) * 5.0
    colors = np.random.randint(0, 255, (num_points, 3), dtype=np.uint8)
    pose = np.eye(4, dtype=np.float32)
    
    # Use a fixed keyframe ID that matches what mesh service expects
    keyframe_id = "test"  # This will create /slam3r_keyframe_test
    
    print(f"Writing keyframe '{keyframe_id}' with {num_points} points to shared memory...")
    
    # Write to shared memory
    shm_key = smm.write_keyframe(keyframe_id, points, colors, pose)
    
    print(f"Keyframe written to shared memory: {shm_key}")
    print(f"Points shape: {points.shape}")
    print(f"Colors shape: {colors.shape}")
    
    # Give mesh service time to process
    print("\nWaiting 3 seconds for mesh service to process...")
    time.sleep(3)
    
    print("\nCheck mesh service logs with: docker logs mesh_service_test")
    print("The keyframe will remain in shared memory for the mesh service to find.")
    
    # Don't cleanup immediately so mesh service can find it
    print("\nKeyframe left in shared memory. Clean up with: smm.cleanup()")

if __name__ == "__main__":
    test_simple_mesh()