#!/usr/bin/env python3
"""Test integration between SLAM3R and mesh service via shared memory."""

import numpy as np
import time
import sys
import os

# Add SLAM3R to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../slam3r/SLAM3R_engine'))

from shared_memory import SharedMemoryManager

def test_mesh_service_integration():
    """Test that mesh service can read keyframes from shared memory."""
    
    print("Testing mesh service shared memory integration...")
    
    # Create shared memory manager
    smm = SharedMemoryManager()
    
    # Create test keyframe data
    num_points = 1000
    points = np.random.randn(num_points, 3).astype(np.float32) * 5.0  # Random points within 5m
    colors = np.random.randint(0, 255, (num_points, 3), dtype=np.uint8)
    
    # Create pose matrix (identity)
    pose = np.eye(4, dtype=np.float32)
    
    # Create keyframe
    keyframe_id = f"test_{int(time.time() * 1000)}"
    
    print(f"Writing keyframe {keyframe_id} with {num_points} points to shared memory...")
    
    # Write to shared memory
    shm_key = smm.write_keyframe(
        keyframe_id,
        points,
        colors,
        pose
    )
    
    print(f"Keyframe written to shared memory: {shm_key}")
    print(f"Points shape: {points.shape}")
    print(f"Colors shape: {colors.shape}")
    print(f"Pose:\n{pose}")
    
    # Give mesh service time to process
    print("\nWaiting 5 seconds for mesh service to process...")
    time.sleep(5)
    
    # Check mesh service logs
    print("\nCheck mesh service logs with: docker logs mesh_service_test")
    
    # Cleanup
    smm.cleanup()
    print("\nShared memory cleaned up")

if __name__ == "__main__":
    test_mesh_service_integration()