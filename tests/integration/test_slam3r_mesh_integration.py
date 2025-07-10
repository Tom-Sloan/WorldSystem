#!/usr/bin/env python3
"""
Test script to verify SLAM3R → Mesh Service integration.
Tests shared memory IPC and RabbitMQ notification flow.
"""

import asyncio
import numpy as np
import sys
import os
import time

# Add SLAM3R to path
slam3r_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../slam3r/SLAM3R_engine'))
sys.path.append(slam3r_path)

# Activate conda environment
os.environ['CONDA_DEFAULT_ENV'] = '3dreconstruction'

async def test_integration():
    print("Testing SLAM3R → Mesh Service Integration")
    print("=" * 50)
    
    # Test 1: Import shared memory module
    print("\n1. Testing shared memory import...")
    try:
        from shared_memory import SharedMemoryManager, StreamingKeyframePublisher
        print("✓ Shared memory modules imported successfully")
    except Exception as e:
        print(f"✗ Failed to import: {e}")
        return
    
    # Test 2: Create test data
    print("\n2. Creating test keyframe data...")
    num_points = 10000
    points = np.random.randn(num_points, 3).astype(np.float32) * 10
    colors = np.random.randint(0, 255, size=(num_points, 3), dtype=np.uint8)
    pose = np.eye(4, dtype=np.float32)
    keyframe_id = f"test_keyframe_{int(time.time())}"
    print(f"✓ Created {num_points} points with ID: {keyframe_id}")
    
    # Test 3: Write to shared memory
    print("\n3. Testing shared memory write...")
    shm_manager = SharedMemoryManager()
    try:
        shm_key = shm_manager.write_keyframe(keyframe_id, points, colors, pose)
        print(f"✓ Wrote keyframe to shared memory: {shm_key}")
        
        # Verify we can read it back
        import posix_ipc
        shm = posix_ipc.SharedMemory(shm_key)
        print(f"✓ Verified shared memory exists, size: {shm.size} bytes")
    except Exception as e:
        print(f"✗ Failed to write shared memory: {e}")
        return
    
    # Test 4: Test RabbitMQ connection (optional)
    print("\n4. Testing RabbitMQ connection...")
    try:
        import aio_pika
        connection = await aio_pika.connect_robust("amqp://localhost:5672", timeout=5)
        await connection.close()
        print("✓ RabbitMQ connection successful")
        
        # Test with publisher
        print("\n5. Testing StreamingKeyframePublisher...")
        # Note: This won't work without proper exchange setup
        publisher = StreamingKeyframePublisher(keyframe_exchange=None)
        print("✓ Publisher created (exchange not connected)")
        
    except Exception as e:
        print(f"⚠ RabbitMQ not available: {e}")
        print("  (This is OK if RabbitMQ isn't running)")
    
    # Test 5: Cleanup
    print("\n6. Cleaning up...")
    try:
        shm_manager.cleanup()  # No parameter needed
        print("✓ Shared memory cleaned up")
    except Exception as e:
        print(f"⚠ Cleanup warning: {e}")
    
    print("\n" + "=" * 50)
    print("Integration test complete!")
    print("\nNext steps:")
    print("1. Start mesh_service: docker-compose --profile mesh_service up mesh_service")
    print("2. Start SLAM3R with streaming enabled")
    print("3. Monitor shared memory: ls -la /dev/shm/slam3r_*")

if __name__ == "__main__":
    asyncio.run(test_integration())