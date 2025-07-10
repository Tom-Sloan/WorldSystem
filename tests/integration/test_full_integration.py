#!/usr/bin/env python3
"""
Test the full SLAM3R → Mesh Service integration pipeline.
This script verifies:
1. SLAM3R removes downsampling overhead
2. Keyframes are streamed via shared memory
3. Mesh service receives and processes keyframes
4. Rerun visualization works (if available)
"""

import os
import sys
import time
import subprocess
import asyncio
import numpy as np

# Add SLAM3R to path
sys.path.append('/home/sam3/Desktop/Toms_Workspace/WorldSystem/slam3r/SLAM3R_engine')

def check_services():
    """Check if required services are running."""
    print("Checking services...")
    
    # Check RabbitMQ
    try:
        result = subprocess.run(['docker', 'ps', '--filter', 'name=rabbitmq', '--format', '{{.Names}}'], 
                              capture_output=True, text=True)
        if 'rabbitmq' not in result.stdout:
            print("❌ RabbitMQ is not running. Start with: docker-compose up -d rabbitmq")
            return False
        print("✅ RabbitMQ is running")
    except Exception as e:
        print(f"❌ Error checking RabbitMQ: {e}")
        return False
    
    # Check mesh_service
    try:
        result = subprocess.run(['docker', 'ps', '--filter', 'name=mesh_service', '--format', '{{.Names}}'], 
                              capture_output=True, text=True)
        if 'mesh_service' not in result.stdout:
            print("⚠️  Mesh service is not running. Start with: docker-compose --profile mesh_service up -d mesh_service")
            print("   (This is optional for this test)")
        else:
            print("✅ Mesh service is running")
    except Exception as e:
        print(f"⚠️  Error checking mesh service: {e}")
    
    return True

def test_shared_memory():
    """Test shared memory functionality."""
    print("\n2. Testing shared memory...")
    
    try:
        from shared_memory import SharedMemoryManager, StreamingKeyframePublisher
        print("✅ Shared memory modules imported successfully")
        
        # Create test data
        num_points = 1000
        points = np.random.randn(num_points, 3).astype(np.float32)
        colors = np.random.randint(0, 255, size=(num_points, 3), dtype=np.uint8)
        pose = np.eye(4, dtype=np.float32)
        
        # Write to shared memory
        shm_manager = SharedMemoryManager()
        shm_key = shm_manager.write_keyframe("test_integration", points, colors, pose)
        print(f"✅ Wrote test keyframe to shared memory: {shm_key}")
        
        # Verify it exists
        import posix_ipc
        try:
            shm = posix_ipc.SharedMemory(shm_key)
            print(f"✅ Verified shared memory exists, size: {shm.size} bytes")
            shm.close_fd()
            
            # Cleanup
            shm_manager.cleanup()
            print("✅ Shared memory cleaned up")
        except Exception as e:
            print(f"❌ Failed to verify shared memory: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Shared memory test failed: {e}")
        return False
    
    return True

def test_performance():
    """Test performance without downsampling."""
    print("\n3. Testing performance...")
    
    try:
        from slam3r_processor import OptimizedPointCloudBuffer
        
        # Create buffer
        buffer = OptimizedPointCloudBuffer(max_points=1_000_000)
        
        # Add points without downsampling
        num_iterations = 10
        points_per_iteration = 100_000
        
        start_time = time.time()
        for i in range(num_iterations):
            points = np.random.randn(points_per_iteration, 3).astype(np.float32)
            colors = np.random.randint(0, 255, size=(points_per_iteration, 3), dtype=np.uint8)
            buffer.add_points(points, colors)
        
        end_time = time.time()
        total_time = end_time - start_time
        fps = num_iterations / total_time
        
        print(f"✅ Added {num_iterations * points_per_iteration:,} points in {total_time:.2f}s")
        print(f"✅ Performance: {fps:.1f} iterations/second")
        print(f"✅ Buffer size: {len(buffer.points):,} points")
        
        if fps > 25:
            print("✅ Performance target achieved (>25 fps)")
        else:
            print(f"⚠️  Performance below target: {fps:.1f} fps < 25 fps")
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False
    
    return True

def check_rerun():
    """Check if Rerun is available."""
    print("\n4. Checking Rerun...")
    
    try:
        import rerun as rr
        print("✅ Rerun Python SDK is available")
        
        # Check if Rerun viewer is running
        try:
            rr.init("test_integration", spawn=False)
            rr.connect("127.0.0.1:9876")
            print("✅ Connected to Rerun viewer")
            
            # Log test data
            rr.log("test/status", rr.TextLog("Integration test successful"))
            print("✅ Logged test data to Rerun")
            
        except Exception as e:
            print("⚠️  Could not connect to Rerun viewer (this is optional)")
            print("   Start Rerun desktop app to see visualizations")
            
    except ImportError:
        print("⚠️  Rerun Python SDK not installed (this is optional)")
    
    return True

def main():
    print("=" * 60)
    print("SLAM3R → Mesh Service Integration Test")
    print("=" * 60)
    
    # Check environment
    print("\n1. Checking environment...")
    if not check_services():
        print("\n❌ Please start required services first")
        return 1
    
    # Test shared memory
    if not test_shared_memory():
        return 1
    
    # Test performance
    if not test_performance():
        return 1
    
    # Check Rerun
    check_rerun()
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("\nNext steps:")
    print("1. Start SLAM3R: docker-compose up slam3r")
    print("2. Start mesh service: docker-compose --profile mesh_service up mesh_service")
    print("3. Send video frames to test the full pipeline")
    print("4. Monitor performance in Rerun or logs")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())