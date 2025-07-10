#!/usr/bin/env python3
"""Test full integration between SLAM3R and mesh service with RabbitMQ."""

import numpy as np
import time
import sys
import os
import asyncio
import aio_pika
import msgpack

# Add SLAM3R to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../slam3r/SLAM3R_engine'))

from shared_memory import SharedMemoryManager

async def test_full_integration():
    """Test complete mesh service integration with shared memory and RabbitMQ."""
    
    print("Testing full mesh service integration...")
    
    # Connect to RabbitMQ
    connection = await aio_pika.connect_robust("amqp://localhost")
    channel = await connection.channel()
    
    # Declare keyframe exchange
    keyframe_exchange = await channel.declare_exchange(
        "slam3r_keyframe_exchange",
        aio_pika.ExchangeType.TOPIC,
        durable=True
    )
    
    # Create shared memory manager
    smm = SharedMemoryManager()
    
    # Create test keyframe data
    num_points = 1000
    points = np.random.randn(num_points, 3).astype(np.float32) * 5.0
    colors = np.random.randint(0, 255, (num_points, 3), dtype=np.uint8)
    pose = np.eye(4, dtype=np.float32)
    
    keyframe_id = f"test_{int(time.time() * 1000)}"
    timestamp_ns = time.time_ns()
    
    print(f"\nWriting keyframe {keyframe_id} with {num_points} points...")
    
    # Write to shared memory
    shm_key = smm.write_keyframe(keyframe_id, points, colors, pose)
    
    # Calculate bounding box
    bbox = [
        float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 2].min()),
        float(points[:, 0].max()), float(points[:, 1].max()), float(points[:, 2].max())
    ]
    
    # Create RabbitMQ message
    msg = {
        'type': 'keyframe.new',
        'keyframe_id': keyframe_id,
        'timestamp_ns': timestamp_ns,
        'pose_matrix': pose.tolist(),
        'shm_key': shm_key,
        'point_count': len(points),
        'bbox': bbox
    }
    
    print(f"Publishing keyframe notification to RabbitMQ...")
    print(f"  SHM Key: {shm_key}")
    print(f"  Points: {num_points}")
    print(f"  BBox: [{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}] to [{bbox[3]:.2f}, {bbox[4]:.2f}, {bbox[5]:.2f}]")
    
    # Publish to RabbitMQ
    await keyframe_exchange.publish(
        aio_pika.Message(body=msgpack.packb(msg)),
        routing_key='keyframe.new'
    )
    
    print("\nWaiting 5 seconds for mesh service to process...")
    await asyncio.sleep(5)
    
    # Cleanup
    await connection.close()
    smm.cleanup()
    
    print("\nTest complete! Check mesh service logs with: docker logs mesh_service_test")
    print("Look for 'Simple mesh generation completed' message")

if __name__ == "__main__":
    asyncio.run(test_full_integration())