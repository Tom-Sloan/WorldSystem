#!/usr/bin/env python3
"""
Test script to verify mesh_service Rerun connectivity.
Creates a simple mesh and publishes it via RabbitMQ to trigger mesh_service visualization.
"""

import numpy as np
import time
import struct
import os
import sys
import msgpack
import pika
from multiprocessing import shared_memory

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_test_cube_points():
    """Create a simple cube point cloud for testing."""
    # Define cube vertices
    cube_vertices = np.array([
        # Front face
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        # Back face  
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        # Additional points for density
        [0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0],
        [1, 0, 0], [-1, 0, 0]
    ], dtype=np.float32)
    
    # Scale and translate
    cube_vertices *= 0.5
    cube_vertices[:, 2] += 2.0  # Move forward in Z
    
    # Generate more points by subdividing faces
    points = []
    colors = []
    
    # Add original vertices
    for v in cube_vertices:
        points.append(v)
        # Color based on position
        color = ((v + 1) * 127.5).astype(np.uint8)
        colors.append(color)
    
    # Add interpolated points on edges
    for i in range(len(cube_vertices)):
        for j in range(i + 1, len(cube_vertices)):
            # Check if vertices share an edge (differ in only one coordinate)
            diff = np.abs(cube_vertices[i] - cube_vertices[j])
            if np.sum(diff > 0.1) == 1:
                # Interpolate points along edge
                for t in np.linspace(0.2, 0.8, 3):
                    p = cube_vertices[i] * (1 - t) + cube_vertices[j] * t
                    points.append(p)
                    color = ((p + 1) * 127.5).astype(np.uint8)
                    colors.append(color)
    
    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)
    
    return points, colors


def create_shared_memory_keyframe(points, colors, shm_name):
    """Create a shared memory segment with keyframe data."""
    # SharedKeyframe structure from C++
    header_size = 128  # Fixed header size
    point_data_size = points.shape[0] * 3 * 4  # float32
    color_data_size = points.shape[0] * 3  # uint8
    total_size = header_size + point_data_size + color_data_size
    
    # Create shared memory
    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=total_size)
    
    try:
        # Write header
        header_data = struct.pack(
            '<Q',  # timestamp_ns (uint64)
            int(time.time() * 1e9)
        )
        header_data += struct.pack('<I', points.shape[0])  # point_count
        header_data += struct.pack('<I', 3)  # color_channels (RGB)
        
        # Pose matrix (4x4 identity)
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = 0.0  # X translation
        pose[1, 3] = 0.0  # Y translation  
        pose[2, 3] = 0.0  # Z translation
        header_data += pose.tobytes()
        
        # Bounding box
        bbox = np.array([
            points[:, 0].min(), points[:, 1].min(), points[:, 2].min(),
            points[:, 0].max(), points[:, 1].max(), points[:, 2].max()
        ], dtype=np.float32)
        header_data += bbox.tobytes()
        
        # Pad header to 128 bytes
        header_data += b'\x00' * (header_size - len(header_data))
        
        # Write to shared memory
        shm.buf[:header_size] = header_data
        
        # Write point data (already in x,y,z format)
        points_flat = points.flatten()
        shm.buf[header_size:header_size + point_data_size] = points_flat.tobytes()
        
        # Write color data  
        colors_flat = colors.flatten()
        shm.buf[header_size + point_data_size:] = colors_flat.tobytes()
        
        print(f"Created shared memory segment '{shm_name}' with {points.shape[0]} points")
        
        return shm
        
    except Exception as e:
        shm.close()
        shm.unlink()
        raise e


def publish_keyframe_message(shm_name, keyframe_id, point_count):
    """Publish keyframe message to RabbitMQ."""
    try:
        # Connect to RabbitMQ
        connection = pika.BlockingConnection(
            pika.ConnectionParameters('localhost')
        )
        channel = connection.channel()
        
        # Declare exchange
        exchange = os.environ.get('SLAM3R_KEYFRAME_EXCHANGE', 'slam3r_keyframe_exchange')
        channel.exchange_declare(exchange=exchange, exchange_type='fanout', durable=True)
        
        # Create message
        message = {
            'shm_key': shm_name,
            'keyframe_id': keyframe_id,
            'timestamp': time.time(),
            'type': 'keyframe',
            'point_count': point_count
        }
        
        # Publish
        channel.basic_publish(
            exchange=exchange,
            routing_key='',
            body=msgpack.packb(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
            )
        )
        
        print(f"Published keyframe message: {message}")
        
        connection.close()
        return True
        
    except Exception as e:
        print(f"Failed to publish message: {e}")
        return False


def main():
    """Main test function."""
    print("Mesh Service Rerun Test")
    print("=======================")
    
    # Generate test data
    points, colors = create_test_cube_points()
    print(f"Generated {points.shape[0]} test points")
    
    # Create unique shared memory name
    timestamp = int(time.time() * 1000)
    shm_name = f"mesh_test_{timestamp}"
    keyframe_id = f"test_keyframe_{timestamp}"
    
    # Create shared memory segment
    shm = None
    try:
        shm = create_shared_memory_keyframe(points, colors, shm_name)
        
        # Publish to RabbitMQ
        if publish_keyframe_message(shm_name, keyframe_id, points.shape[0]):
            print("\nWaiting for mesh_service to process...")
            print("Check Rerun viewer at http://localhost:9876")
            print("You should see a colored cube mesh appear")
            
            # Keep shared memory alive for processing
            time.sleep(5)
            
            print("\nTest completed!")
        else:
            print("\nFailed to publish message to RabbitMQ")
            print("Make sure RabbitMQ is running")
            
    except Exception as e:
        print(f"\nError: {e}")
        
    finally:
        # Cleanup
        if shm:
            try:
                shm.close()
                # Don't unlink if mesh_service should do it
                if os.environ.get('MESH_SERVICE_UNLINK_SHM', 'false') == 'false':
                    time.sleep(2)  # Give mesh_service time to process
                shm.unlink()
                print(f"Cleaned up shared memory '{shm_name}'")
            except:
                pass


if __name__ == "__main__":
    main()