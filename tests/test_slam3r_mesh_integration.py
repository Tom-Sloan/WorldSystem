#!/usr/bin/env python3
"""
Test SLAM3R to mesh_service integration by simulating keyframe publishing.
"""

import numpy as np
import time
import msgpack
import pika
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'slam3r', 'SLAM3R_engine'))

from shared_memory import SharedMemoryManager


def create_test_point_cloud(num_points=1000):
    """Create a test point cloud with a simple shape."""
    # Create a sphere
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    costheta = np.random.uniform(-1, 1, num_points)
    theta = np.arccos(costheta)
    
    radius = 2.0
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta) + 3.0  # Offset in Z
    
    points = np.column_stack((x, y, z)).astype(np.float32)
    
    # Create colors based on position
    colors = ((points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0)) * 255).astype(np.uint8)
    
    return points, colors


class TestKeyframePublisher:
    """Test publisher that simulates SLAM3R keyframe publishing."""
    
    def __init__(self):
        self.shm_manager = SharedMemoryManager()
        self.keyframe_counter = 0
        self.connection = None
        self.channel = None
        self.exchange = None
        
    def connect_rabbitmq(self):
        """Connect to RabbitMQ."""
        try:
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters('localhost')
            )
            self.channel = self.connection.channel()
            
            # Declare exchange (use topic type to match existing)
            exchange_name = os.environ.get('SLAM3R_KEYFRAME_EXCHANGE', 'slam3r_keyframe_exchange')
            self.channel.exchange_declare(exchange=exchange_name, exchange_type='topic', durable=True)
            self.exchange = exchange_name
            
            print(f"Connected to RabbitMQ, exchange: {exchange_name}")
            return True
        except Exception as e:
            print(f"Failed to connect to RabbitMQ: {e}")
            return False
    
    def publish_keyframe(self, points, colors, pose=None):
        """Publish keyframe with shared memory and RabbitMQ notification."""
        if pose is None:
            pose = np.eye(4, dtype=np.float32)
        
        # Generate unique keyframe ID
        timestamp_ns = int(time.time() * 1e9)
        keyframe_id = f"test_kf_{self.keyframe_counter}_{timestamp_ns}"
        self.keyframe_counter += 1
        
        # Note: shm_name is created by write_keyframe based on the keyframe_id
        
        try:
            # Write to shared memory
            shm_key = self.shm_manager.write_keyframe(
                keyframe_id=str(timestamp_ns),  # Use timestamp as ID
                points=points,
                colors=colors,
                pose=pose
            )
            
            print(f"Created shared memory segment: {shm_key}")
            print(f"  Points: {points.shape}")
            print(f"  Colors: {colors.shape}")
            
            # Publish RabbitMQ message
            if self.channel and self.exchange:
                message = {
                    'shm_key': shm_key,
                    'keyframe_id': keyframe_id,
                    'timestamp': time.time() * 1000,  # RabbitMQ expects milliseconds
                    'type': 'keyframe',
                    'point_count': len(points)
                }
                
                self.channel.basic_publish(
                    exchange=self.exchange,
                    routing_key='keyframe.new',  # Need routing key for topic exchange
                    body=msgpack.packb(message),
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # Persistent
                    )
                )
                
                print(f"Published RabbitMQ notification for keyframe {keyframe_id}")
            
            return True
            
        except Exception as e:
            print(f"Error publishing keyframe: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        if self.connection and not self.connection.is_closed:
            self.connection.close()


def main():
    """Test SLAM3R to mesh_service integration."""
    print("SLAM3R to Mesh Service Integration Test")
    print("=======================================\n")
    
    # Create publisher
    publisher = TestKeyframePublisher()
    
    # Connect to RabbitMQ
    if not publisher.connect_rabbitmq():
        print("Failed to connect to RabbitMQ. Make sure it's running.")
        return
    
    print("\nMake sure mesh_service is running with:")
    print("  docker-compose --profile slam3r up mesh_service")
    print("\nStarting keyframe publishing in 3 seconds...\n")
    time.sleep(3)
    
    try:
        # Publish multiple keyframes
        for i in range(5):
            print(f"\n--- Publishing keyframe {i} ---")
            
            # Create point cloud with varying size
            num_points = 500 + i * 200
            points, colors = create_test_point_cloud(num_points)
            
            # Create pose with some movement
            pose = np.eye(4, dtype=np.float32)
            pose[0, 3] = i * 0.5  # Move in X
            pose[1, 3] = np.sin(i * 0.5) * 0.3  # Oscillate in Y
            
            # Publish
            if publisher.publish_keyframe(points, colors, pose):
                print(f"Successfully published keyframe {i}")
            else:
                print(f"Failed to publish keyframe {i}")
            
            # Wait a bit
            time.sleep(1)
        
        print("\n\nTest completed!")
        print("Check the following:")
        print("1. mesh_service logs - should show received keyframes")
        print("2. Rerun viewer at http://localhost:9876")
        print("3. Prometheus metrics at http://localhost:9091/metrics")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        publisher.cleanup()


if __name__ == "__main__":
    main()