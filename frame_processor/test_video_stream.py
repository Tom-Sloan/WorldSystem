#!/usr/bin/env python3
"""
Test script to send video frames to frame_processor via RabbitMQ
This simulates what the server would send to test object tracking
"""

import cv2
import pika
import json
import time
import numpy as np
import sys
import os

def create_test_video():
    """Create a simple test video with moving objects"""
    print("Creating test video with moving objects...")
    
    # Video parameters
    width, height = 640, 480
    fps = 30
    duration = 10  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('/tmp/test_tracking.mp4', fourcc, fps, (width, height))
    
    # Object parameters
    objects = [
        {'x': 50, 'y': 100, 'vx': 2, 'vy': 0, 'size': 50, 'color': (255, 0, 0), 'shape': 'rect'},
        {'x': 300, 'y': 200, 'vx': -1, 'vy': 1, 'size': 40, 'color': (0, 255, 0), 'shape': 'circle'},
        {'x': 200, 'y': 350, 'vx': 1.5, 'vy': -0.5, 'size': 60, 'color': (0, 0, 255), 'shape': 'rect'},
    ]
    
    for frame_num in range(total_frames):
        # Create blank frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw objects
        for obj in objects:
            # Update position
            obj['x'] += obj['vx']
            obj['y'] += obj['vy']
            
            # Bounce off walls
            if obj['x'] <= obj['size']//2 or obj['x'] >= width - obj['size']//2:
                obj['vx'] = -obj['vx']
            if obj['y'] <= obj['size']//2 or obj['y'] >= height - obj['size']//2:
                obj['vy'] = -obj['vy']
            
            # Draw object
            x, y = int(obj['x']), int(obj['y'])
            if obj['shape'] == 'rect':
                cv2.rectangle(frame, 
                            (x - obj['size']//2, y - obj['size']//2),
                            (x + obj['size']//2, y + obj['size']//2),
                            obj['color'], -1)
            else:
                cv2.circle(frame, (x, y), obj['size']//2, obj['color'], -1)
        
        # Add frame number
        cv2.putText(frame, f"Frame {frame_num}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        out.write(frame)
    
    out.release()
    print("Test video created at /tmp/test_tracking.mp4")
    return '/tmp/test_tracking.mp4'

def stream_video_to_rabbitmq(video_path=None, use_webcam=False):
    """Stream video frames to RabbitMQ"""
    
    # RabbitMQ setup
    print("Connecting to RabbitMQ...")
    connection = pika.BlockingConnection(pika.URLParameters('amqp://127.0.0.1:5672'))
    channel = connection.channel()
    
    # Declare exchange
    exchange_name = 'video_frames_exchange'
    channel.exchange_declare(exchange=exchange_name, exchange_type='fanout', durable=True)
    print(f"Connected to RabbitMQ, exchange: {exchange_name}")
    
    # Open video source
    if use_webcam:
        print("Opening webcam...")
        cap = cv2.VideoCapture(0)
    else:
        if video_path is None:
            video_path = create_test_video()
        print(f"Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1.0 / fps
    
    print(f"Video FPS: {fps}")
    print("Streaming frames to RabbitMQ... Press Ctrl+C to stop")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                if use_webcam:
                    print("Error reading from webcam")
                    break
                else:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            
            # Encode frame
            _, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            frame_bytes = encoded.tobytes()
            
            # Create timestamp
            timestamp_ns = int(time.time() * 1e9)
            
            # Headers with metadata
            headers = {
                'timestamp_ns': timestamp_ns,
                'server_received': timestamp_ns,
                'ntp_time': timestamp_ns,
                'frame_number': frame_count,
                'width': frame.shape[1],
                'height': frame.shape[0]
            }
            
            # Publish frame
            channel.basic_publish(
                exchange=exchange_name,
                routing_key='',
                body=frame_bytes,
                properties=pika.BasicProperties(
                    content_type='application/octet-stream',
                    headers=headers
                )
            )
            
            frame_count += 1
            
            # Print progress every second
            if frame_count % int(fps) == 0:
                elapsed = time.time() - start_time
                actual_fps = frame_count / elapsed
                print(f"Sent {frame_count} frames, {actual_fps:.1f} fps")
            
            # Maintain frame rate
            time.sleep(frame_delay)
            
    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        cap.release()
        connection.close()
        print(f"Total frames sent: {frame_count}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stream video to frame_processor')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--webcam', action='store_true', help='Use webcam instead of video file')
    parser.add_argument('--create-test', action='store_true', help='Just create test video and exit')
    
    args = parser.parse_args()
    
    if args.create_test:
        create_test_video()
    else:
        stream_video_to_rabbitmq(args.video, args.webcam)

if __name__ == '__main__':
    main()