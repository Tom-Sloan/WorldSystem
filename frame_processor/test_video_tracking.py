#!/usr/bin/env python3
"""
Test script for SAM2 video tracking functionality.

This script simulates H.264 video streaming to test the frame processor's
video tracking capabilities using SAM2.
"""

import asyncio
import cv2
import numpy as np
import aio_pika
import time
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys

# For H.264 encoding
try:
    import av
except ImportError:
    print("PyAV not installed. Install with: pip install av")
    sys.exit(1)


class VideoStreamSimulator:
    """Simulates H.264 video streaming to RabbitMQ."""
    
    def __init__(self, video_path: str, rabbitmq_url: str = "amqp://localhost:5672"):
        self.video_path = video_path
        self.rabbitmq_url = rabbitmq_url
        self.websocket_id = f"test_stream_{int(time.time())}"
        
        # Video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {self.width}x{self.height} @ {self.fps}fps, {self.total_frames} frames")
        
        # H.264 encoder setup
        self.codec = av.CodecContext.create('h264', 'w')
        self.codec.width = self.width
        self.codec.height = self.height
        self.codec.pix_fmt = 'yuv420p'
        self.codec.framerate = self.fps
        self.codec.time_base = av.Fraction(1, self.fps)
        self.codec.gop_size = 30  # Keyframe every 30 frames
        self.codec.bit_rate = 2000000  # 2 Mbps
        
        # RabbitMQ connection
        self.connection = None
        self.channel = None
        self.exchange = None
        
        # Statistics
        self.frames_sent = 0
        self.bytes_sent = 0
        self.start_time = None
    
    async def connect(self):
        """Connect to RabbitMQ."""
        print(f"Connecting to RabbitMQ at {self.rabbitmq_url}")
        
        self.connection = await aio_pika.connect_robust(
            self.rabbitmq_url,
            heartbeat=3600
        )
        self.channel = await self.connection.channel()
        
        # Declare video stream exchange
        self.exchange = await self.channel.declare_exchange(
            'video_stream_exchange',
            aio_pika.ExchangeType.FANOUT,
            durable=True
        )
        
        print("Connected to RabbitMQ")
    
    async def stream_video(self, loop: bool = False, max_frames: Optional[int] = None):
        """Stream video as H.264 chunks."""
        if not self.connection:
            await self.connect()
        
        print(f"\nStreaming video as websocket_id: {self.websocket_id}")
        print("Press Ctrl+C to stop\n")
        
        self.start_time = time.time()
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    if loop:
                        # Reset to beginning
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break
                
                # Encode frame to H.264
                h264_data = await self._encode_frame(frame)
                
                if h264_data:
                    # Create message with headers
                    headers = {
                        'websocket_id': self.websocket_id,
                        'timestamp_ns': int(time.time() * 1e9),
                        'frame_number': frame_count,
                        'resolution': f"{self.width}x{self.height}",
                        'fps': self.fps
                    }
                    
                    # Publish to exchange
                    message = aio_pika.Message(
                        body=h264_data,
                        headers=headers,
                        content_type='video/h264'
                    )
                    
                    await self.exchange.publish(message, routing_key='')
                    
                    # Update statistics
                    self.frames_sent += 1
                    self.bytes_sent += len(h264_data)
                    frame_count += 1
                    
                    # Print progress
                    if frame_count % 30 == 0:
                        elapsed = time.time() - self.start_time
                        actual_fps = self.frames_sent / elapsed
                        bandwidth_mbps = (self.bytes_sent * 8 / elapsed) / 1e6
                        
                        print(f"\rFrames: {self.frames_sent}/{self.total_frames} | "
                              f"FPS: {actual_fps:.1f} | "
                              f"Bandwidth: {bandwidth_mbps:.1f} Mbps | "
                              f"Progress: {(frame_count/self.total_frames)*100:.1f}%", 
                              end='', flush=True)
                
                # Control frame rate
                await asyncio.sleep(1.0 / self.fps)
                
                # Check max frames
                if max_frames and frame_count >= max_frames:
                    break
                
        except KeyboardInterrupt:
            print("\n\nStreaming interrupted by user")
        finally:
            print(f"\n\nStreaming complete:")
            print(f"  Total frames sent: {self.frames_sent}")
            print(f"  Total data sent: {self.bytes_sent / 1e6:.1f} MB")
            print(f"  Average FPS: {self.frames_sent / (time.time() - self.start_time):.1f}")
    
    async def _encode_frame(self, frame: np.ndarray) -> Optional[bytes]:
        """Encode a frame to H.264."""
        try:
            # Convert BGR to YUV420p
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            av_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
            av_frame = av_frame.reformat(format='yuv420p')
            
            # Encode
            packets = self.codec.encode(av_frame)
            
            # Concatenate packet data
            data = b''
            for packet in packets:
                data += bytes(packet)
            
            return data if data else None
            
        except Exception as e:
            print(f"\nError encoding frame: {e}")
            return None
    
    async def monitor_results(self, duration: int = 60):
        """Monitor processed results from frame processor."""
        print(f"\nMonitoring results for {duration} seconds...")
        
        # Create new channel for monitoring
        monitor_channel = await self.connection.channel()
        
        # Declare and bind to processed frames exchange
        processed_exchange = await monitor_channel.declare_exchange(
            'processed_frames_exchange',
            aio_pika.ExchangeType.FANOUT,
            durable=True
        )
        
        monitor_queue = await monitor_channel.declare_queue(
            exclusive=True
        )
        
        await monitor_queue.bind(processed_exchange)
        
        # Statistics
        results_received = 0
        total_tracks = 0
        
        async def process_result(message: aio_pika.IncomingMessage):
            nonlocal results_received, total_tracks
            
            async with message.process():
                headers = message.headers or {}
                
                # Count results
                results_received += 1
                detection_count = headers.get('detection_count', 0)
                
                # Parse detection data
                if headers.get('class_summary'):
                    class_summary = json.loads(headers['class_summary'])
                    total_tracks = sum(class_summary.values())
                
                # Print summary
                if results_received % 10 == 0:
                    print(f"\rProcessed frames: {results_received} | "
                          f"Active tracks: {total_tracks} | "
                          f"FPS: {headers.get('fps', 0):.1f}",
                          end='', flush=True)
        
        # Start consuming
        await monitor_queue.consume(process_result)
        
        # Wait for duration
        await asyncio.sleep(duration)
        
        # Stop consuming
        await monitor_queue.cancel()
        
        print(f"\n\nResults summary:")
        print(f"  Total processed frames: {results_received}")
        print(f"  Average tracks per frame: {total_tracks / max(results_received, 1):.1f}")
    
    async def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        
        if self.connection and not self.connection.is_closed:
            await self.connection.close()


async def create_test_video(output_path: str, duration: int = 10, fps: int = 30):
    """Create a simple test video with moving objects."""
    print(f"Creating test video: {output_path}")
    
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create moving objects
    objects = [
        {'x': 100, 'y': 100, 'vx': 5, 'vy': 3, 'size': 80, 'color': (255, 0, 0)},
        {'x': 500, 'y': 300, 'vx': -4, 'vy': 2, 'size': 60, 'color': (0, 255, 0)},
        {'x': 800, 'y': 500, 'vx': 3, 'vy': -4, 'size': 100, 'color': (0, 0, 255)},
    ]
    
    total_frames = duration * fps
    
    for frame_idx in range(total_frames):
        # Create frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
        
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
            
            # Draw circle
            cv2.circle(frame, (int(obj['x']), int(obj['y'])), 
                      obj['size']//2, obj['color'], -1)
            
            # Add some texture
            cv2.circle(frame, (int(obj['x']), int(obj['y'])), 
                      obj['size']//4, (255, 255, 255), 2)
        
        # Add frame number
        cv2.putText(frame, f"Frame {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Write frame
        out.write(frame)
        
        if frame_idx % fps == 0:
            print(f"\rGenerating: {(frame_idx/total_frames)*100:.1f}%", end='', flush=True)
    
    out.release()
    print(f"\nTest video created: {output_path}")


async def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test SAM2 video tracking')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--create-test', action='store_true', 
                       help='Create a test video')
    parser.add_argument('--duration', type=int, default=10,
                       help='Test video duration in seconds')
    parser.add_argument('--loop', action='store_true',
                       help='Loop the video')
    parser.add_argument('--max-frames', type=int,
                       help='Maximum frames to stream')
    parser.add_argument('--monitor', action='store_true',
                       help='Monitor results after streaming')
    parser.add_argument('--rabbitmq', type=str, default='amqp://localhost:5672',
                       help='RabbitMQ URL')
    
    args = parser.parse_args()
    
    # Create test video if requested
    if args.create_test:
        test_video_path = "test_moving_objects.mp4"
        await create_test_video(test_video_path, args.duration)
        if not args.video:
            args.video = test_video_path
    
    # Check if video is provided
    if not args.video:
        print("Error: Please provide a video file with --video or use --create-test")
        parser.print_help()
        return
    
    # Check if video exists
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return
    
    # Create simulator
    simulator = VideoStreamSimulator(args.video, args.rabbitmq)
    
    try:
        # Connect to RabbitMQ
        await simulator.connect()
        
        # Start streaming
        stream_task = asyncio.create_task(
            simulator.stream_video(loop=args.loop, max_frames=args.max_frames)
        )
        
        # Optionally monitor results
        if args.monitor:
            monitor_task = asyncio.create_task(
                simulator.monitor_results(duration=60)
            )
            
            # Wait for both tasks
            await asyncio.gather(stream_task, monitor_task)
        else:
            await stream_task
            
    finally:
        await simulator.cleanup()


if __name__ == "__main__":
    print("SAM2 Video Tracking Test Script")
    print("================================\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")