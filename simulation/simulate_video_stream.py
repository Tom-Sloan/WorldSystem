#!/usr/bin/env python3
"""
simulate_video_stream.py

Simulates video streaming by reading video files and sending them as H.264 streams
to the server via WebSocket. This sends standard H.264 streams compatible with
real devices.
"""

import os
import asyncio
import json
import time
import sys
from pathlib import Path
from typing import Optional
import subprocess
import cv2
import websockets

# Configuration
SERVER_WS_URL = os.getenv("SERVER_WS_URL", "ws://127.0.0.1:5001/ws/video")

# Video streaming configuration
CHUNK_SIZE = 32 * 1024  # 32KB chunks for H.264 streaming
FRAME_CHUNK_SIZE = 30  # Number of frames to encode at once

# Determine data source
USE_FOLDER = os.getenv("USE_FOLDER", "")
if USE_FOLDER and os.path.exists("/simulation_data"):
    DATA_ROOT = Path("/simulation_data")
    print(f"[INFO] Using simulation data from: {DATA_ROOT}")
else:
    DATA_ROOT = Path("/data")
    print(f"[INFO] Using default data path: {DATA_ROOT}")


class VideoStreamSimulator:
    """Handles video streaming simulation with standard H.264"""
    
    def __init__(self, websocket_url: str):
        self.websocket_url = websocket_url
        self.websocket = None
        
    async def connect(self):
        """Connect to the server WebSocket"""
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            print(f"[WebSocket] Connected to {self.websocket_url}")
            
            # Send initial configuration
            config_msg = {
                "type": "video_config",
                "format": "h264",
                "source": "simulator"
            }
            await self.websocket.send(json.dumps(config_msg))
            
        except Exception as e:
            print(f"[WebSocket] Failed to connect: {e}")
            raise
            
    async def disconnect(self):
        """Disconnect from the server"""
        if self.websocket:
            await self.websocket.close()
            print("[WebSocket] Disconnected")
            
    async def extract_h264_stream(self, video_path: Path) -> Optional[Path]:
        """Extract raw H.264 stream from video file using ffmpeg"""
        h264_path = video_path.with_suffix('.h264')
        
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-c:v', 'copy',  # Copy video codec (no re-encoding)
            '-bsf:v', 'h264_mp4toannexb',  # Convert to Annex B format
            '-an',  # No audio
            '-f', 'h264',  # Raw H.264 format
            str(h264_path),
            '-y'  # Overwrite if exists
        ]
        
        print(f"[H.264] Extracting H.264 stream from {video_path.name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[H.264] FFmpeg extraction failed: {result.stderr}")
            return None
            
        if h264_path.exists() and h264_path.stat().st_size > 0:
            print(f"[H.264] Successfully extracted {h264_path.stat().st_size / 1024 / 1024:.2f} MB")
            return h264_path
        else:
            print(f"[H.264] Extraction failed - no output file")
            return None
            
    async def stream_h264_file(self, h264_path: Path):
        """Stream raw H.264 file in chunks"""
        print(f"[Stream] Streaming {h264_path.name}")
        
        file_size = h264_path.stat().st_size
        bytes_sent = 0
        start_time = asyncio.get_event_loop().time()
        
        with open(h264_path, 'rb') as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                    
                # Send raw H.264 data
                await self.websocket.send(chunk)
                bytes_sent += len(chunk)
                
                # Progress logging
                if bytes_sent % (1024 * 1024) == 0:  # Every MB
                    progress = (bytes_sent / file_size) * 100
                    elapsed = asyncio.get_event_loop().time() - start_time
                    rate = bytes_sent / elapsed / 1024 / 1024  # MB/s
                    print(f"[Stream] Progress: {progress:.1f}%, Rate: {rate:.2f} MB/s")
                    
                # Small delay to prevent overwhelming the network
                await asyncio.sleep(0.001)
                
        duration = asyncio.get_event_loop().time() - start_time
        avg_rate = bytes_sent / duration / 1024 / 1024
        print(f"[Stream] Completed: {bytes_sent / 1024 / 1024:.2f} MB in {duration:.2f}s ({avg_rate:.2f} MB/s)")
        
    async def stream_video_file(self, video_path: Path):
        """Process and stream a video file"""
        print(f"\n[Video] Processing: {video_path.name}")
        
        # First try to extract H.264 stream
        h264_path = await self.extract_h264_stream(video_path)
        
        if h264_path:
            # Stream the extracted H.264
            await self.stream_h264_file(h264_path)
            # Clean up
            h264_path.unlink()
        else:
            # Fallback: encode frames to H.264
            await self.encode_and_stream_frames(video_path)
            
    async def encode_and_stream_frames(self, video_path: Path):
        """Fallback: Read frames and encode to H.264 for streaming"""
        print(f"[Encode] Encoding frames from {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[Encode] Failed to open video: {video_path}")
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[Encode] Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Create temporary H.264 file for encoding
        temp_h264 = video_path.parent / f"temp_{video_path.stem}.h264"
        
        # Use ffmpeg to create H.264 encoder pipe
        cmd = [
            'ffmpeg',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', str(fps),
            '-i', '-',  # Input from pipe
            '-c:v', 'libx264',
            '-preset', 'ultrafast',  # Fast encoding
            '-tune', 'zerolatency',  # Low latency
            '-profile:v', 'baseline',  # Compatible profile
            '-level', '3.0',
            '-f', 'h264',
            '-'  # Output to pipe
        ]
        
        # Start ffmpeg process
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        frame_count = 0
        chunk_buffer = bytearray()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Write frame to ffmpeg
                process.stdin.write(frame.tobytes())
                process.stdin.flush()
                
                frame_count += 1
                
                # Read encoded H.264 data from ffmpeg
                # This is non-blocking, so we might not get data immediately
                while True:
                    try:
                        chunk = process.stdout.read(CHUNK_SIZE)
                        if chunk:
                            chunk_buffer.extend(chunk)
                            
                            # Send accumulated data when we have enough
                            if len(chunk_buffer) >= CHUNK_SIZE:
                                await self.websocket.send(bytes(chunk_buffer[:CHUNK_SIZE]))
                                chunk_buffer = chunk_buffer[CHUNK_SIZE:]
                        else:
                            break
                    except:
                        break
                        
                # Progress
                if frame_count % 30 == 0:
                    print(f"[Encode] Processed {frame_count}/{total_frames} frames")
                    
        finally:
            # Close ffmpeg stdin to signal end of input
            process.stdin.close()
            
            # Read any remaining output
            remaining = process.stdout.read()
            if remaining:
                chunk_buffer.extend(remaining)
                
            # Send any remaining data
            if chunk_buffer:
                await self.websocket.send(bytes(chunk_buffer))
                
            # Clean up
            process.terminate()
            cap.release()
            
        print(f"[Encode] Completed encoding {frame_count} frames")
        
    async def stream_from_frames_directory(self, frames_path: Path):
        """Convert frame images to H.264 stream"""
        print(f"[Frames] Converting frames from {frames_path}")
        
        # Get list of frame files
        frame_files = sorted(frames_path.glob("*.jpg"))
        if not frame_files:
            frame_files = sorted(frames_path.glob("*.png"))
            
        if not frame_files:
            print(f"[Frames] No frames found in {frames_path}")
            return
            
        print(f"[Frames] Found {len(frame_files)} frames")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frame_files[0]))
        if first_frame is None:
            print(f"[Frames] Failed to read first frame")
            return
            
        height, width = first_frame.shape[:2]
        fps = 30.0  # Assume 30fps for frame sequences
        
        # Create video from frames using ffmpeg
        temp_video = frames_path.parent / f"temp_{frames_path.name}.mp4"
        
        cmd = [
            'ffmpeg',
            '-framerate', str(fps),
            '-pattern_type', 'glob',
            '-i', str(frames_path / '*.jpg'),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-profile:v', 'baseline',
            '-level', '3.0',
            '-pix_fmt', 'yuv420p',
            str(temp_video),
            '-y'
        ]
        
        print(f"[Frames] Creating video from frames...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and temp_video.exists():
            # Stream the created video
            await self.stream_video_file(temp_video)
            # Clean up
            temp_video.unlink()
        else:
            print(f"[Frames] Failed to create video: {result.stderr}")


async def stream_all_videos():
    """Main function to stream all videos from the data directory"""
    
    # Create video streamer
    streamer = VideoStreamSimulator(SERVER_WS_URL)
    
    try:
        # Connect to server
        await streamer.connect()
        
        # Find all video files
        video_files = []
        
        # Look for video files in root
        for pattern in ['*.mp4', '*.avi', '*.mov', '*.h264']:
            video_files.extend(DATA_ROOT.glob(pattern))
            
        # Look for video files in subdirectories
        for folder in DATA_ROOT.iterdir():
            if folder.is_dir():
                # Check mav0/video_segments
                video_segments_path = folder / "mav0" / "video_segments"
                if video_segments_path.exists():
                    video_files.extend(sorted(video_segments_path.glob("*.mp4")))
                    
                # Check for frame directories
                frames_path = folder / "mav0" / "cam0" / "data"
                if frames_path.exists() and list(frames_path.glob("*.jpg")):
                    print(f"[Info] Found frames directory: {frames_path}")
                    await streamer.stream_from_frames_directory(frames_path)
                    
        print(f"[Info] Found {len(video_files)} video files to stream")
        
        # Stream each video file
        for video_file in sorted(video_files):
            await streamer.stream_video_file(video_file)
            
            # Wait between videos
            await asyncio.sleep(2)
            
    except Exception as e:
        print(f"[Error] Streaming failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await streamer.disconnect()


if __name__ == "__main__":
    print("WorldSystem Video Stream Simulator")
    print("=" * 50)
    print(f"Server URL: {SERVER_WS_URL}")
    print(f"Data root: {DATA_ROOT}")
    sys.stdout.flush()
    
    # Check for ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✓ FFmpeg is available")
    except:
        print("✗ FFmpeg not found - this is required for H.264 streaming")
        sys.exit(1)
    
    # Wait for services to start
    print("\nWaiting 15 seconds for services to start...")
    time.sleep(15)
    
    try:
        asyncio.run(stream_all_videos())
        print("\nVideo streaming simulation completed!")
    except Exception as e:
        print(f"\n[ERROR] Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)