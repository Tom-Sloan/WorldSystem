#!/usr/bin/env python3
"""
simulate_video_stream.py

Simulates video streaming by reading video files from test_data and sending them 
as H.264 streams to the server via WebSocket.

Supports two modes:
- video: Streams the test_video.mp4 file
- segments: Streams all segment files from the 20250617_211214_segments folder
"""

import os
import asyncio
import json
import time
import sys
from pathlib import Path
import subprocess
import websockets

# Configuration
SERVER_WS_URL = os.getenv("SERVER_WS_URL", "ws://127.0.0.1:5001/ws/video")
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "video").lower()  # 'video' or 'segments'

# Video streaming configuration
CHUNK_SIZE = 32 * 1024  # 32KB chunks for H.264 streaming

# Test data location (copied into container)
TEST_DATA_ROOT = Path("/test_data")
TEST_VIDEO_PATH = TEST_DATA_ROOT / "test_video.mp4"
SEGMENTS_PATH = TEST_DATA_ROOT / "20250617_211214_segments"


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
                "source": "simulator",
                "mode": SIMULATION_MODE
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
            
    async def extract_h264_stream(self, video_path: Path) -> Path:
        """Extract or re-encode video to H.264 stream"""
        h264_path = video_path.parent / f"{video_path.stem}_stream.h264"
        
        # First try to copy if it's already H.264
        probe_cmd = ['ffmpeg', '-i', str(video_path), '-hide_banner']
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        
        # Check if video is already H.264
        is_h264 = 'Video: h264' in probe_result.stderr
        
        if is_h264:
            # Try to copy H.264 stream
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-c:v', 'copy',  # Copy video codec (no re-encoding)
                '-bsf:v', 'h264_mp4toannexb',  # Convert to Annex B format
                '-an',  # No audio
                '-f', 'h264',  # Raw H.264 format
                str(h264_path),
                '-y'  # Overwrite if exists
            ]
            print(f"[H.264] Extracting existing H.264 stream from {video_path.name}...")
        else:
            # Re-encode to H.264
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-c:v', 'libx264',  # Encode to H.264
                '-preset', 'ultrafast',  # Fast encoding
                '-tune', 'zerolatency',  # Low latency
                '-profile:v', 'baseline',  # Compatible profile
                '-level', '3.0',
                '-an',  # No audio
                '-f', 'h264',  # Raw H.264 format
                str(h264_path),
                '-y'  # Overwrite if exists
            ]
            print(f"[H.264] Re-encoding {video_path.name} to H.264...")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[H.264] FFmpeg failed: {result.stderr}")
            return None
            
        if h264_path.exists() and h264_path.stat().st_size > 0:
            print(f"[H.264] Successfully created {h264_path.stat().st_size / 1024 / 1024:.2f} MB")
            return h264_path
        else:
            print(f"[H.264] Failed - no output file")
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
            print(f"[Video] Failed to process {video_path.name}")


async def main():
    """Main function to stream video based on mode"""
    
    print("WorldSystem Video Stream Simulator")
    print("=" * 50)
    print(f"Server URL: {SERVER_WS_URL}")
    print(f"Mode: {SIMULATION_MODE}")
    print(f"Test data root: {TEST_DATA_ROOT}")
    
    # Verify test data exists
    if not TEST_DATA_ROOT.exists():
        print(f"[ERROR] Test data directory not found: {TEST_DATA_ROOT}")
        sys.exit(1)
    
    # Create video streamer
    streamer = VideoStreamSimulator(SERVER_WS_URL)
    
    try:
        # Connect to server
        await streamer.connect()
        
        if SIMULATION_MODE == "video":
            # Stream single test video
            if not TEST_VIDEO_PATH.exists():
                print(f"[ERROR] Test video not found: {TEST_VIDEO_PATH}")
                sys.exit(1)
                
            print(f"[Mode] Streaming single video: {TEST_VIDEO_PATH.name}")
            await streamer.stream_video_file(TEST_VIDEO_PATH)
            
        elif SIMULATION_MODE == "segments":
            # Stream all segments
            if not SEGMENTS_PATH.exists():
                print(f"[ERROR] Segments directory not found: {SEGMENTS_PATH}")
                sys.exit(1)
                
            # Find all segment files
            segment_files = sorted(SEGMENTS_PATH.glob("*_segment_*.mp4"))
            
            if not segment_files:
                print(f"[ERROR] No segment files found in {SEGMENTS_PATH}")
                sys.exit(1)
                
            print(f"[Mode] Streaming {len(segment_files)} segments from {SEGMENTS_PATH.name}")
            
            # Stream each segment with a small delay between them
            for segment_file in segment_files:
                await streamer.stream_video_file(segment_file)
                await asyncio.sleep(2)  # 2 second delay between segments
                
        else:
            print(f"[ERROR] Invalid mode: {SIMULATION_MODE}. Use 'video' or 'segments'")
            sys.exit(1)
            
    except Exception as e:
        print(f"[Error] Streaming failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await streamer.disconnect()


if __name__ == "__main__":
    print("Waiting 15 seconds for services to start...")
    sys.stdout.flush()
    time.sleep(15)
    
    # Check for ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✓ FFmpeg is available")
    except:
        print("✗ FFmpeg not found - this is required for H.264 streaming")
        sys.exit(1)
    
    try:
        asyncio.run(main())
        print("\nVideo streaming simulation completed!")
    except Exception as e:
        print(f"\n[ERROR] Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)