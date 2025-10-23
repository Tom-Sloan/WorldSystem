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
import re
import struct

# Configuration
SERVER_WS_URL = os.getenv("SERVER_WS_URL", "ws://127.0.0.1:5001/ws/video")
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "video").lower()  # 'video' or 'segments'

# Video streaming configuration
CHUNK_SIZE = 32 * 1024  # 32KB chunks for H.264 streaming

# Android packet types (matching H264VideoStreamVM.kt)
PACKET_TYPE_SPS = 0x01
PACKET_TYPE_PPS = 0x02
PACKET_TYPE_KEYFRAME = 0x03
PACKET_TYPE_FRAME = 0x04
PACKET_TYPE_CONFIG = 0x05

# Test data location (copied into container)
# TEST_DATA_ROOT = Path("/test_continuous_hallway")
# TEST_VIDEO_PATH = TEST_DATA_ROOT / "continuous hallway walk.mp4"
TEST_DATA_ROOT = Path("/test_data")
TEST_VIDEO_PATH = TEST_DATA_ROOT / "test_video.mp4"
SEGMENTS_PATH = TEST_DATA_ROOT / "20250617_211214_segments"


class VideoStreamSimulator:
    """Handles video streaming simulation with standard H.264"""
    
    def __init__(self, websocket_url: str):
        self.websocket_url = websocket_url
        self.websocket = None
        self.sps_data = None
        self.pps_data = None
        self.frame_count = 0
        
    async def connect(self):
        """Connect to the server WebSocket"""
        try:
            # Increased timeouts to handle slow processing
            self.websocket = await websockets.connect(
                self.websocket_url,
                ping_timeout=30,  # 30 seconds for ping timeout
                close_timeout=10,  # 10 seconds for close timeout
                max_size=10 * 1024 * 1024  # 10MB max message size
            )
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
            
    async def get_video_info(self, video_path: Path) -> dict:
        """Extract video information using ffprobe"""
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=duration,r_frame_rate,bit_rate',
            '-of', 'json',
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ffprobe] Failed to get video info: {result.stderr}")
            return None
            
        try:
            info = json.loads(result.stdout)
            stream = info['streams'][0] if info.get('streams') else {}
            
            # Parse frame rate (e.g., "30/1" -> 30.0)
            fps_str = stream.get('r_frame_rate', '30/1')
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                fps = num / den if den != 0 else 30.0
            else:
                fps = float(fps_str)
            
            duration = float(stream.get('duration', 0))
            bit_rate = int(stream.get('bit_rate', 0))
            
            return {
                'duration': duration,
                'fps': fps,
                'bit_rate': bit_rate
            }
        except Exception as e:
            print(f"[ffprobe] Error parsing video info: {e}")
            return None
            
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
            
    def create_android_packet(self, data: bytes, packet_type: int, timestamp_us: int = None, flags: int = 0) -> bytes:
        """Create an Android-formatted packet with 21-byte header"""
        if timestamp_us is None:
            timestamp_us = int(time.time() * 1_000_000)  # Current time in microseconds
            
        # Calculate total packet size
        header_size = 21
        total_size = header_size + len(data)
        
        # Build packet with big-endian format
        packet = bytearray()
        packet.extend(struct.pack('>I', total_size))      # 4 bytes: total packet size
        packet.extend(struct.pack('B', packet_type))       # 1 byte: packet type
        packet.extend(struct.pack('>Q', timestamp_us))     # 8 bytes: timestamp (microseconds)
        packet.extend(struct.pack('>I', len(data)))        # 4 bytes: data size
        packet.extend(struct.pack('>I', flags))            # 4 bytes: flags
        packet.extend(data)                                # N bytes: H.264 data
        
        return bytes(packet)
    
    def create_config_packet(self, data: bytes, packet_type: int) -> bytes:
        """Create a config packet (SPS/PPS) with 13-byte header"""
        timestamp_us = int(time.time() * 1_000_000)
        
        header_size = 13
        total_size = header_size + len(data)
        
        packet = bytearray()
        packet.extend(struct.pack('>I', total_size))      # 4 bytes: total packet size
        packet.extend(struct.pack('B', packet_type))       # 1 byte: packet type
        packet.extend(struct.pack('>Q', timestamp_us))     # 8 bytes: timestamp
        packet.extend(data)                                # N bytes: SPS/PPS data
        
        return bytes(packet)
    
    def find_nal_units(self, data: bytes):
        """Find NAL units in H.264 stream and return list of (offset, size, nal_type)"""
        nal_units = []
        offset = 0
        data_len = len(data)
        
        while offset < data_len - 4:
            # Look for start code (0x00 0x00 0x00 0x01 or 0x00 0x00 0x01)
            if (data[offset:offset+4] == b'\x00\x00\x00\x01'):
                start_offset = offset
                offset += 4
                if offset < data_len:
                    nal_type = data[offset] & 0x1F
                    # Find next start code
                    next_offset = offset + 1
                    while next_offset < data_len - 4:
                        if (data[next_offset:next_offset+4] == b'\x00\x00\x00\x01' or
                            data[next_offset:next_offset+3] == b'\x00\x00\x01'):
                            break
                        next_offset += 1
                    else:
                        next_offset = data_len
                    
                    nal_units.append((start_offset, next_offset - start_offset, nal_type))
                    offset = next_offset
            elif (offset < data_len - 3 and data[offset:offset+3] == b'\x00\x00\x01'):
                start_offset = offset
                offset += 3
                if offset < data_len:
                    nal_type = data[offset] & 0x1F
                    # Find next start code
                    next_offset = offset + 1
                    while next_offset < data_len - 3:
                        if (data[next_offset:next_offset+4] == b'\x00\x00\x00\x01' or
                            data[next_offset:next_offset+3] == b'\x00\x00\x01'):
                            break
                        next_offset += 1
                    else:
                        next_offset = data_len
                    
                    nal_units.append((start_offset, next_offset - start_offset, nal_type))
                    offset = next_offset
            else:
                offset += 1
                
        return nal_units
            
    async def stream_h264_file(self, h264_path: Path, video_info: dict = None):
        """Stream raw H.264 file with Android packet format"""
        print(f"[Stream] Streaming {h264_path.name} with Android packet format")
        
        # Read entire file to parse NAL units
        with open(h264_path, 'rb') as f:
            h264_data = f.read()
            
        file_size = len(h264_data)
        print(f"[Stream] Loaded {file_size / 1024 / 1024:.2f} MB of H.264 data")
        
        # Find all NAL units
        nal_units = self.find_nal_units(h264_data)
        print(f"[Stream] Found {len(nal_units)} NAL units")
        
        # NAL unit types
        NAL_TYPE_SPS = 7
        NAL_TYPE_PPS = 8
        NAL_TYPE_IDR = 5  # IDR frame (keyframe)
        NAL_TYPE_NON_IDR = 1  # Non-IDR frame
        
        bytes_sent = 0
        start_time = asyncio.get_event_loop().time()
        frame_timestamps = []
        
        # Calculate frame timing
        if video_info and video_info.get('fps', 0) > 0:
            frame_duration_ms = 1000.0 / video_info['fps']
            print(f"[Stream] FPS: {video_info['fps']:.1f}, frame duration: {frame_duration_ms:.1f}ms")
        else:
            frame_duration_ms = 33.33  # Default to 30 FPS
            print("[Stream] Using default 30 FPS timing")
            
        # Process NAL units
        for i, (offset, size, nal_type) in enumerate(nal_units):
            nal_data = h264_data[offset:offset + size]
            timestamp_us = int(self.frame_count * frame_duration_ms * 1000)  # Convert to microseconds
            
            # Send based on NAL type
            if nal_type == NAL_TYPE_SPS:
                # Store SPS and send as config packet
                self.sps_data = nal_data
                packet = self.create_config_packet(nal_data, PACKET_TYPE_SPS)
                await self.websocket.send(packet)
                bytes_sent += len(packet)
                print(f"[Stream] Sent SPS packet: {len(nal_data)} bytes")
                
            elif nal_type == NAL_TYPE_PPS:
                # Store PPS and send as config packet
                self.pps_data = nal_data
                packet = self.create_config_packet(nal_data, PACKET_TYPE_PPS)
                await self.websocket.send(packet)
                bytes_sent += len(packet)
                print(f"[Stream] Sent PPS packet: {len(nal_data)} bytes")
                
            elif nal_type == NAL_TYPE_IDR:
                # IDR frame (keyframe)
                # First send SPS/PPS if we have them (Android app does this before keyframes)
                if self.sps_data:
                    sps_packet = self.create_config_packet(self.sps_data, PACKET_TYPE_SPS)
                    await self.websocket.send(sps_packet)
                    bytes_sent += len(sps_packet)
                    
                if self.pps_data:
                    pps_packet = self.create_config_packet(self.pps_data, PACKET_TYPE_PPS)
                    await self.websocket.send(pps_packet)
                    bytes_sent += len(pps_packet)
                
                # Send keyframe
                flags = 1  # MediaCodec.BUFFER_FLAG_KEY_FRAME
                packet = self.create_android_packet(nal_data, PACKET_TYPE_KEYFRAME, timestamp_us, flags)
                await self.websocket.send(packet)
                bytes_sent += len(packet)
                self.frame_count += 1
                frame_timestamps.append(asyncio.get_event_loop().time())
                
                if self.frame_count % 30 == 0:  # Log every 30 frames
                    print(f"[Stream] Sent keyframe #{self.frame_count}: {len(nal_data)} bytes")
                    
            elif nal_type == NAL_TYPE_NON_IDR:
                # Regular frame
                flags = 0
                packet = self.create_android_packet(nal_data, PACKET_TYPE_FRAME, timestamp_us, flags)
                await self.websocket.send(packet)
                bytes_sent += len(packet)
                self.frame_count += 1
                frame_timestamps.append(asyncio.get_event_loop().time())
                
            # Progress logging with percentage
            if bytes_sent > 0 and bytes_sent % (1024 * 1024) < len(packet):  # Every MB
                elapsed = asyncio.get_event_loop().time() - start_time
                rate = bytes_sent / elapsed / 1024 / 1024  # MB/s
                progress_pct = (i / len(nal_units)) * 100  # Progress based on NAL units processed
                print(f"[Stream] Progress: {progress_pct:.1f}%, {bytes_sent / 1024 / 1024:.1f} MB sent, Rate: {rate:.2f} MB/s")
                
            # Frame rate limiting - maintain real-time playback speed
            if len(frame_timestamps) > 1:
                # Calculate how long we should have taken
                expected_time = len(frame_timestamps) * (frame_duration_ms / 1000.0)
                actual_time = asyncio.get_event_loop().time() - start_time
                
                # If we're ahead of schedule, wait
                if actual_time < expected_time:
                    sleep_time = expected_time - actual_time
                    await asyncio.sleep(sleep_time)
                    
        duration = asyncio.get_event_loop().time() - start_time
        avg_rate = bytes_sent / duration / 1024 / 1024
        print(f"[Stream] Progress: 100.0%, {bytes_sent / 1024 / 1024:.1f} MB sent")
        print(f"[Stream] Completed: {self.frame_count} frames, {bytes_sent / 1024 / 1024:.2f} MB in {duration:.2f}s ({avg_rate:.2f} MB/s)")
        
    async def stream_video_file(self, video_path: Path):
        """Process and stream a video file"""
        print(f"\n[Video] Processing: {video_path.name}")
        
        # Reset frame counter for each video
        self.frame_count = 0
        
        # Get video information for real-time streaming
        video_info = await self.get_video_info(video_path)
        if video_info:
            print(f"[Video] Duration: {video_info['duration']:.1f}s, FPS: {video_info['fps']:.1f}")
        
        # First try to extract H.264 stream
        h264_path = await self.extract_h264_stream(video_path)
        
        if h264_path:
            # Stream the extracted H.264 with timing info
            await self.stream_h264_file(h264_path, video_info)
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