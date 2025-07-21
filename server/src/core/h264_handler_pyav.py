#!/usr/bin/env python3
"""
H.264 stream handler using PyAV's CodecContext for proper raw H.264 parsing.
Based on best practices from PyAV documentation and Stack Overflow.
"""

import asyncio
import logging
from typing import Dict, List, Optional
import numpy as np
import av
from dataclasses import dataclass
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class StreamState:
    """Track state for each video stream"""
    codec_context: av.CodecContext
    frames_decoded: int = 0
    bytes_received: int = 0
    last_frame_time: float = 0
    errors: int = 0


class PyAVH264Handler:
    """
    H.264 handler using PyAV's CodecContext.parse() method.
    This is the recommended approach for parsing raw H.264 streams.
    """
    
    def __init__(self):
        self.streams: Dict[str, StreamState] = {}
        self._lock = threading.Lock()
        
    def cleanup_stream(self, websocket_id: str) -> None:
        """Clean up stream resources"""
        with self._lock:
            if websocket_id in self.streams:
                del self.streams[websocket_id]
                logger.info(f"Cleaned up stream for {websocket_id}")
                
    async def process_h264_stream(self, websocket_id: str, data: bytes) -> List[np.ndarray]:
        """
        Process H.264 stream data using PyAV's CodecContext.parse().
        """
        frames = []
        
        # Get or create stream state
        with self._lock:
            if websocket_id not in self.streams:
                # Create codec context for H.264 decoding
                codec_context = av.CodecContext.create("h264", "r")
                # Enable multi-threading for better performance
                codec_context.thread_type = "AUTO"
                codec_context.thread_count = 2
                
                self.streams[websocket_id] = StreamState(codec_context=codec_context)
                logger.info(f"Created H.264 codec context for stream {websocket_id}")
                
            state = self.streams[websocket_id]
            
        try:
            # Parse the raw H.264 data into packets
            packets = state.codec_context.parse(data)
            state.bytes_received += len(data)
            
            # Log progress
            if state.bytes_received % (1024 * 1024) < len(data):  # Every MB
                logger.info(f"Stream {websocket_id}: Received {state.bytes_received // (1024*1024)} MB")
            
            # Decode each packet
            for packet in packets:
                if packet.size == 0:
                    continue
                    
                try:
                    # Decode the packet
                    decoded_frames = state.codec_context.decode(packet)
                    
                    for frame in decoded_frames:
                        # Convert to numpy array (BGR for OpenCV)
                        np_frame = frame.to_ndarray(format='bgr24')
                        frames.append(np_frame)
                        state.frames_decoded += 1
                        state.last_frame_time = time.time()
                        
                        # Log progress
                        if state.frames_decoded == 1:
                            logger.info(f"Stream {websocket_id}: First frame decoded! Resolution: {frame.width}x{frame.height}")
                        elif state.frames_decoded % 30 == 0:
                            logger.info(f"Stream {websocket_id}: Decoded {state.frames_decoded} frames")
                            
                except av.AVError as e:
                    # Individual packet decode errors are normal at the beginning
                    if state.errors < 5:
                        logger.debug(f"Packet decode error for stream {websocket_id}: {e}")
                    state.errors += 1
                except Exception as e:
                    logger.error(f"Unexpected error decoding packet for stream {websocket_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error parsing H.264 stream for {websocket_id}: {e}")
            
        return frames
        
    def get_stats(self) -> dict:
        """Get statistics for all streams"""
        with self._lock:
            stats = {
                "active_streams": len(self.streams),
                "streams": {}
            }
            
            for stream_id, state in self.streams.items():
                stats["streams"][stream_id] = {
                    "bytes_received": state.bytes_received,
                    "frames_decoded": state.frames_decoded,
                    "last_frame_age": time.time() - state.last_frame_time if state.last_frame_time > 0 else -1,
                    "errors": state.errors,
                    "codec_name": state.codec_context.codec.name,
                    "thread_type": state.codec_context.thread_type
                }
                
            return stats


# Helper function for format detection
def is_h264_stream(data: bytes) -> bool:
    """
    Check if data contains H.264 NAL units.
    """
    if len(data) < 4:
        return False
        
    # Check for H.264 start codes
    for i in range(min(len(data) - 4, 100)):
        if (data[i:i+4] == b'\x00\x00\x00\x01' or 
            data[i:i+3] == b'\x00\x00\x01'):
            nal_start = i + (4 if data[i:i+4] == b'\x00\x00\x00\x01' else 3)
            if nal_start < len(data):
                nal_type = data[nal_start] & 0x1F
                if 1 <= nal_type <= 23:
                    return True
                    
    return False