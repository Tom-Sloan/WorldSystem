"""
H.264 video stream decoder for frame_processor.

This module handles decoding of H.264 video streams received from the server.
"""

import av
import numpy as np
from typing import List, Dict, Optional, Generator
import asyncio
from collections import defaultdict
from core.utils import get_logger

logger = get_logger(__name__)


class H264StreamDecoder:
    """
    Handles H.264 stream decoding for multiple concurrent streams.
    
    Uses PyAV's parse() method for proper NAL unit handling.
    """
    
    def __init__(self):
        """Initialize the H.264 decoder."""
        self.streams: Dict[str, av.CodecContext] = {}
        self.buffers: Dict[str, bytearray] = defaultdict(bytearray)
        self._lock = asyncio.Lock()
        
    async def process_stream_chunk(self, stream_id: str, chunk_data: bytes) -> List[np.ndarray]:
        """
        Process an H.264 stream chunk and return decoded frames.
        
        Args:
            stream_id: Unique identifier for the stream
            chunk_data: Raw H.264 data chunk
            
        Returns:
            List of decoded frames as numpy arrays (BGR format)
        """
        frames = []
        
        async with self._lock:
            try:
                # Get or create codec context for this stream
                if stream_id not in self.streams:
                    self.streams[stream_id] = self._create_decoder()
                    logger.info(f"Created H.264 decoder for stream {stream_id}")
                
                codec_ctx = self.streams[stream_id]
                
                # Add chunk to buffer
                self.buffers[stream_id].extend(chunk_data)
                
                # Try to decode frames
                packets = codec_ctx.parse(chunk_data)
                
                for packet in packets:
                    try:
                        frames_decoded = codec_ctx.decode(packet)
                        for frame in frames_decoded:
                            # Convert to numpy array in BGR format
                            img = frame.to_ndarray(format='bgr24')
                            frames.append(img)
                            logger.debug(f"Decoded frame from stream {stream_id}: {img.shape}")
                    except av.AVError as e:
                        logger.warning(f"Error decoding packet from stream {stream_id}: {e}")
                        
            except Exception as e:
                logger.error(f"Error processing H.264 chunk for stream {stream_id}: {e}")
                # Reset the stream on error
                if stream_id in self.streams:
                    self.cleanup_stream(stream_id)
                    
        return frames
    
    def _create_decoder(self) -> av.CodecContext:
        """Create a new H.264 decoder context."""
        codec = av.CodecContext.create('h264', 'r')
        
        # Configure decoder for low latency
        codec.thread_type = 'AUTO'
        codec.thread_count = 1  # Single thread for low latency
        
        # Set options for better compatibility
        codec.options = {
            'flags': 'low_delay',
            'flags2': 'fast',
        }
        
        return codec
    
    def cleanup_stream(self, stream_id: str):
        """Clean up resources for a specific stream."""
        if stream_id in self.streams:
            try:
                # Close codec context
                codec_ctx = self.streams[stream_id]
                # PyAV handles cleanup automatically
                del self.streams[stream_id]
            except Exception as e:
                logger.error(f"Error cleaning up stream {stream_id}: {e}")
                
        if stream_id in self.buffers:
            del self.buffers[stream_id]
            
        logger.info(f"Cleaned up H.264 decoder for stream {stream_id}")
    
    def cleanup_all(self):
        """Clean up all streams."""
        stream_ids = list(self.streams.keys())
        for stream_id in stream_ids:
            self.cleanup_stream(stream_id)
    
    def get_stats(self) -> dict:
        """Get decoder statistics."""
        return {
            "active_streams": len(self.streams),
            "stream_ids": list(self.streams.keys()),
            "buffer_sizes": {sid: len(buf) for sid, buf in self.buffers.items()}
        }