import struct
import asyncio
from typing import Dict, Optional, List
from dataclasses import dataclass
import numpy as np
import logging
import av

logger = logging.getLogger(__name__)

# Packet types matching Android client
PACKET_TYPE_SPS = 0x01
PACKET_TYPE_PPS = 0x02
PACKET_TYPE_KEYFRAME = 0x03
PACKET_TYPE_FRAME = 0x04
PACKET_TYPE_CONFIG = 0x05

# NAL unit types
NAL_TYPE_NON_IDR = 1
NAL_TYPE_IDR = 5
NAL_TYPE_SEI = 6
NAL_TYPE_SPS = 7
NAL_TYPE_PPS = 8

@dataclass
class H264StreamInfo:
    """Information about an H.264 stream"""
    client_id: str
    sps: Optional[bytes] = None
    pps: Optional[bytes] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    total_frames: int = 0
    keyframes: int = 0
    
class H264StreamHandler:
    """Handles H.264 video streams from multiple clients"""
    
    def __init__(self):
        self.streams: Dict[str, H264StreamInfo] = {}
        self.decoders: Dict[str, av.CodecContext] = {}
        
    async def handle_video_packet(self, client_id: str, data: bytes) -> Optional[np.ndarray]:
        """Process an H.264 video packet and return decoded frame if available"""
        
        if len(data) < 21:
            logger.error(f"Packet too small: {len(data)} bytes")
            return None
            
        # Parse packet header
        total_size = struct.unpack('>I', data[:4])[0]
        packet_type = data[4]
        timestamp = struct.unpack('>Q', data[5:13])[0]
        
        if packet_type in [PACKET_TYPE_KEYFRAME, PACKET_TYPE_FRAME]:
            # Video frame packet
            data_size = struct.unpack('>I', data[13:17])[0]
            flags = struct.unpack('>I', data[17:21])[0]
            h264_data = data[21:21+data_size]
            
            return await self._handle_video_frame(
                client_id, h264_data, timestamp, 
                packet_type == PACKET_TYPE_KEYFRAME
            )
            
        elif packet_type == PACKET_TYPE_SPS:
            # SPS packet
            sps_data = data[13:]
            await self._handle_sps(client_id, sps_data)
            
        elif packet_type == PACKET_TYPE_PPS:
            # PPS packet
            pps_data = data[13:]
            await self._handle_pps(client_id, pps_data)
            
        return None
    
    async def _handle_sps(self, client_id: str, sps_data: bytes):
        """Handle Sequence Parameter Set"""
        if client_id not in self.streams:
            self.streams[client_id] = H264StreamInfo(client_id)
            
        self.streams[client_id].sps = sps_data
        
        # Parse SPS for video dimensions
        # This is simplified - full SPS parsing is complex
        logger.info(f"Received SPS for {client_id}: {len(sps_data)} bytes")
        
    async def _handle_pps(self, client_id: str, pps_data: bytes):
        """Handle Picture Parameter Set"""
        if client_id not in self.streams:
            self.streams[client_id] = H264StreamInfo(client_id)
            
        self.streams[client_id].pps = pps_data
        logger.info(f"Received PPS for {client_id}: {len(pps_data)} bytes")
        
        # Initialize decoder after receiving both SPS and PPS
        if self.streams[client_id].sps and self.streams[client_id].pps:
            await self._initialize_decoder(client_id)
    
    async def _initialize_decoder(self, client_id: str):
        """Initialize H.264 decoder with SPS/PPS"""
        try:
            stream_info = self.streams[client_id]
            
            # Create decoder
            decoder = av.CodecContext.create('h264', 'r')
            
            # Set extradata (SPS + PPS)
            extradata = stream_info.sps + stream_info.pps
            decoder.extradata = extradata
            
            self.decoders[client_id] = decoder
            logger.info(f"Initialized H.264 decoder for {client_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize decoder: {e}")
    
    async def _handle_video_frame(
        self, 
        client_id: str, 
        h264_data: bytes, 
        timestamp: int,
        is_keyframe: bool
    ) -> Optional[np.ndarray]:
        """Decode H.264 frame data"""
        
        if client_id not in self.streams:
            logger.warning(f"No stream info for {client_id}")
            return None
            
        stream_info = self.streams[client_id]
        stream_info.total_frames += 1
        if is_keyframe:
            stream_info.keyframes += 1
            
        # Get decoder
        decoder = self.decoders.get(client_id)
        if not decoder:
            logger.warning(f"No decoder for {client_id}")
            return None
            
        try:
            # Create packet
            packet = av.Packet(h264_data)
            packet.pts = timestamp
            packet.dts = timestamp
            
            # Decode
            frames = decoder.decode(packet)
            
            for frame in frames:
                # Convert to numpy array
                img = frame.to_ndarray(format='bgr24')
                return img
                
        except Exception as e:
            logger.error(f"Decode error: {e}")
            
            # Request keyframe on decode error
            return None
    
    def get_stream_info(self, client_id: str) -> Optional[H264StreamInfo]:
        """Get stream information for a client"""
        return self.streams.get(client_id)
    
    def cleanup_stream(self, client_id: str):
        """Clean up resources for a disconnected client"""
        if client_id in self.decoders:
            # Note: PyAV handles cleanup automatically
            del self.decoders[client_id]
            
        if client_id in self.streams:
            del self.streams[client_id]
            
        logger.info(f"Cleaned up stream for {client_id}")

# Helper function to find NAL units in H.264 stream
def find_nal_units(data: bytes) -> List[tuple]:
    """Find NAL unit boundaries in H.264 data"""
    nal_units = []
    i = 0
    
    while i < len(data) - 4:
        # Look for start code (0x00 0x00 0x00 0x01 or 0x00 0x00 0x01)
        if (data[i:i+3] == b'\x00\x00\x01' or 
            data[i:i+4] == b'\x00\x00\x00\x01'):
            
            start_code_length = 4 if data[i:i+4] == b'\x00\x00\x00\x01' else 3
            nal_start = i + start_code_length
            
            # Find next start code
            next_start = len(data)
            j = nal_start
            while j < len(data) - 4:
                if (data[j:j+3] == b'\x00\x00\x01' or 
                    data[j:j+4] == b'\x00\x00\x00\x01'):
                    next_start = j
                    break
                j += 1
            
            if nal_start < len(data):
                nal_type = data[nal_start] & 0x1F
                nal_units.append((nal_start, next_start, nal_type))
            
            i = next_start
        else:
            i += 1
    
    return nal_units

def get_nal_unit_type_name(nal_type: int) -> str:
    """Get human-readable NAL unit type name"""
    types = {
        1: "Non-IDR Slice",
        5: "IDR Slice",
        6: "SEI",
        7: "SPS",
        8: "PPS",
        9: "AUD"
    }
    return types.get(nal_type, f"Unknown ({nal_type})")