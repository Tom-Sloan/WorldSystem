"""
Shared memory manager for SLAM3R to stream keyframes to mesh service.
Uses POSIX shared memory for zero-copy IPC.
"""

import posix_ipc
import numpy as np
import struct
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class SharedMemoryManager:
    """Manages shared memory segments for keyframe data."""
    
    def __init__(self, prefix="/slam3r_keyframe_"):
        self.prefix = prefix
        self.active_segments = {}  # shm_name -> (shm_object, mapfile)
        
    def write_keyframe(self, keyframe_id: str, points: np.ndarray, 
                      colors: np.ndarray, pose: np.ndarray,
                      bbox: Optional[np.ndarray] = None) -> str:
        """
        Write keyframe data to shared memory.
        
        Args:
            keyframe_id: Unique identifier for the keyframe
            points: Nx3 array of 3D points (float32)
            colors: Nx3 array of RGB colors (uint8)
            pose: 4x4 pose matrix (float32)
            bbox: Optional 6-element bounding box [min_x,min_y,min_z,max_x,max_y,max_z]
            
        Returns:
            shm_name: Name of the shared memory segment
        """
        # Ensure correct data types
        points = np.asarray(points, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.uint8)
        pose = np.asarray(pose, dtype=np.float32)
        
        if points.shape[0] != colors.shape[0]:
            raise ValueError("Points and colors must have same number of elements")
            
        # Calculate bounding box if not provided
        if bbox is None:
            bbox = np.array([
                points[:, 0].min(), points[:, 1].min(), points[:, 2].min(),
                points[:, 0].max(), points[:, 1].max(), points[:, 2].max()
            ], dtype=np.float32)
        else:
            bbox = np.asarray(bbox, dtype=np.float32)
            
        # Create shared memory name
        shm_name = f"{self.prefix}{keyframe_id}"
        
        # Calculate total size
        # Header: timestamp(8) + count(4) + color_format(4) + pose(64) + bbox(24) = 104 bytes
        header_format = "QII" + "f" * 16 + "f" * 6  # Q=uint64, I=uint32, f=float32
        header_size = struct.calcsize(header_format)
        
        # Data size
        points_size = points.nbytes  # N*3*4 bytes
        colors_size = colors.nbytes  # N*3*1 bytes
        total_size = header_size + points_size + colors_size
        
        # Close existing segment if any
        self._close_segment(shm_name)
        
        try:
            # Create shared memory segment
            shm = posix_ipc.SharedMemory(shm_name, posix_ipc.O_CREAT | posix_ipc.O_EXCL, 
                                        size=total_size)
            
            # Map to memory using mmap
            import mmap
            mapfile = mmap.mmap(shm.fd, total_size)
            
            # Write header
            timestamp_ns = int(pose[3, 3] * 1e9) if pose.shape[0] > 3 else 0  # Use pose timestamp if available
            header_data = struct.pack(header_format,
                timestamp_ns,              # timestamp_ns
                len(points),              # point_count
                0,                        # color_format (0=RGB)
                *pose.flatten(),          # pose_matrix (16 floats)
                *bbox                     # bbox (6 floats)
            )
            
            # Write data
            offset = 0
            mapfile[offset:offset+header_size] = header_data
            offset += header_size
            
            # Write points
            mapfile[offset:offset+points_size] = points.tobytes()
            offset += points_size
            
            # Write colors
            mapfile[offset:offset+colors_size] = colors.tobytes()
            
            # Store reference
            self.active_segments[shm_name] = (shm, mapfile)
            
            logger.debug(f"Wrote keyframe to shared memory: {shm_name} "
                        f"({len(points)} points, {total_size} bytes)")
            
            return shm_name
            
        except posix_ipc.ExistentialError:
            logger.error(f"Shared memory segment {shm_name} already exists")
            raise
        except Exception as e:
            logger.error(f"Failed to create shared memory: {e}")
            raise
            
    def _close_segment(self, shm_name: str):
        """Close and unlink a shared memory segment."""
        if shm_name in self.active_segments:
            shm, mapfile = self.active_segments[shm_name]
            try:
                mapfile.close()
                shm.close_fd()
                shm.unlink()
            except Exception as e:
                logger.warning(f"Error closing shared memory {shm_name}: {e}")
            del self.active_segments[shm_name]
            
    def cleanup(self):
        """Clean up all active shared memory segments."""
        for shm_name in list(self.active_segments.keys()):
            self._close_segment(shm_name)
            
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()

class StreamingKeyframePublisher:
    """Replaces world_point_cloud_buffer for streaming keyframes."""
    
    def __init__(self, keyframe_exchange=None):
        self.keyframe_exchange = keyframe_exchange
        self.shm_manager = SharedMemoryManager()
        self.keyframe_counter = 0
        
    async def publish_keyframe(self, keyframe_id: str, pose: np.ndarray, 
                              points: np.ndarray, colors: np.ndarray):
        """Publish keyframe to mesh service via shared memory + RabbitMQ."""
        import aio_pika
        import msgpack
        import time
        
        if self.keyframe_exchange is None:
            logger.warning("No keyframe exchange configured")
            return
            
        # Write to shared memory
        shm_key = self.shm_manager.write_keyframe(
            keyframe_id, points, colors, pose
        )
        
        # Calculate bounding box
        bbox = [
            float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 2].min()),
            float(points[:, 0].max()), float(points[:, 1].max()), float(points[:, 2].max())
        ]
        
        # Publish notification to RabbitMQ
        msg = {
            'type': 'keyframe.new',
            'keyframe_id': keyframe_id,
            'timestamp_ns': time.time_ns(),
            'pose_matrix': pose.tolist(),
            'shm_key': shm_key,
            'point_count': len(points),
            'bbox': bbox
        }
        
        await self.keyframe_exchange.publish(
            aio_pika.Message(body=msgpack.packb(msg),
                           content_type="application/msgpack"),
            routing_key='keyframe.new'
        )
        
        self.keyframe_counter += 1
        logger.info(f"Published keyframe {keyframe_id} to mesh service "
                   f"({len(points)} points)")
        
    def cleanup(self):
        """Clean up shared memory."""
        self.shm_manager.cleanup()