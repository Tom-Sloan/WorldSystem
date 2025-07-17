"""
Shared memory manager for SLAM3R to stream keyframes to mesh service.
Uses POSIX shared memory for zero-copy IPC.
"""

import posix_ipc
import numpy as np
import struct
import logging
import os
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class SharedMemoryManager:
    """Manages shared memory segments for keyframe data."""
    
    def __init__(self, prefix="/slam3r_keyframe_"):
        self.prefix = prefix
        self.active_segments = {}  # shm_name -> (shm_object, mapfile)
        self._cleanup_stale_segments()
        
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
            if len(points) > 0:
                bbox = np.array([
                    points[:, 0].min(), points[:, 1].min(), points[:, 2].min(),
                    points[:, 0].max(), points[:, 1].max(), points[:, 2].max()
                ], dtype=np.float32)
            else:
                bbox = np.zeros(6, dtype=np.float32)
        else:
            bbox = np.asarray(bbox, dtype=np.float32)
            
        # Create shared memory name
        shm_name = f"{self.prefix}{keyframe_id}"
        
        # Calculate total size
        # Header: timestamp(8) + count(4) + color_channels(4) + pose(64) + bbox(24) = 104 bytes
        header_format = "QII" + "f" * 16 + "f" * 6  # Q=uint64, I=uint32, f=float32
        header_size = struct.calcsize(header_format)
        
        # Verify header size matches C++ struct
        expected_size = 8 + 4 + 4 + 64 + 24  # Should be 104
        if header_size != expected_size:
            logger.warning(f"Header size mismatch! Python: {header_size}, Expected: {expected_size}")
        
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
            
            # DEBUG: Log the pose matrix being written
            logger.info(f"[SHM SLAM3R DEBUG] Writing pose matrix to shared memory:")
            logger.info(f"[SHM SLAM3R DEBUG] Pose shape: {pose.shape}")
            logger.info("[SHM SLAM3R DEBUG] Pose matrix:")
            for row in range(min(4, pose.shape[0])):
                row_str = "  ["
                for col in range(min(4, pose.shape[1])):
                    row_str += f"{pose[row, col]:10.4f}"
                    if col < 3:
                        row_str += ", "
                row_str += "]"
                logger.info(f"[SHM SLAM3R DEBUG] {row_str}")
            logger.info(f"[SHM SLAM3R DEBUG] Camera position (translation): [{pose[0,3]:.4f}, {pose[1,3]:.4f}, {pose[2,3]:.4f}]")
            
            # Check if pose is identity or invalid
            is_identity = np.allclose(pose, np.eye(4), atol=1e-6)
            is_zero = np.allclose(pose, 0, atol=1e-6)
            if is_identity:
                logger.warning("[SHM SLAM3R WARNING] Pose matrix is identity - no camera transform!")
            elif is_zero:
                logger.warning("[SHM SLAM3R WARNING] Pose matrix is all zeros - invalid!")
            
            # CRITICAL FIX: Flatten pose in row-major order to match C++ struct layout
            # The SharedKeyframe C++ struct expects row-major order where translation is at [12,13,14]
            # The mesh service expects a transposed format where translation is in the bottom row
            pose_row_major = pose.T.flatten()  # Transpose then flatten to get translation at [12,13,14]
            
            # Debug: Verify the translation is now at the correct indices
            logger.info(f"[SHM SLAM3R FIX] Original pose translation at [0,3],[1,3],[2,3]: [{pose[0,3]:.4f}, {pose[1,3]:.4f}, {pose[2,3]:.4f}]")
            logger.info(f"[SHM SLAM3R FIX] Row-major flattened translation at [12],[13],[14]: [{pose_row_major[12]:.4f}, {pose_row_major[13]:.4f}, {pose_row_major[14]:.4f}]")
            
            # Additional debug: Print full flattened array to verify layout
            logger.debug("[SHM SLAM3R FIX] Full row-major flattened pose:")
            for i in range(0, 16, 4):
                logger.debug(f"  [{i:2d}-{i+3:2d}]: {pose_row_major[i]:.4f}, {pose_row_major[i+1]:.4f}, {pose_row_major[i+2]:.4f}, {pose_row_major[i+3]:.4f}")
            
            # Verify the matrix can be reconstructed correctly
            pose_reconstructed = np.array(pose_row_major).reshape(4, 4)  # This creates row-major
            logger.debug(f"[SHM SLAM3R FIX] Reconstructed pose (should match transposed original):")
            logger.debug(f"{pose_reconstructed}")
            logger.debug(f"[SHM SLAM3R FIX] Original pose transposed:")
            logger.debug(f"{pose.T}")
            
            header_data = struct.pack(header_format,
                timestamp_ns,              # timestamp_ns
                len(points),              # point_count
                3,                        # color_channels (3=RGB)
                *pose_row_major,          # pose_matrix (16 floats, row-major)
                *bbox                     # bbox (6 floats)
            )
            
            logger.debug(f"Header format: {header_format}, size: {header_size}")
            logger.debug(f"Points shape: {points.shape}, dtype: {points.dtype}")
            logger.debug(f"Points size: {points_size} bytes, expected: {len(points) * 3 * 4}")
            logger.debug(f"First few points: {points[:3] if len(points) > 0 else 'none'}")
            logger.debug(f"Last few points: {points[-3:] if len(points) > 0 else 'none'}")
            logger.debug(f"Total size calculation: header({header_size}) + points({points_size}) + colors({colors_size}) = {total_size}")
            
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
        else:
            # Try to unlink even if not in active_segments (cleanup stale segments)
            try:
                shm = posix_ipc.SharedMemory(shm_name)
                shm.unlink()
                logger.info(f"Cleaned up stale shared memory segment: {shm_name}")
            except posix_ipc.ExistentialError:
                # Segment doesn't exist, that's fine
                pass
            except Exception as e:
                logger.debug(f"Could not cleanup segment {shm_name}: {e}")
            
    def cleanup(self):
        """Clean up all active shared memory segments."""
        for shm_name in list(self.active_segments.keys()):
            self._close_segment(shm_name)
            
    def _cleanup_stale_segments(self):
        """Clean up any stale shared memory segments from previous runs."""
        try:
            logger.info("Cleaning up stale shared memory segments")
            # List all shared memory segments
            shm_dir = "/dev/shm"
            if os.path.exists(shm_dir):
                for filename in os.listdir(shm_dir):
                    if filename.startswith(self.prefix.lstrip('/')):
                        shm_name = f"/{filename}"
                        try:
                            shm = posix_ipc.SharedMemory(shm_name)
                            shm.unlink()
                            logger.info(f"Cleaned up stale segment on startup: {shm_name}")
                        except Exception as e:
                            logger.debug(f"Could not cleanup {shm_name}: {e}")
        except Exception as e:
            logger.warning(f"Error during startup cleanup: {e}")
            
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
        
        # Always write to shared memory (even if RabbitMQ is not configured)
        shm_key = self.shm_manager.write_keyframe(
            keyframe_id, points, colors, pose
        )
        logger.info(f"Wrote keyframe {keyframe_id} to shared memory: {shm_key}")
        
        if self.keyframe_exchange is None:
            logger.warning("No keyframe exchange configured - skipping RabbitMQ notification")
            return
        
        # Calculate bounding box (handle empty points)
        if len(points) > 0:
            bbox = [
                float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 2].min()),
                float(points[:, 0].max()), float(points[:, 1].max()), float(points[:, 2].max())
            ]
            logger.info(f"Keyframe {keyframe_id} bbox: min=({bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}), "
                       f"max=({bbox[3]:.2f}, {bbox[4]:.2f}, {bbox[5]:.2f})")
        else:
            # Default bbox for empty point cloud
            bbox = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            logger.warning(f"Empty point cloud for keyframe {keyframe_id}")
        
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