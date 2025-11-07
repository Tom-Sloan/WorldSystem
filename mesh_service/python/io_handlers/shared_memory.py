"""
Shared memory reader for SLAM3R keyframes.

Reads point cloud data from /dev/shm shared memory segments written by SLAM3R.
"""

import os
import struct
import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SharedKeyframe:
    """Keyframe data from shared memory"""
    timestamp_ns: int
    point_count: int
    color_channels: int
    pose_matrix: np.ndarray  # 4x4 transformation matrix
    bbox: np.ndarray  # (6,) min/max bounds
    points: np.ndarray  # (N, 3) XYZ coordinates
    colors: np.ndarray  # (N, 3) or (N, 4) RGB(A) colors
    confidence: Optional[np.ndarray] = None  # (N,) per-point confidence scores


class SharedMemoryReader:
    """
    Reader for SLAM3R keyframe data from shared memory.

    Reads data written by C++ SLAM3R in /dev/shm with the structure:
    - Header: 104 bytes (timestamp, counts, pose, bbox)
    - Points: point_count * 3 * 4 bytes (float32 XYZ)
    - Colors: point_count * color_channels bytes (uint8 RGB/RGBA)
    """

    # C++ struct SharedKeyframe layout (updated to include confidence flag)
    HEADER_SIZE = 108  # 8 + 4 + 4 + 4 + 64 + 24 bytes (added has_confidence field)
    HEADER_FORMAT = '<QIII16f6f'  # little-endian: uint64, uint32, uint32, uint32(has_conf), 16 floats (4x4 matrix), 6 floats (bbox)

    def __init__(self, shm_path: str = '/dev/shm'):
        """
        Initialize shared memory reader.

        Args:
            shm_path: Path to shared memory directory (default: /dev/shm)
        """
        self.shm_path = shm_path
        if not os.path.exists(shm_path):
            raise RuntimeError(f"Shared memory path does not exist: {shm_path}")

    def read_keyframe(self, shm_key: str) -> Optional[SharedKeyframe]:
        """
        Read keyframe data from shared memory.

        Args:
            shm_key: Shared memory key (e.g., "slam3r_keyframe_0001" or "/slam3r_keyframe_0001")

        Returns:
            SharedKeyframe object or None if read failed
        """
        # Strip leading slash if present (SLAM3R uses /slam3r_keyframe_X format)
        if shm_key.startswith('/'):
            shm_key = shm_key[1:]

        shm_file = os.path.join(self.shm_path, shm_key)

        if not os.path.exists(shm_file):
            logger.warning(f"Shared memory file does not exist: {shm_file}")
            return None

        try:
            with open(shm_file, 'rb') as f:
                # Read header
                header_bytes = f.read(self.HEADER_SIZE)
                if len(header_bytes) < self.HEADER_SIZE:
                    logger.error(f"Incomplete header read: {len(header_bytes)} < {self.HEADER_SIZE}")
                    return None

                # Parse header
                header_data = struct.unpack(self.HEADER_FORMAT, header_bytes)
                timestamp_ns = header_data[0]
                point_count = header_data[1]
                color_channels = header_data[2]
                has_confidence = header_data[3]  # New field: 1 if confidence data is present, 0 otherwise
                # SLAM3R writes pose.T.flatten(), so we need to reshape and transpose back
                pose_matrix = np.array(header_data[4:20], dtype=np.float32).reshape(4, 4).T
                bbox = np.array(header_data[20:26], dtype=np.float32)

                # Read points (N x 3 float32)
                points_size = point_count * 3 * 4
                points_bytes = f.read(points_size)
                if len(points_bytes) < points_size:
                    logger.error(f"Incomplete points read: {len(points_bytes)} < {points_size}")
                    return None

                points = np.frombuffer(points_bytes, dtype=np.float32).reshape(-1, 3)

                # Read colors (N x color_channels uint8)
                colors_size = point_count * color_channels
                colors_bytes = f.read(colors_size)
                if len(colors_bytes) < colors_size:
                    logger.error(f"Incomplete colors read: {len(colors_bytes)} < {colors_size}")
                    return None

                colors = np.frombuffer(colors_bytes, dtype=np.uint8).reshape(-1, color_channels)

                # Read confidence (N x float32) if present
                confidence = None
                if has_confidence:
                    confidence_size = point_count * 4  # 4 bytes per float32
                    confidence_bytes = f.read(confidence_size)
                    if len(confidence_bytes) < confidence_size:
                        logger.warning(f"Incomplete confidence read: {len(confidence_bytes)} < {confidence_size}")
                        # Continue without confidence rather than failing
                    else:
                        confidence = np.frombuffer(confidence_bytes, dtype=np.float32)
                        logger.debug(f"Read confidence data: min={confidence.min():.2f}, max={confidence.max():.2f}, mean={confidence.mean():.2f}")

                logger.debug(f"Read keyframe {shm_key}: {point_count} points" +
                            (f" with confidence" if confidence is not None else ""))

                return SharedKeyframe(
                    timestamp_ns=timestamp_ns,
                    point_count=point_count,
                    color_channels=color_channels,
                    pose_matrix=pose_matrix,
                    bbox=bbox,
                    points=points,
                    colors=colors,
                    confidence=confidence
                )

        except Exception as e:
            logger.error(f"Error reading shared memory {shm_key}: {e}")
            return None

    def unlink(self, shm_key: str) -> bool:
        """
        Remove shared memory segment.

        Args:
            shm_key: Shared memory key to remove

        Returns:
            True if successfully removed, False otherwise
        """
        # Strip leading slash if present (POSIX SHM names start with /)
        if shm_key.startswith('/'):
            shm_key = shm_key[1:]

        shm_file = os.path.join(self.shm_path, shm_key)
        try:
            if os.path.exists(shm_file):
                os.unlink(shm_file)
                logger.debug(f"Unlinked shared memory: {shm_key}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error unlinking shared memory {shm_key}: {e}")
            return False

    @staticmethod
    def calculate_size(point_count: int, color_channels: int = 3, has_confidence: bool = False) -> int:
        """
        Calculate total size needed for a keyframe in shared memory.

        Args:
            point_count: Number of points
            color_channels: Color channels per point (3=RGB, 4=RGBA)
            has_confidence: Whether confidence data is included

        Returns:
            Total size in bytes
        """
        header_size = 108  # Updated from 104 to include has_confidence flag
        points_size = point_count * 3 * 4  # 3 floats per point
        colors_size = point_count * color_channels  # uint8 per channel
        confidence_size = point_count * 4 if has_confidence else 0  # float32 per point
        return header_size + points_size + colors_size + confidence_size
