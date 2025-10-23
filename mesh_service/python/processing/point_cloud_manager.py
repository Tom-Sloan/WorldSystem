"""
Point cloud manager with Open3D voxel downsampling.

Efficiently accumulates and downsamples SLAM point clouds for real-time visualization.
"""

import open3d as o3d
import numpy as np
import time
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class PointCloudManager:
    """
    Efficient point cloud accumulator using Open3D.

    Features:
    - Incremental point cloud merging
    - Automatic voxel downsampling when memory cap reached
    - Fast extraction for Rerun logging
    - Deduplication of keyframes
    """

    def __init__(
        self,
        voxel_size: float = 0.02,
        max_points: int = 500_000,
        max_raw_points: int = 1_000_000
    ):
        """
        Initialize point cloud manager.

        Args:
            voxel_size: Voxel size in meters for downsampling (default: 2cm)
            max_points: Maximum points to display in Rerun (memory cap)
            max_raw_points: Maximum raw points before auto-downsampling
        """
        self.voxel_size = voxel_size
        self.max_points = max_points
        self.max_raw_points = max_raw_points

        # Accumulated point cloud (Open3D)
        self.accumulated_pcd = o3d.geometry.PointCloud()

        # Tracking
        self.processed_keyframes = set()
        self.total_keyframes = 0
        self.total_points_raw = 0
        self.last_downsample_time = time.time()

        logger.info(
            f"PointCloudManager initialized: voxel_size={voxel_size*100:.1f}cm, "
            f"max_points={max_points:,}, max_raw_points={max_raw_points:,}"
        )

    def add_keyframe(
        self,
        keyframe_id: str,
        points: np.ndarray,
        colors: np.ndarray,
        pose: Optional[np.ndarray] = None
    ) -> bool:
        """
        Add keyframe to accumulated point cloud.

        Args:
            keyframe_id: Unique keyframe identifier
            points: Point coordinates (N, 3) float32
            colors: Point colors (N, 3) or (N, 4) uint8
            pose: Optional 4x4 transformation matrix to world coordinates

        Returns:
            True if keyframe was added (new), False if skipped (duplicate)
        """
        # Skip if already processed
        if keyframe_id in self.processed_keyframes:
            logger.debug(f"Skipping duplicate keyframe: {keyframe_id}")
            return False

        self.processed_keyframes.add(keyframe_id)
        self.total_keyframes += 1

        start_time = time.time()

        # Transform points to world coordinates if pose provided
        if pose is not None:
            points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
            points = (pose @ points_homogeneous.T).T[:, :3]

        # Create Open3D point cloud for this keyframe
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

        # Normalize colors to [0, 1] range for Open3D
        if colors.dtype == np.uint8:
            colors_normalized = colors[:, :3].astype(np.float64) / 255.0
        else:
            colors_normalized = colors[:, :3].astype(np.float64)

        new_pcd.colors = o3d.utility.Vector3dVector(colors_normalized)

        # Merge with accumulated cloud
        self.accumulated_pcd += new_pcd
        self.total_points_raw += len(points)

        # Auto-downsample if exceeding memory cap
        current_point_count = len(self.accumulated_pcd.points)
        if current_point_count > self.max_raw_points:
            downsample_start = time.time()
            logger.info(
                f"⚡ Auto-downsampling: {current_point_count:,} -> target ~{self.max_points:,} points"
            )

            self.accumulated_pcd = self.accumulated_pcd.voxel_down_sample(self.voxel_size)
            self.last_downsample_time = time.time()

            downsample_time = (time.time() - downsample_start) * 1000
            new_count = len(self.accumulated_pcd.points)
            logger.info(
                f"   Downsampled to {new_count:,} points in {downsample_time:.1f}ms "
                f"(compression: {current_point_count/new_count:.1f}x)"
            )

        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"Added keyframe {keyframe_id}: {len(points)} points in {elapsed_ms:.1f}ms"
        )

        return True

    def get_downsampled_cloud(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get downsampled point cloud for visualization.

        Returns:
            Tuple of (points, colors):
            - points: (N, 3) float32 XYZ coordinates
            - colors: (N, 3) uint8 RGB colors
        """
        if len(self.accumulated_pcd.points) == 0:
            return np.array([]), np.array([])

        start_time = time.time()

        # Downsample for display (doesn't modify accumulated_pcd)
        display_pcd = self.accumulated_pcd.voxel_down_sample(self.voxel_size)

        # Extract numpy arrays
        points = np.asarray(display_pcd.points, dtype=np.float32)
        colors = (np.asarray(display_pcd.colors) * 255).astype(np.uint8)

        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"Generated downsampled cloud: {len(points):,} points in {elapsed_ms:.1f}ms"
        )

        return points, colors

    def get_statistics(self) -> dict:
        """
        Get current statistics.

        Returns:
            Dictionary with point cloud statistics
        """
        raw_count = len(self.accumulated_pcd.points)
        display_pcd = self.accumulated_pcd.voxel_down_sample(self.voxel_size)
        display_count = len(display_pcd.points)

        return {
            'total_keyframes': self.total_keyframes,
            'processed_keyframes': len(self.processed_keyframes),
            'raw_point_count': raw_count,
            'display_point_count': display_count,
            'compression_ratio': raw_count / display_count if display_count > 0 else 0,
            'total_points_received': self.total_points_raw,
            'voxel_size_cm': self.voxel_size * 100,
            'memory_estimate_mb': (raw_count * 15) / (1024 * 1024)  # 15 bytes per point
        }

    def clear(self):
        """Clear all accumulated data."""
        self.accumulated_pcd.clear()
        self.processed_keyframes.clear()
        self.total_keyframes = 0
        self.total_points_raw = 0
        logger.info("Point cloud manager cleared")

    def __repr__(self) -> str:
        """String representation"""
        stats = self.get_statistics()
        return (
            f"PointCloudManager(keyframes={stats['total_keyframes']}, "
            f"raw_points={stats['raw_point_count']:,}, "
            f"display_points={stats['display_point_count']:,}, "
            f"voxel_size={self.voxel_size*100:.1f}cm)"
        )
