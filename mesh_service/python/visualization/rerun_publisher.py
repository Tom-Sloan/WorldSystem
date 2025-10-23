"""
Rerun publisher for SLAM visualization.

Handles logging of point clouds, video frames, and statistics to Rerun.
"""

import rerun as rr
import numpy as np
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class RerunPublisher:
    """Publishes SLAM data to Rerun viewer"""

    def __init__(
        self,
        log_interval_seconds: float = 0.5,
        video_log_interval_seconds: float = 0.1
    ):
        """
        Initialize Rerun publisher.

        Args:
            log_interval_seconds: Interval for point cloud updates (default: 500ms)
            video_log_interval_seconds: Interval for video frame updates (default: 100ms)
        """
        self.log_interval = log_interval_seconds
        self.video_log_interval = video_log_interval_seconds

        self.last_point_cloud_log = time.time()
        self.last_video_log = time.time()

        self.point_cloud_update_count = 0
        self.video_frame_count = 0

        logger.info(
            f"RerunPublisher initialized: pc_interval={log_interval_seconds}s, "
            f"video_interval={video_log_interval_seconds}s"
        )

    def should_log_point_cloud(self) -> bool:
        """Check if it's time to log point cloud update"""
        elapsed = time.time() - self.last_point_cloud_log
        return elapsed >= self.log_interval

    def should_log_video(self) -> bool:
        """Check if it's time to log video frame"""
        elapsed = time.time() - self.last_video_log
        return elapsed >= self.video_log_interval

    def log_point_cloud(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        timestamp_ns: Optional[int] = None
    ) -> None:
        """
        Log point cloud to Rerun.

        Args:
            points: (N, 3) float32 XYZ coordinates
            colors: (N, 3) uint8 RGB colors
            timestamp_ns: Optional timestamp in nanoseconds
        """
        if len(points) == 0:
            return

        # Set time context
        if timestamp_ns:
            rr.set_time_nanos("sensor_time", timestamp_ns)

        # Log to single entity (replaces previous)
        rr.log(
            "/slam/world/accumulated_cloud",
            rr.Points3D(points, colors=colors, radii=0.01)
        )

        self.last_point_cloud_log = time.time()
        self.point_cloud_update_count += 1

        logger.debug(f"Logged point cloud: {len(points):,} points")

    def log_video_frame(
        self,
        frame: np.ndarray,
        timestamp_ns: Optional[int] = None
    ) -> None:
        """
        Log video frame to Rerun.

        Args:
            frame: (H, W, 3) uint8 RGB image
            timestamp_ns: Optional timestamp in nanoseconds
        """
        if frame is None or frame.size == 0:
            return

        # Set time context
        if timestamp_ns:
            rr.set_time_nanos("sensor_time", timestamp_ns)

        # Log video frame
        rr.log("/slam/camera/video", rr.Image(frame))

        self.last_video_log = time.time()
        self.video_frame_count += 1

        logger.debug(f"Logged video frame: {frame.shape}")

    def log_camera_pose(
        self,
        pose: np.ndarray,
        keyframe_id: str,
        timestamp_ns: Optional[int] = None
    ) -> None:
        """
        Log camera pose to trajectory.

        Args:
            pose: 4x4 transformation matrix
            keyframe_id: Unique keyframe identifier
            timestamp_ns: Optional timestamp in nanoseconds
        """
        if timestamp_ns:
            rr.set_time_nanos("sensor_time", timestamp_ns)

        # Extract translation and rotation
        translation = pose[:3, 3]
        rotation_matrix = pose[:3, :3]

        # Log transform
        rr.log(
            f"/slam/world/camera_trajectory/{keyframe_id}",
            rr.Transform3D(
                translation=translation,
                mat3x3=rotation_matrix.flatten()
            )
        )

        logger.debug(f"Logged camera pose: {keyframe_id}")

    def log_statistics(self, stats: dict) -> None:
        """
        Log statistics panel.

        Args:
            stats: Dictionary with statistics
        """
        stats_text = f"""# 🗺️ SLAM Point Cloud Statistics

## Point Cloud
- **Total Keyframes:** {stats.get('total_keyframes', 0)}
- **Processed:** {stats.get('processed_keyframes', 0)}
- **Raw Points:** {stats.get('raw_point_count', 0):,}
- **Display Points:** {stats.get('display_point_count', 0):,}
- **Compression:** {stats.get('compression_ratio', 0):.1f}x
- **Memory:** ~{stats.get('memory_estimate_mb', 0):.1f} MB

## Configuration
- **Voxel Size:** {stats.get('voxel_size_cm', 0):.1f} cm
- **Total Points Received:** {stats.get('total_points_received', 0):,}

## Performance
- **Point Cloud Updates:** {self.point_cloud_update_count}
- **Video Frames:** {self.video_frame_count}
"""

        rr.log("/slam/stats", rr.TextDocument(stats_text, media_type=rr.MediaType.MARKDOWN))

    def get_statistics(self) -> dict:
        """Get publisher statistics"""
        return {
            'point_cloud_updates': self.point_cloud_update_count,
            'video_frames': self.video_frame_count
        }
