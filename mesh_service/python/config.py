"""
Configuration management for mesh service.
"""

import os
from dataclasses import dataclass


@dataclass
class MeshServiceConfig:
    """Configuration for mesh service"""

    # RabbitMQ
    rabbitmq_url: str = os.getenv('RABBITMQ_URL', 'amqp://guest:guest@localhost:5672/')

    # Rerun
    rerun_host: str = os.getenv('RERUN_HOST', '127.0.0.1')
    rerun_port: int = int(os.getenv('RERUN_PORT', '9876'))
    rerun_connect: bool = os.getenv('RERUN_CONNECT', 'true').lower() == 'true'

    # Point Cloud Processing
    voxel_size: float = float(os.getenv('VOXEL_SIZE', '0.02'))  # 2cm voxels
    max_points: int = int(os.getenv('MAX_POINTS', '500000'))  # Memory cap
    max_raw_points: int = int(os.getenv('MAX_RAW_POINTS', '1000000'))  # Before downsample
    log_interval_ms: int = int(os.getenv('LOG_INTERVAL_MS', '500'))  # Rerun update rate

    # Shared Memory
    unlink_shm: bool = os.getenv('MESH_SERVICE_UNLINK_SHM', 'true').lower() == 'true'
    shm_path: str = '/dev/shm'

    # Video Processing
    enable_video: bool = os.getenv('ENABLE_VIDEO', 'true').lower() == 'true'
    video_log_interval_ms: int = int(os.getenv('VIDEO_LOG_INTERVAL_MS', '1'))  # 10fps to Rerun

    # Performance
    enable_metrics: bool = os.getenv('ENABLE_METRICS', 'true').lower() == 'true'
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')

    # Optimization (Background high-quality point cloud generation)
    enable_optimization: bool = os.getenv('ENABLE_OPTIMIZATION', 'true').lower() == 'true'
    optimization_interval_seconds: float = float(os.getenv('OPTIMIZATION_INTERVAL_SECONDS', '15.0'))
    optimization_conf_threshold: float = float(os.getenv('OPTIMIZATION_CONF_THRESHOLD', '10.0'))
    optimization_target_points: int = int(os.getenv('OPTIMIZATION_TARGET_POINTS', '1000000'))
    max_cached_keyframes: int = int(os.getenv('MAX_CACHED_KEYFRAMES', '100'))

    @property
    def rerun_url(self) -> str:
        """Get Rerun gRPC URL"""
        return f"rerun+http://{self.rerun_host}:{self.rerun_port}/proxy"

    @property
    def log_interval_seconds(self) -> float:
        """Get log interval in seconds"""
        return self.log_interval_ms / 1000.0

    @property
    def video_log_interval_seconds(self) -> float:
        """Get video log interval in seconds"""
        return self.video_log_interval_ms / 1000.0

    def __str__(self) -> str:
        """Human-readable config"""
        opt_str = f"""
  Optimization: {self.enable_optimization}
    Interval: {self.optimization_interval_seconds}s
    Conf Threshold: {self.optimization_conf_threshold}
    Target Points: {self.optimization_target_points:,}
    Max Cached Keyframes: {self.max_cached_keyframes}""" if self.enable_optimization else "\n  Optimization: Disabled"

        return f"""Mesh Service Configuration:
  RabbitMQ: {self.rabbitmq_url}
  Rerun: {self.rerun_url} (connect={self.rerun_connect})
  Voxel Size: {self.voxel_size * 100:.1f} cm
  Max Points: {self.max_points:,}
  Log Interval: {self.log_interval_ms}ms
  Video Enabled: {self.enable_video}
  Unlink SHM: {self.unlink_shm}{opt_str}
"""


# Global config instance
config = MeshServiceConfig()
