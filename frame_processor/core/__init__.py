"""
Core utilities and configuration for the frame processor.

This module provides:
- Configuration management with Pydantic
- Logging setup and utilities
- NTP time synchronization
- Common helper functions
"""

from .config import Config
from .utils import (
    setup_logging,
    get_logger,
    sync_ntp_time,
    async_sync_ntp_time,
    get_ntp_time_ns,
    format_dimensions,
    encode_frame_for_rabbitmq,
    decode_frame_from_rabbitmq,
    PerformanceTimer,
    JSONFormatter,
    ConsoleFormatter
)

__all__ = [
    'Config',
    'setup_logging',
    'get_logger',
    'sync_ntp_time',
    'async_sync_ntp_time',
    'get_ntp_time_ns',
    'format_dimensions',
    'encode_frame_for_rabbitmq',
    'decode_frame_from_rabbitmq',
    'PerformanceTimer',
    'JSONFormatter',
    'ConsoleFormatter'
]