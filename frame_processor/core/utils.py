"""
Utility functions and logging setup.

This module provides shared utilities including structured logging configuration,
NTP time synchronization, and common helper functions.
"""

import logging
import logging.handlers
import sys
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import json
import ntplib
import socket
import asyncio
from datetime import datetime
import numpy as np
import cv2


# Global NTP client and synchronization variables
ntp_client = ntplib.NTPClient()
ntp_time_offset: float = 0.0
last_ntp_sync: float = 0
ntp_sync_lock = threading.Lock()


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, "extra"):
            log_data.update(record.extra)
            
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors for console output."""
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        # Format the message
        message = super().format(record)
        
        # Reset levelname for other handlers
        record.levelname = levelname
        
        return message


class MessageFilter(logging.Filter):
    """Filter to suppress specific unwanted log messages."""
    
    def __init__(self, patterns_to_suppress: list = None):
        """
        Initialize filter with patterns to suppress.
        
        Args:
            patterns_to_suppress: List of string patterns to filter out
        """
        super().__init__()
        self.patterns_to_suppress = patterns_to_suppress or [
            "For numpy array image",
            "Using cache found in",
            "Downloading: ",
            "to /root/.cache/torch",
        ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter out messages containing specific patterns.
        
        Returns:
            False if message should be suppressed, True otherwise
        """
        message = record.getMessage()
        for pattern in self.patterns_to_suppress:
            if pattern in message:
                return False
        return True


def setup_logging(log_level: str = "INFO", log_dir: Optional[Path] = None, 
                  disable_console_when_rich: bool = True,
                  suppress_external_logs: bool = True) -> None:
    """
    Configure logging for the application.
    
    Sets up both console and file logging with appropriate formatters.
    Replaces all print statements with proper logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (creates if not exists)
        disable_console_when_rich: Disable console handler when rich terminal is enabled
        suppress_external_logs: Apply filters to suppress noisy external library messages
    """
    # Create log directory if specified
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Add message filter to suppress unwanted messages (if enabled)
    if suppress_external_logs:
        message_filter = MessageFilter()
        root_logger.addFilter(message_filter)
    
    # Check if rich terminal is enabled
    import os
    rich_enabled = os.environ.get('ENABLE_RICH_TERMINAL', 'false').lower() == 'true'
    
    # Allow override of suppress_external_logs via environment
    if os.environ.get('SUPPRESS_EXTERNAL_LOGS'):
        suppress_external_logs = os.environ.get('SUPPRESS_EXTERNAL_LOGS', 'true').lower() == 'true'
    
    # Console handler with colored output (unless rich is enabled and we want to disable)
    if not (rich_enabled and disable_console_when_rich):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level))
        console_formatter = ConsoleFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handlers if log_dir is specified
    if log_dir:
        # Main log file with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "frame_processor.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)
        
        # Error log file
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "errors.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(error_handler)
    
    # Set levels for noisy libraries
    logging.getLogger("pika").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    
    # Suppress SAM2 and related libraries
    logging.getLogger("sam2").setLevel(logging.WARNING)
    logging.getLogger("detectron2").setLevel(logging.WARNING)
    logging.getLogger("fvcore").setLevel(logging.WARNING)
    logging.getLogger("iopath").setLevel(logging.WARNING)
    
    # Suppress any library that might be logging "For numpy array image..."
    # This is likely coming from a computer vision library
    logging.getLogger("cv2").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("torchvision").setLevel(logging.WARNING)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging initialized",
        extra={
            "log_level": log_level,
            "log_dir": str(log_dir) if log_dir else None,
            "handlers": len(root_logger.handlers)
        }
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def sync_ntp_time(ntp_server: str = "pool.ntp.org", timeout: int = 5) -> bool:
    """
    Synchronize with NTP server to get accurate time.
    
    Args:
        ntp_server: NTP server hostname
        timeout: Request timeout in seconds
        
    Returns:
        True if synchronization successful
    """
    global ntp_time_offset, last_ntp_sync
    logger = get_logger(__name__)
    
    try:
        with ntp_sync_lock:
            response = ntp_client.request(ntp_server, timeout=timeout)
            ntp_time_offset = response.offset
            last_ntp_sync = time.time()
            
        logger.info(
            "NTP synchronization successful",
            extra={
                "server": ntp_server,
                "offset": ntp_time_offset,
                "stratum": response.stratum
            }
        )
        return True
        
    except (ntplib.NTPException, socket.gaierror, socket.timeout) as e:
        logger.warning(
            "NTP synchronization failed",
            extra={
                "server": ntp_server,
                "error": str(e)
            }
        )
        return False


async def async_sync_ntp_time(ntp_server: str = "pool.ntp.org", timeout: int = 5) -> float:
    """
    Asynchronously synchronize with NTP server.
    
    This runs the blocking NTP request in a thread pool to avoid blocking the event loop.
    
    Args:
        ntp_server: NTP server hostname
        timeout: Request timeout in seconds
        
    Returns:
        Time offset in seconds
    """
    loop = asyncio.get_event_loop()
    
    # Run the blocking NTP request in a thread pool
    def _sync():
        sync_ntp_time(ntp_server, timeout)
        return ntp_time_offset
    
    return await loop.run_in_executor(None, _sync)


def get_ntp_time_ns(sync_interval: int = 60) -> int:
    """
    Get current time in nanoseconds, synchronized with NTP.
    
    Args:
        sync_interval: Seconds between automatic resyncs
        
    Returns:
        Current time in nanoseconds
    """
    global last_ntp_sync
    
    # Check if resync needed
    if time.time() - last_ntp_sync > sync_interval:
        # Start async resync
        threading.Thread(target=sync_ntp_time, daemon=True).start()
    
    # Return adjusted time
    current_time = time.time() + ntp_time_offset
    return int(current_time * 1e9)


def format_dimensions(dimensions: Dict[str, Any]) -> str:
    """
    Format dimension dictionary into readable string.
    
    Args:
        dimensions: Dictionary with width, height, depth, unit
        
    Returns:
        Formatted string like "30.0 x 20.0 x 10.0 cm"
    """
    if not dimensions:
        return "Unknown"
    
    width = dimensions.get("width", 0)
    height = dimensions.get("height", 0)
    depth = dimensions.get("depth", 0)
    unit = dimensions.get("unit", "cm")
    
    return f"{width} x {height} x {depth} {unit}"


def encode_frame_for_rabbitmq(frame: np.ndarray, quality: int = 85) -> bytes:
    """
    Encode frame for RabbitMQ transmission.
    
    Args:
        frame: OpenCV frame to encode
        quality: JPEG quality (0-100)
        
    Returns:
        Encoded frame as bytes
    """
    _, encoded = cv2.imencode(
        ".jpg", 
        frame, 
        [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    return encoded.tobytes()


def decode_frame_from_rabbitmq(data: bytes) -> Optional[np.ndarray]:
    """
    Decode frame from RabbitMQ message.
    
    Args:
        data: Encoded frame data
        
    Returns:
        Decoded frame or None if invalid
    """
    try:
        np_arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning(f"Failed to decode frame from RabbitMQ: {e}")
        return None


class PerformanceTimer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or get_logger(__name__)
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        duration_ms = (self.end_time - self.start_time) * 1000
        
        self.logger.debug(
            f"{self.name} took {duration_ms:.2f}ms",
            extra={
                "operation": self.name,
                "duration_ms": duration_ms
            }
        )
        
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0