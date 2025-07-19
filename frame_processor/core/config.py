"""
Centralized configuration management using Pydantic.

This module provides type-safe configuration with validation, enabling easy
swapping of detection and tracking algorithms via environment variables.
"""

from pydantic import field_validator, Field, model_validator
from pydantic_settings import BaseSettings
from typing import Literal, Optional
import os
from pathlib import Path


class Config(BaseSettings):
    """
    Centralized configuration with modular component selection.
    
    All settings can be overridden via environment variables or .env file.
    """
    
    # ========== Component Selection ==========
    detector_type: Literal["yolo", "detectron2", "grounding_dino"] = Field(
        default="yolo",
        description="Detection algorithm to use"
    )
    tracker_type: Literal["iou", "sort", "deep_sort", "bytetrack"] = Field(
        default="iou",
        description="Tracking algorithm to use"
    )
    
    # ========== Detector Configuration ==========
    detector_model: str = Field(
        default="yolov11l.pt",
        description="Path to detector model weights"
    )
    detector_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for detections"
    )
    detector_device: str = Field(
        default="cuda",
        description="Device to run detector on (cuda/cpu)"
    )
    detector_batch_size: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Batch size for detection inference"
    )
    
    # ========== Tracker Configuration ==========
    tracker_iou_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="IOU threshold for matching detections to tracks"
    )
    tracker_max_lost: int = Field(
        default=10,
        ge=1,
        description="Maximum frames before removing lost track"
    )
    tracker_max_tracks: int = Field(
        default=100,
        ge=1,
        description="Maximum number of simultaneous tracks"
    )
    process_after_seconds: float = Field(
        default=1.5,
        ge=0.1,
        description="Time to wait before processing object for API"
    )
    reprocess_interval_seconds: float = Field(
        default=3.0,
        ge=0.1,
        description="Time between reprocessing attempts"
    )
    
    # ========== API Configuration (Feature Flags) ==========
    use_gcs: bool = Field(
        default=False,
        description="Enable Google Cloud Storage for image uploads"
    )
    use_serpapi: bool = Field(
        default=False,
        description="Enable Google Lens via SerpAPI for object identification"
    )
    use_perplexity: bool = Field(
        default=False,
        description="Enable Perplexity AI for dimension lookup"
    )
    
    # ========== API Keys ==========
    serpapi_key: Optional[str] = Field(
        default=None,
        description="SerpAPI key for Google Lens"
    )
    perplexity_key: Optional[str] = Field(
        default=None,
        description="Perplexity AI API key"
    )
    gcs_bucket_name: str = Field(
        default="worldsystem-frame-processor",
        description="Google Cloud Storage bucket name"
    )
    gcs_credentials_path: Optional[str] = Field(
        default=None,
        description="Path to Google Cloud credentials JSON file"
    )
    
    # ========== Enhancement Configuration ==========
    enhancement_enabled: bool = Field(
        default=True,
        description="Enable image enhancement before API processing"
    )
    enhancement_auto_adjust: bool = Field(
        default=True,
        description="Auto-adjust enhancement parameters based on image statistics"
    )
    enhancement_gamma: float = Field(
        default=1.2,
        ge=0.1,
        le=3.0,
        description="Gamma correction factor"
    )
    enhancement_alpha: float = Field(
        default=1.3,
        ge=0.1,
        le=3.0,
        description="Contrast control factor"
    )
    enhancement_beta: int = Field(
        default=20,
        ge=-100,
        le=100,
        description="Brightness control factor"
    )
    
    # ========== System Configuration ==========
    rabbitmq_url: str = Field(
        default="amqp://rabbitmq",
        description="RabbitMQ connection URL"
    )
    video_frames_exchange: str = Field(
        default="video_frames_exchange",
        description="Exchange name for incoming video frames"
    )
    processed_frames_exchange: str = Field(
        default="processed_frames_exchange",
        description="Exchange name for processed frames"
    )
    scene_scaling_exchange: str = Field(
        default="scene_scaling_exchange",
        description="Exchange name for scene scale updates"
    )
    analysis_mode_exchange: str = Field(
        default="analysis_mode_exchange",
        description="Exchange name for analysis mode control"
    )
    initial_analysis_mode: str = Field(
        default="yolo",
        description="Initial analysis mode (none/yolo)"
    )
    
    # ========== Monitoring Configuration ==========
    metrics_port: int = Field(
        default=8003,
        ge=1024,
        le=65535,
        description="Port for Prometheus metrics"
    )
    log_level: str = Field(
        default="INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description="Logging level"
    )
    log_dir: Path = Field(
        default=Path("/app/logs"),
        description="Directory for log files"
    )
    log_file: str = Field(
        default="/app/logs/frame_processor.log",
        description="Path to log file"
    )
    
    # ========== Visualization Configuration ==========
    rerun_enabled: bool = Field(
        default=True,
        description="Enable Rerun visualization"
    )
    rerun_viewer_address: str = Field(
        default="0.0.0.0:9876",
        description="Rerun viewer address"
    )
    rerun_connect_url: str = Field(
        default="rerun+http://localhost:9876/proxy",
        description="Rerun connection URL"
    )
    
    # ========== Performance Configuration ==========
    queue_size: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Size of internal processing queues"
    )
    worker_threads: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of worker threads for API processing"
    )
    batch_timeout: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Timeout for accumulating detection batches"
    )
    
    # ========== NTP Configuration ==========
    ntp_server: str = Field(
        default="pool.ntp.org",
        description="NTP server for time synchronization"
    )
    ntp_sync_interval: int = Field(
        default=60,
        ge=10,
        description="Seconds between NTP synchronizations"
    )
    
    # ========== Validators ==========
    
    @field_validator('detector_type')
    def validate_detector(cls, v):
        """Ensure detector type is supported."""
        supported = ["yolo", "detectron2", "grounding_dino"]
        if v not in supported:
            raise ValueError(f"Detector must be one of {supported}, got {v}")
        return v
    
    @field_validator('tracker_type')
    def validate_tracker(cls, v):
        """Ensure tracker type is supported."""
        supported = ["iou", "sort", "deep_sort", "bytetrack"]
        if v not in supported:
            raise ValueError(f"Tracker must be one of {supported}, got {v}")
        return v
    
    @field_validator('serpapi_key')
    def validate_serpapi(cls, v):
        """Validate SerpAPI key is provided when SerpAPI is enabled."""
        # Note: In Pydantic v2, we can't access other fields in field_validator
        # This validation would need to be done in model_validator
        return v
    
    @field_validator('perplexity_key')
    def validate_perplexity(cls, v):
        """Validate Perplexity key is provided when Perplexity is enabled."""
        # Note: In Pydantic v2, we can't access other fields in field_validator
        # This validation would need to be done in model_validator
        return v
    
    @field_validator('gcs_credentials_path')
    def validate_gcs_credentials(cls, v):
        """Validate GCS credentials path exists when GCS is enabled."""
        if v:
            path = Path(v)
            if not path.exists():
                # Warning only, don't fail
                print(f"Warning: GCS credentials file not found: {v}")
        return v
    
    @field_validator('detector_model')
    def validate_model_path(cls, v):
        """Check if model file exists (warning only)."""
        if not Path(v).exists():
            print(f"Warning: Model file not found: {v}. Will attempt to download.")
        return v
    
    @model_validator(mode='after')
    def validate_api_keys(self):
        """Validate API keys are provided when services are enabled."""
        if self.use_serpapi and not self.serpapi_key:
            raise ValueError("SERPAPI_API_KEY required when USE_SERPAPI=true")
        if self.use_perplexity and not self.perplexity_key:
            raise ValueError("PERPLEXITY_KEY required when USE_PERPLEXITY=true")
        if self.use_gcs and self.gcs_credentials_path:
            path = Path(self.gcs_credentials_path)
            if not path.exists():
                print(f"Warning: GCS credentials file not found: {self.gcs_credentials_path}")
        return self
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,  # Allow both lowercase and uppercase env vars
        "extra": "ignore"  # Ignore extra fields
    }


# Singleton instance
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """
    Get the configuration singleton.
    
    Returns:
        Config instance with validated settings
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def reload_config() -> Config:
    """
    Force reload configuration from environment.
    
    Returns:
        Fresh Config instance
    """
    global _config_instance
    _config_instance = Config()
    return _config_instance