"""
Centralized configuration management using Pydantic.

This module provides type-safe configuration with validation, enabling easy
swapping of detection and tracking algorithms via environment variables.
"""

from pydantic import field_validator, Field, model_validator
from pydantic_settings import BaseSettings
from typing import Literal, Optional, Dict, Any
import os
from pathlib import Path
from .utils import get_logger
from model_configs import get_model_config, get_default_model, get_model_by_size

logger = get_logger(__name__)


class Config(BaseSettings):
    """
    Centralized configuration with modular component selection.
    
    All settings can be overridden via environment variables or .env file.
    """
    
    # ========== Component Selection ==========
    
    # Model selection - this is the primary way to select a model
    model_name: Optional[str] = Field(
        default=None,
        description="Specific model name (e.g., 'sam2_tiny', 'sam2_base_plus'). If not set, uses default model."
    )
    tracker_type: Literal["iou"] = Field(
        default="iou",
        description="Tracking algorithm to use"
    )
    
    # ========== Detector Configuration ==========
    detector_model: Optional[str] = Field(
        default=None,
        description="Path to detector model weights (auto-determined if not specified)"
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
    video_stream_exchange: str = Field(
        default="video_stream_exchange",
        description="Exchange name for incoming H.264 video streams"
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
    
    # ========== Monitoring Configuration ==========
    metrics_port: int = Field(
        default=8003,
        ge=1024,
        le=65535,
        description="Port for Prometheus metrics"
    )
    log_level: str = Field(
        default="DEBUG",
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
    
    # ========== Output Configuration ==========
    output_enabled: bool = Field(
        default=True,
        description="Enable local output saving"
    )
    output_dir: str = Field(
        default="/app/outputs",
        description="Directory for saving outputs"
    )
    save_visualizations: bool = Field(
        default=True,
        description="Save visualization images"
    )
    save_video_summary: bool = Field(
        default=True,
        description="Create video summary at end of session"
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
    
    # ========== SAM2 Configuration ==========
    # These are now derived from model_name or can be overridden
    sam_model_cfg: Optional[str] = Field(
        default=None,
        description="SAM2 model configuration file (auto-determined from model_name if not set)"
    )
    sam_checkpoint_path: Optional[str] = Field(
        default=None,
        description="Path to SAM2 checkpoint file (auto-determined from model_name if not set)"
    )
    sam_points_per_side: int = Field(
        default=24,
        ge=1,
        le=64,
        description="Number of points sampled per side"
    )
    sam_pred_iou_thresh: float = Field(
        default=0.86,
        ge=0.0,
        le=1.0,
        description="Threshold for predicted IoU"
    )
    sam_stability_score_thresh: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description="Threshold for mask stability"
    )
    sam_min_mask_region_area: int = Field(
        default=500,
        ge=0,
        description="Minimum area for valid masks"
    )
    
    # Video-specific SAM2 thresholds (can override the defaults for video mode)
    sam_video_pred_iou_thresh: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="IoU threshold for video mode (lower than image mode)"
    )
    sam_video_stability_score_thresh: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Stability threshold for video mode (lower than image mode)"
    )
    sam_video_min_area: int = Field(
        default=500,
        ge=0,
        description="Minimum mask area for video mode"
    )
    
    # ========== Removed FastSAM/YOLO Configuration - Using SAM2 exclusively ==========
    
    # ========== Video Processing Configuration ==========
    video_tracker_type: str = Field(
        default="sam2_realtime",
        description="Video tracker to use (sam2_realtime, grounded_sam2)"
    )
    
    # ========== Grounded SAM2 Configuration ==========
    grounded_detection_interval: int = Field(
        default=30,
        ge=1,
        description="Frames between full object detection (1 = every frame)"
    )
    grounded_text_prompt: str = Field(
        default="all objects. item. thing. stuff.",
        description="Text prompt for open-vocabulary detection"
    )
    grounded_box_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Box confidence threshold for GroundingDINO"
    )
    grounded_text_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Text matching threshold for GroundingDINO"
    )
    grounded_iou_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="IOU threshold for matching detections to tracks"
    )
    
    # Configuration profile
    config_profile: str = Field(
        default="balanced",
        description="Configuration profile: performance, balanced, quality"
    )
    
    # Performance settings
    target_fps: int = Field(
        default=15,
        description="Target FPS for real-time processing"
    )
    
    processing_resolution: int = Field(
        default=720,
        description="Max resolution for processing (scales down if needed)"
    )
    
    # SAM2 Video configuration
    sam2_model_size: str = Field(
        default="small",
        description="Model size: tiny, small, base, large"
    )
    
    enable_model_compilation: bool = Field(
        default=True,
        description="Compile model for better performance (PyTorch 2.0+)"
    )
    
    # Memory tree configuration
    memory_tree_branches: int = Field(
        default=3,
        description="Number of hypothesis branches in memory tree"
    )
    
    memory_tree_consistency_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Consistency threshold for memory tree branches"
    )
    
    # Prompting configuration
    sam2_prompt_strategy: str = Field(
        default="grid",
        description="Prompting strategy: grid, motion, saliency, hybrid"
    )
    
    grid_prompt_density: int = Field(
        default=16,
        description="Points per side for grid prompting"
    )
    
    reprompt_interval: int = Field(
        default=60,
        description="Frames between re-prompting for new objects"
    )
    
    # Object filtering
    min_object_area: int = Field(
        default=1000,
        description="Minimum pixel area for tracking"
    )
    
    # Google Lens configuration
    lens_api_rate_limit: int = Field(
        default=10,
        description="Max Google Lens API calls per second"
    )
    
    lens_cache_size: int = Field(
        default=1000,
        description="Size of visual similarity cache"
    )
    
    lens_cache_similarity_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for cache hits"
    )
    
    lens_batch_size: int = Field(
        default=10,
        description="Maximum items per Lens API batch"
    )
    
    lens_batch_wait_ms: int = Field(
        default=500,
        description="Maximum wait time before processing batch"
    )
    
    lens_enable_similar_dedup: bool = Field(
        default=True,
        description="Deduplicate visually similar items in batch"
    )
    
    # Processing intervals
    reprocess_interval_ms: int = Field(
        default=3000,
        description="Minimum time between reprocessing same object (ms)"
    )
    
    # Video buffer configuration
    video_buffer_size: int = Field(
        default=30,
        description="Frames to buffer per stream"
    )
    
    # Stream lifecycle
    stream_stale_timeout_seconds: int = Field(
        default=30,
        description="Seconds before marking a stream as stale"
    )
    
    stream_cleanup_timeout_seconds: int = Field(
        default=120,
        description="Seconds before cleaning up an inactive stream"
    )
    
    # GPU memory management
    model_switch_threshold_mb: int = Field(
        default=3000,
        description="GPU memory threshold for model switching (MB)"
    )
    
    enable_dynamic_model_switching: bool = Field(
        default=True,
        description="Enable automatic model size switching on GPU pressure"
    )
    
    # Error recovery
    max_retry_attempts: int = Field(
        default=2,
        description="Maximum retry attempts for GPU OOM errors"
    )
    
    retry_delay_ms: int = Field(
        default=100,
        description="Delay between retry attempts (ms)"
    )
    
    # Metrics
    metrics_enabled: bool = Field(
        default=True,
        description="Enable Prometheus metrics export"
    )
    
    # ========== Validators ==========
    
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
        if v is None:
            # Model path will be auto-determined later
            return v
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
    
    @model_validator(mode='after')
    def auto_determine_model_config(self):
        """Automatically determine model configuration based on model_name or detector_type."""
        # Determine which model to use
        if self.model_name:
            # Use specific model if provided
            try:
                model_config = get_model_config(self.model_name)
            except ValueError:
                # Try to interpret as a size for SAM2
                try:
                    model_config = get_model_by_size(self.model_name)
                except ValueError:
                    raise ValueError(f"Unknown model: {self.model_name}")
        else:
            # Use default SAM2 model
            model_config = get_default_model("sam2")
        
        # Apply model configuration if fields not explicitly set
        if self.detector_model is None:
            self.detector_model = model_config.checkpoint_path
            logger.info(f"Using model: {model_config.name} at {self.detector_model}")
        
        # Set SAM2-specific configs
        if model_config.model_type == "sam2":
            if self.sam_model_cfg is None:
                self.sam_model_cfg = model_config.config_file
            if self.sam_checkpoint_path is None:
                self.sam_checkpoint_path = model_config.checkpoint_path
            
            # Apply recommended parameters if not overridden
            params = model_config.parameters
            if hasattr(self, 'sam_points_per_side') and self.sam_points_per_side == 24:  # default
                self.sam_points_per_side = params.get('points_per_side', 24)
            if hasattr(self, 'sam_pred_iou_thresh') and self.sam_pred_iou_thresh == 0.86:  # default
                self.sam_pred_iou_thresh = params.get('pred_iou_thresh', 0.86)
            if hasattr(self, 'sam_stability_score_thresh') and self.sam_stability_score_thresh == 0.92:  # default
                self.sam_stability_score_thresh = params.get('stability_score_thresh', 0.92)
            if hasattr(self, 'sam_min_mask_region_area') and self.sam_min_mask_region_area == 500:  # default
                self.sam_min_mask_region_area = params.get('min_mask_region_area', 500)
        
        # FastSAM configuration removed - using SAM2 exclusively
        
        return self
    
    @model_validator(mode='before')
    def apply_profile(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Apply configuration profile presets."""
        profile = values.get('config_profile', 'balanced')
        
        profiles = {
            "performance": {
                "sam2_model_size": "tiny",
                "target_fps": 25,
                "processing_resolution": 480,
                "grid_prompt_density": 8,
                "reprompt_interval": 120,
                "min_object_area": 2000,
                "memory_tree_branches": 2,
                "enable_model_compilation": True,
                "lens_batch_size": 20,
                "lens_batch_wait_ms": 1000,
                "enhancement_enabled": False  # Disable for speed
            },
            "balanced": {
                "sam2_model_size": "small",
                "target_fps": 15,
                "processing_resolution": 720,
                "grid_prompt_density": 16,
                "reprompt_interval": 60,
                "min_object_area": 1000,
                "memory_tree_branches": 3,
                "enable_model_compilation": True,
                "lens_batch_size": 10,
                "lens_batch_wait_ms": 500
            },
            "quality": {
                "sam2_model_size": "base",
                "target_fps": 10,
                "processing_resolution": 1080,
                "grid_prompt_density": 24,
                "reprompt_interval": 30,
                "min_object_area": 500,
                "memory_tree_branches": 4,
                "enable_model_compilation": False,  # Prefer accuracy
                "lens_batch_size": 5,
                "lens_batch_wait_ms": 250,
                "enhancement_enabled": True,
                "enhancement_auto_adjust": True
            }
        }
        
        if profile in profiles:
            # Apply profile defaults (can be overridden by explicit settings)
            profile_settings = profiles[profile]
            for key, value in profile_settings.items():
                if key not in values or values[key] is None:
                    values[key] = value
            logger.info(f"Applied configuration profile: {profile}")
        
        return values
    
    @field_validator('processing_resolution')
    def validate_resolution(cls, v):
        """Ensure resolution is reasonable."""
        valid_resolutions = [480, 720, 1080, 1440, 2160]
        if v not in valid_resolutions:
            # Find closest valid resolution
            closest = min(valid_resolutions, key=lambda x: abs(x - v))
            logger.warning(f"Invalid resolution {v}, using closest: {closest}")
            return closest
        return v
    
    @field_validator('sam2_model_size')
    def validate_sam2_model_size(cls, v):
        """Ensure model size is valid."""
        valid_sizes = ["tiny", "small", "base", "large"]
        if v not in valid_sizes:
            raise ValueError(f"Invalid SAM2 model size: {v}. Must be one of {valid_sizes}")
        return v
    
    @field_validator('target_fps')
    def validate_target_fps(cls, v):
        """Ensure FPS target is reasonable."""
        if v < 5:
            logger.warning("Target FPS < 5 may cause tracking issues")
        elif v > 30:
            logger.warning("Target FPS > 30 may not be achievable")
        return v
    
    @field_validator('sam2_prompt_strategy')
    def validate_prompt_strategy(cls, v):
        """Ensure prompt strategy is valid."""
        valid_strategies = ["grid", "motion", "saliency", "hybrid"]
        if v not in valid_strategies:
            raise ValueError(f"Invalid prompt strategy: {v}. Must be one of {valid_strategies}")
        return v
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,  # Allow both lowercase and uppercase env vars
        "extra": "ignore"  # Ignore extra fields
    }


class ConfigFactory:
    """Factory for creating configurations with different profiles."""
    
    @staticmethod
    def create_config(profile: str = "balanced", **overrides) -> Config:
        """Create a configuration with a specific profile and optional overrides."""
        config_dict = {"config_profile": profile}
        config_dict.update(overrides)
        return Config(**config_dict)
    
    @staticmethod
    def create_performance_config(**overrides) -> Config:
        """Create a performance-optimized configuration."""
        return ConfigFactory.create_config("performance", **overrides)
    
    @staticmethod
    def create_quality_config(**overrides) -> Config:
        """Create a quality-optimized configuration."""
        return ConfigFactory.create_config("quality", **overrides)


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