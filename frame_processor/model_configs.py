"""
Unified model configuration system.

This is the single source of truth for all model configurations including:
- Model paths
- Config file names
- Memory requirements
- Model-specific parameters

All other parts of the system should reference this file.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Complete configuration for a model."""
    name: str
    model_type: str  # "sam2", "fastsam", etc.
    config_file: Optional[str]  # YAML config file name
    checkpoint_path: str  # Path to model weights
    memory_mb: int  # GPU memory requirement in MB
    parameters: Dict[str, Any]  # Model-specific parameters
    

# SAM2 Model Configurations
SAM2_MODELS = {
    "sam2_tiny": ModelConfig(
        name="sam2_tiny",
        model_type="sam2",
        config_file="sam2_hiera_t.yaml",
        checkpoint_path="/app/models/sam2_hiera_tiny.pt",
        memory_mb=500,
        parameters={
            "encoder_embed_dim": 96,
            "encoder_depth": 12,
            "encoder_num_heads": 1,
            "encoder_global_attn_indexes": [5, 11],
            "points_per_side": 16,  # Reduced for performance
            "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.95,
            "min_mask_region_area": 1000,
        }
    ),
    "sam2_small": ModelConfig(
        name="sam2_small",
        model_type="sam2",
        config_file="sam2_hiera_s.yaml",
        checkpoint_path="/app/models/sam2_hiera_small.pt",
        memory_mb=1000,
        parameters={
            "encoder_embed_dim": 384,
            "encoder_depth": 16,
            "encoder_num_heads": 6,
            "encoder_global_attn_indexes": [7, 15],
            "points_per_side": 24,  # Balanced
            "pred_iou_thresh": 0.86,
            "stability_score_thresh": 0.92,
            "min_mask_region_area": 500,
        }
    ),
    "sam2_base_plus": ModelConfig(
        name="sam2_base_plus",
        model_type="sam2",
        config_file="sam2_hiera_b+.yaml",
        checkpoint_path="/app/models/sam2_hiera_base_plus.pt",
        memory_mb=2000,
        parameters={
            "encoder_embed_dim": 640,
            "encoder_depth": 24,
            "encoder_num_heads": 10,
            "encoder_global_attn_indexes": [7, 15, 23],
            "points_per_side": 32,  # Higher quality
            "pred_iou_thresh": 0.86,
            "stability_score_thresh": 0.92,
            "min_mask_region_area": 500,
        }
    ),
    "sam2_large": ModelConfig(
        name="sam2_large",
        model_type="sam2",
        config_file="sam2_hiera_l.yaml",
        checkpoint_path="/app/models/sam2_hiera_large.pt",
        memory_mb=4000,
        parameters={
            "encoder_embed_dim": 1024,
            "encoder_depth": 48,
            "encoder_num_heads": 16,
            "encoder_global_attn_indexes": [7, 15, 23, 31, 39, 47],
            "points_per_side": 48,  # Maximum quality
            "pred_iou_thresh": 0.86,
            "stability_score_thresh": 0.92,
            "min_mask_region_area": 300,
        }
    ),
}

# FastSAM Model Configurations
FASTSAM_MODELS = {
    "fastsam_x": ModelConfig(
        name="fastsam_x",
        model_type="fastsam",
        config_file=None,  # FastSAM doesn't use YAML configs
        checkpoint_path="/app/models/FastSAM-x.pt",
        memory_mb=1500,
        parameters={
            "conf_threshold": 0.4,
            "iou_threshold": 0.9,
            "max_det": 300,
            "retina_masks": True,
        }
    ),
}

# All models registry
ALL_MODELS = {
    **SAM2_MODELS,
    **FASTSAM_MODELS,
}

# Default models for each type
DEFAULT_MODELS = {
    "sam2": "sam2_base_plus",
    "fastsam": "fastsam_x",
}

# Model size mappings for dynamic switching
MODEL_SIZE_MAP = {
    "tiny": "sam2_tiny",
    "small": "sam2_small", 
    "base": "sam2_base_plus",
    "large": "sam2_large",
}


def get_model_config(model_name: str) -> ModelConfig:
    """
    Get model configuration by name.
    
    Args:
        model_name: Name of the model (e.g., "sam2_tiny", "fastsam_x")
        
    Returns:
        ModelConfig object
        
    Raises:
        ValueError: If model not found
    """
    if model_name not in ALL_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(ALL_MODELS.keys())}"
        )
    return ALL_MODELS[model_name]


def get_default_model(model_type: str) -> ModelConfig:
    """
    Get default model for a given type.
    
    Args:
        model_type: Type of model ("sam2", "fastsam")
        
    Returns:
        ModelConfig object
    """
    if model_type not in DEFAULT_MODELS:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_name = DEFAULT_MODELS[model_type]
    return get_model_config(model_name)


def get_model_by_size(size: str) -> ModelConfig:
    """
    Get SAM2 model by size designation.
    
    Args:
        size: Size designation ("tiny", "small", "base", "large")
        
    Returns:
        ModelConfig object
    """
    if size not in MODEL_SIZE_MAP:
        raise ValueError(
            f"Unknown size: {size}. "
            f"Available sizes: {list(MODEL_SIZE_MAP.keys())}"
        )
    
    model_name = MODEL_SIZE_MAP[size]
    return get_model_config(model_name)


def get_available_models(model_type: Optional[str] = None) -> Dict[str, ModelConfig]:
    """
    Get all available models, optionally filtered by type.
    
    Args:
        model_type: Optional model type filter
        
    Returns:
        Dictionary of model configurations
    """
    if model_type is None:
        return ALL_MODELS
    
    return {
        name: config 
        for name, config in ALL_MODELS.items() 
        if config.model_type == model_type
    }


def validate_model_files() -> Dict[str, bool]:
    """
    Check which model files actually exist on disk.
    
    Returns:
        Dictionary mapping model names to existence status
    """
    results = {}
    for name, config in ALL_MODELS.items():
        path = Path(config.checkpoint_path)
        results[name] = path.exists()
        if not path.exists():
            print(f"Warning: Model file not found for {name}: {config.checkpoint_path}")
    return results