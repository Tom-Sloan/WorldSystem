"""
SAM2 model configuration mappings.

Since the config files might not be accessible directly, 
we provide the necessary mappings here.
"""

# Model configuration mappings
SAM2_MODEL_CONFIGS = {
    "sam2_hiera_t.yaml": {
        "model_path": "sam2_hiera_tiny.pt",
        "encoder_embed_dim": 96,
        "encoder_depth": 12,
        "encoder_num_heads": 1,
        "encoder_global_attn_indexes": [5, 11],
    },
    "sam2_hiera_s.yaml": {
        "model_path": "sam2_hiera_small.pt", 
        "encoder_embed_dim": 384,
        "encoder_depth": 16,
        "encoder_num_heads": 6,
        "encoder_global_attn_indexes": [7, 15],
    },
    "sam2_hiera_b+.yaml": {
        "model_path": "sam2_hiera_base_plus.pt",
        "encoder_embed_dim": 640,
        "encoder_depth": 24,
        "encoder_num_heads": 10,
        "encoder_global_attn_indexes": [7, 15, 23],
    },
    "sam2_hiera_l.yaml": {
        "model_path": "sam2_hiera_large.pt",
        "encoder_embed_dim": 1024,
        "encoder_depth": 48,
        "encoder_num_heads": 16,
        "encoder_global_attn_indexes": [7, 15, 23, 31, 39, 47],
    },
}

def get_sam2_config(config_name: str) -> dict:
    """Get SAM2 model configuration by name."""
    if config_name not in SAM2_MODEL_CONFIGS:
        raise ValueError(f"Unknown SAM2 config: {config_name}. Available: {list(SAM2_MODEL_CONFIGS.keys())}")
    return SAM2_MODEL_CONFIGS[config_name]