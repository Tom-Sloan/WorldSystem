#!/usr/bin/env python3
"""Debug script to inspect SLAM3R model architecture and find dimension mismatch."""

import torch
import sys
sys.path.append('.')

from slam3r.models import Image2PointsModel, Local2WorldModel

def inspect_model_architecture():
    """Inspect the architecture of pretrained SLAM3R models."""
    
    print("Loading pretrained models...")
    try:
        # Load models
        i2p_model = Image2PointsModel.from_pretrained("siyan824/slam3r_i2p")
        l2w_model = Local2WorldModel.from_pretrained("siyan824/slam3r_l2w")
        
        print("\n=== Image2Points Model Architecture ===")
        print(f"Encoder embed dim: {i2p_model.enc_embed_dim if hasattr(i2p_model, 'enc_embed_dim') else 'Not found'}")
        print(f"Decoder embed dim: {i2p_model.dec_embed_dim if hasattr(i2p_model, 'dec_embed_dim') else 'Not found'}")
        
        # Check decoder blocks
        if hasattr(i2p_model, 'mv_dec_blocks1') and len(i2p_model.mv_dec_blocks1) > 0:
            first_block = i2p_model.mv_dec_blocks1[0]
            print(f"\nFirst decoder block type: {type(first_block)}")
            
            # Check cross attention dimensions
            if hasattr(first_block, 'batched_cross_attn'):
                print("Has batched_cross_attn method")
            if hasattr(first_block, 'cross_attn'):
                cross_attn = first_block.cross_attn
                print(f"Cross attention projq: {cross_attn.projq}")
                print(f"  - Input features: {cross_attn.projq.in_features}")
                print(f"  - Output features: {cross_attn.projq.out_features}")
                print(f"  - Num heads: {cross_attn.num_heads if hasattr(cross_attn, 'num_heads') else 'Not found'}")
        
        # Check patch embed
        if hasattr(i2p_model, 'patch_embed'):
            print(f"\nPatch embed:")
            print(f"  - Type: {type(i2p_model.patch_embed)}")
            if hasattr(i2p_model.patch_embed, 'proj'):
                print(f"  - Proj in_channels: {i2p_model.patch_embed.proj.in_channels}")
                print(f"  - Proj out_channels: {i2p_model.patch_embed.proj.out_channels}")
        
        # Calculate expected tensor sizes
        print("\n=== Tensor Size Analysis ===")
        Vx_B = 25  # From error message
        Nx = 196   # 14x14 patches for 224x224 image
        expected_C = 768  # Expected channel dimension
        actual_elements = 752640  # From error message
        
        actual_C = actual_elements // (Vx_B * Nx)
        print(f"Expected elements for shape [25, 196, 12, 64]: {Vx_B * Nx * 12 * 64}")
        print(f"Actual elements: {actual_elements}")
        print(f"Actual channel dimension: {actual_C}")
        print(f"Ratio: {expected_C / actual_C}")
        
        # Try to run a forward pass with dummy data
        print("\n=== Testing Forward Pass ===")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        i2p_model = i2p_model.to(device).eval()
        
        # Create dummy input
        batch_size = 1
        num_views = 5
        img_size = 224
        
        # Create dummy views
        dummy_views = []
        for i in range(num_views):
            view = {
                'img': torch.randn(batch_size, 3, img_size, img_size).to(device),
                'true_shape': torch.tensor([[img_size, img_size]]).to(device),
                'instance': [f'dummy_{i}']
            }
            dummy_views.append(view)
        
        print(f"Input shape: {dummy_views[0]['img'].shape}")
        
        try:
            with torch.no_grad():
                # Image2PointsModel expects ref_id (singular)
                output = i2p_model(dummy_views, ref_id=0)
                print("Forward pass successful!")
                print(f"Output type: {type(output)}")
                if isinstance(output, dict):
                    print(f"Output keys: {output.keys()}")
                elif isinstance(output, list):
                    print(f"Output length: {len(output)}")
                    if output and isinstance(output[0], dict):
                        print(f"First output keys: {output[0].keys()}")
        except Exception as e:
            print(f"Forward pass failed: {e}")
            print(f"Error type: {type(e).__name__}")
            
            # Try to trace where the error occurs
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"Failed to load models: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_model_architecture()