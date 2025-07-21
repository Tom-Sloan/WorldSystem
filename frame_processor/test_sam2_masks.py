#!/usr/bin/env python3
"""Test script to verify SAM2 mask generation approaches."""

import numpy as np
import cv2
import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_sam2_approaches():
    """Test different approaches to get SAM2 to generate masks."""
    
    print("Testing SAM2 mask generation approaches...")
    
    # Create a test image
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray background
    # Add some rectangles to segment
    cv2.rectangle(test_image, (50, 50), (200, 200), (255, 0, 0), -1)  # Blue
    cv2.rectangle(test_image, (300, 100), (450, 250), (0, 255, 0), -1)  # Green
    cv2.rectangle(test_image, (100, 300), (250, 450), (0, 0, 255), -1)  # Red
    
    print(f"Test image shape: {test_image.shape}")
    
    try:
        # Test 1: Try importing SAM2
        print("\n1. Testing SAM2 imports...")
        from sam2.build_sam import build_sam2_video_predictor, build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("✓ Basic imports successful")
        
        # Try automatic mask generator import
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            print("✓ SAM2AutomaticMaskGenerator imported")
            has_auto_generator = True
        except ImportError:
            print("✗ SAM2AutomaticMaskGenerator not available")
            has_auto_generator = False
        
        # Test 2: Load model
        print("\n2. Loading SAM2 model...")
        model_cfg = "sam2_hiera_b+.yaml"
        checkpoint = "/app/models/sam2_hiera_base_plus.pt"
        
        # Check if files exist
        if not os.path.exists(checkpoint):
            print(f"✗ Checkpoint not found: {checkpoint}")
            return
        
        # Build model
        sam_model = build_sam2(model_cfg, checkpoint, device='cuda' if torch.cuda.is_available() else 'cpu')
        print("✓ Model loaded successfully")
        
        # Test 3: Create image predictor
        print("\n3. Creating image predictor...")
        image_predictor = SAM2ImagePredictor(sam_model)
        print("✓ Image predictor created")
        
        # Test 4: Try different mask generation approaches
        print("\n4. Testing mask generation approaches...")
        
        # Approach A: Check for generate method
        if hasattr(image_predictor, 'generate'):
            print("✓ Image predictor has 'generate' method")
            try:
                masks = image_predictor.generate(test_image)
                print(f"  Generated {len(masks)} masks")
            except Exception as e:
                print(f"  ✗ Generate failed: {e}")
        else:
            print("✗ Image predictor does not have 'generate' method")
        
        # Approach B: Try automatic mask generator
        if has_auto_generator:
            try:
                print("\n  Testing SAM2AutomaticMaskGenerator...")
                mask_generator = SAM2AutomaticMaskGenerator(sam_model)
                masks = mask_generator.generate(test_image)
                print(f"  ✓ Generated {len(masks)} masks with automatic generator")
                
                # Print mask details
                if masks:
                    print("\n  Sample mask info:")
                    for i, mask in enumerate(masks[:3]):
                        print(f"    Mask {i}: area={mask.get('area', 'N/A')}, "
                              f"bbox={mask.get('bbox', 'N/A')}, "
                              f"predicted_iou={mask.get('predicted_iou', 'N/A')}")
            except Exception as e:
                print(f"  ✗ Automatic mask generator failed: {e}")
        
        # Approach C: Grid prompting with predict
        print("\n  Testing grid prompting...")
        image_predictor.set_image(test_image)
        
        # Generate grid points
        h, w = test_image.shape[:2]
        points_per_side = 8
        x_coords = np.linspace(20, w - 20, points_per_side)
        y_coords = np.linspace(20, h - 20, points_per_side)
        
        points = []
        for y in y_coords:
            for x in x_coords:
                points.append([int(x), int(y)])
        
        print(f"  Generated {len(points)} grid points")
        
        # Test single point
        test_point = np.array([[320, 240]])  # Center of image
        test_label = np.array([1])
        
        try:
            masks, scores, logits = image_predictor.predict(
                point_coords=test_point,
                point_labels=test_label,
                multimask_output=True
            )
            print(f"  ✓ Single point prediction: {len(masks)} masks, scores: {scores}")
        except Exception as e:
            print(f"  ✗ Single point prediction failed: {e}")
        
        # Test batch points
        batch_points = np.array(points[:16])  # First 16 points
        batch_labels = np.ones(16, dtype=np.int32)
        
        try:
            masks, scores, logits = image_predictor.predict(
                point_coords=batch_points,
                point_labels=batch_labels,
                multimask_output=False
            )
            print(f"  ✓ Batch prediction: shape={masks.shape}, scores shape={scores.shape}")
            print(f"    Scores: min={scores.min():.3f}, max={scores.max():.3f}, mean={scores.mean():.3f}")
        except Exception as e:
            print(f"  ✗ Batch prediction failed: {e}")
        
        # Test 5: Check available methods
        print("\n5. Available methods on image_predictor:")
        methods = [m for m in dir(image_predictor) if not m.startswith('_') and callable(getattr(image_predictor, m))]
        for method in sorted(methods)[:10]:
            print(f"  - {method}")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sam2_approaches()