#!/usr/bin/env python3
"""
Simple test to verify image resizing logic.
"""

import numpy as np
import cv2

def test_resize_logic():
    """Test the resize logic from both implementations."""
    
    # Test sizes
    test_sizes = [
        (640, 480),   # Standard VGA
        (1920, 1080), # Full HD
        (224, 224),   # Target size
        (256, 256),   # Multiple of 16
        (300, 400),   # Arbitrary size
    ]
    
    print("Testing OLD implementation logic (always 224x224):")
    TARGET_WIDTH = 224
    TARGET_HEIGHT = 224
    
    for h, w in test_sizes:
        # Create test image
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Old implementation resize
        img_resized = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))
        
        print(f"  Input: {h}x{w} -> Output: {img_resized.shape[0]}x{img_resized.shape[1]}")
    
    print("\nTesting BROKEN implementation logic (round to multiple of 16):")
    
    for h, w in test_sizes:
        # Create test image  
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Broken implementation logic
        if h % 16 != 0 or w % 16 != 0:
            new_h = ((h + 15) // 16) * 16
            new_w = ((w + 15) // 16) * 16
            img_resized = cv2.resize(img, (new_w, new_h))
        else:
            img_resized = img
            
        print(f"  Input: {h}x{w} -> Output: {img_resized.shape[0]}x{img_resized.shape[1]}")
        
    print("\n" + "="*60)
    print("PROBLEM IDENTIFIED:")
    print("The new implementation resizes to different dimensions based on input,")
    print("while SLAM3R models REQUIRE 224x224 input as they were trained on this size.")
    print("="*60)

if __name__ == "__main__":
    test_resize_logic()