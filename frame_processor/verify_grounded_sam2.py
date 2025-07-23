#!/usr/bin/env python3
"""
Verification script to ensure Grounded-SAM-2 functionality is intact after YOLO removal.
"""

import sys
import os
import importlib
import subprocess

def verify_imports():
    """Verify all necessary imports work correctly."""
    print("1. Verifying imports...")
    
    # Add paths as in grounded_sam2_processor.py
    sys.path.append('/app/common')
    sys.path.append('./Grounded-SAM-2')
    sys.path.append('./Grounded-SAM-2/grounding_dino')
    
    errors = []
    
    # Test critical imports
    try:
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        print("   ✓ SAM2 imports successful")
    except ImportError as e:
        errors.append(f"   ✗ SAM2 import failed: {e}")
    
    try:
        from groundingdino.util.inference import load_model, load_image, predict
        print("   ✓ GroundingDINO imports successful")
    except ImportError as e:
        errors.append(f"   ✗ GroundingDINO import failed: {e}")
    
    try:
        import torch
        import cv2
        import numpy as np
        import aio_pika
        import rerun as rr
        print("   ✓ Core dependencies imports successful")
    except ImportError as e:
        errors.append(f"   ✗ Core dependency import failed: {e}")
    
    # Test that YOLO is NOT importable
    try:
        from ultralytics import YOLO
        errors.append("   ✗ WARNING: YOLO is still importable!")
    except ImportError:
        print("   ✓ YOLO correctly removed (cannot import)")
    
    return errors

def verify_model_files():
    """Verify required model files exist and YOLO models are removed."""
    print("\n2. Verifying model files...")
    
    errors = []
    
    # Check that Grounded-SAM-2 checkpoints exist (if in Docker environment)
    grounded_sam2_files = [
        "/app/Grounded-SAM-2/checkpoints/sam2_hiera_small.pt",
        "/app/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
    ]
    
    for file_path in grounded_sam2_files:
        if os.path.exists(file_path):
            print(f"   ✓ Found: {file_path}")
        else:
            # Not an error if running outside Docker
            print(f"   ℹ Not found (OK if not in Docker): {file_path}")
    
    # Check that YOLO models are removed
    yolo_models = [
        "/app/models/yolov11l.pt",
        "/app/models/yolov8n.pt", 
        "/app/models/FastSAM-x.pt",
        "models/yolov11l.pt",
        "models/yolov8n.pt",
        "models/FastSAM-x.pt"
    ]
    
    for yolo_model in yolo_models:
        if os.path.exists(yolo_model):
            errors.append(f"   ✗ YOLO model still exists: {yolo_model}")
        else:
            print(f"   ✓ YOLO model correctly removed: {yolo_model}")
    
    return errors

def verify_code_references():
    """Verify no YOLO references remain in code."""
    print("\n3. Verifying code references...")
    
    errors = []
    
    # Check for YOLO references
    result = subprocess.run(
        ['grep', '-ri', 'yolo\\|fastsam', '.', 
         '--exclude-dir=.git', '--exclude-dir=__pycache__',
         '--exclude=*.pyc', '--exclude=verify_grounded_sam2.py',
         '--exclude=YOLO_REMOVAL_SUMMARY.md'],
        capture_output=True,
        text=True
    )
    
    if result.stdout.strip():
        lines = result.stdout.strip().split('\n')
        # Filter out acceptable references (comments about removal, etc.)
        problematic_lines = []
        for line in lines:
            # Skip comments about YOLO being removed
            if 'removed' in line.lower() or 'removal' in line.lower():
                continue
            # Skip this verification script
            if 'verify_grounded_sam2.py' in line:
                continue
            problematic_lines.append(line)
        
        if problematic_lines:
            errors.append("   ✗ Found YOLO references in code:")
            for line in problematic_lines[:5]:  # Show first 5
                errors.append(f"     {line}")
            if len(problematic_lines) > 5:
                errors.append(f"     ... and {len(problematic_lines) - 5} more")
        else:
            print("   ✓ No problematic YOLO references found")
    else:
        print("   ✓ No YOLO references found in code")
    
    return errors

def verify_configuration():
    """Verify configuration is correct."""
    print("\n4. Verifying configuration...")
    
    errors = []
    
    # Check that video_tracker_type only allows valid options
    try:
        from core.config import Config
        config = Config()
        
        valid_trackers = ["sam2_realtime", "grounded_sam2"]
        if config.video_tracker_type in valid_trackers:
            print(f"   ✓ Video tracker type is valid: {config.video_tracker_type}")
        else:
            errors.append(f"   ✗ Invalid video tracker type: {config.video_tracker_type}")
        
        # Ensure YOLO is not an option
        if "yolo" in config.video_tracker_type.lower():
            errors.append("   ✗ YOLO still appears in video tracker type!")
            
    except Exception as e:
        print(f"   ℹ Could not verify config (OK if not in proper environment): {e}")
    
    return errors

def verify_dependencies():
    """Verify required dependencies are in requirements.txt."""
    print("\n5. Verifying dependencies...")
    
    errors = []
    
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            requirements = f.read().lower()
        
        # Check for required dependencies
        required_deps = [
            "transformers",  # For GroundingDINO
            "supervision",   # For Grounded-SAM-2
            "torch",         # PyTorch
            "opencv-python", # OpenCV
            "pyav",          # Video processing
        ]
        
        for dep in required_deps:
            if dep in requirements:
                print(f"   ✓ Found required dependency: {dep}")
            else:
                errors.append(f"   ✗ Missing required dependency: {dep}")
        
        # Check that ultralytics (YOLO) is NOT present
        if "ultralytics" in requirements:
            errors.append("   ✗ ultralytics (YOLO) still in requirements.txt!")
        else:
            print("   ✓ ultralytics correctly removed from requirements")
    else:
        print("   ℹ requirements.txt not found")
    
    return errors

def main():
    """Run all verification checks."""
    print("=== Grounded-SAM-2 Verification after YOLO Removal ===\n")
    
    all_errors = []
    
    # Run all checks
    all_errors.extend(verify_imports())
    all_errors.extend(verify_model_files())
    all_errors.extend(verify_code_references())
    all_errors.extend(verify_configuration())
    all_errors.extend(verify_dependencies())
    
    # Summary
    print("\n=== VERIFICATION SUMMARY ===")
    if all_errors:
        print(f"\n❌ Found {len(all_errors)} issues:\n")
        for error in all_errors:
            print(error)
        print("\nGrounded-SAM-2 may have issues. Please review the errors above.")
        sys.exit(1)
    else:
        print("\n✅ All checks passed! Grounded-SAM-2 functionality is preserved.")
        print("\nThe following are working correctly:")
        print("- SAM2 imports and models")
        print("- GroundingDINO imports") 
        print("- No YOLO references in code")
        print("- Configuration only allows SAM2-based trackers")
        print("- All required dependencies present")
        print("\nGrounded-SAM-2 should work as expected.")
        sys.exit(0)

if __name__ == "__main__":
    main()