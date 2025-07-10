#!/usr/bin/env python3
"""Test script to verify segment detection logic"""

from pathlib import Path
import sys

# Test path
test_path = Path("/home/sam3/Desktop/Toms_Workspace/WorldSystem/20250617_211214_segments")

print(f"Testing path: {test_path}")
print(f"Path exists: {test_path.exists()}")
print(f"Is directory: {test_path.is_dir()}")

if test_path.exists() and test_path.is_dir():
    # Test glob patterns
    video_segments = list(test_path.glob("*_segment_*.mp4"))
    print(f"\nFound {len(video_segments)} video segments:")
    for seg in sorted(video_segments)[:5]:  # Show first 5
        print(f"  - {seg.name}")
    
    # Also test other patterns
    all_mp4 = list(test_path.glob("*.mp4"))
    print(f"\nTotal MP4 files: {len(all_mp4)}")
    
    # Check for subdirectories
    subdirs = [d for d in test_path.iterdir() if d.is_dir()]
    print(f"\nSubdirectories: {len(subdirs)}")
    for d in subdirs[:5]:
        print(f"  - {d.name}")