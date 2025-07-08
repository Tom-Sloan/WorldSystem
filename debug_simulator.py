#!/usr/bin/env python3
"""Debug simulator path logic"""

from pathlib import Path

# Test the exact logic from the simulator
DATA_ROOT = Path("/simulation_data")

print(f"DATA_ROOT: {DATA_ROOT}")
print(f"DATA_ROOT exists: {DATA_ROOT.exists()}")
print(f"DATA_ROOT is_dir: {DATA_ROOT.is_dir()}")

# Check for single video files
single_videos = list(DATA_ROOT.glob("*.mp4"))
print(f"\nAll MP4 files: {len(single_videos)}")
non_segment_videos = [v for v in single_videos if "_segment_" not in v.name]
print(f"Non-segment MP4 files: {len(non_segment_videos)}")

# Check for video segments
video_segments = list(DATA_ROOT.glob("*_segment_*.mp4"))
print(f"\nVideo segments: {len(video_segments)}")
if video_segments:
    for seg in sorted(video_segments)[:3]:
        print(f"  - {seg.name}")

# Check for subdirectories
try:
    folders = sorted([f for f in DATA_ROOT.iterdir() if f.is_dir()])
    print(f"\nSubdirectories: {len(folders)}")
    for f in folders[:3]:
        print(f"  - {f.name}")
except Exception as e:
    print(f"Error listing subdirs: {e}")

print("\n--- Simulator Logic Test ---")
if single_videos and not any("_segment_" in v.name for v in single_videos):
    print("Would process as single videos")
elif video_segments:
    print("Would process as video segments")
else:
    print("Would look for recording folders")