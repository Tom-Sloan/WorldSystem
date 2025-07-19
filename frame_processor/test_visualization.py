#!/usr/bin/env python3
"""
Test script to verify the updated Rerun visualization.

This script creates test detections and tracks to verify that:
1. Colorful segmentation masks are displayed correctly
2. Enhanced objects grid updates properly
3. Statistics are shown correctly
"""

import numpy as np
import cv2
import time
from dataclasses import dataclass
from typing import List, Tuple

from core.config import Config
from detection.base import Detection
from tracking.base import TrackedObject
from visualization.rerun_client import RerunClient


def create_test_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a test frame with some objects."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some colored rectangles to simulate objects
    cv2.rectangle(frame, (50, 50), (150, 150), (0, 255, 0), -1)  # Green
    cv2.rectangle(frame, (200, 100), (350, 200), (255, 0, 0), -1)  # Blue
    cv2.rectangle(frame, (400, 250), (550, 400), (0, 0, 255), -1)  # Red
    cv2.rectangle(frame, (100, 300), (250, 450), (255, 255, 0), -1)  # Cyan
    
    # Add some text
    cv2.putText(frame, "Test Frame", (width//2 - 50, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame


def create_test_detections(frame_shape: Tuple[int, int]) -> List[Detection]:
    """Create test detections for the objects in the frame."""
    height, width = frame_shape
    
    detections = [
        Detection(
            bbox=(50, 50, 150, 150),
            confidence=0.95,
            class_id=0,
            class_name="object_1"
        ),
        Detection(
            bbox=(200, 100, 350, 200),
            confidence=0.88,
            class_id=1,
            class_name="object_2"
        ),
        Detection(
            bbox=(400, 250, 550, 400),
            confidence=0.92,
            class_id=2,
            class_name="object_3"
        ),
        Detection(
            bbox=(100, 300, 250, 450),
            confidence=0.85,
            class_id=3,
            class_name="object_4"
        ),
    ]
    
    # Add some smaller detections to test filtering
    for i in range(5, 10):
        x = np.random.randint(0, width - 50)
        y = np.random.randint(0, height - 50)
        w = np.random.randint(30, 80)
        h = np.random.randint(30, 80)
        
        detections.append(Detection(
            bbox=(x, y, x + w, y + h),
            confidence=np.random.uniform(0.7, 0.95),
            class_id=i,
            class_name=f"small_object_{i}"
        ))
    
    return detections


def create_test_tracks(detections: List[Detection], frame: np.ndarray) -> List[TrackedObject]:
    """Create test tracked objects from detections."""
    tracks = []
    
    for i, det in enumerate(detections[:4]):  # Only track the main objects
        track = TrackedObject(
            id=i + 1,
            class_name=det.class_name,
            bbox=det.bbox,
            confidence=det.confidence
        )
        
        # Extract ROI as best_frame
        x1, y1, x2, y2 = det.bbox
        track.best_frame = frame[y1:y2, x1:x2].copy()
        
        # Simulate some tracks having API results
        if i < 2:
            track.api_result = {
                'dimensions': {'width': 0.15, 'height': 0.10, 'depth': 0.08},
                'product_name': f'Test Product {i+1}'
            }
            track.estimated_dimensions = track.api_result['dimensions']
        
        tracks.append(track)
    
    return tracks


def main():
    """Run the visualization test."""
    # Create config
    config = Config()
    config.rerun_enabled = True
    config.detector_type = 'sam'  # Test SAM visualization
    
    # Create Rerun client
    print("Initializing Rerun client...")
    rerun_client = RerunClient(config)
    
    print("Starting visualization test...")
    print("Open Rerun viewer at http://localhost:9876 to see the results")
    
    # Run test loop
    frame_count = 0
    while frame_count < 300:  # Run for 300 frames
        # Create test frame
        frame = create_test_frame()
        
        # Add some variation to make it interesting
        if frame_count % 30 == 0:
            # Shift colors every 30 frames
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        
        # Create detections and tracks
        detections = create_test_detections(frame.shape[:2])
        tracks = create_test_tracks(detections, frame)
        
        # Log frame
        timestamp_ns = int(time.time() * 1e9)
        rerun_client.log_frame(frame, detections, tracks, frame_count, timestamp_ns)
        
        # Simulate enhanced object logging every 10 frames
        if frame_count % 10 == 0 and tracks:
            track_to_enhance = tracks[frame_count % len(tracks)]
            # Simulate enhancement by brightening the image
            enhanced = cv2.convertScaleAbs(track_to_enhance.best_frame, alpha=1.2, beta=20)
            track_to_enhance.best_frame = enhanced
            rerun_client.log_enhanced_object(track_to_enhance)
        
        # Log scene scale periodically
        if frame_count % 50 == 0:
            scale_info = {
                'scale_factor': 0.0015,
                'confidence': 0.85,
                'num_estimates': len(tracks),
                'avg_dimensions_m': {
                    'width': 0.12,
                    'height': 0.09,
                    'depth': 0.07
                }
            }
            rerun_client.log_scene_scale(scale_info)
        
        frame_count += 1
        time.sleep(0.033)  # ~30 FPS
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    
    print("Test completed!")
    print("Check the Rerun viewer to see:")
    print("1. Colorful segmentation masks overlaid on frames")
    print("2. Enhanced objects grid showing processed objects")
    print("3. Statistics panel with detection and tracking info")


if __name__ == "__main__":
    main()