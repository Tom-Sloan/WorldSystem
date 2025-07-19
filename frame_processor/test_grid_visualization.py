#!/usr/bin/env python3
"""Test script to verify grid visualization works correctly."""

import numpy as np
import cv2
import asyncio
from core.config import Config
from visualization.rerun_client import RerunClient
from detection.base import Detection
from tracking.base import TrackedObject

async def test_grid_visualization():
    """Test the grid visualization with sample frames."""
    # Create config
    config = Config()
    config.rerun_enabled = True
    
    # Create Rerun client
    client = RerunClient(config)
    
    # Generate test frames
    for frame_num in range(50):
        # Create a test frame with different colors
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some variation to each frame
        color = (frame_num * 5 % 255, 100, 200 - frame_num * 3 % 200)
        cv2.rectangle(frame, (100, 100), (540, 380), color, -1)
        cv2.putText(frame, f"Frame {frame_num}", (250, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Create some test detections (minimal to keep it clean)
        detections = []
        if frame_num % 10 == 0:  # Only add detections every 10 frames
            detections.append(Detection(
                bbox=(200, 150, 400, 300),
                confidence=0.9,
                class_id=0,
                class_name="test_object"
            ))
        
        # Create tracked objects
        tracks = []
        if detections:
            track = TrackedObject(
                id=1,
                bbox=(200, 150, 400, 300),
                class_id=0,
                class_name="test_object",
                confidence=0.9
            )
            tracks.append(track)
        
        # Log frame
        client.log_frame(frame, detections, tracks, frame_num)
        
        # Small delay to simulate real processing
        await asyncio.sleep(0.1)
    
    print("Test completed! Check Rerun viewer for grid visualization.")

if __name__ == "__main__":
    asyncio.run(test_grid_visualization())