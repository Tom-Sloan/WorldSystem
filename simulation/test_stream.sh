#!/bin/bash
# Test H.264 video streaming

echo "Testing H.264 Video Streaming"
echo "============================"

# Test single video mode
echo "Testing single video mode..."
SIMULATION_MODE=video python3 simulate_video_stream.py

# Test segments mode
echo -e "\n\nTesting segments mode..."
SIMULATION_MODE=segments python3 simulate_video_stream.py