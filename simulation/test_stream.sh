#!/bin/bash
# Test H.264 video streaming

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Testing H.264 Video Streaming"
echo "============================"

# Change to the simulation directory
cd "$SCRIPT_DIR"

# Check if video exists
if [ ! -f "test_video.mp4" ]; then
    echo "Error: test_video.mp4 not found!"
    echo "Please copy a video file to simulation/test_video.mp4"
    echo "Current directory: $(pwd)"
    exit 1
fi

echo "Found test video: test_video.mp4"
echo "Starting H.264 stream to server..."

# Run the video streaming script
python3 simulate_video_stream.py