#!/bin/bash
# Script to run SAM2 tests locally with conda environment

echo "SAM2 Local Testing Setup"
echo "========================"
echo

# Activate conda environment
echo "Activating conda environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate 3d_reconstruction

# Check Python environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo

# Install additional dependencies if needed
echo "Checking dependencies..."
pip install aio-pika prometheus-client av

# Set environment variables
export PYTHONPATH=/home/sam3/Desktop/Toms_Workspace/WorldSystem/frame_processor:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export VIDEO_MODE=true
export CONFIG_PROFILE=balanced
export SAM2_MODEL_SIZE=small
export PROCESSING_RESOLUTION=720

# Check if RabbitMQ is running
if docker ps | grep -q rabbitmq; then
    echo "✓ RabbitMQ is running"
else
    echo "Starting RabbitMQ..."
    docker-compose up -d rabbitmq
    sleep 5
fi

echo
echo "Running tests..."
echo

# Option 1: Run simple test (no RabbitMQ needed)
echo "1. Running SAM2 component test..."
python test_sam2_simple.py

echo
echo "2. Running H.264 streaming test..."
echo "   This requires RabbitMQ to be running"
python test_video_tracking.py --create-test --duration 5

echo
echo "Tests complete!"