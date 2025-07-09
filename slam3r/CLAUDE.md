# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SLAM3R is part of the WorldSystem project - a real-time RGB-based dense scene reconstruction system that regresses 3D points from video frames using feed-forward neural networks, without explicitly estimating camera parameters. It processes video streams from drones to create 3D reconstructions for AR/VR applications.

## Build Commands

### Docker (Recommended)
```bash
# Build SLAM3R service
docker-compose build slam3r

# Run SLAM3R service  
docker-compose up slam3r

# Run without SLAM3R (if using other SLAM services)
docker compose up --detach $(docker compose config --services | grep -v slam3r)
```

### Local Development
```bash
# Create conda environment
conda create -n slam3r python=3.11 cmake=3.14.0
conda activate slam3r

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
cd SLAM3R_engine
pip install -r requirements.txt
pip install -r requirements_optional.txt  # For Open3D mesh generation

# Optional: Performance acceleration
pip install xformers==0.0.28.post2
cd slam3r/pos_embed/curope/ && python setup.py build_ext --inplace && cd ../../../
```

## Demo/Testing Commands

```bash
cd SLAM3R_engine

# Run on Replica dataset
bash scripts/demo_replica.sh

# Run on custom video/images
bash scripts/demo_wild.sh

# Visualize reconstruction (after running with --save_preds)
bash scripts/demo_vis_wild.sh

# Launch Gradio interface
python app.py

# Evaluate on Replica
python evaluation/process_gt.py  # Generate ground truth first
bash scripts/eval_replica.sh
```

## Training Commands

```bash
cd SLAM3R_engine

# Download pretrained weights
mkdir -p checkpoints
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth -P checkpoints/

# Train models
bash scripts/train_i2p.sh  # Image-to-Points model
bash scripts/train_l2w.sh  # Local-to-World model
```

## Architecture

### Neural Network Pipeline
1. **Image-to-Points (I2P)**: Processes RGB frames → local 3D point clouds
2. **Local-to-World (L2W)**: Transforms local points → global world coordinates

### Key Components
- `slam3r_processor.py`: RabbitMQ service entry point for real-time processing
- `recon.py`: Offline reconstruction pipeline
- `app.py`: Gradio web interface
- `slam3r/models.py`: Core model definitions (Multiview3D)
- `slam3r/inference.py`: Inference logic and keyframe management

### Real-time Processing Flow
```
RabbitMQ (frames) → Tokenization → I2P Inference → L2W Registration →
Keyframe Selection → Point Accumulation → Mesh Generation → Visualization
```

### Performance Considerations

**Current Bottlenecks** (from plan.md):
- Point cloud downsampling: 47% of CPU time
- Python GIL limitations with ThreadPoolExecutor
- Synchronous mesh generation blocking main thread
- Inefficient data structures (numpy ↔ list conversions)

**Optimization Opportunities**:
- Remove downsampling (mesh generation already reduces volume)
- Increase RabbitMQ prefetch_count (currently 1)
- Fix INFERENCE_WINDOW_BATCH for proper GPU batching
- Use process pool for true async mesh generation
- Implement adaptive mesh generation based on camera motion

## Configuration

### Environment Variables
- `RABBITMQ_HOST`: Message queue host
- `ENABLE_MESH_GENERATION`: Enable Open3D mesh generation
- `SAVE_SEGMENTS`: Save point clouds/trajectories per segment
- `SEGMENT_OUTPUT_DIR`: Directory for segment outputs

### Key Parameters
- `configs/camera_intrinsics.yaml`: Camera calibration
- `configs/wild.yaml`: Reconstruction parameters
- Max points: 2M (configurable in SpatialPointCloudBuffer)
- Keyframe stride: Adaptive (1-10 based on motion)

## Development Guidelines

### Code Organization
- Keep processing logic in `slam3r_processor.py` lean
- Separate concerns: SLAM processing vs visualization
- Use numpy arrays consistently (avoid list conversions)
- Implement proper error handling with logging

### Performance Tips
- Batch frame processing when possible
- Use msgpack for RabbitMQ serialization
- Offload heavy computation to separate processes
- Monitor memory usage (point cloud accumulation)

### Testing
- Use Replica dataset for benchmarking
- Check reconstruction quality with Rerun visualization
- Monitor processing FPS (target: 25fps)
- Validate mesh generation with Open3D viewer