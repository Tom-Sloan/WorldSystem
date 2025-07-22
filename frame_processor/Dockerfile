# Grounded-SAM-2 Frame Processor Dockerfile
# Using CUDA 12.1 development image for build tools
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies matching Grounded-SAM-2 requirements
RUN apt-get update && apt-get install -y \
      wget \
      ffmpeg \
      libsm6 \
      libxext6 \
      git \
      nano \
      vim \
      ninja-build \
      gcc \
      g++ \
      libgl1-mesa-glx \
      libglib2.0-0 \
      libgomp1 \
      libxrender-dev \
      libavcodec-dev \
      libavformat-dev \
      libavutil-dev \
      libswscale-dev \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for CUDA and build
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV BUILD_WITH_CUDA=1
ENV SAM2_BUILD_CUDA=1
ENV SAM2_BUILD_ALLOW_ERRORS=0

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY frame_processor/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Clone Grounded-SAM-2
RUN git clone https://github.com/IDEA-Research/Grounded-SAM-2.git

# Install Segment Anything 2 in editable mode
WORKDIR /app/Grounded-SAM-2
RUN pip install -e .

# Install Grounding DINO with special flag
RUN pip install --no-build-isolation -e grounding_dino

# Download model checkpoints
RUN mkdir -p checkpoints gdino_checkpoints && \
    cd checkpoints && \
    # Download SAM2 checkpoint
    wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt && \
    cd ../gdino_checkpoints && \
    # Download Grounding DINO checkpoint
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Additional model download script
WORKDIR /app/Grounded-SAM-2/checkpoints
RUN bash download_ckpts.sh || true

WORKDIR /app/Grounded-SAM-2/gdino_checkpoints  
RUN bash download_ckpts.sh || true

# Back to app directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/models /app/logs

# Copy common utilities
COPY common/ /app/common/

# Copy the existing application structure (for API integration)
COPY frame_processor/core /app/core/
COPY frame_processor/external /app/external/
COPY frame_processor/pipeline /app/pipeline/
COPY frame_processor/visualization /app/visualization/

# Copy the new Grounded-SAM-2 processor and model configs
COPY frame_processor/grounded_sam2_processor.py /app/
COPY frame_processor/model_configs.py /app/

# Copy Google Cloud credentials if available
# This is optional - can also be mounted at runtime
COPY frame_processor/credentials/worldsystem-23f7306a1a75.json* /app/

# Set Python path to include app directory and Grounded-SAM-2
ENV PYTHONPATH=/app:/app/Grounded-SAM-2
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Create a startup script to verify installation
RUN echo '#!/bin/bash\n\
echo "Verifying PyTorch installation..."\n\
python3 -c "import torch; print(f\"PyTorch version: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA version: {torch.version.cuda}\")"\n\
echo "Verifying Grounded-SAM-2 installation..."\n\
python3 -c "import sys; sys.path.append(\"/app/Grounded-SAM-2\"); from sam2.build_sam import build_sam2; print(\"SAM2 import successful\")"\n\
python3 -c "import timm; print(f\"timm version: {timm.__version__}\")"\n\
echo "Starting frame processor..."\n\
exec python3 -u /app/grounded_sam2_processor.py' > /app/start.sh && chmod +x /app/start.sh

# Run the Grounded-SAM-2 processor
CMD ["/app/start.sh"]