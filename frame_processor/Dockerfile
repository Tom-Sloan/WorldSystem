# Frame Processor Dockerfile based on Grounded-SAM-2
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies matching Grounded-SAM-2
RUN apt-get update && apt-get install -y \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    nano \
    vim \
    ninja-build \
    gcc-10 \
    g++-10 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set gcc-10 as default compiler
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100

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
RUN pip install --no-cache-dir -r requirements.txt

# Clone Grounded-SAM-2
RUN git clone https://github.com/IDEA-Research/Grounded-SAM-2.git

# Install SAM2
WORKDIR /app/Grounded-SAM-2
RUN pip install -e .

# Install Grounding DINO
RUN pip install --no-build-isolation -e grounding_dino

# Download checkpoints
RUN mkdir -p checkpoints gdino_checkpoints && \
    cd checkpoints && \
    wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt && \
    wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt && \
    wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt && \
    wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt

WORKDIR /app/Grounded-SAM-2/gdino_checkpoints
RUN wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Back to app directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/outputs

# Copy application code
COPY common/ /app/common/
COPY frame_processor/core /app/core/
COPY frame_processor/external /app/external/
COPY frame_processor/pipeline /app/pipeline/
COPY frame_processor/video /app/video/
COPY frame_processor/main.py /app/
COPY frame_processor/grounded_sam2_processor.py /app/
COPY frame_processor/model_configs.py /app/

# Copy Google Cloud credentials if needed
COPY frame_processor/credentials/worldsystem-23f7306a1a75.json /app/credentials/worldsystem-23f7306a1a75.json

# Verification script
RUN echo '#!/bin/bash\necho "Verifying PyTorch installation..."\npython -c "import torch; print(f\"PyTorch version: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA version: {torch.version.cuda}\")"\necho "Verifying Grounded-SAM-2 installation..."\npython -c "from sam2.build_sam import build_sam2; print(\"SAM2 import successful\")"' > /app/verify_install.sh && chmod +x /app/verify_install.sh

# Run verification
RUN /app/verify_install.sh

# Set Python path
ENV PYTHONPATH=/app:/app/Grounded-SAM-2

# Default command - run the main processor
CMD ["python", "main.py"]