# Dockerfile for SLAM3R service
# This Dockerfile sets up an environment to run the SLAM3R application,
# consuming RGB images from RabbitMQ and producing SLAM outputs.
# Influenced by user prompts regarding CUDA version,
# SLAM3R GitHub repository, checkpoint handling, and a subsequent
# prompt to remove Conda-based environment management in favor of system Python.

# Base image with CUDA 11.8.1
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1

ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV PYTHONPATH=/app:$PYTHONPATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    sudo \
    build-essential \
    cmake \
    curl \
    gnupg2 \
    lsb-core \
    vim \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    ffmpeg \
    '^libxcb.*-dev' \
    libx11-xcb-dev \
    libglu1-mesa-dev \
    libxrender-dev \
    libxi-dev \
    libxkbcommon-dev \
    libxkbcommon-x11-dev \
    x11-apps \
    # OpenCV dependencies
    libgl1-mesa-dev \
    libgtk2.0-dev \
    # Python
    python3-pip \
    python3-dev \
    python3-opencv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Install PyTorch (cu118 as per SLAM3R README, ensure compatibility with CUDA 12.4 base or update if needed)
# Note: If issues arise, consider a PyTorch build for CUDA 12.1 or using a CUDA 11.8 base image.
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install xformers (memory-efficient attention)
RUN pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118

# Set working directory
WORKDIR /app

# Copy SLAM3R engine code (submodule) and requirements files
# This assumes the submodule is cloned into SLAM3R_engine within the build context (./slam3r)
COPY ./SLAM3R_engine /app/SLAM3R_engine/

# Install SLAM3R Python dependencies
RUN pip3 install -r /app/SLAM3R_engine/requirements.txt

# Ensure libGL is available for Rerun (even if running headlessly, some parts might link against it)
RUN apt-get update && apt-get install -y libgl1-mesa-glx && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install other necessary Python packages for the processor script
RUN pip3 install aio-pika PyYAML

# --- Rerun SDK (visualization) ---
RUN pip3 install --no-cache-dir "numpy>=2.2,<3" rerun-sdk==0.23.2

# Build RoPE2D CUDA extension
RUN cd /app/SLAM3R_engine/slam3r/pos_embed/curope && \
    python3 setup.py build_ext --inplace

    
WORKDIR /app

RUN mkdir -p /app/models
RUN python3 -c "from SLAM3R_engine.slam3r.models import Image2PointsModel, Local2WorldModel; \
    i2p = Image2PointsModel.from_pretrained('siyan824/slam3r_i2p'); \
    l2w = Local2WorldModel.from_pretrained('siyan824/slam3r_l2w')"

WORKDIR /app
ARG USERNAME
ARG UID
ARG GID
ARG HOME
RUN groupadd -g $GID $USERNAME && \
    useradd -m -u $UID -g $GID -s /bin/bash $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER $USERNAME
RUN echo "*customization: -color" > $HOME/.Xdefaults

# Rerun visualization configuration
ENV RERUN_VIEWER_ADDRESS="0.0.0.0:9876"
ENV RERUN_CONNECT_URL="rerun+http://127.0.0.1:9876/proxy"
ENV RERUN_ENABLED="true"

# Expose Rerun viewer gRPC/Web port
EXPOSE 9876

# Set the default command to run the processor script
CMD ["python3", "./SLAM3R_engine/slam3r_processor.py"] 