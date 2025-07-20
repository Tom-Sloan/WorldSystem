# Refactored Frame Processor Dockerfile
FROM nvidia/cuda:12.6.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
      python3 \
      python3-pip \
      libgl1-mesa-glx \
      libglib2.0-0 \
      libgomp1 \
      libsm6 \
      libxext6 \
      libxrender-dev \
      libgomp1 \
      wget \
      git \
      ffmpeg \
      libavcodec-dev \
      libavformat-dev \
      libavutil-dev \
      libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Clone SAM2 repository temporarily to get config files
RUN git clone https://github.com/facebookresearch/sam2.git /tmp/sam2 && \
    mkdir -p /app/sam2_configs && \
    cp /tmp/sam2/sam2_configs/*.yaml /app/sam2_configs/ || true && \
    rm -rf /tmp/sam2

# Create models directory
RUN mkdir -p /app/models

# Copy all model files from local models directory
COPY models/ /app/models/

# Copy the refactored application structure
COPY core /app/core/
COPY detection /app/detection/
COPY tracking /app/tracking/
COPY external /app/external/
COPY pipeline /app/pipeline/
COPY visualization /app/visualization/
COPY main.py /app/main.py
COPY sam2_configs.py /app/sam2_configs.py

# Copy additional documentation if needed
COPY docs /app/docs/

# All modules are now part of the refactored structure



# Copy Google Cloud credentials if available
# This is optional - can also be mounted at runtime
COPY credentials/worldsystem-23f7306a1a75.json* /app/

# Set Python path to include app directory
ENV PYTHONPATH=/app

# Set Docker environment flag for performance monitor
ENV IS_DOCKER=true

# Run the refactored main entry point
CMD ["python3", "-u", "/app/main.py"]