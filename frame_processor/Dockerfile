# Video-Only Frame Processor Dockerfile
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
# We need to copy them to where SAM2 expects them - in the package directory
RUN git clone https://github.com/facebookresearch/sam2.git /tmp/sam2 && \
    mkdir -p /app/sam2_configs && \
    cp /tmp/sam2/sam2_configs/*.yaml /app/sam2_configs/ || true && \
    # Also copy to the SAM2 package directory where Hydra looks for them
    python3 -c "import sam2; import os; print(os.path.dirname(sam2.__file__))" > /tmp/sam2_path.txt && \
    SAM2_PATH=$(cat /tmp/sam2_path.txt) && \
    cp /tmp/sam2/sam2_configs/*.yaml ${SAM2_PATH}/ || true && \
    rm -rf /tmp/sam2

# Create models directory
RUN mkdir -p /app/models

# Copy all model files from local models directory
COPY models/ /app/models/

# Copy the refactored application structure
COPY core /app/core/
COPY video /app/video/
COPY external /app/external/
COPY pipeline /app/pipeline/
COPY visualization /app/visualization/
COPY main.py /app/main.py
COPY model_configs.py /app/model_configs.py

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