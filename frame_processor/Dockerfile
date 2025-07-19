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
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the refactored application structure
COPY core /app/core/
COPY detection /app/detection/
COPY tracking /app/tracking/
COPY external /app/external/
COPY pipeline /app/pipeline/
COPY visualization /app/visualization/
COPY main.py /app/main.py

# All modules are now part of the refactored structure

# Copy YOLO models
COPY models/yolov11l.pt /app/yolov11l.pt

# Copy Google Cloud credentials if available
# This is optional - can also be mounted at runtime
COPY credentials/worldsystem-23f7306a1a75.json* /app/

# Set Python path to include app directory
ENV PYTHONPATH=/app

# Run the refactored main entry point
CMD ["python3", "-u", "/app/main.py"]