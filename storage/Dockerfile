FROM python:3.9-slim

# Install system dependencies including FFmpeg for RTSP
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy common utilities first
COPY common/ /app/common/

# Copy storage specific code
COPY storage/ /app/storage/

# Install Python dependencies
# Note: opencv-python-headless doesn't include video codecs
# We need the full opencv-python for RTSP support
RUN pip install --no-cache-dir \
    opencv-contrib-python==4.8.1.78 \
    aio-pika==9.0.0 \
    numpy==1.24.3 \
    prometheus-client==0.17.1

# Create recordings directory
RUN mkdir -p /app/recordings

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Run RTSP storage processor
CMD ["python", "/app/storage/rtsp_storage.py"] 