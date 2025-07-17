FROM nvidia/cuda:12.6.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
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

# Copy requirements first for better caching
WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Install dependencies including rerun-sdk version 0.23.2
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY frame_processor.py /app/frame_processor.py
COPY *.py /app/

CMD ["python3", "/app/frame_processor.py"]