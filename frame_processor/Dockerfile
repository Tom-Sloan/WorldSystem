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
COPY frame_processor_original.py /app/frame_processor_original.py
COPY enhanced_rerun_visualizer.py /app/enhanced_rerun_visualizer.py
COPY integrate_enhanced_visualizer.py /app/integrate_enhanced_visualizer.py
COPY modules /app/modules/
RUN touch /app/modules/__init__.py

# Copy YOLO model and credentials (these must exist at build time)
COPY yolov11l.pt /app/yolov11l.pt
COPY worldsystem-23f7306a1a75.json /app/worldsystem-23f7306a1a75.json

CMD ["python3", "-u", "/app/frame_processor.py"]