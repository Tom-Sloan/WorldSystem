# frame_processor/Dockerfile
FROM nvidia/cuda:12.6.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Install any dependencies needed (like ultralytics, etc.)
RUN pip3 install ultralytics numpy opencv-python-headless pika

WORKDIR /app
COPY frame_processor.py /app/frame_processor.py
CMD ["python3", "/app/frame_processor.py"]
