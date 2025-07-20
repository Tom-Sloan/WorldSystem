FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Ensure Python output is unbuffered for real-time logging
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . .
RUN pip install aio-pika opencv-python-headless numpy websockets

CMD ["python", "simulate_video_stream.py"] 