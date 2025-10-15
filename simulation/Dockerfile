FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Ensure Python output is unbuffered for real-time logging
ENV PYTHONUNBUFFERED=1

WORKDIR /app
RUN pip install aio-pika opencv-python-headless numpy websockets
COPY . .

# Copy test data into the container
COPY test_continuous_hallway /test_continuous_hallway

CMD ["python", "simulate_video_stream.py"] 