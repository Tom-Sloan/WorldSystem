FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
RUN pip install aio-pika opencv-python-headless numpy pika prometheus_client

# Copy and install shared package
COPY shared /tmp/shared
RUN pip install -e /tmp/shared

# Copy application code
COPY . .

CMD ["python", "data_storage.py"] 