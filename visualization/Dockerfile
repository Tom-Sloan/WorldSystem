# Visualization service using Rerun SDK
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY visualization/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY visualization/ /app/

# Copy common utilities
COPY common/ /app/common

# Expose Rerun viewer port
EXPOSE 9876

# Run the visualization service
CMD ["python", "main.py"]