# Mesh Service v2.0 - Python-based with Open3D
FROM python:3.10-slim

# Install system dependencies
# Open3D requires: libgomp1, libgl1, libglib2.0-0
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY python/ /app/

# Expose Rerun viewer port (if running locally)
EXPOSE 9876

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the mesh service
CMD ["python", "main.py"]
