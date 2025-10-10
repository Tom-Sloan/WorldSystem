# Base image: NVIDIA CUDA devel for GPU support
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libssl-dev \
    libgrpc++-dev \
    protobuf-compiler-grpc \
    git \
    curl \
    librabbitmq-dev \
    libmsgpack-dev \
    cargo \
    pkg-config \
    libclang-dev \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Apache Arrow (required by Rerun SDK)
RUN apt-get update && apt-get install -y \
    lsb-release \
    && wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr '[:upper:]' '[:lower:]')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb \
    && apt-get install -y ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb \
    && apt-get update \
    && apt-get install -y libarrow-dev \
    && rm -rf /var/lib/apt/lists/*

# Install RabbitMQ-C from source for completeness
RUN git clone https://github.com/alanxz/rabbitmq-c.git /rabbitmq-c && \
    cd /rabbitmq-c && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && make install && \
    ldconfig

# Install Open3D
RUN apt-get update && apt-get install -y libopen3d-dev

# Set up working directory
WORKDIR /app

# Copy project files
COPY CMakeLists.txt /app/
COPY src /app/src
COPY include /app/include

# Build the application
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j$(nproc)

# Run the executable
CMD ["./build/mesh_service"]