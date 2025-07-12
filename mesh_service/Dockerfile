# mesh_service/Dockerfile
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    # SSL/Crypto for AMQP
    libssl-dev \
    # Event loop for AMQP
    libev-dev \
    # Mesh generation libraries
    libcgal-dev \
    libeigen3-dev \
    # Networking
    libwebsocketpp-dev \
    libboost-all-dev \
    # Additional tools
    pkg-config \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Draco compression library from source
RUN cd /tmp && \
    git clone https://github.com/google/draco.git && \
    cd draco && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd / && rm -rf /tmp/draco

# Install prometheus-cpp from source
RUN cd /tmp && \
    git clone https://github.com/jupp0r/prometheus-cpp.git && \
    cd prometheus-cpp && \
    git submodule init && \
    git submodule update && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_PUSH=OFF -DENABLE_COMPRESSION=OFF && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd / && rm -rf /tmp/prometheus-cpp

# Install AMQP-CPP for better RabbitMQ support
RUN cd /tmp && \
    git clone https://github.com/CopernicaMarketingSoftware/AMQP-CPP.git && \
    cd AMQP-CPP && \
    mkdir build && cd build && \
    cmake .. -DAMQP-CPP_BUILD_SHARED=ON -DAMQP-CPP_LINUX_TCP=ON && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd / && rm -rf /tmp/AMQP-CPP

# Install latest msgpack-c (C++ version)
RUN cd /tmp && \
    git clone https://github.com/msgpack/msgpack-c.git && \
    cd msgpack-c && \
    git checkout cpp_master && \
    cmake -S . -B build -DMSGPACK_BUILD_TESTS=OFF -DMSGPACK_BUILD_EXAMPLES=OFF && \
    cmake --build build -j$(nproc) && \
    cmake --install build && \
    ldconfig && \
    cd / && rm -rf /tmp/msgpack-c

# Install Apache Arrow (required by Rerun SDK)
RUN apt-get update && apt-get install -y \
    lsb-release \
    wget \
    && wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr '[:upper:]' '[:lower:]')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb \
    && apt-get install -y ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb \
    && apt-get update \
    && apt-get install -y libarrow-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up user (matches WorldSystem pattern)
ARG USERNAME=developer
ARG UID=1000
ARG GID=1000

RUN groupadd -g ${GID} ${USERNAME} && \
    useradd -u ${UID} -g ${GID} -s /bin/bash -m ${USERNAME}

# Build mesh service
WORKDIR /app
COPY . .

# Fix permissions
RUN chown -R ${USERNAME}:${USERNAME} /app

# Create build directory and compile
USER ${USERNAME}
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc) && \
    # Verify the binary was created
    test -f mesh_service || (echo "Build failed: mesh_service binary not found" && exit 1) && \
    ls -la /app/build/

# Runtime environment
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

# Expose ports
EXPOSE 8005 8006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
    CMD curl -f http://localhost:8006/health || exit 1

# Set working directory
WORKDIR /app/build

# Switch back to non-root user for runtime
USER ${USERNAME}

# Run the mesh service
CMD ["./mesh_service"]