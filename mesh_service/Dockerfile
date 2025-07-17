# mesh_service/Dockerfile
# Mesh reconstruction service using NVIDIA Marching Cubes implementation
# with algorithm selector for future Poisson/NKSR support
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Build argument for Open3D support (ON by default for Poisson reconstruction)
ARG USE_OPEN3D=ON

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

# Conditionally install Open3D for high-quality normal estimation
RUN if [ "$USE_OPEN3D" = "ON" ]; then \
        echo "Installing Open3D for normal estimation..." && \
        apt-get update && apt-get install -y \
            xorg-dev \
            libglu1-mesa-dev \
            python3-dev \
            && cd /tmp && \
        git clone --depth 1 --branch v0.17.0 https://github.com/isl-org/Open3D && \
        cd Open3D && \
        mkdir build && cd build && \
        cmake .. \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_SHARED_LIBS=ON \
            -DBUILD_CUDA_MODULE=ON \
            -DBUILD_GUI=OFF \
            -DBUILD_WEBRTC=OFF \
            -DBUILD_EXAMPLES=OFF \
            -DBUILD_UNIT_TESTS=OFF \
            -DBUILD_BENCHMARKS=OFF \
            -DBUILD_PYTHON_MODULE=OFF \
            -DUSE_SYSTEM_EIGEN3=ON && \
        make -j$(nproc) && \
        make install && \
        ldconfig && \
        cd / && rm -rf /tmp/Open3D && \
        rm -rf /var/lib/apt/lists/*; \
    else \
        echo "Skipping Open3D installation (USE_OPEN3D=OFF)"; \
    fi

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
    cmake -DCMAKE_BUILD_TYPE=Release -DUSE_OPEN3D=${USE_OPEN3D} .. && \
    make -j$(nproc) && \
    # Verify the binary was created
    test -f mesh_service || (echo "Build failed: mesh_service binary not found" && exit 1) && \
    ls -la /app/build/

# Clean up build artifacts to reduce image size
RUN cd /app/build && \
    find . -name "*.o" -delete && \
    find . -name "*.a" -delete && \
    strip mesh_service || true

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