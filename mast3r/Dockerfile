FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG USERNAME
ARG UID
ARG GID
ENV USERNAME=${USERNAME} UID=${UID} GID=${GID}
ENV MAST3R_CHECKPOINTS_DIR=/app/checkpoints

# ───────────── Runtime environment ──────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    TORCH_CUDA_ARCH_LIST="8.6" \
    PYTHONPATH=/app:$PYTHONPATH \
    FORCE_CUDA=1                

# ───────────── System packages ──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl build-essential cmake ninja-build \
        libgl1-mesa-glx libjpeg-dev libpng-dev libtiff-dev \
        ffmpeg python3-pip python3-dev python3-opencv \
        x11-apps libx11-xcb-dev libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# ───────────── PyTorch & friends (CUDA 12.4 wheels) ─────────────────
RUN pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
        --index-url https://download.pytorch.org/whl/cu124
RUN pip install xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu124

WORKDIR /app
# ───────────── Download MASt3R checkpoints once ─────
RUN mkdir -p /tmp/mast3r_checkpoints && \
    wget -nv -P /tmp/mast3r_checkpoints https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth && \
    wget -nv -P /tmp/mast3r_checkpoints https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth && \
    wget -nv -P /tmp/mast3r_checkpoints https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl
# Move the cached checkpoints into the repo so runtime can see them
RUN mkdir -p /app/checkpoints && cp /tmp/mast3r_checkpoints/* /app/checkpoints/

COPY . /app

RUN pip install -e thirdparty/mast3r \
 && pip install -e thirdparty/in3d \
 && pip install --no-build-isolation -e . \
 && pip install torchcodec==0.1


# build gn.cpp/cu
RUN python3 setup.py build_ext --inplace

# ───────────── Pre‑compile torch JIT extension so the container starts instantly ─────
RUN python3 - <<'PY'
import torch                      # ensures CUDA/Torch ready
import mast3r_slam_backends       # triggers JIT build at image‑build time
print("✓ mast3r_slam_backends compiled into the image layer.")
PY

# ───────────── Misc runtime deps ────────────────────────────────────
RUN pip install numpy rerun-sdk aio-pika opencv-python-headless pyyaml
RUN pip install pyrealsense2 natsort plyfile tqdm
# RUN pip install --no-cache-dir --no-binary=lietorch
RUN pip install --no-build-isolation git+https://github.com/princeton-vl/lietorch.git

# ───────────── Optional unprivileged user (match host IDs) ──────────
RUN groupadd -g ${GID} ${USERNAME} \
 && useradd  -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME} \
 && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER ${USERNAME}
WORKDIR /app

# ───────────── Entrypoint ───────────────────────────────────────────
COPY mast3r_processor.py /app/mast3r_processor.py
CMD ["python3", "mast3r_processor.py"]