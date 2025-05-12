FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev \
        libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev \
        x11-apps git ffmpeg python3-pip sudo && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip \
 && pip install --no-cache-dir "numpy>=2.2,<3" rerun-sdk==0.23.2

COPY test_rerun.py /workspace/test_rerun.py
WORKDIR /workspace
CMD ["python3", "test_rerun.py"]