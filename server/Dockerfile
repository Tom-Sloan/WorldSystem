FROM nvidia/cuda:12.6.0-base-ubuntu22.04

# Set up environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including python3-pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    '^libxcb.*-dev' \
    libx11-xcb-dev \
    libglu1-mesa-dev \
    libxrender-dev \
    libxi-dev \
    libxkbcommon-dev \
    libxkbcommon-x11-dev \
    x11-apps \ 
    git \
    python3-pyqt5 \
    python3-pip \
    ffmpeg \
    sudo \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

# Create environment
COPY environment.yml .
RUN conda env create -f environment.yml

# Install Python dependencies
RUN conda run -n drone_server pip install --no-cache-dir \
    opencv-python-headless \
    numpy \
    pathfinding3d \
    transforms3d \
    --upgrade-strategy only-if-needed

WORKDIR /app

ARG USERNAME
ARG UID
ARG GID
ARG HOME

# Create a group and user with the specified GID and UID
RUN groupadd -g $GID $USERNAME && \
    useradd -m -u $UID -g $GID -s /bin/bash $USERNAME

# Grant the user sudo privileges
RUN echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set the default user to the new user
USER $USERNAME
RUN echo "*customization: -color" > $HOME/.Xdefaults

CMD ["bash", "-c", "source activate drone_server && python main.py"]