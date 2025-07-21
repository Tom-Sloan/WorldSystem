FROM nvidia/cuda:12.6.0-base-ubuntu22.04

# Set up environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
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
    ffmpeg \
    sudo \
    && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1 \
    libglu1-mesa \
    freeglut3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --no-binary pyrender

ARG USERNAME
ARG UID
ARG GID
ARG HOME

# Set up .Xdefaults for X11 color customization
# Create a group and user with the specified GID and UID
RUN groupadd -g $GID $USERNAME && \
    useradd -m -u $UID -g $GID -s /bin/bash $USERNAME

# Grant the user sudo privileges (optional)
RUN echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set the default user to the new user
USER $USERNAME
RUN echo "*customization: -color" > $HOME/.Xdefaults

CMD ["python3", "main.py"] 