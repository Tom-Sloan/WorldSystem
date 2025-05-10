# Base image from nvidia NGC
FROM nvcr.io/nvidia/tensorflow:24.07-tf2-py3

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

RUN python3 -m pip install --upgrade pip

# First ensure numpy is installed correctly to avoid binary compatibility issues
RUN pip uninstall -y numpy
RUN pip install --no-cache-dir numpy>=1.24.0

# Install Python packages with the specific version of rerun-sdk
RUN pip install --no-cache-dir \
    mne \
    pyqt6 \
    matplotlib \
    rerun-sdk==0.23.2

# Upgrade ipykernel
RUN /usr/bin/python3 -m pip install ipykernel -U --user --force-reinstall

ARG USERNAME
ARG UID
ARG GID
ARG HOME

# Set up .Xdefaults for X11 color customization
# Create a group and user with the specified GID and UID
RUN groupadd -g $GID $USERNAME && \
    useradd -m -u $UID -g $GID -s /bin/bash $USERNAME

# Grant the user sudo privileges
RUN echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set the default user to the new user
USER $USERNAME
RUN echo "*customization: -color" > $HOME/.Xdefaults

# Make sure workspace directory exists and is owned by the user
RUN mkdir -p /home/$USERNAME/workspace
WORKDIR /home/$USERNAME/workspace

# Copy the Rerun test script
COPY --chown=$USERNAME:$USERNAME test_rerun.py /home/$USERNAME/workspace/test_rerun.py

# For external network access, make sure we're setting up proper network environment
ENV RERUN_VIEWER_ADDRESS=0.0.0.0:9090

# Set up the container to start with the Rerun test script
CMD ["python3", "test_rerun.py"]