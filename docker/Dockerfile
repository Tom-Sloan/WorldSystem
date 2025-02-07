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

# Install Python packages
RUN pip install --no-cache-dir \
    mne \
    pyqt6 \
    matplotlib

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

# Grant the user sudo privileges (optional)
RUN echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set the default user to the new user
USER $USERNAME
RUN echo "*customization: -color" > $HOME/.Xdefaults
RUN mkdir -p /home/$USERNAME/workspace
WORKDIR $HOME/workspace

# Set up the container to start with bash by default
CMD ["/bin/bash"]
