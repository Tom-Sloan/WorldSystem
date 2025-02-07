#!/bin/bash

# Dynamically set environment variables
echo "USERNAME=$(whoami)" > .env
echo "HOME=$HOME" >> .env
echo "WORKSPACE=/home/sam3/Desktop/Toms_Workspace/WorldSystem" >> .env
echo "X11SOCKET=/tmp/.X11-unix" >> .env
echo "XAUTHORITY=${HOME}/.Xauthority" >> .env
echo "UID=$(id -u)" >> .env
echo "GID=$(id -g)" >> .env
echo "CUDA_PATH=/usr/local/cuda" >> .env
echo "LIBGL_ALWAYS_INDIRECT=1" >> .env
echo "DISPLAY=$DISPLAY" >> .env
echo "BIND_HOST=0.0.0.0" >> .env

# Determine the Docker Compose command
if command -v docker-compose &> /dev/null; then
    compose_cmd="docker-compose"
else
    compose_cmd="docker compose"
fi

# Add cleanup of all containers before building
echo "Removing any existing containers..."
$compose_cmd down --remove-orphans --volumes --rmi local

# Run Docker Compose with build and remove orphans
$compose_cmd up --build --force-recreate --remove-orphans -d

# Remove the container_id check and simplify cleanup
# Since we're checking all services, we don't need single-container checks
if [ $? -ne 0 ]; then
  echo "Docker Compose failed to start services"
  $compose_cmd down
  exit 1
fi

echo "All services rebuilt and started:"
echo "- slam (3D SLAM)"
echo "- fantasy (Fantasy World Generator)"
echo "- server (Backend API)"
echo "- reconstruction (3D Reconstruction)"
echo "- rabbitmq (Message Broker)"
echo "- nginx (Web Server)"
echo "- website (Frontend)"
