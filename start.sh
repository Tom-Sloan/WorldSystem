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

# Determine the Docker Compose command
if command -v docker-compose &> /dev/null; then
    compose_cmd="docker-compose"
else
    compose_cmd="docker compose"
fi

# Run Docker Compose
$compose_cmd up --build -d slam

# Capture the container ID or name
container_id=$($compose_cmd ps -q slam)

# Function to clean up the specific container
cleanup() {
  echo "Cleaning up container $container_id..."
  if [ -n "$container_id" ]; then
    docker container rm -f "$container_id" 2>/dev/null || echo "Failed to remove container $container_id"
  else
    echo "No container ID found to clean up."
  fi
}

# Check if the container is running (i.e., deployment failed)
if ! docker ps -q -f "id=$container_id" > /dev/null; then
  echo "Docker container failed to start."
  cleanup
  exit 1
fi

echo "Docker Compose build and deployment succeeded."
