#!/bin/bash
# Run docker-compose up with dynamic branch name

export BRANCH_NAME=$(git branch --show-current)
echo "Starting services with branch name: $BRANCH_NAME"
docker compose up "$@"