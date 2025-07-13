#!/bin/bash
# Build docker-compose with dynamic branch name

export BRANCH_NAME=$(git branch --show-current)
echo "Building with branch name: $BRANCH_NAME"
docker-compose build "$@"