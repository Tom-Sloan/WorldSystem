#!/bin/bash
# Entrypoint script for mesh_service

echo "Starting Mesh Service..."
cd /app/build
exec ./mesh_service