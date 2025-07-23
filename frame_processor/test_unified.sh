#!/bin/bash
# Quick test script for unified WebSocket frame processor

echo "=== Testing Unified Frame Processor ==="
echo ""

# Check if services are running
echo "Checking required services..."
MISSING=()

if ! docker ps | grep -q "server"; then
    MISSING+=("server")
fi

if ! docker ps | grep -q "rabbitmq"; then
    MISSING+=("rabbitmq")
fi

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "❌ Missing services: ${MISSING[*]}"
    echo "Start them with: docker-compose up -d server rabbitmq"
    exit 1
fi

echo "✅ Required services running"
echo ""

# Build the frame processor
echo "Building frame processor..."
if docker-compose --profile frame_processor build; then
    echo "✅ Build successful"
else
    echo "❌ Build failed"
    exit 1
fi

echo ""
echo "Starting frame processor..."
docker-compose --profile frame_processor up -d

# Wait for startup
sleep 5

# Check if running
if docker ps | grep -q "frame_processor"; then
    echo "✅ Frame processor started"
    echo ""
    echo "=== Container Logs ==="
    docker logs worldsystem-frame_processor-1 --tail 20
    echo ""
    echo "=== Test Instructions ==="
    echo "1. Send video from Android app or simulator"
    echo "2. Watch logs: docker logs -f worldsystem-frame_processor-1"
    echo "3. Check metrics: curl http://localhost:8003/metrics | grep frame"
    echo "4. Look for 'Connected to WebSocket' in logs"
    echo ""
    echo "To stop: docker-compose --profile frame_processor down"
else
    echo "❌ Frame processor failed to start"
    echo "Check logs: docker logs worldsystem-frame_processor-1"
    exit 1
fi