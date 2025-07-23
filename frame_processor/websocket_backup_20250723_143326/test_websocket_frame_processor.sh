#!/bin/bash
# Test script for WebSocket frame processor with Rerun visualization

echo "=== WebSocket Frame Processor Test ==="
echo "This will test the frame processor consuming H.264 video via WebSocket"
echo "and visualizing the stream in Rerun"
echo ""

# Check if Rerun is running
echo "Checking Rerun viewer..."
if ! nc -z localhost 9876 2>/dev/null; then
    echo "⚠️  Rerun viewer not detected on localhost:9876"
    echo "Starting Rerun viewer..."
    rerun --port 9876 &
    RERUN_PID=$!
    sleep 2
else
    echo "✓ Rerun viewer is running"
fi

# Check if core services are running
echo ""
echo "Checking required services..."
MISSING_SERVICES=()

if ! docker ps | grep -q "server"; then
    MISSING_SERVICES+=("server")
fi

if ! docker ps | grep -q "rabbitmq"; then
    MISSING_SERVICES+=("rabbitmq")
fi

if [ ${#MISSING_SERVICES[@]} -gt 0 ]; then
    echo "❌ Missing required services: ${MISSING_SERVICES[*]}"
    echo "Please start the core services first:"
    echo "  docker-compose up -d server rabbitmq"
    exit 1
fi

echo "✓ Core services are running"

# Stop any existing frame processor
echo ""
echo "Stopping existing frame processor..."
docker-compose --profile frame_processor down 2>/dev/null
docker-compose -f docker-compose.websocket.yml down 2>/dev/null

# Build the WebSocket frame processor
echo ""
echo "Building WebSocket frame processor..."
docker-compose -f docker-compose.websocket.yml build

# Start the WebSocket frame processor
echo ""
echo "Starting WebSocket frame processor..."
docker-compose -f docker-compose.websocket.yml up -d

# Wait for it to start
echo ""
echo "Waiting for frame processor to initialize..."
sleep 5

# Check if it's running
if docker ps | grep -q "worldsystem-frame_processor_websocket"; then
    echo "✓ WebSocket frame processor is running"
    
    # Show logs
    echo ""
    echo "=== Frame Processor Logs ==="
    docker logs worldsystem-frame_processor_websocket --tail 20
    
    echo ""
    echo "=== Test Instructions ==="
    echo "1. Open Rerun viewer at: http://localhost:9876"
    echo "2. Start sending video from Android app or simulator"
    echo "3. You should see:"
    echo "   - video/h264_stream: Raw H.264 video stream"
    echo "   - video/decoded_frame: Decoded frames"
    echo "   - video/tracked_frame: Frames with object tracking"
    echo "   - tracks/mask_*: Individual object masks"
    echo ""
    echo "4. Monitor logs:"
    echo "   docker logs -f worldsystem-frame_processor_websocket"
    echo ""
    echo "5. Check metrics:"
    echo "   curl http://localhost:8003/metrics | grep frame_processor"
    echo ""
    echo "6. To stop the test:"
    echo "   docker-compose -f docker-compose.websocket.yml down"
else
    echo "❌ Failed to start WebSocket frame processor"
    echo "Check logs with: docker logs worldsystem-frame_processor_websocket"
    exit 1
fi