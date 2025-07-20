#!/bin/bash
# Integration test script for SAM2 video tracking

echo "SAM2 Video Tracking Integration Test"
echo "===================================="
echo

# Check if frame_processor is running
if docker ps | grep -q "frame_processor"; then
    echo "✓ Frame processor is running"
else
    echo "✗ Frame processor is not running"
    echo "  Please start it with: docker-compose --profile frame_processor up"
    exit 1
fi

# Check if RabbitMQ is accessible
if curl -s http://localhost:15672 > /dev/null; then
    echo "✓ RabbitMQ management UI is accessible"
else
    echo "✗ Cannot access RabbitMQ management UI"
    echo "  Please ensure RabbitMQ is running"
    exit 1
fi

# Check environment variables
echo
echo "Checking configuration..."
if [[ -f .env ]]; then
    echo "✓ .env file found"
    
    # Check critical variables
    if grep -q "VIDEO_MODE=true" .env; then
        echo "✓ VIDEO_MODE is enabled"
    else
        echo "⚠ VIDEO_MODE is not enabled in .env"
        echo "  Add: VIDEO_MODE=true"
    fi
    
    if grep -q "SERPAPI_API_KEY=" .env && ! grep -q "SERPAPI_API_KEY=$" .env; then
        echo "✓ SERPAPI_API_KEY is set"
    else
        echo "⚠ SERPAPI_API_KEY is not set"
    fi
else
    echo "⚠ No .env file found"
    echo "  Copy .env.example to .env and configure"
fi

echo
echo "Testing video tracking..."
echo

# Option 1: Test with synthetic video
echo "1. Creating synthetic test video..."
docker exec -it worldsystem-frame_processor-1 python3 /app/test_sam2_simple.py

echo
echo "2. Testing H.264 streaming (requires PyAV)..."
echo "   Installing PyAV in container..."
docker exec -it worldsystem-frame_processor-1 pip3 install av

echo
echo "3. Running video stream test..."
docker exec -it worldsystem-frame_processor-1 python3 -c "
import sys
sys.path.append('/app')
exec(open('/app/test_video_tracking.py').read())
" --create-test --duration 5 --monitor

echo
echo "Integration test complete!"
echo
echo "Check the following:"
echo "1. Rerun viewer at http://localhost:9876 for live visualization"
echo "2. Metrics at http://localhost:8003/metrics"
echo "3. RabbitMQ queues at http://localhost:15672"
echo "4. Logs: docker logs worldsystem-frame_processor-1"