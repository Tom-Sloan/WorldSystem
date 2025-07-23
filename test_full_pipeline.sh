#!/bin/bash
# Test the full WebSocket video pipeline

echo "WebSocket Video Pipeline Test"
echo "============================="

# Check if server is running
echo -n "Checking server health... "
if curl -s http://localhost:5001/health/video > /dev/null 2>&1; then
    echo "✓ Server is running"
else
    echo "✗ Server not running. Start with: docker-compose up server"
    exit 1
fi

# Check video status
echo -e "\nVideo streaming status:"
curl -s http://localhost:5001/video/status | python3 -m json.tool

# Start a consumer in background
echo -e "\nStarting test consumer..."
python3 test_websocket_streaming.py &
CONSUMER_PID=$!

# Give consumer time to connect
sleep 2

# Check status again
echo -e "\nVideo status after consumer connected:"
curl -s http://localhost:5001/video/status | python3 -m json.tool

# Start simulator
echo -e "\nStarting simulator..."
docker-compose run --rm simulator

# Check final status
echo -e "\nFinal video status:"
curl -s http://localhost:5001/video/status | python3 -m json.tool

# Clean up
kill $CONSUMER_PID 2>/dev/null

echo -e "\nTest complete!"