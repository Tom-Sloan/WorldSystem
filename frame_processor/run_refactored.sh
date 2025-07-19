#!/bin/bash
# Script to run the refactored frame processor without docker-compose conflicts

# Build the image if needed
echo "Building refactored frame processor image..."
docker build -t frame_processor_refactored:latest -f Dockerfile.refactored .

# Run the container
echo "Starting refactored frame processor..."
docker run --rm -it \
  --name frame_processor_refactored \
  --runtime=nvidia \
  --network=worldsystem_default \
  -e RABBITMQ_URL="amqp://rabbitmq:5672/" \
  -e VIDEO_FRAMES_EXCHANGE="video_frames_exchange" \
  -e PROCESSED_FRAMES_EXCHANGE="processed_frames_exchange_refactored" \
  -e ANALYSIS_MODE_EXCHANGE="analysis_mode_exchange" \
  -e INITIAL_ANALYSIS_MODE="yolo" \
  -e DETECTOR_TYPE="yolo" \
  -e TRACKER_TYPE="iou" \
  -e YOLO_MODEL="yolov11l.pt" \
  -e YOLO_CONFIDENCE="0.5" \
  -e DEVICE="cuda" \
  -e TRACKER_IOU_THRESHOLD="0.3" \
  -e TRACKER_MAX_LOST="10" \
  -e PROCESS_AFTER_SECONDS="1.5" \
  -e MAX_TRACKS="100" \
  -e USE_GCS="false" \
  -e USE_SERPAPI="false" \
  -e USE_PERPLEXITY="false" \
  -e GCS_BUCKET_NAME="worldsystem" \
  -e GOOGLE_APPLICATION_CREDENTIALS="/app/worldsystem-23f7306a1a75.json" \
  -e ENHANCEMENT_ENABLED="true" \
  -e ENHANCEMENT_AUTO_ADJUST="true" \
  -e ENHANCEMENT_GAMMA="1.2" \
  -e ENHANCEMENT_ALPHA="1.3" \
  -e ENHANCEMENT_BETA="20" \
  -e RERUN_ENABLED="true" \
  -e RERUN_VIEWER_ADDRESS="0.0.0.0:9877" \
  -e RERUN_CONNECT_URL="http://host.docker.internal:9877/proxy" \
  -e NTP_SERVER="pool.ntp.org" \
  -e METRICS_PORT="8004" \
  -e LOG_LEVEL="INFO" \
  -e LOG_FILE="/app/logs/frame_processor_refactored.log" \
  -p 8004:8004 \
  -p 9877:9877 \
  -v frame_processor_logs:/app/logs \
  -v /dev/shm:/dev/shm \
  frame_processor_refactored:latest