# Testing SAM2 Video Tracking

This guide explains how to test the SAM2 video tracking functionality in the frame processor.

## Prerequisites

1. **Environment Setup**
   ```bash
   # Copy and configure .env
   cp .env.example .env
   
   # Set these variables in .env:
   VIDEO_MODE=true
   CONFIG_PROFILE=balanced  # or performance/quality
   SERPAPI_API_KEY=your_key_here  # Optional for object identification
   ```

2. **Start Services**
   ```bash
   # Start RabbitMQ
   docker-compose up -d rabbitmq
   
   # Start frame processor with GPU support
   docker-compose --profile frame_processor up
   ```

## Test Scripts

### 1. Simple Component Test
Tests SAM2 components directly without RabbitMQ:

```bash
# Run inside container
docker exec -it worldsystem-frame_processor-1 python3 /app/test_sam2_simple.py
```

This test:
- Initializes SAM2 tracker with video buffer
- Generates synthetic moving objects
- Tests tracking consistency
- Measures FPS and memory usage
- Saves annotated sample frames

### 2. H.264 Stream Test
Tests the complete pipeline with H.264 video streaming:

```bash
# Install PyAV first
docker exec -it worldsystem-frame_processor-1 pip3 install av

# Run test with synthetic video
docker exec -it worldsystem-frame_processor-1 python3 /app/test_video_tracking.py \
  --create-test --duration 10 --monitor
```

Options:
- `--create-test`: Generate synthetic test video
- `--duration N`: Test video duration in seconds
- `--video path/to/video.mp4`: Use existing video
- `--loop`: Loop the video continuously
- `--max-frames N`: Limit frames to process
- `--monitor`: Monitor processed results

### 3. Integration Test Script
Automated test that checks everything:

```bash
chmod +x test_integration.sh
./test_integration.sh
```

## Performance Benchmarks

Expected performance on different GPUs:

| GPU | Model Size | Resolution | Expected FPS |
|-----|------------|------------|--------------|
| RTX 3090 | Small | 720p | 20-25 |
| RTX 3090 | Base | 720p | 15-20 |
| RTX 3090 | Large | 720p | 8-12 |
| RTX 4090 | Small | 1080p | 25-30 |
| RTX 4090 | Base | 1080p | 18-22 |
| RTX 4090 | Large | 1080p | 12-15 |

## Monitoring During Tests

### 1. Rerun Viewer
Open http://localhost:9876 to see:
- Live video feed with tracking overlays
- Object trajectories
- Performance metrics
- Memory usage graphs

### 2. Prometheus Metrics
Check http://localhost:8003/metrics for:
- `frame_processor_frames_processed_total`
- `frame_processor_processing_time_seconds`
- `frame_processor_active_tracks`

### 3. RabbitMQ Management
Monitor at http://localhost:15672 (guest/guest):
- Check `video_stream_exchange` for incoming streams
- Check `processed_frames_exchange` for outputs
- Monitor queue depths and rates

### 4. Container Logs
```bash
# Follow logs
docker logs -f worldsystem-frame_processor-1

# Filter for tracking info
docker logs worldsystem-frame_processor-1 2>&1 | grep -E "(FPS|tracks|Track_)"
```

## Common Issues

### 1. Out of Memory (OOM)
If you get CUDA OOM errors:
- Reduce `PROCESSING_RESOLUTION` (480, 720, 1080)
- Use smaller model: `SAM2_MODEL_SIZE=tiny`
- Enable dynamic switching: `ENABLE_DYNAMIC_MODEL_SWITCHING=true`

### 2. Low FPS
To improve performance:
- Use `CONFIG_PROFILE=performance`
- Reduce `GRID_PROMPT_DENSITY` (4, 9, 16)
- Increase `REPROMPT_INTERVAL` (60, 120, 180)

### 3. Poor Tracking
For better tracking quality:
- Use `CONFIG_PROFILE=quality`
- Decrease `MIN_OBJECT_AREA` for small objects
- Adjust `VIDEO_BUFFER_SIZE` for longer memory

## Testing with Real Video

To test with your own video:

```bash
# Copy video to container
docker cp your_video.mp4 worldsystem-frame_processor-1:/tmp/

# Run test
docker exec -it worldsystem-frame_processor-1 python3 /app/test_video_tracking.py \
  --video /tmp/your_video.mp4 --monitor
```

## Debugging

Enable debug logging:
```bash
# In docker-compose.yml or .env
LOG_LEVEL=DEBUG
```

Save debug frames:
```bash
# Frames will be saved to /app/test_output/
docker exec -it worldsystem-frame_processor-1 ls -la /app/test_output/

# Copy them out
docker cp worldsystem-frame_processor-1:/app/test_output ./test_output
```

## Next Steps

After successful testing:

1. **Optimize for your use case**:
   - Adjust configuration profile
   - Tune detection parameters
   - Configure API integration

2. **Production deployment**:
   - Set up proper API keys
   - Configure GCS bucket
   - Enable monitoring/alerting

3. **Integration with other services**:
   - Connect SLAM3R for camera poses
   - Enable mesh service visualization
   - Configure storage service