# SLAM3R Debugging Guide

## Current Status
The SLAM3R v2 processor starts successfully and connects to RabbitMQ, but doesn't appear to be processing frames when running the test.

## Things to Check

### 1. **RabbitMQ Connection**
Check if messages are being received:
```bash
# Check RabbitMQ management UI
http://localhost:15672
# Look for:
# - video_frames_exchange exists
# - slam3r_queue is bound to the exchange
# - Messages are being published to the exchange
```

### 2. **Queue Name Conflict**
The queue name 'slam3r_queue' might already exist and be bound to a different exchange:
```bash
# In RabbitMQ management UI, check:
# - Queues tab → slam3r_queue
# - See what exchange it's bound to
# - Check message count
```

### 3. **Docker Networking**
Since slam3r uses `network_mode: host`, ensure:
- RabbitMQ is accessible on localhost:5672
- Not using container names for connection

### 4. **Test the Processor**
Run the simple test script:
```bash
cd slam3r/SLAM3R_engine
python3 test_slam3r_v2.py
```

Then check docker logs:
```bash
docker logs slam3r -f
```

### 5. **Common Issues**

#### No Messages Received
- Queue might be bound to wrong exchange
- Messages might be in wrong format
- Network connectivity issues

#### Messages Received but No Processing
- Image decoding might be failing
- Initialization might be taking longer than expected
- Batch accumulator might be waiting for more frames

### 6. **Debug Steps**

1. **Check if messages are reaching the handler**:
   Add logging at the very start of `on_video_frame_message`

2. **Check if frames are being accumulated**:
   The BatchAccumulator waits for 5 frames by default before processing

3. **Check initialization**:
   SLAM3R needs 5 frames for initialization before it can process

4. **Check for exceptions**:
   Look for any error messages in the logs

### 7. **Expected Log Output**
When working correctly, you should see:
```
Received frame with shape (480, 640, 3), timestamp 1234567890
Processing frame 0 with timestamp 1234567890
Handling initialization, have 0 frames
...
Processing frame 4 with timestamp 1234567894
Handling initialization, have 4 frames
SLAM initialization successful with 5 frames
Processing batch of 5 frames
Got result from SLAM3R: dict_keys(['pose', 'points', 'confidence', ...])
```

### 8. **Quick Fixes to Try**

1. **Delete existing queue**:
   ```python
   # In RabbitMQ management UI, delete slam3r_queue
   # Or use different queue name in slam3r_processor_v2.py
   ```

2. **Increase logging**:
   ```bash
   # Set environment variable
   export LOG_LEVEL=DEBUG
   ```

3. **Reduce batch size**:
   ```bash
   export SLAM3R_BATCH_SIZE=1
   ```

4. **Check GPU memory**:
   ```bash
   nvidia-smi
   ```

### 9. **Alternative: Use Original Processor**
If v2 continues to have issues, the original processor should work:
```bash
# Edit Dockerfile to use:
CMD ["python3", "./SLAM3R_engine/slam3r_processor.py"]
```