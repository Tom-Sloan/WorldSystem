# Grounded-SAM-2 with RTSP Streaming Implementation Plan

## Overview
This plan modifies your drone server to stream H.264 video via RTSP to 3 concurrent Grounded-SAM-2 processors for open-vocabulary object detection and tracking.

## Architecture
```
Drone → WebSocket → server/main.py → RTSP Server → 3× Grounded-SAM-2 Processors
                            ↓
                        RabbitMQ (for IMU/telemetry data)
```

## Phase 1: Modify server/main.py

### 1.1 Add Required Imports
```python
# Add these imports at the top of server/main.py
import subprocess
import queue
import threading
import shutil  # For checking if ffmpeg is installed
```

### 1.2 Add Global Variables
```python
# Add near other global variables (after line ~90)
rtsp_process = None
rtsp_queue = queue.Queue(maxsize=100)
rtsp_thread = None
RTSP_PORT = int(os.getenv('RTSP_PORT', 8554))
```

### 1.3 Add RTSP Server Functions
```python
# Add after the global variables section

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    if not shutil.which('ffmpeg'):
        logger.error("FFmpeg not found! Please install FFmpeg in the container.")
        return False
    return True

def start_rtsp_server():
    """Start FFmpeg RTSP server in a separate thread"""
    global rtsp_process, rtsp_thread
    
    if not check_ffmpeg():
        return False
    
    try:
        # Create RTSP server using FFmpeg
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-f', 'h264',  # Input format
            '-i', '-',  # Read from stdin
            '-c:v', 'copy',  # Copy codec (no re-encoding)
            '-f', 'rtsp',  # Output format
            '-rtsp_transport', 'tcp',  # Use TCP for reliability
            f'rtsp://0.0.0.0:{RTSP_PORT}/drone'  # Bind to all interfaces
        ]
        
        rtsp_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        
        logger.info(f"RTSP server started on rtsp://0.0.0.0:{RTSP_PORT}/drone")
        
        # Process queue in a separate thread
        def process_rtsp_queue():
            while True:
                try:
                    chunk = rtsp_queue.get(timeout=1)
                    if chunk is None:  # Shutdown signal
                        break
                    if rtsp_process and rtsp_process.stdin and rtsp_process.poll() is None:
                        rtsp_process.stdin.write(chunk)
                        rtsp_process.stdin.flush()
                except queue.Empty:
                    continue
                except BrokenPipeError:
                    logger.error("RTSP server pipe broken, restarting...")
                    start_rtsp_server()
                    break
                except Exception as e:
                    logger.error(f"Error writing to RTSP server: {e}")
        
        rtsp_thread = threading.Thread(target=process_rtsp_queue, daemon=True)
        rtsp_thread.start()
        
        # Monitor FFmpeg stderr in another thread
        def monitor_ffmpeg():
            while rtsp_process and rtsp_process.poll() is None:
                line = rtsp_process.stderr.readline()
                if line:
                    logger.debug(f"FFmpeg: {line.decode().strip()}")
        
        monitor_thread = threading.Thread(target=monitor_ffmpeg, daemon=True)
        monitor_thread.start()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to start RTSP server: {e}")
        return False

def stop_rtsp_server():
    """Stop the RTSP server gracefully"""
    global rtsp_process, rtsp_thread
    
    logger.info("Stopping RTSP server...")
    
    # Signal thread to stop
    if rtsp_queue:
        rtsp_queue.put(None)
    
    # Wait for thread
    if rtsp_thread and rtsp_thread.is_alive():
        rtsp_thread.join(timeout=5)
    
    # Terminate FFmpeg
    if rtsp_process:
        rtsp_process.terminate()
        try:
            rtsp_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            rtsp_process.kill()
```

### 1.4 Modify Lifespan Context Manager
```python
# Modify the existing lifespan function (~line 130)
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server startup...")

    # NTP sync
    logger.info("[NTP] Initializing time synchronization...")
    sync_ntp_time()

    # RabbitMQ & exchanges
    await setup_amqp()
    
    # Start RTSP server
    if not start_rtsp_server():
        logger.warning("RTSP server failed to start, video streaming will not work")

    # background tasks we need to cancel on shutdown
    bg_tasks = [
        asyncio.create_task(log_imu_rate()),
        asyncio.create_task(consume_processed_frames_async()),
        asyncio.create_task(consume_restart_messages())
    ]
    
    yield
    
    # Shutdown
    logger.info("Server shutdown...")
    
    # Stop RTSP server
    stop_rtsp_server()
    
    # Cancel background tasks
    for task in bg_tasks:
        task.cancel()
    
    try:
        await asyncio.gather(*bg_tasks, return_exceptions=True)
    except Exception as e:
        logger.error(f"Error during task cleanup: {e}")
    
    # Close AMQP
    if amqp_channel:
        await amqp_channel.close()
    if amqp_connection:
        await amqp_connection.close()
```

### 1.5 Modify Video WebSocket Handler
```python
# Replace the existing websocket_video_endpoint function (~line 580)
@app.websocket("/ws/video")
async def websocket_video_endpoint(websocket: WebSocket):
    """Handle H.264 video streaming from phones/drones"""
    await websocket.accept()
    websocket_id = str(id(websocket))
    connected_phones.add(websocket)
    logger.info(f"New video stream connected: {websocket.client.host} (ID: {websocket_id})")
    
    # Track statistics
    bytes_received = 0
    chunks_sent_to_rtsp = 0
    
    try:
        while True:
            data = await websocket.receive()
            if data["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()
                
            # Handle binary messages (H.264 video streams)
            if "bytes" in data:
                frame_bytes = data["bytes"]
                bytes_received += len(frame_bytes)
                
                # Send to RTSP server (non-blocking)
                try:
                    rtsp_queue.put_nowait(frame_bytes)
                    chunks_sent_to_rtsp += 1
                    
                    # Log progress every 100 chunks
                    if chunks_sent_to_rtsp % 100 == 0:
                        logger.info(f"[{websocket_id}] Sent {chunks_sent_to_rtsp} chunks to RTSP, {bytes_received/1024/1024:.2f} MB total")
                        
                except queue.Full:
                    logger.warning("RTSP queue full, dropping frame")
                
                # Optional: Still publish to RabbitMQ for recording/analysis
                # await publish_video_stream_chunk(frame_bytes, websocket_id)
                    
            # Handle text messages (control/config)
            elif "text" in data:
                try:
                    msg = json.loads(data["text"])
                    if msg.get("type") == "video_config":
                        logger.info(f"Video config received: {msg}")
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from video stream: {data['text']}")
                    
    except WebSocketDisconnect:
        logger.info(f"Video stream disconnected: {websocket.client.host}")
        logger.info(f"[{websocket_id}] Total: {bytes_received/1024/1024:.2f} MB received, {chunks_sent_to_rtsp} chunks sent to RTSP")
    except Exception as e:
        logger.error(f"Error in video websocket: {e}")
    finally:
        connected_phones.discard(websocket)
```

### 1.6 Add RTSP Status Endpoint
```python
# Add after the H.264 test endpoint (~line 570)
@app.get("/rtsp/status")
async def rtsp_status():
    """Check RTSP server status"""
    return {
        "rtsp_running": rtsp_process is not None and rtsp_process.poll() is None,
        "rtsp_url": f"rtsp://localhost:{RTSP_PORT}/drone",
        "queue_size": rtsp_queue.qsize() if rtsp_queue else 0,
        "thread_alive": rtsp_thread.is_alive() if rtsp_thread else False
    }
```

## Phase 2: Create Grounded-SAM-2 Frame Processor

### 2.1 Create frame_processor/grounded_sam2_processor.py
```python
#!/usr/bin/env python3
"""
Grounded-SAM-2 Frame Processor
Consumes RTSP stream and performs open-vocabulary object detection and tracking
"""

import os
import cv2
import torch
import numpy as np
import time
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

# Grounded-SAM-2 imports
import sys
sys.path.append('./Grounded-SAM-2')  # Adjust path as needed

from sam2.build_sam import build_sam2_camera_predictor
from grounding_dino.util.inference import load_model, load_image, predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroundedSAM2Processor:
    def __init__(self, processor_id: int, rtsp_url: str):
        self.processor_id = processor_id
        self.rtsp_url = rtsp_url
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.setup_models()
        
        # Tracking state
        self.if_init = False
        self.object_tracks = defaultdict(list)  # obj_id -> [frames]
        self.frame_count = 0
        
        # Connect to stream
        self.cap = None
        self.connect_to_stream()
    
    def setup_models(self):
        """Initialize Grounded-SAM-2 models"""
        logger.info(f"[Processor {self.processor_id}] Setting up models on {self.device}")
        
        # SAM2 setup
        sam2_checkpoint = "./checkpoints/sam2_hiera_small.pt"
        model_cfg = "sam2_hiera_s.yaml"
        self.sam2_predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
        
        # Grounding DINO setup
        grounding_dino_config = "./grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        grounding_dino_checkpoint = "./gdino_checkpoints/groundingdino_swint_ogc.pth"
        self.grounding_model = load_model(grounding_dino_config, grounding_dino_checkpoint)
        
        logger.info(f"[Processor {self.processor_id}] Models loaded successfully")
    
    def connect_to_stream(self) -> bool:
        """Connect to RTSP stream"""
        logger.info(f"[Processor {self.processor_id}] Connecting to {self.rtsp_url}")
        
        # OpenCV VideoCapture with optimized settings
        self.cap = cv2.VideoCapture(self.rtsp_url)
        
        # Set buffer size to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Check if opened
        if not self.cap.isOpened():
            logger.error(f"[Processor {self.processor_id}] Failed to connect to RTSP stream")
            return False
        
        # Get stream properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"[Processor {self.processor_id}] Connected! Stream: {width}x{height} @ {fps} FPS")
        return True
    
    def detect_objects(self, image: np.ndarray, text_prompt: str, 
                      box_threshold: float = 0.25, text_threshold: float = 0.2):
        """Detect objects using Grounding DINO"""
        # Convert to PIL Image for Grounding DINO
        from PIL import Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Run detection
        boxes, logits, phrases = predict(
            model=self.grounding_model,
            image=pil_image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        # Convert to numpy for SAM2
        h, w = image.shape[:2]
        boxes_scaled = boxes * torch.tensor([w, h, w, h])
        boxes_xyxy = boxes_scaled.cpu().numpy()
        
        return boxes_xyxy
    
    def process_stream(self):
        """Main processing loop"""
        logger.info(f"[Processor {self.processor_id}] Starting processing loop")
        
        # Text prompt for open-vocabulary detection
        text_prompt = "all objects. item. thing. stuff."
        
        while True:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning(f"[Processor {self.processor_id}] Failed to read frame, reconnecting...")
                    self.cap.release()
                    time.sleep(1)
                    if not self.connect_to_stream():
                        time.sleep(5)
                    continue
                
                self.frame_count += 1
                
                # First frame initialization
                if not self.if_init:
                    logger.info(f"[Processor {self.processor_id}] Initializing SAM2 with first frame")
                    self.sam2_predictor.load_first_frame(frame)
                    self.if_init = True
                    
                    # Detect objects with Grounding DINO
                    boxes = self.detect_objects(frame, text_prompt)
                    logger.info(f"[Processor {self.processor_id}] Detected {len(boxes)} objects")
                    
                    if len(boxes) > 0:
                        # Initialize tracking with detected boxes
                        _, out_obj_ids, out_mask_logits = self.sam2_predictor.add_new_prompt(
                            frame_idx=0,
                            obj_id=0,
                            bbox=boxes
                        )
                else:
                    # Track objects in subsequent frames
                    out_obj_ids, out_mask_logits = self.sam2_predictor.track(frame)
                
                # Process tracked objects
                if out_obj_ids is not None and len(out_obj_ids) > 0:
                    masks = (out_mask_logits > 0.0).cpu().numpy()
                    
                    for obj_id, mask in zip(out_obj_ids, masks):
                        # Crop object with padding
                        cropped = self.crop_object_with_padding(frame, mask, padding=0.2)
                        
                        if cropped is not None:
                            # Store frames for each object
                            self.object_tracks[obj_id].append({
                                'frame': cropped,
                                'frame_num': self.frame_count,
                                'timestamp': time.time()
                            })
                            
                            # Create enhanced image after collecting enough frames
                            if len(self.object_tracks[obj_id]) >= 30:  # ~1 second at 30fps
                                self.create_enhanced_image(obj_id)
                                # Keep only recent frames
                                self.object_tracks[obj_id] = self.object_tracks[obj_id][-30:]
                
                # Log progress
                if self.frame_count % 100 == 0:
                    logger.info(f"[Processor {self.processor_id}] Processed {self.frame_count} frames, tracking {len(self.object_tracks)} objects")
                
                # Optional: Display or save results
                # self.visualize_results(frame, masks, out_obj_ids)
                
            except Exception as e:
                logger.error(f"[Processor {self.processor_id}] Error processing frame: {e}")
                import traceback
                traceback.print_exc()
    
    def crop_object_with_padding(self, frame: np.ndarray, mask: np.ndarray, 
                                padding: float = 0.2) -> np.ndarray:
        """Crop object from frame with padding"""
        if mask.sum() == 0:  # Empty mask
            return None
        
        # Find bounding box of mask
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None
        
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        
        # Add padding
        width = x_max - x_min
        height = y_max - y_min
        
        pad_x = int(width * padding / 2)
        pad_y = int(height * padding / 2)
        
        x_min = max(0, x_min - pad_x)
        x_max = min(frame.shape[1], x_max + pad_x)
        y_min = max(0, y_min - pad_y)
        y_max = min(frame.shape[0], y_max + pad_y)
        
        return frame[y_min:y_max, x_min:x_max].copy()
    
    def create_enhanced_image(self, obj_id: int):
        """Create enhanced image from multiple views"""
        frames_data = self.object_tracks[obj_id]
        frames = [fd['frame'] for fd in frames_data]
        
        logger.info(f"[Processor {self.processor_id}] Creating enhanced image for object {obj_id} from {len(frames)} frames")
        
        # TODO: Implement your multi-frame enhancement algorithm here
        # For now, just save the middle frame as an example
        if frames:
            middle_idx = len(frames) // 2
            enhanced = frames[middle_idx]
            
            # Save enhanced image
            output_dir = f"./enhanced_objects/processor_{self.processor_id}"
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = int(time.time())
            filename = f"{output_dir}/object_{obj_id}_{timestamp}.jpg"
            cv2.imwrite(filename, enhanced)
            logger.info(f"[Processor {self.processor_id}] Saved enhanced image: {filename}")
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        logger.info(f"[Processor {self.processor_id}] Cleaned up resources")

def main():
    """Main entry point"""
    # Get processor ID from environment
    processor_id = int(os.getenv('PROCESSOR_ID', 1))
    
    # RTSP URL - use host.docker.internal for Docker on Mac/Windows
    # or use service name for docker-compose
    rtsp_host = os.getenv('RTSP_HOST', 'server')
    rtsp_port = os.getenv('RTSP_PORT', 8554)
    rtsp_url = f"rtsp://{rtsp_host}:{rtsp_port}/drone"
    
    # Create processor
    processor = GroundedSAM2Processor(processor_id, rtsp_url)
    
    try:
        # Run processing loop
        processor.process_stream()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        processor.cleanup()

if __name__ == "__main__":
    main()
```

### 2.2 Create frame_processor/Dockerfile
```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone Grounded-SAM-2
RUN git clone https://github.com/IDEA-Research/Grounded-SAM-2.git

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Grounded-SAM-2 dependencies
WORKDIR /app/Grounded-SAM-2
RUN pip install -e .
RUN pip install --no-build-isolation -e grounding_dino

# Download model checkpoints
RUN cd checkpoints && bash download_ckpts.sh
RUN cd gdino_checkpoints && bash download_ckpts.sh

# Copy processor code
WORKDIR /app
COPY grounded_sam2_processor.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Run processor
CMD ["python", "grounded_sam2_processor.py"]
```

### 2.3 Create frame_processor/requirements.txt
```txt
opencv-python==4.8.1.78
numpy==1.24.3
torch==2.0.1
torchvision==0.15.2
Pillow==10.1.0
matplotlib==3.7.2
supervision==0.16.0
transformers==4.35.2
```

## Phase 3: Docker Compose Configuration

### 3.1 Update docker-compose.yml
```yaml
version: '3.8'

services:
  rabbitmq:
    image: rabbitmq:3-management
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      - RABBITMQ_DEFAULT_USER=admin
      - RABBITMQ_DEFAULT_PASS=admin
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq

  server:
    build: 
      context: ./server
      dockerfile: Dockerfile
    ports:
      - "5001:5001"  # API/WebSocket
      - "8554:8554"  # RTSP
    environment:
      - RABBITMQ_URL=amqp://admin:admin@rabbitmq:5672/
      - RTSP_PORT=8554
      - BIND_HOST=0.0.0.0
      - API_PORT=5001
      - PYTHONUNBUFFERED=1
    depends_on:
      - rabbitmq
    volumes:
      - ./server:/app
      - ./models:/app/models
    command: ["python", "main.py"]

  frame_processor_1:
    build: 
      context: ./frame_processor
      dockerfile: Dockerfile
    environment:
      - PROCESSOR_ID=1
      - RTSP_HOST=server
      - RTSP_PORT=8554
      - PYTHONUNBUFFERED=1
    depends_on:
      - server
    volumes:
      - ./enhanced_objects:/app/enhanced_objects
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  frame_processor_2:
    build: 
      context: ./frame_processor
      dockerfile: Dockerfile
    environment:
      - PROCESSOR_ID=2
      - RTSP_HOST=server
      - RTSP_PORT=8554
      - PYTHONUNBUFFERED=1
    depends_on:
      - server
    volumes:
      - ./enhanced_objects:/app/enhanced_objects
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  frame_processor_3:
    build: 
      context: ./frame_processor
      dockerfile: Dockerfile
    environment:
      - PROCESSOR_ID=3
      - RTSP_HOST=server
      - RTSP_PORT=8554
      - PYTHONUNBUFFERED=1
    depends_on:
      - server
    volumes:
      - ./enhanced_objects:/app/enhanced_objects
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  rabbitmq_data:
```

### 3.2 Update server/Dockerfile
```dockerfile
# Add FFmpeg installation to your existing server Dockerfile
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
```

## Phase 4: Testing & Verification

### 4.1 Test RTSP Server
```bash
# Check if RTSP server is running
curl http://localhost:5001/rtsp/status

# Test RTSP stream with ffplay
ffplay rtsp://localhost:8554/drone

# Or with VLC
vlc rtsp://localhost:8554/drone
```

### 4.2 Monitor Logs
```bash
# View server logs
docker-compose logs -f server

# View processor logs
docker-compose logs -f frame_processor_1
```

### 4.3 Debug Connection Issues
If processors can't connect to RTSP:
1. Check if server started RTSP successfully
2. Verify port 8554 is exposed
3. Try using IP address instead of hostname
4. Check firewall rules

## Key Resources

### Grounded-SAM-2 Camera Example
The official implementation we're adapting from:
- **Repository**: https://github.com/IDEA-Research/Grounded-SAM-2
- **Camera Script**: https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/grounded_sam2_tracking_camera_with_continuous_id.py
- **Documentation**: https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/README.md

### Important Notes
1. **GPU Memory**: Each processor needs ~4GB VRAM (using sam2_hiera_small)
2. **Latency**: Expect 20-50ms processing latency per frame
3. **Object Persistence**: Objects maintain IDs across frames automatically
4. **Text Prompts**: Modify `text_prompt` for specific object types

## Troubleshooting

### RTSP Connection Issues
```python
# Add to frame_processor for debugging
import subprocess

def test_rtsp_connection(url):
    """Test if RTSP URL is accessible"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 
             'stream=width,height', '-of', 'default=noprint_wrappers=1:nokey=1', url],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except:
        return False
```

### Memory Issues
If running out of GPU memory:
1. Use smaller SAM2 model (sam2_hiera_tiny.pt)
2. Reduce number of concurrent processors
3. Add frame skipping (process every Nth frame)

### Performance Optimization
```python
# Add frame skipping for better performance
if self.frame_count % 2 != 0:  # Process every other frame
    continue
```

## Next Steps

1. **Implement Enhanced Image Algorithm**: Add your multi-frame fusion algorithm in `create_enhanced_image()`
2. **Add Object Classification**: Integrate CLIP for object type classification
3. **Optimize Prompts**: Fine-tune text prompts for your specific use case
4. **Add Metrics**: Integrate Prometheus for monitoring
5. **Scale Horizontally**: Add more processors as needed