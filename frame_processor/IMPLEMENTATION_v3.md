# Grounded-SAM-2 with WebSocket Streaming Implementation Plan

## Overview
This plan simplifies the video streaming architecture by using WebSocket connections directly between the server and consumers, eliminating the need for FFmpeg, MediaMTX, and RTSP protocol complexity.

## Architecture
```
Drone/Simulator → WebSocket → server/main.py → WebSocket ├─→ frame_processor (Grounded-SAM-2)
                     ↓              ↓           (H.264)   ├─→ slam3r (SLAM processing)
                (custom headers)  (strip headers)         └─→ storage (Recording/archival)
                     ↓              ↓
                  /ws/video      /ws/h264_stream

Separate data path:
Phone → WebSocket → /ws/phone → RabbitMQ (IMU/telemetry data)
```

Key points:
- Server receives custom header packets from Android/Simulator
- Server strips headers and relays pure H.264 to consumers
- Each consumer connects via WebSocket and receives raw H.264 stream
- OpenCV can directly consume WebSocket H.264 streams

## Phase 1: Modify server/main.py

### 1.1 Add Required Imports
```python
# Add these imports at the top of server/main.py
import asyncio
from typing import Set, Dict
import weakref
```

### 1.2 Add Global Variables for WebSocket Broadcasting
```python
# Add near other global variables (after line ~90)
# Store WebSocket connections for H.264 consumers
h264_consumers: Set[weakref.ref] = set()
consumer_queues: Dict[int, asyncio.Queue] = {}
```

### 1.3 Add Prometheus Metrics
```python
# Add after imports section
from prometheus_client import Counter, Gauge, Histogram, generate_latest

# Prometheus metrics
h264_consumers_count = Gauge('h264_consumers_connected', 'Number of connected H.264 consumers')
h264_frames_relayed = Counter('h264_frames_relayed_total', 'Total H.264 frames relayed to consumers')
h264_bytes_relayed = Counter('h264_bytes_relayed_total', 'Total H.264 bytes relayed to consumers')
video_bandwidth_metric = Gauge('video_bandwidth_mbps', 'Current video bandwidth in Mbps')
consumer_queue_size = Gauge('consumer_queue_size', 'Average consumer queue size')
```

### 1.4 Add WebSocket Broadcasting Functions
```python
# Add after the global variables section

async def broadcast_h264_to_consumers(h264_data: bytes):
    """Broadcast H.264 data to all connected consumers"""
    if not h264_data:
        return
    
    # Update metrics
    h264_frames_relayed.inc()
    h264_bytes_relayed.inc(len(h264_data))
    
    # Send to all consumer queues
    dead_queues = []
    for queue_id, queue in consumer_queues.items():
        try:
            # Non-blocking put with small queue to prevent memory issues
            queue.put_nowait(h264_data)
        except asyncio.QueueFull:
            logger.warning(f"Consumer {queue_id} queue full, dropping frame")
            dead_queues.append(queue_id)
    
    # Clean up dead queues
    for queue_id in dead_queues:
        consumer_queues.pop(queue_id, None)
    
    # Update consumer count metric
    h264_consumers_count.set(len(consumer_queues))
    
    # Update average queue size
    if consumer_queues:
        avg_size = sum(q.qsize() for q in consumer_queues.values()) / len(consumer_queues)
        consumer_queue_size.set(avg_size)

def start_rtsp_server():
    """Start FFmpeg to push H.264 stream to MediaMTX RTSP server"""
    global rtsp_process, rtsp_thread
    
    if not check_ffmpeg():
        return False
    
    try:
        # FFmpeg command to push to MediaMTX
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-f', 'h264',  # Input format
            '-i', '-',  # Read from stdin
            '-c:v', 'copy',  # Copy codec (no re-encoding)
            '-f', 'rtsp',  # Output format
            '-rtsp_transport', 'tcp',  # Use TCP for reliability
            '-fflags', 'nobuffer',  # Disable buffering
            '-flags', 'low_delay',  # Low latency mode
            '-strict', 'experimental',
            f'rtsp://127.0.0.1:{RTSP_PORT}/drone'  # Push to MediaMTX on localhost
        ]
        
        rtsp_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0  # Unbuffered for low latency
        )
        
        logger.info(f"FFmpeg started, pushing to MediaMTX at rtsp://127.0.0.1:{RTSP_PORT}/drone")
        rtsp_restarts_metric.inc()
        
        # Process queue in a separate thread
        def process_rtsp_queue():
            consecutive_errors = 0
            max_errors = 5
            
            while True:
                try:
                    chunk = rtsp_queue.get(timeout=1)
                    if chunk is None:  # Shutdown signal
                        break
                    if rtsp_process and rtsp_process.stdin and rtsp_process.poll() is None:
                        rtsp_process.stdin.write(chunk)
                        rtsp_process.stdin.flush()
                        consecutive_errors = 0  # Reset on success
                except queue.Empty:
                    continue
                except BrokenPipeError:
                    consecutive_errors += 1
                    logger.error(f"FFmpeg pipe broken (error {consecutive_errors}/{max_errors})")
                    if consecutive_errors >= max_errors:
                        logger.error("Max errors reached, restarting RTSP server...")
                        start_rtsp_server()
                        break
                except Exception as e:
                    logger.error(f"Error writing to FFmpeg: {e}")
                    consecutive_errors += 1
        
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
        logger.error(f"Failed to start FFmpeg: {e}")
        return False

def stop_rtsp_server():
    """Stop FFmpeg gracefully"""
    global rtsp_process, rtsp_thread
    
    logger.info("Stopping FFmpeg...")
    
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

# Health check endpoint for FFmpeg/RTSP
@app.get("/health/rtsp")
async def rtsp_health_check():
    """Health check endpoint for monitoring systems"""
    ffmpeg_healthy = rtsp_process is not None and rtsp_process.poll() is None
    queue_size = rtsp_queue.qsize() if rtsp_queue else 0
    
    # Update Prometheus metrics
    rtsp_health_metric.set(1 if ffmpeg_healthy else 0)
    rtsp_queue_size_metric.set(queue_size)
    
    return {
        "ffmpeg_healthy": ffmpeg_healthy,
        "queue_size": queue_size,
        "queue_capacity": 100,
        "queue_usage_percent": (queue_size / 100) * 100
    }
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
    
    # Start FFmpeg to push to MediaMTX
    if not start_rtsp_server():
        logger.warning("FFmpeg failed to start, video streaming will not work")

    # background tasks we need to cancel on shutdown
    bg_tasks = [
        asyncio.create_task(log_imu_rate()),
        asyncio.create_task(consume_processed_frames_async()),
        asyncio.create_task(consume_restart_messages())
    ]
    
    yield
    
    # Shutdown
    logger.info("Server shutdown...")
    
    # Stop FFmpeg
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
    
    # Network quality monitoring
    bandwidth_samples = []
    sample_window = 5.0  # seconds
    last_sample_time = time.time()
    bytes_in_window = 0
    
    # Track SPS/PPS for RTSP initialization
    sps_data = None
    pps_data = None
    
    try:
        while True:
            data = await websocket.receive()
            if data["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()
                
            # Handle binary messages (H.264 video streams)
            if "bytes" in data:
                packet = data["bytes"]
                packet_size = len(packet)
                bytes_received += packet_size
                bytes_in_window += packet_size
                
                # Parse Android packet header
                if packet_size >= 5:
                    packet_type = packet[4]
                    
                    # Extract H.264 NAL units based on packet type
                    if packet_type == 0x01:  # PACKET_TYPE_SPS
                        sps_data = packet[13:]  # Skip header
                        logger.info(f"Received SPS: {len(sps_data)} bytes")
                        
                    elif packet_type == 0x02:  # PACKET_TYPE_PPS
                        pps_data = packet[13:]  # Skip header
                        logger.info(f"Received PPS: {len(pps_data)} bytes")
                        
                    elif packet_type in [0x03, 0x04]:  # KEYFRAME or FRAME
                        # Extract H.264 data (skip 21-byte header)
                        h264_data = packet[21:]
                        
                        # For RTSP, ensure SPS/PPS are sent with keyframes
                        if packet_type == 0x03 and sps_data and pps_data:
                            # Send SPS+PPS before keyframe
                            try:
                                rtsp_queue.put_nowait(sps_data)
                                rtsp_queue.put_nowait(pps_data)
                            except queue.Full:
                                logger.warning("RTSP queue full, dropping SPS/PPS")
                        
                        # Send H.264 NAL units to RTSP
                        try:
                            rtsp_queue.put_nowait(h264_data)
                            chunks_sent_to_rtsp += 1
                        except queue.Full:
                            logger.warning("RTSP queue full, dropping frame")
                            frames_dropped_metric.inc()
                
                # Calculate bandwidth every 5 seconds
                current_time = time.time()
                if current_time - last_sample_time >= sample_window:
                    bandwidth_mbps = (bytes_in_window * 8) / (sample_window * 1_000_000)
                    bandwidth_samples.append(bandwidth_mbps)
                    
                    # Log bandwidth and update metrics
                    avg_bandwidth = sum(bandwidth_samples[-10:]) / min(len(bandwidth_samples), 10)
                    logger.info(f"[{websocket_id}] Video bandwidth: {bandwidth_mbps:.2f} Mbps (avg: {avg_bandwidth:.2f} Mbps)")
                    video_bandwidth_metric.set(bandwidth_mbps)
                    
                    # Alert if bandwidth is consistently low
                    if avg_bandwidth < 3.0 and len(bandwidth_samples) > 5:
                        logger.warning(f"[{websocket_id}] Low bandwidth detected ({avg_bandwidth:.2f} Mbps), consider reducing bitrate")
                    
                    # Reset window
                    bytes_in_window = 0
                    last_sample_time = current_time
                    
                # Log progress every 100 chunks
                if chunks_sent_to_rtsp % 100 == 0:
                    logger.info(f"[{websocket_id}] Sent {chunks_sent_to_rtsp} chunks to RTSP, {bytes_received/1024/1024:.2f} MB total")
                    
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
        logger.info(f"[{websocket_id}] Final stats: {bytes_received/1024/1024:.2f} MB received, {chunks_sent_to_rtsp} chunks sent")
        if bandwidth_samples:
            final_avg = sum(bandwidth_samples) / len(bandwidth_samples)
            logger.info(f"[{websocket_id}] Average bandwidth: {final_avg:.2f} Mbps")
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
    """Check FFmpeg and RTSP status"""
    return {
        "ffmpeg_running": rtsp_process is not None and rtsp_process.poll() is None,
        "rtsp_url": f"rtsp://localhost:{RTSP_PORT}/drone",
        "queue_size": rtsp_queue.qsize() if rtsp_queue else 0,
        "thread_alive": rtsp_thread.is_alive() if rtsp_thread else False
    }

# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics"""
    return Response(generate_latest(), media_type="text/plain")
```

## Phase 2: Create Base RTSP Consumer Class

### 2.1 Create common/rtsp_consumer.py
```python
#!/usr/bin/env python3
"""
Base RTSP consumer class for all services
"""
import cv2
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)

class RTSPConsumer(ABC):
    """Base class for services that consume RTSP streams with configurable frame skipping"""
    
    def __init__(self, rtsp_url: str, service_name: str, frame_skip: int = 0):
        self.rtsp_url = rtsp_url
        self.service_name = service_name
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.reconnect_delay = 2
        self.frame_count = 0
        self.frame_skip = frame_skip  # Process every Nth frame (0 = no skip)
        self.frames_processed = 0
        
    def connect(self) -> bool:
        """Connect to RTSP stream"""
        try:
            logger.info(f"[{self.service_name}] Connecting to {self.rtsp_url}")
            
            self.cap = cv2.VideoCapture(self.rtsp_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            
            # Test connection
            ret, _ = self.cap.read()
            if ret:
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(f"[{self.service_name}] Connected! {width}x{height} @ {fps} FPS")
                return True
            
            self.cap.release()
            return False
            
        except Exception as e:
            logger.error(f"[{self.service_name}] Connection failed: {e}")
            return False
    
    def run(self):
        """Main processing loop"""
        self.is_running = True
        
        while self.is_running:
            if not self.cap or not self.cap.isOpened():
                if not self.connect():
                    time.sleep(self.reconnect_delay)
                    self.reconnect_delay = min(self.reconnect_delay * 2, 30)
                    continue
                self.reconnect_delay = 2
            
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning(f"[{self.service_name}] Failed to read frame")
                    self.cap.release()
                    self.cap = None
                    continue
                
                self.frame_count += 1
                
                # Apply frame skipping
                if self.frame_skip > 0 and self.frame_count % self.frame_skip != 0:
                    continue
                
                # Process frame
                self.process_frame(frame, self.frame_count)
                self.frames_processed += 1
                
                # Log progress
                if self.frames_processed % 100 == 0:
                    skip_ratio = f" (skipping {self.frame_skip-1} of {self.frame_skip})" if self.frame_skip > 1 else ""
                    logger.info(f"[{self.service_name}] Processed {self.frames_processed} frames{skip_ratio}")
                    
            except Exception as e:
                logger.error(f"[{self.service_name}] Error processing frame: {e}")
    
    @abstractmethod
    def process_frame(self, frame, frame_number: int):
        """Process a single frame - implement in subclass"""
        pass
    
    def stop(self):
        """Stop processing"""
        self.is_running = False
        if self.cap:
            self.cap.release()
```

## Phase 3: Update Frame Processor with Grounded-SAM-2

### 3.1 Create frame_processor/grounded_sam2_processor.py
```python
#!/usr/bin/env python3
"""
Grounded-SAM-2 Frame Processor with Rerun and API Integration
Consumes RTSP stream and performs open-vocabulary object detection and tracking
Maintains all existing Rerun visualization and API functionality
"""

import os
import cv2
import torch
import numpy as np
import time
import logging
import asyncio
import aio_pika
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Add common to path
sys.path.append('/app/common')
from rtsp_consumer import RTSPConsumer

# Grounded-SAM-2 imports
import sys
sys.path.append('./Grounded-SAM-2')
from sam2.build_sam import build_sam2_camera_predictor
from grounding_dino.util.inference import load_model, load_image, predict

# Your existing imports for Rerun and API
import rerun as rr
from external.api_client import APIClient
from pipeline.enhancer import ImageEnhancer
from visualization.rerun_client import RerunVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroundedSAM2Processor(RTSPConsumer):
    def __init__(self, rtsp_url: str):
        # Frame processor: process every frame for best tracking
        frame_skip = int(os.getenv('FRAME_PROCESSOR_SKIP', '1'))
        super().__init__(rtsp_url, "FrameProcessor", frame_skip)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.setup_models()
        
        # Initialize Rerun
        self.setup_rerun()
        
        # Initialize API clients and enhancement
        self.api_client = APIClient()
        self.image_enhancer = ImageEnhancer()
        self.rerun_viz = RerunVisualizer()
        
        # Tracking state
        self.if_init = False
        self.object_tracks = defaultdict(list)  # obj_id -> [frames]
        
        # RabbitMQ setup
        self.setup_rabbitmq()
    
    def setup_rerun(self):
        """Initialize Rerun connection"""
        try:
            rr.init("frame_processor", spawn=False)
            rr.connect("localhost:9876")
            logger.info("Connected to Rerun viewer")
        except Exception as e:
            logger.warning(f"Could not connect to Rerun: {e}")
    
    def setup_rabbitmq(self):
        """Setup RabbitMQ connections"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        self.rabbit_connection = loop.run_until_complete(
            aio_pika.connect_robust(os.getenv('RABBITMQ_URL'))
        )
        self.rabbit_channel = loop.run_until_complete(
            self.rabbit_connection.channel()
        )
        
        # Declare exchanges
        self.processed_exchange = loop.run_until_complete(
            self.rabbit_channel.declare_exchange(
                'processed_frames_exchange',
                aio_pika.ExchangeType.FANOUT
            )
        )
        self.api_results_exchange = loop.run_until_complete(
            self.rabbit_channel.declare_exchange(
                'api_results_exchange',
                aio_pika.ExchangeType.FANOUT
            )
        )
    
    def setup_models(self):
        """Initialize Grounded-SAM-2 models"""
        logger.info(f"Setting up models on {self.device}")
        
        # SAM2 setup
        sam2_checkpoint = "./checkpoints/sam2_hiera_small.pt"
        model_cfg = "sam2_hiera_s.yaml"
        self.sam2_predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
        
        # Grounding DINO setup
        grounding_dino_config = "./grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        grounding_dino_checkpoint = "./gdino_checkpoints/groundingdino_swint_ogc.pth"
        self.grounding_model = load_model(grounding_dino_config, grounding_dino_checkpoint)
        
        logger.info("Models loaded successfully")
    
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
    
    def process_frame(self, frame: np.ndarray, frame_number: int):
        """Process frame through Grounded-SAM-2 pipeline"""
        
        # Text prompt for open-vocabulary detection - detect everything
        text_prompt = "all objects. item. thing. stuff."
        
        try:
            # First frame initialization
            if not self.if_init:
                logger.info("Initializing SAM2 with first frame")
                self.sam2_predictor.load_first_frame(frame)
                self.if_init = True
                
                # Detect objects with Grounding DINO
                boxes = self.detect_objects(frame, text_prompt)
                logger.info(f"Detected {len(boxes)} objects")
                
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
            
            # Visualize in Rerun
            self.visualize_frame_in_rerun(frame, out_mask_logits, out_obj_ids, frame_number)
            
            # Process tracked objects
            if out_obj_ids is not None and len(out_obj_ids) > 0:
                masks = (out_mask_logits > 0.0).cpu().numpy()
                
                # Publish tracking results to RabbitMQ
                self.publish_tracking_results(out_obj_ids, masks, frame_number)
                
                for obj_id, mask in zip(out_obj_ids, masks):
                    # Crop object with padding
                    cropped = self.crop_object_with_padding(frame, mask, padding=0.2)
                    
                    if cropped is not None:
                        # Store frames for each object
                        self.object_tracks[obj_id].append({
                            'frame': cropped,
                            'frame_num': frame_number,
                            'timestamp': time.time(),
                            'mask': mask
                        })
                        
                        # Process with enhancement and API after collecting enough frames
                        if len(self.object_tracks[obj_id]) >= 30:  # ~1 second at 30fps
                            asyncio.create_task(self.process_object_with_api(obj_id))
                            # Keep only recent frames
                            self.object_tracks[obj_id] = self.object_tracks[obj_id][-30:]
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
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
    
    def visualize_frame_in_rerun(self, frame: np.ndarray, mask_logits, obj_ids, frame_number: int):
        """Visualize frame and detections in Rerun"""
        try:
            # Log the frame
            rr.log("frame", rr.Image(frame))
            
            # Log masks if available
            if mask_logits is not None:
                masks = (mask_logits > 0.0).cpu().numpy()
                for i, (obj_id, mask) in enumerate(zip(obj_ids, masks)):
                    rr.log(f"masks/object_{obj_id}", rr.SegmentationImage(mask.astype(np.uint8) * 255))
            
            # Log frame number
            rr.log("frame_number", rr.TextLog(f"Frame {frame_number}"))
            
        except Exception as e:
            logger.debug(f"Rerun visualization error: {e}")
    
    def publish_tracking_results(self, obj_ids, masks, frame_number: int):
        """Publish tracking results to RabbitMQ"""
        message = {
            "frame_number": frame_number,
            "timestamp": time.time(),
            "objects": [
                {
                    "id": int(obj_id),
                    "mask_shape": mask.shape,
                    "area": int(mask.sum())
                }
                for obj_id, mask in zip(obj_ids, masks)
            ]
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        loop.run_until_complete(
            self.processed_exchange.publish(
                aio_pika.Message(body=json.dumps(message).encode()),
                routing_key=''
            )
        )
    
    async def process_object_with_api(self, obj_id: int):
        """Process object with enhancement and API calls"""
        frames_data = self.object_tracks[obj_id]
        frames = [fd['frame'] for fd in frames_data]
        
        logger.info(f"Processing object {obj_id} with {len(frames)} frames")
        
        # Apply enhancement
        enhanced = self.image_enhancer.enhance_from_multiple_frames(frames)
        
        # Call APIs (Google Lens, Perplexity, etc.)
        api_results = await self.api_client.identify_object(enhanced)
        
        # Publish results
        await self.publish_api_results(obj_id, api_results)
        
        # Log to Rerun
        rr.log(f"api_results/object_{obj_id}", rr.TextLog(json.dumps(api_results)))
    
    async def publish_api_results(self, obj_id: int, results: dict):
        """Publish API results to RabbitMQ"""
        message = {
            "object_id": obj_id,
            "timestamp": time.time(),
            "results": results
        }
        
        await self.api_results_exchange.publish(
            aio_pika.Message(body=json.dumps(message).encode()),
            routing_key=''
        )

def main():
    """Main entry point"""
    # RTSP URL
    rtsp_host = os.getenv('RTSP_HOST', 'server')
    rtsp_port = os.getenv('RTSP_PORT', 8554)
    rtsp_url = f"rtsp://{rtsp_host}:{rtsp_port}/drone"
    
    # Create processor
    processor = GroundedSAM2Processor(rtsp_url)
    
    try:
        # Run processing loop
        processor.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        processor.stop()

if __name__ == "__main__":
    main()
```

## Phase 4: Create SLAM3R RTSP Consumer

### 4.1 Create slam3r/rtsp_slam_processor.py
```python
#!/usr/bin/env python3
"""
SLAM3R RTSP Processor
Consumes RTSP stream and performs SLAM processing
"""

import os
import sys
import cv2
import numpy as np
import time
import asyncio
import aio_pika
import json

# Add common to path
sys.path.append('/app/common')
from rtsp_consumer import RTSPConsumer

# Your existing SLAM3R imports
from slam3r_core import SLAM3RCore
from shared_memory_writer import SharedMemoryWriter

class SLAM3RProcessor(RTSPConsumer):
    def __init__(self, rtsp_url: str):
        # SLAM: process every 3rd frame by default (10fps from 30fps)
        frame_skip = int(os.getenv('SLAM_FRAME_SKIP', '3'))
        super().__init__(rtsp_url, "SLAM3R", frame_skip)
        
        # Initialize SLAM3R
        self.slam = SLAM3RCore()
        self.shared_memory = SharedMemoryWriter()
        
        # RabbitMQ setup
        self.rabbitmq_url = os.getenv('RABBITMQ_URL', 'amqp://admin:admin@rabbitmq:5672/')
        self.setup_rabbitmq()
        
    def setup_rabbitmq(self):
        """Setup RabbitMQ connection"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        self.rabbit_connection = loop.run_until_complete(
            aio_pika.connect_robust(self.rabbitmq_url)
        )
        self.rabbit_channel = loop.run_until_complete(
            self.rabbit_connection.channel()
        )
        
        # Declare exchange for SLAM results
        self.slam_exchange = loop.run_until_complete(
            self.rabbit_channel.declare_exchange(
                'slam_results_exchange',
                aio_pika.ExchangeType.FANOUT
            )
        )
    
    def process_frame(self, frame: np.ndarray, frame_number: int):
        """Process frame through SLAM"""
        try:
            # Convert to grayscale for SLAM
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Process through SLAM
            pose, keypoints, is_keyframe = self.slam.process_frame(
                gray_frame, 
                timestamp=time.time()
            )
            
            if pose is not None:
                # Write to shared memory for mesh service
                self.shared_memory.write_pose(frame_number, pose)
                
                # Publish to RabbitMQ
                self.publish_slam_result(frame_number, pose, is_keyframe)
                
                if is_keyframe:
                    # Save keyframe
                    self.save_keyframe(frame, frame_number, pose)
                    
        except Exception as e:
            logger.error(f"SLAM processing error: {e}")
    
    def publish_slam_result(self, frame_number: int, pose: np.ndarray, is_keyframe: bool):
        """Publish SLAM results to RabbitMQ"""
        message = {
            "frame_number": frame_number,
            "timestamp": time.time(),
            "pose": pose.tolist(),
            "is_keyframe": is_keyframe
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        loop.run_until_complete(
            self.slam_exchange.publish(
                aio_pika.Message(body=json.dumps(message).encode()),
                routing_key=''
            )
        )
    
    def save_keyframe(self, frame: np.ndarray, frame_number: int, pose: np.ndarray):
        """Save keyframe to disk"""
        keyframe_dir = "/app/data/keyframes"
        os.makedirs(keyframe_dir, exist_ok=True)
        
        filename = f"{keyframe_dir}/keyframe_{frame_number:06d}.jpg"
        cv2.imwrite(filename, frame)
        
        # Save pose
        pose_filename = f"{keyframe_dir}/pose_{frame_number:06d}.npy"
        np.save(pose_filename, pose)

def main():
    rtsp_host = os.getenv('RTSP_HOST', 'server')
    rtsp_port = os.getenv('RTSP_PORT', 8554)
    rtsp_url = f"rtsp://{rtsp_host}:{rtsp_port}/drone"
    
    processor = SLAM3RProcessor(rtsp_url)
    
    try:
        processor.run()
    except KeyboardInterrupt:
        logger.info("Shutting down SLAM3R...")
    finally:
        processor.stop()

if __name__ == "__main__":
    main()
```

## Phase 5: Create Storage RTSP Consumer

### 5.1 Create storage/rtsp_storage.py
```python
#!/usr/bin/env python3
"""
Storage Service with RTSP
Records video stream and saves frames/metadata
"""

import os
import sys
import cv2
import time
import json
import asyncio
import aio_pika
from datetime import datetime
from pathlib import Path

# Add common to path
sys.path.append('/app/common')
from rtsp_consumer import RTSPConsumer

class StorageProcessor(RTSPConsumer):
    def __init__(self, rtsp_url: str):
        # Storage: record every frame by default (no skipping)
        frame_skip = int(os.getenv('STORAGE_FRAME_SKIP', '1'))
        super().__init__(rtsp_url, "Storage", frame_skip)
        
        # Storage configuration
        self.storage_path = os.getenv('STORAGE_PATH', '/app/recordings')
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_path = Path(self.storage_path) / self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)
        
        # Video writer
        self.video_writer = None
        self.video_path = self.session_path / "video.mp4"
        
        # Frame saving settings
        self.save_every_n_frames = 30  # Save individual frame every second
        
        # Metadata
        self.metadata = {
            "session_id": self.session_id,
            "start_time": time.time(),
            "frames": []
        }
    
    def process_frame(self, frame: np.ndarray, frame_number: int):
        """Store frame and metadata"""
        
        # Initialize video writer on first frame
        if self.video_writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                str(self.video_path), fourcc, 30.0, (w, h)
            )
            logger.info(f"Started recording to {self.video_path}")
        
        # Write frame to video
        self.video_writer.write(frame)
        
        # Save individual frames periodically
        if frame_number % self.save_every_n_frames == 0:
            frame_path = self.session_path / f"frame_{frame_number:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            
            # Add to metadata
            self.metadata["frames"].append({
                "frame_number": frame_number,
                "timestamp": time.time(),
                "path": str(frame_path)
            })
        
        # Save metadata periodically
        if frame_number % 300 == 0:  # Every 10 seconds
            self.save_metadata()
    
    def save_metadata(self):
        """Save session metadata"""
        metadata_path = self.session_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def stop(self):
        """Clean up and finalize storage"""
        super().stop()
        
        if self.video_writer:
            self.video_writer.release()
            logger.info(f"Finished recording to {self.video_path}")
        
        # Final metadata save
        self.metadata["end_time"] = time.time()
        self.metadata["total_frames"] = self.frame_count
        self.save_metadata()
        
        logger.info(f"Storage session complete: {self.session_id}")

def main():
    rtsp_host = os.getenv('RTSP_HOST', 'server')
    rtsp_port = os.getenv('RTSP_PORT', 8554)
    rtsp_url = f"rtsp://{rtsp_host}:{rtsp_port}/drone"
    
    processor = StorageProcessor(rtsp_url)
    
    try:
        processor.run()
    except KeyboardInterrupt:
        logger.info("Shutting down Storage...")
    finally:
        processor.stop()

if __name__ == "__main__":
    main()
```

## Phase 6: Update Dockerfiles

### 6.1 Create frame_processor/Dockerfile
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

### 6.2 Create slam3r/Dockerfile
```dockerfile
FROM ubuntu:22.04

# Install dependencies including OpenCV
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    ffmpeg \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy common utilities
COPY common/ /app/common/

# Copy SLAM3R specific code
COPY slam3r/ /app/slam3r/

# Install Python dependencies
RUN pip3 install opencv-python aio-pika numpy

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Run RTSP SLAM processor
CMD ["python3", "slam3r/rtsp_slam_processor.py"]
```

### 6.3 Create storage/Dockerfile
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy common utilities
COPY common/ /app/common/

# Copy storage specific code
COPY storage/ /app/storage/

# Install Python dependencies
RUN pip install --no-cache-dir \
    opencv-python==4.8.1.78 \
    aio-pika==9.0.0 \
    numpy==1.24.3

# Create recordings directory
RUN mkdir -p /app/recordings

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run RTSP storage processor
CMD ["python", "storage/rtsp_storage.py"]
```

### 6.4 Update server/Dockerfile
Add FFmpeg installation to your existing server Dockerfile:
```dockerfile
# Add FFmpeg installation to your existing server Dockerfile
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
```

### 6.5 Create frame_processor/requirements.txt
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

## Phase 7: Docker Compose Configuration

### 7.1 Update docker-compose.yml
```yaml
version: '3.8'

services:
  mediamtx:
    image: bluenviron/mediamtx:latest-ffmpeg
    container_name: mediamtx
    network_mode: host
    environment:
      - MTX_PROTOCOLS=tcp
      - MTX_RTSPADDRESS=:8554
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "nc", "-z", "localhost", "8554"]
      interval: 5s
      timeout: 3s
      retries: 5

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
      - mediamtx
    volumes:
      - ./server:/app
      - ./models:/app/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5001/health/rtsp"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    labels:
      - "prometheus.io/scrape=true"
      - "prometheus.io/port=5001"
      - "prometheus.io/path=/metrics"

  frame_processor:
    build: 
      context: ./frame_processor
      dockerfile: Dockerfile
    environment:
      - RTSP_HOST=server
      - RTSP_PORT=8554
      - RABBITMQ_URL=amqp://admin:admin@rabbitmq:5672/
      - PYTHONUNBUFFERED=1
      - FRAME_PROCESSOR_SKIP=1  # Process every frame
    depends_on:
      - server
      - rabbitmq
      - mediamtx
    volumes:
      - ./enhanced_objects:/app/enhanced_objects
      - ./common:/app/common
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "python", "-c", "import cv2; cv2.VideoCapture('rtsp://server:8554/drone').isOpened()"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  slam3r:
    build:
      context: ./slam3r
      dockerfile: Dockerfile
    environment:
      - RTSP_HOST=server
      - RTSP_PORT=8554
      - RABBITMQ_URL=amqp://admin:admin@rabbitmq:5672/
      - PYTHONUNBUFFERED=1
      - SLAM_FRAME_SKIP=3  # Process every 3rd frame (10fps)
    depends_on:
      - server
      - rabbitmq
      - mediamtx
    volumes:
      - ./slam3r/data:/app/data
      - slam3r_shared_memory:/dev/shm
      - ./common:/app/common
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "python3", "-c", "import os; exit(0 if os.path.exists('/dev/shm/slam3r_keyframes') else 1)"]
      interval: 30s
      timeout: 10s
      retries: 3

  storage:
    build:
      context: ./storage
      dockerfile: Dockerfile
    environment:
      - RTSP_HOST=server
      - RTSP_PORT=8554
      - RABBITMQ_URL=amqp://admin:admin@rabbitmq:5672/
      - STORAGE_PATH=/app/recordings
      - PYTHONUNBUFFERED=1
      - STORAGE_FRAME_SKIP=1  # Record every frame
    depends_on:
      - server
      - rabbitmq
      - mediamtx
    volumes:
      - ./recordings:/app/recordings
      - ./common:/app/common
    healthcheck:
      test: ["CMD", "python", "-c", "import os; exit(0 if os.path.exists('/app/recordings') else 1)"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  rabbitmq_data:
  slam3r_shared_memory:
```

### 7.2 Add Prometheus Configuration
```yaml
# prometheus.yml - Add server metrics scraping
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'worldsystem-server'
    static_configs:
      - targets: ['server:5001']
    metrics_path: '/metrics'
    
  # Existing scrape configs...
```

### 7.3 Grafana Dashboard Configuration
Create a dashboard to monitor:
- RTSP server health status
- Video bandwidth (current and average)
- RTSP queue size and usage
- Frames dropped count
- RTSP server restart count
- Frame processing rates per service
```

## Phase 8: Testing & Verification

### 8.1 Test MediaMTX RTSP Server
```bash
# Check if MediaMTX is running
docker logs mediamtx

# Check if FFmpeg is pushing to MediaMTX
curl http://localhost:5001/rtsp/status

# Test RTSP stream with ffplay
ffplay rtsp://localhost:8554/drone

# Or with VLC
vlc rtsp://localhost:8554/drone

# Check MediaMTX stats
curl http://localhost:9997/v3/stats
```

### 8.2 Monitor Logs
```bash
# View server logs
docker-compose logs -f server

# View frame processor logs
docker-compose logs -f frame_processor

# View SLAM3R logs
docker-compose logs -f slam3r

# View storage logs
docker-compose logs -f storage
```

### 8.3 Debug Connection Issues
If processors can't connect to RTSP:
1. Check if MediaMTX is running: `docker ps | grep mediamtx`
2. Check if FFmpeg is pushing data: `curl http://localhost:5001/rtsp/status`
3. Verify MediaMTX is listening on port 8554: `netstat -an | grep 8554`
4. Try connecting directly to MediaMTX: `ffplay rtsp://localhost:8554/drone`
5. Check MediaMTX logs: `docker logs mediamtx`
6. Ensure FFmpeg has data to push (simulator/app is sending video)

## Key Architecture Benefits

### Advantages of This Design
1. **Service Specialization**: Each service has a single responsibility
   - frame_processor: Object detection and tracking
   - slam3r: Camera pose estimation and mapping
   - storage: Recording and archival

2. **Scalability**: Can add more services consuming the RTSP stream
3. **Fault Isolation**: If one service fails, others continue
4. **Resource Efficiency**: Services process frames at their optimal rates
5. **Clean Integration**: All services integrate through RabbitMQ

### Resource Requirements
- **frame_processor**: ~4GB VRAM (Grounded-SAM-2)
- **slam3r**: ~2GB VRAM (SLAM algorithms)
- **storage**: Minimal GPU, mainly disk I/O
- **Total GPU Memory**: ~6-8GB for all services

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

## Implementation Timeline

### Phase 1 (Day 1): Server RTSP Setup

- Modify server/main.py to add RTSP streaming
- Test RTSP output with VLC/ffplay
- Ensure stable H.264 streaming

### Phase 2 (Day 2): Base Infrastructure

- Create common/rtsp_consumer.py base class
- Set up directory structure
- Update docker-compose.yml

### Phase 3 (Day 3-4): Service Implementation

- Implement Grounded-SAM-2 in frame_processor
- Adapt SLAM3R to consume RTSP
- Create storage service for recording

### Phase 4 (Day 5): Integration Testing

- Test all services together
- Verify RabbitMQ message flow
- Check resource utilization

### Phase 5 (Day 6-7): Optimization

- Fine-tune frame skipping rates
- Optimize prompt strategies
- Add monitoring and metrics

## Summary

This implementation provides a clean, scalable architecture with MediaMTX as the RTSP server:

### Core Features
1. **MediaMTX RTSP Server** - A proper RTSP server that handles multiple clients
2. **FFmpeg Push Architecture** - Server uses FFmpeg to push H.264 stream to MediaMTX
3. **Three specialized services** consuming the RTSP stream independently:
   - **frame_processor**: Grounded-SAM-2 for object detection and tracking
   - **slam3r**: SLAM processing for camera pose estimation  
   - **storage**: Video recording and archival
4. **Configurable frame skipping** per service via environment variables
5. **Comprehensive monitoring** with Prometheus metrics and health checks

### Key Improvements with MediaMTX
- **Proper RTSP Server**: MediaMTX handles RTSP protocol correctly (FFmpeg cannot act as RTSP server)
- **Multiple Concurrent Clients**: All services can connect simultaneously
- **Reliable Stream Distribution**: MediaMTX buffers and distributes the stream efficiently
- **Standard RTSP URLs**: Services connect to `rtsp://mediamtx:8554/drone`
- **Health Monitoring**: Built-in health checks for MediaMTX container

### Architecture Flow
1. Android/Simulator sends custom packet format via WebSocket to server
2. Server extracts H.264 NAL units from packets (strips headers)
3. FFmpeg receives H.264 via stdin and pushes to MediaMTX RTSP server
4. MediaMTX serves RTSP stream to all consumers (frame_processor, slam3r, storage)
5. Each service processes frames at its own rate using the RTSPConsumer base class

### Monitoring Integration
The system integrates with your existing Grafana/Prometheus stack to provide:
- FFmpeg health status and queue metrics
- MediaMTX connection statistics  
- Real-time video bandwidth monitoring
- Per-service frame processing rates
- Network quality indicators