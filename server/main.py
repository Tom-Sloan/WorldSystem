#!/usr/bin/env python3
"""
server/main.py – Fully asynchronous version using aio_pika for both publishing and consuming.
"""

import os
import time
import json
import asyncio
import struct
import base64
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource

import aio_pika
import numpy as np
import cv2
import ntplib
import socket
import threading
from contextlib import asynccontextmanager
import av  # For H.264 testing endpoint

# No longer need RTSP/FFmpeg imports

from src.config.settings import logger, API_PORT, BIND_HOST, AnalysisMode
from src.api.routes import router

# ------------- WebSocket Broadcasting imports -------------
from typing import Set, Dict
import weakref

# ------------- Prometheus metrics for WebSocket streaming -------------
# Use a try-except block to handle metric re-registration during hot reload
from prometheus_client import CollectorRegistry, REGISTRY

# Initialize metrics to None
h264_consumers_count = None
h264_frames_relayed = None
h264_bytes_relayed = None
video_bandwidth_metric = None
consumer_queue_size = None
frames_dropped_metric = None

try:
    h264_consumers_count = Gauge('h264_consumers_connected', 'Number of connected H.264 consumers')
    h264_frames_relayed = Counter('h264_frames_relayed_total', 'Total H.264 frames relayed to consumers')
    h264_bytes_relayed = Counter('h264_bytes_relayed_total', 'Total H.264 bytes relayed to consumers')
    video_bandwidth_metric = Gauge('video_bandwidth_mbps', 'Current video bandwidth in Mbps')
    consumer_queue_size = Gauge('consumer_queue_size', 'Average consumer queue size')
    frames_dropped_metric = Counter('consumer_frames_dropped_total', 'Total frames dropped due to queue full')
except ValueError as e:
    # Metrics already registered - get existing ones
    for collector in REGISTRY._collector_to_names:
        if hasattr(collector, '_name'):
            if collector._name == 'h264_consumers_connected':
                h264_consumers_count = collector
            elif collector._name == 'h264_frames_relayed_total':
                h264_frames_relayed = collector
            elif collector._name == 'h264_bytes_relayed_total':
                h264_bytes_relayed = collector
            elif collector._name == 'video_bandwidth_mbps':
                video_bandwidth_metric = collector
            elif collector._name == 'consumer_queue_size':
                consumer_queue_size = collector
            elif collector._name == 'consumer_frames_dropped_total':
                frames_dropped_metric = collector

# If any metrics are still None, create them with a fallback approach
if h264_consumers_count is None:
    h264_consumers_count = Gauge('h264_consumers_connected_fallback', 'Number of connected H.264 consumers')
if h264_frames_relayed is None:
    h264_frames_relayed = Counter('h264_frames_relayed_total_fallback', 'Total H.264 frames relayed to consumers')
if h264_bytes_relayed is None:
    h264_bytes_relayed = Counter('h264_bytes_relayed_total_fallback', 'Total H.264 bytes relayed to consumers')
if video_bandwidth_metric is None:
    video_bandwidth_metric = Gauge('video_bandwidth_mbps_fallback', 'Current video bandwidth in Mbps')
if consumer_queue_size is None:
    consumer_queue_size = Gauge('consumer_queue_size_fallback', 'Average consumer queue size')
if frames_dropped_metric is None:
    frames_dropped_metric = Counter('consumer_frames_dropped_total_fallback', 'Total frames dropped due to queue full')

# ------------- OTLP / Tracing -------------
resource = Resource(attributes={
    "service.name": "my-fastapi-server",
    "service.version": "1.0.0"
})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer_provider = trace.get_tracer_provider()
otlp_exporter = OTLPSpanExporter(endpoint="http://jaeger:4318/v1/traces", timeout=5)
tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

# ------------- FastAPI App -------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server startup...")

    # NTP sync
    logger.info("[NTP] Initializing time synchronization...")
    sync_ntp_time()

    # RabbitMQ & exchanges
    await setup_amqp()
    
    # WebSocket video broadcasting is handled per-connection
    logger.info("WebSocket video broadcasting ready")

    # background tasks we need to cancel on shutdown
    bg_tasks = [
        asyncio.create_task(log_imu_rate()),
        # Removed consume_processed_frames_async() - visualization service handles this now
        asyncio.create_task(consume_trajectory_updates()),
        asyncio.create_task(consume_imu_data()),
        asyncio.create_task(consume_restart_messages()),
    ]

    # Send restart broadcast
    try:
        restart_message = aio_pika.Message(
            body=b'{"type":"restart"}',
            content_type="application/json",
        )
        await amqp_exchanges[RESTART_EXCHANGE].publish(restart_message, routing_key="")
        logger.info("Published restart command on startup")
    except Exception as e:
        logger.error(f"Failed to publish restart command on startup: {e}")

    # ——— hand control back to FastAPI ———
    yield

    # ---------- graceful shutdown ----------
    logger.info("Server shutting down…")
    
    # Close all video consumer connections
    for ws in list(video_manager._video_consumers):
        await video_manager.remove_consumer(ws)
    
    # Clean up tasks
    for task in bg_tasks:
        task.cancel()
    await asyncio.gather(*bg_tasks, return_exceptions=True)
    if amqp_connection:
        await amqp_connection.close()


app = FastAPI(lifespan=lifespan)

# Add this new middleware before other middleware
@app.middleware("http")
async def add_timestamp_to_log(request: Request, call_next):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    response = await call_next(request)
    # Log with timestamp
    logger.info(f"[{timestamp}] {request.client.host}:{request.client.port} - \"{request.method} {request.url.path} HTTP/{request.scope['http_version']}\" {response.status_code}")
    return response

# ------------- Instrumentation -------------
FastAPIInstrumentor.instrument_app(app)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# ------------- RabbitMQ Connection & Exchange Setup -------------
# Environment variables for RabbitMQ and exchange names:
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://rabbitmq")
VIDEO_FRAMES_EXCHANGE = os.getenv("VIDEO_FRAMES_EXCHANGE", "video_frames_exchange")
VIDEO_STREAM_EXCHANGE = os.getenv("VIDEO_STREAM_EXCHANGE", "video_stream_exchange")
IMU_DATA_EXCHANGE = os.getenv("IMU_DATA_EXCHANGE", "imu_data_exchange")
PROCESSED_FRAMES_EXCHANGE = os.getenv("PROCESSED_FRAMES_EXCHANGE", "processed_frames_exchange")
ANALYSIS_MODE_EXCHANGE = os.getenv("ANALYSIS_MODE_EXCHANGE", "analysis_mode_exchange")
TRAJECTORY_DATA_EXCHANGE = os.getenv("TRAJECTORY_DATA_EXCHANGE", "trajectory_data_exchange")
PLY_FANOUT_EXCHANGE = os.getenv("PLY_FANOUT_EXCHANGE", "ply_fanout_exchange")
RESTART_EXCHANGE = os.getenv("RESTART_EXCHANGE", "restart_exchange")


# Global variables for the connection, channel, and exchanges:
amqp_connection: aio_pika.RobustConnection = None
amqp_channel: aio_pika.Channel = None
amqp_exchanges = {}  # Dict to hold declared exchanges



# Add these near the top with other global variables
imu_counter = 0
last_imu_time = time.time()
imu_timestamps = []  # Store recent timestamps for calculating intervals
imu_buffer = []  # Buffer to hold incoming IMU data for sorting
imu_buffer_max_size = 100  # Maximum buffer size
imu_buffer_flush_interval = 0.05  # Flush buffer every 50ms
frame_latencies = []  # Store (timestamp, latency) pairs for the last 10 seconds

# Add NTP client and time offset tracking
ntp_client = ntplib.NTPClient()
ntp_time_offset = 0.0  # Offset between system time and NTP time in seconds
last_ntp_sync = 0
NTP_SYNC_INTERVAL = 60  # Sync NTP every 60 seconds

# List of NTP servers to try in order
NTP_SERVERS = [
    "pool.ntp.org",  # Move this to first position since it works in simulation
    "1.pool.ntp.org", # IP-based alternatives
    "2.pool.ntp.org",
    "time.google.com",
    "time.cloudflare.com",
    "time.windows.com",
    "time.apple.com",
    os.getenv("NTP_SERVER", "0.pool.ntp.org")  # Use custom server if specified
]

# WebSocket Broadcasting global variables
class WebSocketVideoManager:
    def __init__(self):
        # Track connected video consumers
        self._video_consumers: Set[WebSocket] = weakref.WeakSet()
        self.frame_buffer = None  # Optional: buffer last I-frame for new consumers
        self.sps_pps_data = None  # Store SPS/PPS for new consumers
        self._lock = asyncio.Lock()
        
    async def add_consumer(self, websocket: WebSocket):
        """Add a new video consumer"""
        await websocket.accept()
        async with self._lock:
            self._video_consumers.add(websocket)
        
        # Send SPS/PPS immediately to new consumer if available
        if self.sps_pps_data:
            try:
                await websocket.send_bytes(self.sps_pps_data)
            except:
                pass
                
        logger.info(f"Video consumer connected. Total consumers: {len(self._video_consumers)}")
        h264_consumers_count.set(len(self._video_consumers))
        
    async def remove_consumer(self, websocket: WebSocket):
        """Remove a video consumer"""
        async with self._lock:
            self._video_consumers.discard(websocket)
        try:
            await websocket.close()
        except:
            pass
        logger.info(f"Video consumer disconnected. Total consumers: {len(self._video_consumers)}")
        h264_consumers_count.set(len(self._video_consumers))
        
    async def broadcast_h264_frame(self, frame_data: bytes, is_sps_pps: bool = False):
        """Broadcast H.264 frame to all consumers"""
        if is_sps_pps:
            self.sps_pps_data = frame_data
            
        # Update metrics
        h264_frames_relayed.inc()
        h264_bytes_relayed.inc(len(frame_data))
        
        # Broadcast to all connected consumers
        disconnected = []
        consumers = list(self._video_consumers)
        
        for ws in consumers:
            try:
                await ws.send_bytes(frame_data)
            except Exception as e:
                logger.debug(f"Failed to send to consumer: {e}")
                disconnected.append(ws)
                
        # Clean up disconnected consumers
        for ws in disconnected:
            await self.remove_consumer(ws)
    
    def get_consumer_count(self):
        """Get current number of consumers"""
        return len(self._video_consumers)

# Global video manager instance
video_manager = WebSocketVideoManager()

def sync_ntp_time():
    """Update the NTP time offset by querying NTP servers."""
    global ntp_time_offset, last_ntp_sync
    
    for server in NTP_SERVERS:
        try:
            response = ntp_client.request(server, timeout=5)
            # Calculate offset: NTP time - local time
            ntp_time_offset = response.offset
            last_ntp_sync = time.time()
            logger.info(f"[NTP] Synchronized time with {server}, offset: {ntp_time_offset:.6f} seconds")
            return True
        except (ntplib.NTPException, socket.gaierror, socket.timeout) as e:
            # logger.warning(f"[NTP] Failed to sync with {server}: {str(e)}")
            continue
    
    # logger.error("[NTP] Failed to sync with all NTP servers")
    return False

def get_ntp_time_ns():
    """Get current time in nanoseconds, synchronized with NTP."""
    current_time = time.time() + ntp_time_offset
    # Check if we need to resync
    if time.time() - last_ntp_sync > NTP_SYNC_INTERVAL:
        # Start a thread to sync NTP time without blocking
        threading.Thread(target=sync_ntp_time, daemon=True).start()
    return int(current_time * 1e9)  # Convert to nanoseconds

# WebSocket broadcasting functions are now part of WebSocketVideoManager class above

async def setup_amqp():
    """
    Establish an asynchronous RabbitMQ connection and channel, and declare exchanges.
    """
    global amqp_connection, amqp_channel, amqp_exchanges
    
    max_retries = 5
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            amqp_connection = await aio_pika.connect_robust(
                RABBITMQ_URL,
                heartbeat=3600,
                connection_attempts=3,
                retry_delay=2
            )
            amqp_channel = await amqp_connection.channel()
            
            # Declare all exchanges as fanout and durable:
            amqp_exchanges[VIDEO_FRAMES_EXCHANGE] = await amqp_channel.declare_exchange(
                VIDEO_FRAMES_EXCHANGE, aio_pika.ExchangeType.FANOUT, durable=True)
            amqp_exchanges[VIDEO_STREAM_EXCHANGE] = await amqp_channel.declare_exchange(
                VIDEO_STREAM_EXCHANGE, aio_pika.ExchangeType.FANOUT, durable=True)
            amqp_exchanges[IMU_DATA_EXCHANGE] = await amqp_channel.declare_exchange(
                IMU_DATA_EXCHANGE, aio_pika.ExchangeType.FANOUT, durable=True)
            amqp_exchanges[PROCESSED_FRAMES_EXCHANGE] = await amqp_channel.declare_exchange(
                PROCESSED_FRAMES_EXCHANGE, aio_pika.ExchangeType.FANOUT, durable=True)
            amqp_exchanges[ANALYSIS_MODE_EXCHANGE] = await amqp_channel.declare_exchange(
                ANALYSIS_MODE_EXCHANGE, aio_pika.ExchangeType.FANOUT, durable=True)
            amqp_exchanges[TRAJECTORY_DATA_EXCHANGE] = await amqp_channel.declare_exchange(
                TRAJECTORY_DATA_EXCHANGE, aio_pika.ExchangeType.FANOUT, durable=True)
            amqp_exchanges[PLY_FANOUT_EXCHANGE] = await amqp_channel.declare_exchange(
                PLY_FANOUT_EXCHANGE, aio_pika.ExchangeType.FANOUT, durable=True)
            amqp_exchanges[RESTART_EXCHANGE] = await amqp_channel.declare_exchange(
                RESTART_EXCHANGE, aio_pika.ExchangeType.FANOUT, durable=True)
                
            logger.info("AMQP setup complete.")
            return
            
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Failed to connect to RabbitMQ (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect to RabbitMQ after {max_retries} attempts: {e}")
                raise

# ------------- Global State -------------
connected_viewers = set()   # Set of connected viewer WebSocket objects
connected_phones = set()    # Set of connected phone WebSocket objects
partial_uploads = {}

# ------------- CORS -------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to your domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------- Include Existing Routes -------------
app.include_router(router, prefix="/api")

# ------------- H.264 Endpoints -------------
@app.get("/h264/test")
async def test_h264_decoder():
    """Test H.264 decoder availability"""
    try:
        import av
        codec = av.CodecContext.create('h264', 'r')
        
        return {
            "status": "ok",
            "av_version": av.__version__,
            "codec_name": codec.name,
            "codec_long_name": codec.codec.long_name if hasattr(codec, 'codec') else "H.264",
            "thread_type": codec.thread_type if hasattr(codec, 'thread_type') else "unknown"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "suggestion": "Make sure PyAV is installed with: conda install av>=10.0.0"
        }

# Basic health check endpoint
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "service": "worldsystem-server",
        "timestamp": time.time()
    }

# Health check endpoint for RTSP
@app.get("/health/video")
async def video_health_check():
    """Health check endpoint for video streaming"""
    return {
        "status": "healthy",
        "video_url": "ws://server:5001/ws/video/consume",
        "active_consumers": video_manager.get_consumer_count(),
        "has_sps_pps": video_manager.sps_pps_data is not None
    }

@app.get("/video/status")
async def video_status():
    """Check video broadcasting status"""
    return {
        "websocket_consumers": video_manager.get_consumer_count(),
        "consumer_endpoint": "/ws/video/consume",
        "has_sps_pps_cached": video_manager.sps_pps_data is not None,
        "sps_pps_size": len(video_manager.sps_pps_data) if video_manager.sps_pps_data else 0
    }

# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics"""
    return Response(generate_latest(), media_type="text/plain")

# ------------- WebSocket: VIDEO STREAMING -> /ws/video -------------
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
                    
                    # Log packet type for debugging
                    if chunks_sent_to_rtsp < 10 or chunks_sent_to_rtsp % 100 == 0:
                        logger.info(f"Packet type: 0x{packet_type:02X}, size: {packet_size}")
                    
                    # Extract H.264 NAL units based on packet type
                    if packet_type == 0x01:  # PACKET_TYPE_SPS
                        sps_data = packet[13:]  # Skip header
                        logger.info(f"Received SPS: {len(sps_data)} bytes")
                        # Broadcast SPS to consumers
                        await video_manager.broadcast_h264_frame(sps_data, is_sps_pps=True)
                        
                    elif packet_type == 0x02:  # PACKET_TYPE_PPS
                        pps_data = packet[13:]  # Skip header
                        logger.info(f"Received PPS: {len(pps_data)} bytes")
                        # Combine SPS+PPS for consumers
                        if sps_data:
                            combined_sps_pps = sps_data + pps_data
                            await video_manager.broadcast_h264_frame(combined_sps_pps, is_sps_pps=True)
                        
                    elif packet_type in [0x03, 0x04]:  # KEYFRAME or FRAME
                        # Extract H.264 data (skip 21-byte header)
                        h264_data = packet[21:]
                        
                        # Broadcast H.264 frame to all WebSocket consumers
                        await video_manager.broadcast_h264_frame(h264_data)
                        chunks_sent_to_rtsp += 1
                
                # Calculate bandwidth every 5 seconds
                current_time = time.time()
                if current_time - last_sample_time >= sample_window:
                    bandwidth_mbps = (bytes_in_window * 8) / (sample_window * 1_000_000)
                    bandwidth_samples.append(bandwidth_mbps)
                    
                    # Log bandwidth and update metrics
                    avg_bandwidth = sum(bandwidth_samples[-10:]) / min(len(bandwidth_samples), 10)
                    logger.info(f"[{websocket_id}] Video bandwidth: {bandwidth_mbps:.2f} Mbps (avg: {avg_bandwidth:.2f} Mbps)")
                    try:
                        video_bandwidth_metric.set(bandwidth_mbps)
                    except Exception:
                        pass
                    
                    # Alert if bandwidth is consistently low
                    if avg_bandwidth < 3.0 and len(bandwidth_samples) > 5:
                        logger.warning(f"[{websocket_id}] Low bandwidth detected ({avg_bandwidth:.2f} Mbps), consider reducing bitrate")
                    
                    # Reset window
                    bytes_in_window = 0
                    last_sample_time = current_time
                    
                # Log progress every 100 chunks
                if chunks_sent_to_rtsp % 100 == 0:
                    logger.info(f"[{websocket_id}] Broadcast {chunks_sent_to_rtsp} chunks to {video_manager.get_consumer_count()} consumers, {bytes_received/1024/1024:.2f} MB total")
                    
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

# ------------- WebSocket: VIDEO CONSUMER -> /ws/video/consume -------------
@app.websocket("/ws/video/consume")
async def websocket_video_consumer(websocket: WebSocket):
    """WebSocket endpoint for video consumers (frame_processor, storage, etc.)"""
    await video_manager.add_consumer(websocket)
    
    try:
        # Keep connection alive and wait for any messages (e.g., heartbeat)
        while True:
            # We don't expect consumers to send data, but we need to handle disconnects
            data = await websocket.receive()
            if data["type"] == "websocket.disconnect":
                break
            # Optional: handle heartbeat messages
            elif "text" in data:
                msg = data["text"]
                if msg == "ping":
                    await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Error in video consumer: {e}")
    finally:
        await video_manager.remove_consumer(websocket)

# ------------- WebSocket: PHONE -> /ws/phone -------------
@app.websocket("/ws/phone")
async def phone_ws(ws: WebSocket):
    await ws.accept()
    connected_phones.add(ws)
    logger.info(f"New phone connected: {ws.client.host}")
    
    # Create tasks for buffer flushing and latency logging
    buffer_flush_task = asyncio.create_task(flush_imu_buffer_periodically())
    latency_logging_task = asyncio.create_task(log_frame_latency())
    
    # Set to track already seen timestamps to prevent duplicates
    seen_timestamps = set()
    
    try:
        while True:
            data = await ws.receive()
            if data["type"] == "websocket.disconnect":
                buffer_flush_task.cancel()
                latency_logging_task.cancel()
                raise WebSocketDisconnect()

            # Handle binary messages (video frames)
            if "bytes" in data:
                frame_bytes = data["bytes"]
                # Check if the received frame has at least 8 bytes for the timestamp.
                if len(frame_bytes) >= 8:
                    # Extract timestamp and image data
                    timestamp_ns = int.from_bytes(frame_bytes[:8], byteorder="big")
                    image_data = frame_bytes[8:]
                    
                    # Get NTP-synchronized server timestamp
                    server_timestamp_ns = get_ntp_time_ns()
                    
                    # Convert image bytes to numpy array and get dimensions
                    nparr = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    height, width = img.shape[:2]
                    # logger.info(f"Received frame: {width}x{height}")
                    
                    # Calculate latency using NTP time
                    latency_ms = (server_timestamp_ns - timestamp_ns) / 1_000_000  # Convert ns to ms
                    
                    # Update latency tracking
                    global frame_latencies
                    frame_latencies.append((server_timestamp_ns, latency_ms))

                    message = aio_pika.Message(
                        body=image_data,
                        content_type="application/octet-stream",
                        headers={
                            "timestamp_ns": str(timestamp_ns),
                            "server_received": str(server_timestamp_ns),
                            "ntp_time": str(server_timestamp_ns),
                            "ntp_offset": str(ntp_time_offset),
                            "width": str(width),
                            "height": str(height)
                        }
                    )
                    await amqp_exchanges[VIDEO_FRAMES_EXCHANGE].publish(message, routing_key="")
                else:
                    logger.error("Received frame is too short to contain a timestamp")

            # Handle text messages (control and IMU data)
            # Handle text messages (control and IMU data)
            elif "text" in data:
                try:
                    msg = json.loads(data["text"])
                    if msg.get("type") == "imu_data":
                        # Check if this timestamp has been seen before
                        timestamp = msg["timestamp"]
                        if timestamp in seen_timestamps:
                            # Skip duplicate timestamps
                            logger.debug(f"Skipping duplicate IMU timestamp: {timestamp}")
                            continue
                            
                        # Add to seen timestamps set
                        seen_timestamps.add(timestamp)
                        
                        # Limit the size of seen_timestamps to prevent memory growth
                        if len(seen_timestamps) > 10000:
                            # Keep only the most recent 5000 timestamps
                            seen_timestamps = set(sorted(seen_timestamps)[-5000:])
                            
                        # Add timestamp tracking for interval calculation
                        global imu_timestamps, imu_buffer, imu_counter
                        imu_timestamps.append(timestamp)
                        imu_counter += 1
                        
                        # Add to buffer instead of publishing directly
                        imu_buffer.append(msg)
                        
                            
                    elif msg.get("type") == "analysis_mode":
                        mode = msg.get("data", "none").lower()
                        if mode == "yolo":
                            AnalysisMode = AnalysisMode.YOLO
                            logger.info("Frame processor mode set to YOLO")
                        else:
                            AnalysisMode = AnalysisMode.NONE
                            logger.info("Frame processor mode set to NONE")
                        message = aio_pika.Message(
                            body=json.dumps(msg).encode(),
                            content_type="application/json"
                        )
                        await amqp_exchanges[ANALYSIS_MODE_EXCHANGE].publish(message, routing_key="")
                    
                    elif msg.get("type") == "file_upload":
                        # Retrieve file details from the JSON message.
                        file_name = msg.get("fileName", f"upload_{int(time.time())}.csv")
                        file_content = msg.get("fileContent", "")
                        # Ensure that the upload directory exists.
                        upload_dir = "uploads"
                        os.makedirs(upload_dir, exist_ok=True)
                        file_path = os.path.join(upload_dir, file_name)
                        with open(file_path, "w") as f:
                            f.write(file_content)
                        logger.info(f"File received and saved locally as {file_path}")
                    elif msg.get("type") == "file_upload_start":
                        file_name = msg.get("fileName", f"upload_{int(time.time())}.csv")
                        upload_dir = "uploads"
                        os.makedirs(upload_dir, exist_ok=True)
                        file_path = os.path.join(upload_dir, file_name)
                        # Open the file for writing in text mode
                        partial_uploads[file_name] = open(file_path, "w", encoding="utf-8")
                        logger.info(f"Started receiving file {file_name} in chunks.")

                    elif msg.get("type") == "file_upload_chunk":
                        file_name = msg.get("fileName")
                        chunk_index = msg.get("chunkIndex", -1)
                        chunk_data = msg.get("chunk", "")
                        if file_name in partial_uploads:
                            partial_uploads[file_name].write(chunk_data)
                            logger.debug(f"Wrote chunk {chunk_index} for file {file_name}")
                        else:
                            logger.warning(f"Received chunk for unknown file {file_name}")

                    elif msg.get("type") == "file_upload_complete":
                        file_name = msg.get("fileName")
                        if file_name in partial_uploads:
                            partial_uploads[file_name].close()
                            del partial_uploads[file_name]
                            logger.info(f"File fully received: {file_name}")
                        else:
                            logger.warning(f"Received file_upload_complete for unknown file {file_name}")

                    else:
                        logger.info(f"Phone JSON: {msg}")
                except json.JSONDecodeError:
                    print(f"Phone JSON:{data}")
                    logger.warning("Received invalid JSON from phone.")
                except Exception as e:
                    logger.error(f"Error processing phone text message: {e}")
    except WebSocketDisconnect:
        buffer_flush_task.cancel()
        connected_phones.discard(ws)
        logger.info(f"Phone disconnected: {ws.client.host}")
    except Exception as e:
        buffer_flush_task.cancel()
        connected_phones.discard(ws)
        logger.error(f"Error in phone websocket: {e}")

async def flush_imu_buffer():
    """Sort and publish IMU data from buffer by timestamp order"""
    global imu_buffer
    
    if not imu_buffer:
        return
        
    try:
        # Get a snapshot of the current buffer and clear it immediately
        # to prevent any race conditions with new incoming data
        current_buffer = imu_buffer.copy()
        imu_buffer.clear()
        
        if not current_buffer:
            return
            
        # Log buffer size before deduplication
        buffer_size_before = len(current_buffer)
        
        # Sort buffer by timestamp
        sorted_buffer = sorted(current_buffer, key=lambda x: x["timestamp"])
        
        # Use a set to track seen timestamps during this flush operation
        seen_timestamps = set()
        deduped_messages = []
        
        # Deduplicate the buffer
        for msg in sorted_buffer:
            timestamp = msg["timestamp"]
            if timestamp not in seen_timestamps:
                seen_timestamps.add(timestamp)
                deduped_messages.append(msg)
        
        # Log deduplication results
        buffer_size_after = len(deduped_messages)
        if buffer_size_before > buffer_size_after:
            logger.info(f"Deduped {buffer_size_before - buffer_size_after} messages in buffer flush")
        
        # Process and publish the deduplicated messages
        for msg in deduped_messages:
            # Extract IMU data
            accel = msg["accelerometer"]
            gyro = msg["gyroscope"]
            
            # Convert to numpy arrays
            phone_accel = np.array([accel["x"], accel["y"], accel["z"]])
            phone_gyro = np.array([gyro["x"], gyro["y"], gyro["z"]])
            
            # Use values directly without transformation
            accel_cam = phone_accel
            gyro_cam = phone_gyro
            
            # Create message with transformed data
            transformed_msg = {
                "type": "imu_data",
                "timestamp": msg["timestamp"],
                "accelerometer": {
                    "x": float(accel_cam[0]),
                    "y": float(accel_cam[1]),
                    "z": float(accel_cam[2])
                },
                "gyroscope": {
                    "x": float(gyro_cam[0]),
                    "y": float(gyro_cam[1]),
                    "z": float(gyro_cam[2])
                }
            }
            
            # Publish to RabbitMQ
            message = aio_pika.Message(
                body=json.dumps(transformed_msg).encode(),
                content_type="application/json"
            )
            await amqp_exchanges[IMU_DATA_EXCHANGE].publish(message, routing_key="")
            
    except Exception as e:
        logger.error(f"Error in flush_imu_buffer: {e}")
        # Log the structure of the first message to help debug
        if sorted_buffer and len(sorted_buffer) > 0:
            logger.error(f"First message structure: {sorted_buffer[0].keys()}")

async def flush_imu_buffer_periodically():
    """Periodically flush the IMU buffer to ensure data is published even if buffer isn't full"""
    try:
        # Log startup of the periodic flush task
        logger.info(f"Starting periodic IMU buffer flush task (interval: {imu_buffer_flush_interval}s)")
        
        while True:
            await asyncio.sleep(imu_buffer_flush_interval)
            # Log buffer size before flush if it's not empty
            if imu_buffer:
                logger.debug(f"Periodic flush: buffer has {len(imu_buffer)} messages")
            await flush_imu_buffer()
    except asyncio.CancelledError:
        # Task was cancelled, clean up
        logger.info("Periodic IMU buffer flush task cancelled")
        pass
    except Exception as e:
        logger.error(f"Error in flush_imu_buffer_periodically: {e}")

# ------------- WebSocket: VIEWER -> /ws/viewer -------------
@app.websocket("/ws/viewer")
async def viewer_ws(ws: WebSocket):
    await ws.accept()
    connected_viewers.add(ws)
    logger.info(f"New viewer connected: {ws.client.host}")
    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                print(data)
                if msg.get("type") == "joystick":
                    # Forward joystick commands to all connected phones
                    for phone in list(connected_phones):
                        try:
                            await phone.send_text(data)
                        except Exception as e:
                            logger.error(f"Error sending joystick command to phone {phone.client.host}: {e}")
                            connected_phones.discard(phone)
                # Handle new message types
                elif msg.get("type") in ["movement", "rotation", "camera", "flightmode"]:
                    # Forward all control commands to connected phones
                    for phone in list(connected_phones):
                        try:
                            await phone.send_text(data)
                        except Exception as e:
                            logger.error(f"Error sending {msg.get('type')} command to phone {phone.client.host}: {e}")
                            connected_phones.discard(phone)
                    
                    # Log specific actions
                    if msg.get("type") == "camera":
                        logger.info(f"Camera {msg.get('action')} command received")
                    elif msg.get("type") == "flightmode":
                        logger.info(f"Flight mode {msg.get('action')} command received")
                
                # Log latency if timestamp is present
                if "timestamp" in data:
                    timestamp = int(msg["timestamp"])
                    current_time = get_ntp_time_ns()
                    latency = (current_time - timestamp) / 1_000_000
                    # logger.info(f"Latency: {latency} ms")
            except json.JSONDecodeError:
                logger.warning(f"Received invalid JSON from viewer: {data}")
            except Exception as e:
                logger.error(f"Error processing viewer message: {e}")
                logger.info(f"Viewer said: {data}")
    except WebSocketDisconnect:
        logger.info(f"Viewer disconnected: {ws.client.host}")
    finally:
        connected_viewers.discard(ws)

# ------------- Consumers using aio_pika -------------

# Modify this function to handle nanosecond timestamps
async def log_imu_rate():
    """Periodically log the IMU message rate and average interval between samples"""
    global imu_counter, last_imu_time, imu_timestamps
    while True:
        await asyncio.sleep(1)  # Wait for 1 second
        current_time = time.time()
        elapsed_time = current_time - last_imu_time
        rate = imu_counter / elapsed_time if elapsed_time > 0 else 0
        
        # Calculate average time between IMU samples from timestamps
        avg_interval_ms = 0
        if len(imu_timestamps) > 1:
            # Get only timestamps from the last second
            # Convert current time to nanoseconds to match IMU timestamp format
            current_ns = int(time.time() * 1_000_000_000)
            one_second_ago_ns = current_ns - 1_000_000_000  # 1 second in ns
            
            # Filter recent timestamps, ensure uniqueness, and sort them
            recent_timestamps = sorted(set([ts for ts in imu_timestamps if ts > one_second_ago_ns]))
            
            if len(recent_timestamps) > 1:
                # Calculate intervals in nanoseconds, then convert to milliseconds for display
                intervals = [recent_timestamps[i] - recent_timestamps[i-1] for i in range(1, len(recent_timestamps))]
                avg_interval_ns = sum(intervals) / len(intervals)
                avg_interval_ms = avg_interval_ns / 1_000_000  # Convert ns to ms
            
            # Prune old timestamps to prevent memory growth
            imu_timestamps = [ts for ts in imu_timestamps if ts > one_second_ago_ns]
        
        # logger.info(f"IMU message rate: {rate:.2f} msgs/sec, Avg interval: {avg_interval_ms:.2f} ms")
        imu_counter = 0
        last_imu_time = current_time

async def log_frame_latency():
    """Periodically log the average frame latency over the last 10 seconds"""
    while True:
        await asyncio.sleep(1)  # Log every second
        current_time_ns = time.time_ns()
        cutoff_time_ns = current_time_ns - (10 * 1_000_000_000)  # 10 seconds in ns
        
        # Calculate average latency over last 10 seconds
        recent_latencies = [lat for ts, lat in frame_latencies if ts > cutoff_time_ns]
        if recent_latencies:
            avg_latency = sum(recent_latencies) / len(recent_latencies)
            # logger.info(f"Average frame latency over last 10 seconds: {avg_latency:.2f} ms")


# Removed consume_processed_frames_async() - visualization service is the only consumer now

async def consume_trajectory_updates():
    """
    Asynchronously consume messages from the trajectory_updates queue
    and broadcast each trajectory update to all connected viewers immediately.
    """
    try:
        connection = await aio_pika.connect_robust(RABBITMQ_URL, heartbeat=3600)
        channel = await connection.channel()
        # Add server_ prefix to queue name
        queue = await channel.declare_queue('server_trajectory_updates', exclusive=True)
        await queue.bind(exchange=TRAJECTORY_DATA_EXCHANGE)
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    try:
                        pose = json.loads(message.body)
                        data_to_send = {
                            "type": "trajectory_update",
                            "pose": pose
                        }
                        msg = json.dumps(data_to_send)
                        for vw in list(connected_viewers):
                            try:
                                await vw.send_text(msg)
                            except Exception as e:
                                logger.error(f"Error sending trajectory update to {vw.client.host}: {e}")
                                connected_viewers.discard(vw)
                    except Exception as e:
                        logger.error(f"Error processing trajectory update: {e}")
    except Exception as e:
        logger.error(f"Failed to consume trajectory updates: {e}")

async def consume_imu_data():
    """
    Asynchronously consume messages from the IMU data exchange and rebroadcast each
    message as one of type "imu_data" to all connected viewers.
    """
    try:
        connection = await aio_pika.connect_robust(RABBITMQ_URL, heartbeat=3600)
        channel = await connection.channel()
        # Add server_ prefix to queue name
        queue = await channel.declare_queue('server_imu_data', exclusive=True)
        await queue.bind(exchange=IMU_DATA_EXCHANGE)
        
        imu_count = 0  # Counter for IMU readings
        
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    try:
                        imu_msg = json.loads(message.body)
                        data_to_send = {
                            "type": "imu_data",
                            "imu_data": imu_msg
                        }
                        msg_str = json.dumps(data_to_send)
                        for vw in list(connected_viewers):
                            try:
                                await vw.send_text(msg_str)
                            except Exception as e:
                                logger.error(f"Error sending IMU data to viewer {vw.client.host}: {e}")
                                connected_viewers.discard(vw)
                        
                        # Increment counter and log every 1000 readings
                        imu_count += 1
                        if imu_count % 1000 == 0:
                            logger.info(f"Processed {imu_count} IMU readings")
                            
                    except Exception as e:
                        logger.error(f"Error processing IMU data: {e}")
    except Exception as e:
        logger.error(f"Failed to consume IMU data: {e}")

async def consume_restart_messages():
    """
    Asynchronously consume messages from the restart exchange and broadcast them to all viewers.
    """
    try:
        connection = await aio_pika.connect_robust(RABBITMQ_URL, heartbeat=3600)
        channel = await connection.channel()
        # Add server_ prefix to queue name
        queue = await channel.declare_queue('server_restart', exclusive=True)
        await queue.bind(exchange=RESTART_EXCHANGE)
        
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    try:
                        # Forward the restart message to all connected viewers
                        restart_msg = json.dumps({"type": "restart"})
                        for vw in list(connected_viewers):
                            try:
                                await vw.send_text(restart_msg)
                            except Exception as e:
                                logger.error(f"Error sending restart message to viewer {vw.client.host}: {e}")
                                connected_viewers.discard(vw)
                        logger.info("Broadcast restart message to all viewers")
                    except Exception as e:
                        logger.error(f"Error processing restart message: {e}")
    except Exception as e:
        logger.error(f"Failed to consume restart messages: {e}")

async def publish_video_stream_chunk(chunk_bytes: bytes, websocket_id: str) -> None:
    """
    Publish raw H.264 video stream chunks to RabbitMQ for frame_processor.
    
    Args:
        chunk_bytes: Raw H.264 stream chunk data
        websocket_id: ID of the websocket connection
    """
    try:
        # Get current timestamp
        timestamp_ns = get_ntp_time_ns()
        
        # Create message with metadata
        message = aio_pika.Message(
            body=chunk_bytes,
            content_type="video/h264",
            headers={
                "timestamp_ns": str(timestamp_ns),
                "server_received": str(timestamp_ns),
                "ntp_time": str(timestamp_ns),
                "ntp_offset": str(ntp_time_offset),
                "websocket_id": websocket_id,
                "stream_type": "h264",
                "chunk_size": str(len(chunk_bytes))
            }
        )
        
        # Publish to video stream exchange
        await amqp_exchanges[VIDEO_STREAM_EXCHANGE].publish(message, routing_key="")
        
    except Exception as e:
        logger.error(f"Error publishing video stream chunk: {e}")

# Removed broadcast_to_all_viewers() - visualization service handles processed frames now

if __name__ == "__main__":
    uvicorn.run("main:app", host=BIND_HOST, port=API_PORT, reload=True)
