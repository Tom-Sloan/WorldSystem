#!/usr/bin/env python3
"""
Storage Service with WebSocket
Records video stream in time-based chunks with dynamic FPS support
Includes idle timeout to save partial chunks when stream stops
"""

import os
import sys
import cv2
import time
import json
import asyncio
import aio_pika
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
from threading import Lock, Timer
from typing import Optional, Dict, Any

# Add common to path
sys.path.append('/app/common')
from websocket_video_consumer import WebSocketVideoConsumer

# Prometheus metrics
from prometheus_client import start_http_server, Counter, Histogram, Gauge

# Define metrics
video_chunks_saved_counter = Counter(
    "storage_video_chunks_saved_total",
    "Total number of video chunks successfully saved"
)
video_frames_written_counter = Counter(
    "storage_video_frames_written_total",
    "Total number of frames written to video files"
)
video_chunk_duration_hist = Histogram(
    "storage_video_chunk_duration_seconds",
    "Duration of saved video chunks",
    buckets=[30, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600]
)
current_chunk_size = Gauge(
    "storage_current_chunk_size_bytes",
    "Current size of the video chunk being written"
)
actual_fps_gauge = Gauge(
    "storage_actual_fps",
    "Actual FPS being received from stream"
)
storage_queue_size = Gauge(
    "storage_queue_size",
    "Number of frames waiting to be written"
)
partial_chunks_saved_counter = Counter(
    "storage_partial_chunks_saved_total",
    "Total number of partial chunks saved due to idle timeout"
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WebSocketStorageProcessor(WebSocketVideoConsumer):
    """Efficient storage processor for video streams with chunking support and idle timeout"""
    
    def __init__(self, ws_url: str):
        # Storage: record every frame (no skipping)
        super().__init__(ws_url, "Storage", frame_skip=1)
        
        # Configuration
        self.storage_path = Path(os.getenv('STORAGE_PATH', '/app/recordings'))
        self.chunk_duration_seconds = int(os.getenv('VIDEO_CHUNK_DURATION_SECONDS', '60'))
        self.idle_timeout_seconds = float(os.getenv('VIDEO_IDLE_TIMEOUT_SECONDS', '10.0'))
        self.min_chunk_frames = int(os.getenv('MIN_CHUNK_FRAMES', '30'))  # Don't save chunks with fewer frames
        self.use_h264_codec = os.getenv('USE_H264_CODEC', 'false').lower() == 'true'
        
        # Session setup
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_path = self.storage_path / self.session_id
        self.video_segments_path = self.session_path / "video_segments"
        self._create_directories()
        
        # Video codec selection
        if self.use_h264_codec:
            # Try H.264 codec (better compression but may require additional codecs)
            self.video_codec = cv2.VideoWriter_fourcc(*'H264')
            self.file_extension = 'mp4'
        else:
            # Use MJPEG codec (works out-of-the-box, larger files)
            self.video_codec = cv2.VideoWriter_fourcc(*'MJPG')
            self.file_extension = 'avi'
        
        # Video writer state
        self.current_writer: Optional[cv2.VideoWriter] = None
        self.current_chunk_index = 0
        self.current_chunk_start_time: Optional[float] = None
        self.current_chunk_frame_count = 0
        self.writer_lock = Lock()
        
        # Idle timeout management
        self.idle_timer: Optional[Timer] = None
        self.last_frame_time_storage: Optional[float] = None  # Separate from base class last_frame_time
        
        # FPS tracking - extend base class functionality
        self.fps_sample_window = 5.0  # Calculate FPS over 5 second windows
        self.estimated_fps = 15.0  # Initial estimate
        
        # Metadata
        self.metadata = self._initialize_metadata(ws_url)
        self.metadata_update_interval = 300  # Update every ~10 seconds at 30fps
        
        # RabbitMQ setup
        self.rabbit_connection: Optional[aio_pika.Connection] = None
        self.rabbit_channel: Optional[aio_pika.Channel] = None
        self.storage_exchange: Optional[aio_pika.Exchange] = None
        self.async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._setup_rabbitmq()
        
        # Performance tracking
        self.last_metric_update = time.time()
        self.metric_update_interval = 5.0  # Update metrics every 5 seconds
        
        logger.info(f"Storage session initialized: {self.session_id}")
        logger.info(f"Storage path: {self.session_path}")
        logger.info(f"Video chunk duration: {self.chunk_duration_seconds} seconds")
        logger.info(f"Idle timeout: {self.idle_timeout_seconds} seconds")
        logger.info(f"Minimum chunk frames: {self.min_chunk_frames}")
        logger.info(f"Codec: {'H.264' if self.use_h264_codec else 'MJPEG'}")
    
    def _create_directories(self):
        """Create necessary directories for storage"""
        try:
            self.session_path.mkdir(parents=True, exist_ok=True)
            self.video_segments_path.mkdir(exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            raise
    
    def _initialize_metadata(self, ws_url: str) -> Dict[str, Any]:
        """Initialize session metadata"""
        return {
            "session_id": self.session_id,
            "start_time": time.time(),
            "video_url": ws_url,
            "chunk_duration_seconds": self.chunk_duration_seconds,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "codec": "H.264" if self.use_h264_codec else "MJPEG",
            "chunks": [],
            "partial_chunks": [],  # Track chunks saved due to idle timeout
            "total_frames": 0,
            "estimated_fps": self.estimated_fps
        }
    
    def _setup_rabbitmq(self):
        """Setup RabbitMQ connections for publishing storage events"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def _async_setup():
                self.rabbit_connection = await aio_pika.connect_robust(
                    os.getenv('RABBITMQ_URL', 'amqp://127.0.0.1:5672'),
                    loop=loop
                )
                self.rabbit_channel = await self.rabbit_connection.channel()
                
                # Declare storage events exchange
                self.storage_exchange = await self.rabbit_channel.declare_exchange(
                    'storage_events_exchange',
                    aio_pika.ExchangeType.FANOUT,
                    durable=True
                )
            
            loop.run_until_complete(_async_setup())
            self.async_loop = loop
            logger.info("RabbitMQ setup complete")
            
        except Exception as e:
            logger.warning(f"Failed to setup RabbitMQ: {e}. Storage will work without event publishing.")
            self.rabbit_connection = None
            self.async_loop = None
    
    def _reset_idle_timer(self):
        """Reset the idle timer - called whenever a frame is received"""
        # Cancel existing timer if any
        if self.idle_timer is not None:
            self.idle_timer.cancel()
        
        # Only set timer if we have an active chunk with frames
        if self.current_writer and self.current_chunk_frame_count > 0:
            self.idle_timer = Timer(self.idle_timeout_seconds, self._handle_idle_timeout)
            self.idle_timer.start()
    
    def _handle_idle_timeout(self):
        """Handle idle timeout - save current chunk if it has enough frames"""
        with self.writer_lock:
            if self.current_writer and self.current_chunk_frame_count >= self.min_chunk_frames:
                idle_duration = time.time() - self.last_frame_time_storage if self.last_frame_time_storage else 0
                logger.warning(
                    f"Idle timeout reached after {idle_duration:.1f}s of no frames. "
                    f"Saving partial chunk {self.current_chunk_index} with {self.current_chunk_frame_count} frames."
                )
                
                # Mark this as a partial chunk in metadata
                self.metadata["partial_chunks"].append(self.current_chunk_index)
                partial_chunks_saved_counter.inc()
                
                # Close the current chunk
                self._close_current_chunk()
            elif self.current_writer and self.current_chunk_frame_count > 0:
                logger.info(
                    f"Idle timeout reached but chunk has only {self.current_chunk_frame_count} frames "
                    f"(minimum: {self.min_chunk_frames}). Keeping chunk open."
                )
                # Restart the timer to check again later
                self._reset_idle_timer()
    
    def calculate_fps(self):
        """Calculate actual FPS from recent frame timestamps"""
        current_time = time.time()
        
        # Add current timestamp
        self.fps_samples.append(current_time)
        
        # Keep only recent samples within window
        cutoff_time = current_time - self.fps_sample_window
        self.fps_samples = [t for t in self.fps_samples if t > cutoff_time]
        
        # Calculate FPS if we have enough samples
        if len(self.fps_samples) >= 2:
            time_span = self.fps_samples[-1] - self.fps_samples[0]
            if time_span > 0:
                fps = (len(self.fps_samples) - 1) / time_span
                
                # Smooth FPS changes with exponential moving average
                alpha = 0.3  # Smoothing factor
                self.estimated_fps = alpha * fps + (1 - alpha) * self.estimated_fps
                
                # Update metric
                actual_fps_gauge.set(self.estimated_fps)
    
    def should_start_new_chunk(self) -> bool:
        """Check if it's time to start a new video chunk"""
        if self.current_chunk_start_time is None:
            return True
        
        elapsed = time.time() - self.current_chunk_start_time
        return elapsed >= self.chunk_duration_seconds
    
    def _close_current_chunk(self):
        """Close the current video chunk and update metadata (assumes lock is held)"""
        if not self.current_writer:
            return
        
        # Cancel idle timer
        if self.idle_timer:
            self.idle_timer.cancel()
            self.idle_timer = None
            
        # Release video writer
        self.current_writer.release()
        self.current_writer = None
        
        # Calculate chunk metrics
        chunk_duration = time.time() - self.current_chunk_start_time
        chunk_filename = f"chunk_{self.current_chunk_index:04d}.{self.file_extension}"
        chunk_path = self.video_segments_path / chunk_filename
        
        # Get file size (with retry for filesystem sync)
        chunk_size = 0
        for _ in range(3):
            try:
                chunk_size = chunk_path.stat().st_size
                break
            except Exception:
                time.sleep(0.1)
        
        # Create chunk info
        chunk_info = {
            "index": self.current_chunk_index,
            "filename": chunk_filename,
            "start_time": self.current_chunk_start_time,
            "duration": chunk_duration,
            "frame_count": self.current_chunk_frame_count,
            "average_fps": self.current_chunk_frame_count / chunk_duration if chunk_duration > 0 else 0,
            "size_bytes": chunk_size,
            "partial": self.current_chunk_index in self.metadata.get("partial_chunks", [])
        }
        
        # Update metadata
        self.metadata["chunks"].append(chunk_info)
        
        # Update metrics
        video_chunks_saved_counter.inc()
        video_chunk_duration_hist.observe(chunk_duration)
        
        # Publish event
        self._publish_storage_event("chunk_saved", {
            "session_id": self.session_id,
            "chunk_info": chunk_info
        })
        
        chunk_type = "partial" if chunk_info["partial"] else "video"
        logger.info(
            f"Saved {chunk_type} chunk {self.current_chunk_index}: "
            f"{chunk_duration:.1f}s, {self.current_chunk_frame_count} frames, "
            f"{chunk_size/1024/1024:.1f}MB"
        )
    
    def _start_new_chunk(self, frame_shape: tuple):
        """Start a new video chunk (assumes lock is held)"""
        # Close previous chunk if exists
        if self.current_writer:
            self._close_current_chunk()
        
        # Update chunk state
        self.current_chunk_index += 1
        self.current_chunk_start_time = time.time()
        self.current_chunk_frame_count = 0
        
        # Create new writer
        chunk_filename = f"chunk_{self.current_chunk_index:04d}.{self.file_extension}"
        chunk_path = self.video_segments_path / chunk_filename
        height, width = frame_shape[:2]
        
        # Use appropriate FPS for video writer
        fps_to_use = max(10.0, min(60.0, self.estimated_fps))
        
        self.current_writer = cv2.VideoWriter(
            str(chunk_path),
            self.video_codec,
            fps_to_use,
            (width, height)
        )
        
        if not self.current_writer.isOpened():
            logger.error(f"Failed to open video writer for chunk {self.current_chunk_index}")
            self.current_writer = None
            return
        
        logger.info(
            f"Started new video chunk {self.current_chunk_index} "
            f"({width}x{height} @ {fps_to_use:.1f} FPS)"
        )
    
    def process_frame(self, frame: np.ndarray, frame_number: int):
        """Store frame in current video chunk"""
        # Log first frame
        if self.frame_count == 1:
            logger.info(f"✓ Storage service is receiving video frames! Frame shape: {frame.shape}")
            logger.info(f"Starting video recording to: {self.session_path}")
        
        # Update last frame time and reset idle timer
        self.last_frame_time_storage = time.time()
        self._reset_idle_timer()
        
        # Update FPS tracking
        self.calculate_fps()
        
        # Handle chunking and frame writing
        with self.writer_lock:
            # Check if new chunk needed
            if self.should_start_new_chunk():
                if self.current_chunk_start_time:
                    logger.info(f"Starting new chunk after {time.time() - self.current_chunk_start_time:.1f} seconds")
                self._start_new_chunk(frame.shape)
            
            # Write frame
            if self.current_writer and self.current_writer.isOpened():
                self.current_writer.write(frame)
                self.current_chunk_frame_count += 1
                video_frames_written_counter.inc()
            else:
                logger.error(f"No active video writer for frame {frame_number}")
        
        # Periodic updates (outside lock to prevent blocking)
        self._handle_periodic_updates(frame_number)
    
    def _handle_periodic_updates(self, frame_number: int):
        """Handle periodic logging and metric updates"""
        current_time = time.time()
        
        # Update metrics periodically
        if current_time - self.last_metric_update >= self.metric_update_interval:
            self.last_metric_update = current_time
            storage_queue_size.set(self.frame_queue.qsize())
            
            # Update chunk size metric
            with self.writer_lock:
                if self.current_chunk_index > 0:
                    chunk_path = self.video_segments_path / f"chunk_{self.current_chunk_index:04d}.{self.file_extension}"
                    try:
                        size = chunk_path.stat().st_size
                        current_chunk_size.set(size)
                    except Exception:
                        pass
        
        # Log progress every 150 frames (~5 seconds at 30fps)
        if frame_number % 150 == 0 and frame_number > 0:
            with self.writer_lock:
                chunk_frames = self.current_chunk_frame_count
            logger.info(
                f"📹 Recording progress: {chunk_frames} frames in current chunk, "
                f"{frame_number} total frames, {self.estimated_fps:.1f} FPS"
            )
        
        # Save metadata periodically
        if frame_number % self.metadata_update_interval == 0:
            self.save_metadata()
    
    def _publish_storage_event(self, event_type: str, data: dict):
        """Publish storage event to RabbitMQ asynchronously"""
        if not self.rabbit_connection or not self.async_loop:
            return
        
        try:
            message = {
                "type": event_type,
                "timestamp": time.time(),
                "data": data
            }
            
            # Schedule coroutine in event loop
            asyncio.run_coroutine_threadsafe(
                self.storage_exchange.publish(
                    aio_pika.Message(
                        body=json.dumps(message).encode(),
                        content_type="application/json"
                    ),
                    routing_key=''
                ),
                self.async_loop
            )
        except Exception as e:
            logger.debug(f"Failed to publish storage event: {e}")
    
    def save_metadata(self):
        """Save session metadata to disk"""
        try:
            # Update metadata with current values
            with self.writer_lock:
                current_chunk_frames = self.current_chunk_frame_count
            
            self.metadata.update({
                "last_updated": time.time(),
                "total_frames": self.frame_count,
                "estimated_fps": self.estimated_fps,
                "current_chunk_frames": current_chunk_frames,
                "last_frame_time": self.last_frame_time_storage
            })
            
            # Write metadata atomically
            metadata_path = self.session_path / "metadata.json"
            temp_path = metadata_path.with_suffix('.tmp')
            
            with open(temp_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            # Atomic rename
            temp_path.replace(metadata_path)
            
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def stop(self):
        """Clean up and finalize storage"""
        logger.info("Stopping storage processor...")
        self.is_running = False
        
        # Cancel idle timer
        if self.idle_timer:
            self.idle_timer.cancel()
            self.idle_timer = None
        
        # Close current chunk
        with self.writer_lock:
            if self.current_writer and self.current_chunk_frame_count > 0:
                logger.info(f"Saving final chunk with {self.current_chunk_frame_count} frames")
                self._close_current_chunk()
        
        # Final metadata update
        end_time = time.time()
        duration = end_time - self.metadata["start_time"]
        
        # Calculate total size
        total_size = sum(chunk.get("size_bytes", 0) for chunk in self.metadata["chunks"])
        
        # Update final metadata
        self.metadata.update({
            "end_time": end_time,
            "duration": duration,
            "total_frames": self.frame_count,
            "total_chunks": self.current_chunk_index,
            "total_size_bytes": total_size,
            "partial_chunks_count": len(self.metadata.get("partial_chunks", []))
        })
        
        self.save_metadata()
        
        # Publish session complete event
        self._publish_storage_event("session_complete", {
            "session_id": self.session_id,
            "duration": duration,
            "total_frames": self.frame_count,
            "total_chunks": self.current_chunk_index,
            "partial_chunks": len(self.metadata.get("partial_chunks", [])),
            "total_size_mb": total_size / 1024 / 1024
        })
        
        # Clean up async resources
        if self.async_loop and not self.async_loop.is_closed():
            if self.rabbit_channel:
                self.async_loop.run_until_complete(self.rabbit_channel.close())
            if self.rabbit_connection:
                self.async_loop.run_until_complete(self.rabbit_connection.close())
            self.async_loop.close()
        
        # Call parent stop
        super().stop()
        
        partial_info = f", {len(self.metadata.get('partial_chunks', []))} partial" if self.metadata.get('partial_chunks') else ""
        logger.info(f"Storage session complete: {self.session_id}")
        logger.info(
            f"Total: {self.frame_count} frames, {self.current_chunk_index} chunks{partial_info}, "
            f"{total_size/1024/1024:.1f}MB, {duration:.1f}s"
        )


def main():
    """Main entry point"""
    # Start Prometheus metrics server
    metrics_port = int(os.getenv('METRICS_PORT', 8005))
    start_http_server(metrics_port)
    logger.info(f"Prometheus metrics server started on port {metrics_port}")
    
    # WebSocket URL
    ws_url = os.getenv('VIDEO_STREAM_URL', 'ws://server:5001/ws/video/consume')
    
    logger.info(f"Connecting to WebSocket stream at {ws_url}")
    
    # Create and run processor
    processor = WebSocketStorageProcessor(ws_url)
    
    try:
        processor.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        processor.stop()


if __name__ == "__main__":
    main()