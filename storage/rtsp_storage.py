#!/usr/bin/env python3
"""
Storage Service with RTSP
Records video stream in time-based chunks with dynamic FPS support
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
from threading import Lock

# Add common to path
sys.path.append('/app/common')
from rtsp_consumer import RTSPConsumer

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
    "Actual FPS being received from RTSP stream"
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RTSPStorageProcessor(RTSPConsumer):
    def __init__(self, rtsp_url: str):
        # Storage: record every frame (no skipping)
        super().__init__(rtsp_url, "Storage", frame_skip=1)
        
        # Storage configuration
        self.storage_path = os.getenv('STORAGE_PATH', '/app/recordings')
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_path = Path(self.storage_path) / self.session_id
        
        # Create directory structure
        self.session_path.mkdir(parents=True, exist_ok=True)
        self.video_segments_path = self.session_path / "video_segments"
        self.video_segments_path.mkdir(exist_ok=True)
        
        # Video chunking configuration
        self.chunk_duration_seconds = int(os.getenv('VIDEO_CHUNK_DURATION_SECONDS', '60'))  # Default 1 minute chunks
        self.video_codec = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Video writer state
        self.current_writer = None
        self.current_chunk_index = 0
        self.current_chunk_start_time = None
        self.current_chunk_frame_count = 0
        self.writer_lock = Lock()
        
        # FPS tracking
        self.fps_samples = []  # Recent frame timestamps for FPS calculation
        self.fps_sample_window = 5.0  # Calculate FPS over 5 second windows
        self.estimated_fps = 30.0  # Initial estimate
        
        # Session metadata
        self.metadata = {
            "session_id": self.session_id,
            "start_time": time.time(),
            "rtsp_url": rtsp_url,
            "chunk_duration_seconds": self.chunk_duration_seconds,
            "chunks": []
        }
        
        # RabbitMQ setup for publishing storage events
        self.setup_rabbitmq()
        
        logger.info(f"Storage session initialized: {self.session_id}")
        logger.info(f"Storage path: {self.session_path}")
        logger.info(f"Video chunk duration: {self.chunk_duration_seconds} seconds")
    
    def setup_rabbitmq(self):
        """Setup RabbitMQ connections for publishing storage events"""
        try:
            async def _async_setup():
                self.rabbit_connection = await aio_pika.connect_robust(
                    os.getenv('RABBITMQ_URL', 'amqp://127.0.0.1:5672')
                )
                self.rabbit_channel = await self.rabbit_connection.channel()
                
                # Declare storage events exchange
                self.storage_exchange = await self.rabbit_channel.declare_exchange(
                    'storage_events_exchange',
                    aio_pika.ExchangeType.FANOUT,
                    durable=True
                )
            
            # Run the async setup
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_async_setup())
            finally:
                self.async_loop = loop
                
            logger.info("RabbitMQ setup complete")
        except Exception as e:
            logger.warning(f"Failed to setup RabbitMQ: {e}. Storage will work without event publishing.")
            self.rabbit_connection = None
            self.async_loop = None
    
    def calculate_fps(self):
        """Calculate actual FPS from recent frame timestamps"""
        current_time = time.time()
        
        # Add current timestamp
        self.fps_samples.append(current_time)
        
        # Remove old samples outside the window
        cutoff_time = current_time - self.fps_sample_window
        self.fps_samples = [t for t in self.fps_samples if t > cutoff_time]
        
        # Calculate FPS if we have enough samples
        if len(self.fps_samples) >= 2:
            time_span = self.fps_samples[-1] - self.fps_samples[0]
            if time_span > 0:
                fps = (len(self.fps_samples) - 1) / time_span
                self.estimated_fps = fps
                actual_fps_gauge.set(fps)
                
                # Log significant FPS changes
                if abs(fps - self.estimated_fps) > 5:
                    logger.info(f"FPS changed significantly: {self.estimated_fps:.1f} -> {fps:.1f}")
    
    def should_start_new_chunk(self):
        """Check if it's time to start a new video chunk"""
        if self.current_chunk_start_time is None:
            return True
        
        elapsed = time.time() - self.current_chunk_start_time
        return elapsed >= self.chunk_duration_seconds
    
    def close_current_chunk(self):
        """Close the current video chunk and update metadata"""
        with self.writer_lock:
            if self.current_writer:
                self.current_writer.release()
                
                # Calculate chunk duration
                chunk_duration = time.time() - self.current_chunk_start_time
                
                # Get file size
                chunk_path = self.video_segments_path / f"chunk_{self.current_chunk_index:04d}.mp4"
                try:
                    chunk_size = os.path.getsize(chunk_path)
                except:
                    chunk_size = 0
                
                # Update metadata
                chunk_info = {
                    "index": self.current_chunk_index,
                    "filename": chunk_path.name,
                    "start_time": self.current_chunk_start_time,
                    "duration": chunk_duration,
                    "frame_count": self.current_chunk_frame_count,
                    "average_fps": self.current_chunk_frame_count / chunk_duration if chunk_duration > 0 else 0,
                    "size_bytes": chunk_size
                }
                self.metadata["chunks"].append(chunk_info)
                
                # Update metrics
                video_chunks_saved_counter.inc()
                video_chunk_duration_hist.observe(chunk_duration)
                
                # Publish event
                self._publish_storage_event("chunk_saved", {
                    "session_id": self.session_id,
                    "chunk_info": chunk_info
                })
                
                logger.info(f"Saved video chunk {self.current_chunk_index}: "
                          f"{chunk_duration:.1f}s, {self.current_chunk_frame_count} frames, "
                          f"{chunk_size/1024/1024:.1f}MB")
                
                self.current_writer = None
    
    def start_new_chunk(self, frame_shape):
        """Start a new video chunk"""
        with self.writer_lock:
            # Close previous chunk if exists
            if self.current_writer:
                self.close_current_chunk()
            
            # Increment chunk index
            self.current_chunk_index += 1
            self.current_chunk_start_time = time.time()
            self.current_chunk_frame_count = 0
            
            # Create new writer
            chunk_path = self.video_segments_path / f"chunk_{self.current_chunk_index:04d}.mp4"
            height, width = frame_shape[:2]
            
            # Use estimated FPS for video writer
            fps_to_use = max(10.0, min(60.0, self.estimated_fps))  # Clamp between 10-60 FPS
            
            self.current_writer = cv2.VideoWriter(
                str(chunk_path),
                self.video_codec,
                fps_to_use,
                (width, height)
            )
            
            logger.info(f"Started new video chunk {self.current_chunk_index} "
                      f"({width}x{height} @ {fps_to_use:.1f} FPS)")
    
    def process_frame(self, frame: np.ndarray, frame_number: int):
        """Store frame in current video chunk"""
        # Calculate actual FPS
        self.calculate_fps()
        
        # Check if we need a new chunk
        if self.should_start_new_chunk():
            self.start_new_chunk(frame.shape)
        
        # Write frame to current chunk
        with self.writer_lock:
            if self.current_writer:
                self.current_writer.write(frame)
                self.current_chunk_frame_count += 1
                video_frames_written_counter.inc()
                
                # Update chunk size metric periodically
                if self.current_chunk_frame_count % 30 == 0:
                    chunk_path = self.video_segments_path / f"chunk_{self.current_chunk_index:04d}.mp4"
                    try:
                        size = os.path.getsize(chunk_path)
                        current_chunk_size.set(size)
                    except:
                        pass
        
        # Save metadata periodically
        if frame_number % 300 == 0:  # Every ~10 seconds at 30fps
            self.save_metadata()
    
    def _publish_storage_event(self, event_type: str, data: dict):
        """Publish storage event to RabbitMQ"""
        if not self.rabbit_connection or not self.async_loop:
            return
            
        try:
            message = {
                "type": event_type,
                "timestamp": time.time(),
                "data": data
            }
            
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
        """Save session metadata"""
        try:
            metadata_path = self.session_path / "metadata.json"
            
            # Update metadata
            self.metadata["last_updated"] = time.time()
            self.metadata["total_frames"] = self.frame_count
            self.metadata["estimated_fps"] = self.estimated_fps
            
            # Write metadata
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def cleanup(self):
        """Clean up async resources"""
        if hasattr(self, 'async_loop') and self.async_loop:
            async def _async_cleanup():
                try:
                    if hasattr(self, 'rabbit_channel') and self.rabbit_channel:
                        await self.rabbit_channel.close()
                    if hasattr(self, 'rabbit_connection') and self.rabbit_connection:
                        await self.rabbit_connection.close()
                except Exception as e:
                    logger.error(f"Error during cleanup: {e}")
            
            self.async_loop.run_until_complete(_async_cleanup())
            self.async_loop.close()
    
    def stop(self):
        """Clean up and finalize storage"""
        logger.info("Stopping storage processor...")
        
        # Close current chunk
        self.close_current_chunk()
        
        # Final metadata save
        self.metadata["end_time"] = time.time()
        self.metadata["duration"] = self.metadata["end_time"] - self.metadata["start_time"]
        self.metadata["total_frames"] = self.frame_count
        self.metadata["total_chunks"] = self.current_chunk_index
        
        # Calculate total size
        total_size = 0
        for chunk in self.metadata["chunks"]:
            total_size += chunk.get("size_bytes", 0)
        self.metadata["total_size_bytes"] = total_size
        
        self.save_metadata()
        
        # Publish session complete event
        if self.rabbit_connection:
            self._publish_storage_event("session_complete", {
                "session_id": self.session_id,
                "duration": self.metadata["duration"],
                "total_frames": self.metadata["total_frames"],
                "total_chunks": self.metadata["total_chunks"],
                "total_size_mb": total_size / 1024 / 1024
            })
        
        # Clean up async resources
        self.cleanup()
        
        # Call parent stop
        super().stop()
        
        logger.info(f"Storage session complete: {self.session_id}")
        logger.info(f"Total: {self.frame_count} frames, {self.current_chunk_index} chunks, "
                   f"{total_size/1024/1024:.1f}MB, {self.metadata['duration']:.1f}s")


def main():
    """Main entry point"""
    # Start Prometheus metrics server
    metrics_port = int(os.getenv('METRICS_PORT', 8005))
    start_http_server(metrics_port)
    logger.info(f"Prometheus metrics server started on port {metrics_port}")
    
    # RTSP URL
    rtsp_host = os.getenv('RTSP_HOST', '127.0.0.1')
    rtsp_port = os.getenv('RTSP_PORT', 8554)
    rtsp_url = f"rtsp://{rtsp_host}:{rtsp_port}/drone"
    
    logger.info(f"Connecting to RTSP stream at {rtsp_url}")
    
    # Create processor
    processor = RTSPStorageProcessor(rtsp_url)
    
    try:
        # Run processing loop
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