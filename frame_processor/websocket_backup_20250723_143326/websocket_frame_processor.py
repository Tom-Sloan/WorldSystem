#!/usr/bin/env python3
"""
WebSocket Frame Processor with Rerun Video Streaming
Consumes H.264 video stream via WebSocket and performs object tracking with SAM2
Includes Rerun video streaming visualization for testing
"""

import os
import sys
import cv2
import numpy as np
import time
import asyncio
import aio_pika
import json
import logging
import av
import rerun as rr
from typing import Optional, Dict, Any

# Add common to path
sys.path.append('/app/common')
from websocket_video_consumer import WebSocketVideoConsumer

# Add frame processor modules
from core.config import Config
from core.utils import get_logger, sync_ntp_time
from video.processor import VideoProcessor
from pipeline.publisher import RabbitMQPublisher
from external.lens_identifier import LensIdentifier

# Prometheus metrics
from prometheus_client import start_http_server, Counter, Gauge, Histogram

# Define metrics
frames_processed = Counter('frame_processor_frames_processed_total', 
                         'Total number of frames processed')
video_tracks = Counter('frame_processor_video_tracks_total', 
                        'Total number of video tracks')
processing_time = Histogram('frame_processor_processing_time_seconds', 
                          'Frame processing time in seconds')
active_tracks = Gauge('frame_processor_active_tracks', 
                    'Number of active tracks')
api_calls = Counter('frame_processor_api_calls_total',
                  'Total number of API calls made')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSocketFrameProcessor(WebSocketVideoConsumer):
    """Frame processor that consumes WebSocket H.264 stream with Rerun visualization"""
    
    def __init__(self, ws_url: str, config: Config):
        # Process every frame by default (frame_skip=1)
        frame_skip = int(os.getenv('FRAME_PROCESSOR_SKIP', '1'))
        super().__init__(ws_url, "FrameProcessor", frame_skip)
        
        self.config = config
        self.video_processor: Optional[VideoProcessor] = None
        self.publisher: Optional[RabbitMQPublisher] = None
        self.lens_identifier: Optional[LensIdentifier] = None
        
        # Rerun setup
        self.rerun_initialized = False
        self.av_encoder = None
        self.av_stream = None
        self.encoded_packets = []
        
        # Stream tracking
        self.stream_id = "websocket_stream"
        self.first_frame = True
        self.frame_width = None
        self.frame_height = None
        
        # Performance tracking
        self.processing_times = []
        
    async def initialize_services(self):
        """Initialize processing services"""
        try:
            # Sync NTP time
            logger.info("[NTP] Synchronizing time...")
            sync_ntp_time()
            
            # Initialize video processor
            self.video_processor = VideoProcessor(self.config)
            await self.video_processor.initialize()
            
            # Initialize publisher
            self.publisher = RabbitMQPublisher(self.config)
            await self.publisher.connect()
            
            # Initialize API client if enabled
            if self.config.use_serpapi or self.config.use_perplexity:
                self.lens_identifier = LensIdentifier(self.config)
                
            # Initialize Rerun
            self.initialize_rerun()
            
            logger.info("All services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise
            
    def initialize_rerun(self):
        """Initialize Rerun for video streaming visualization"""
        try:
            # Initialize Rerun
            rr.init("frame_processor_websocket", spawn=False)
            
            # Try to connect to Rerun viewer
            try:
                rr.connect("127.0.0.1:9876")
                logger.info("Connected to Rerun viewer at localhost:9876")
            except Exception as e:
                logger.warning(f"Could not connect to Rerun viewer: {e}")
                logger.info("Starting Rerun in spawn mode")
                rr.spawn()
            
            # Initialize video stream with H.264 codec (static=True for one-time setup)
            rr.log("video/h264_stream", rr.VideoStream(codec=rr.VideoCodec.H264), static=True)
            
            self.rerun_initialized = True
            logger.info("Rerun video streaming initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Rerun: {e}")
            self.rerun_initialized = False
            
    def initialize_h264_encoder(self, width: int, height: int):
        """Initialize PyAV H.264 encoder for Rerun streaming"""
        try:
            # Create encoder for H.264
            codec = av.CodecContext.create('libx264', 'w')
            codec.width = width
            codec.height = height
            codec.pix_fmt = 'yuv420p'
            codec.time_base = av.Fraction(1, 30)  # 30 FPS
            codec.framerate = av.Fraction(30, 1)
            codec.options = {
                'preset': 'ultrafast',
                'tune': 'zerolatency',
                'crf': '23',
            }
            codec.gop_size = 30  # Keyframe every 30 frames
            codec.max_b_frames = 0  # No B-frames for Rerun compatibility
            
            self.av_encoder = codec
            logger.info(f"Initialized H.264 encoder: {width}x{height}")
            
        except Exception as e:
            logger.error(f"Failed to initialize H.264 encoder: {e}")
            
    def encode_frame_to_h264(self, frame: np.ndarray, pts: int) -> bytes:
        """Encode a frame to H.264 and return the encoded data"""
        if self.av_encoder is None:
            return None
            
        try:
            # Convert BGR to YUV420p
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            av_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
            av_frame = av_frame.reformat(format='yuv420p')
            av_frame.pts = pts
            av_frame.time_base = self.av_encoder.time_base
            
            # Encode the frame
            packets = self.av_encoder.encode(av_frame)
            
            # Collect encoded data
            encoded_data = b''
            for packet in packets:
                encoded_data += bytes(packet)
                
            return encoded_data
            
        except Exception as e:
            logger.error(f"Error encoding frame: {e}")
            return None
            
    def process_frame(self, frame: np.ndarray, frame_number: int):
        """Process a single frame through the pipeline with Rerun visualization"""
        start_time = time.time()
        
        try:
            # Initialize encoder on first frame
            if self.first_frame:
                self.frame_height, self.frame_width = frame.shape[:2]
                self.initialize_h264_encoder(self.frame_width, self.frame_height)
                self.first_frame = False
                logger.info(f"First frame received: {self.frame_width}x{self.frame_height}")
            
            # Log to Rerun - Original decoded frame
            if self.rerun_initialized:
                rr.set_time("frame", frame_number)
                rr.log("video/decoded_frame", rr.Image(frame))
                
                # Also encode and log as H.264 stream
                if self.av_encoder:
                    h264_data = self.encode_frame_to_h264(frame, frame_number)
                    if h264_data:
                        # Set time context for H.264 stream
                        timestamp = float(frame_number) / 30.0  # Assuming 30 FPS
                        rr.set_time("time", timestamp)
                        
                        # Log H.264 packet
                        rr.log(
                            "video/h264_stream",
                            rr.VideoStream.from_fields(sample=h264_data)
                        )
            
            # Process through video tracking pipeline
            if self.video_processor:
                # Run async processing in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                result = loop.run_until_complete(
                    self.video_processor.process_frame(
                        frame,
                        self.stream_id,
                        frame_number,
                        timestamp=time.time()
                    )
                )
                
                # Publish results if we have detections
                if result and self.publisher:
                    loop.run_until_complete(
                        self.publisher.publish_frame_result(result)
                    )
                    
                    # Update metrics
                    if result.get("objects"):
                        active_tracks.set(len(result["objects"]))
                        video_tracks.inc(len(result["objects"]))
                    
                    # Log tracking results to Rerun
                    if self.rerun_initialized and result.get("objects"):
                        self.log_tracking_to_rerun(frame, result)
                
                loop.close()
            
            # Update metrics
            frames_processed.inc()
            process_time = time.time() - start_time
            processing_time.observe(process_time)
            self.processing_times.append(process_time)
            
            # Log performance every 100 frames
            if frame_number % 100 == 0:
                avg_time = sum(self.processing_times[-100:]) / min(100, len(self.processing_times))
                logger.info(f"Processed {frame_number} frames, avg processing time: {avg_time:.3f}s")
                
        except Exception as e:
            logger.error(f"Error processing frame {frame_number}: {e}")
            import traceback
            traceback.print_exc()
            
    def log_tracking_to_rerun(self, frame: np.ndarray, result: Dict[str, Any]):
        """Log tracking results to Rerun with visualization"""
        try:
            # Create visualization frame
            vis_frame = frame.copy()
            
            # Draw all tracked objects
            for obj in result.get("objects", []):
                if "mask" in obj and obj["mask"] is not None:
                    mask = obj["mask"]
                    track_id = obj.get("track_id", -1)
                    confidence = obj.get("confidence", 0.0)
                    
                    # Create colored overlay
                    color = self.get_track_color(track_id)
                    overlay = np.zeros_like(vis_frame)
                    overlay[mask > 0] = color
                    
                    # Blend with original
                    vis_frame = cv2.addWeighted(vis_frame, 0.7, overlay, 0.3, 0)
                    
                    # Add track info
                    y_indices, x_indices = np.where(mask > 0)
                    if len(x_indices) > 0 and len(y_indices) > 0:
                        cx = int(np.mean(x_indices))
                        cy = int(np.mean(y_indices))
                        
                        # Draw text background
                        text = f"ID:{track_id} ({confidence:.2f})"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        thickness = 2
                        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                        
                        cv2.rectangle(vis_frame, 
                                    (cx - text_width//2 - 5, cy - text_height - 5),
                                    (cx + text_width//2 + 5, cy + 5),
                                    (0, 0, 0), -1)
                        
                        cv2.putText(vis_frame, text, 
                                  (cx - text_width//2, cy),
                                  font, font_scale, (255, 255, 255), thickness)
            
            # Log annotated frame
            rr.log("video/tracked_frame", rr.Image(vis_frame))
            
            # Log individual masks
            for obj in result.get("objects", []):
                if "mask" in obj and obj["mask"] is not None:
                    track_id = obj.get("track_id", -1)
                    rr.log(f"tracks/mask_{track_id}", 
                          rr.SegmentationImage(obj["mask"].astype(np.uint8) * 255))
                    
            # Log tracking statistics
            stats_text = f"Active tracks: {len(result.get('objects', []))}"
            rr.log("tracking/stats", rr.TextLog(stats_text))
                    
        except Exception as e:
            logger.error(f"Error logging tracking to Rerun: {e}")
            
    def get_track_color(self, track_id: int) -> tuple:
        """Get a consistent color for a track ID"""
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (255, 192, 203),# Pink
            (165, 42, 42),  # Brown
        ]
        return colors[track_id % len(colors)]
        
    def stop(self):
        """Stop processing and clean up"""
        super().stop()
        
        # Clean up encoder
        if self.av_encoder:
            self.av_encoder.close()
            
        # Clean up services
        if self.video_processor:
            self.video_processor.cleanup()
            
        # Close publisher
        if self.publisher:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.publisher.close())
            loop.close()


def main():
    """Main entry point"""
    # Load configuration
    config = Config()
    
    # WebSocket URL
    ws_host = os.getenv('WS_HOST', 'server')
    ws_port = os.getenv('WS_PORT', '5001')
    ws_url = f"ws://{ws_host}:{ws_port}/ws/video/consume"
    
    # Start Prometheus metrics server
    metrics_port = int(os.getenv('METRICS_PORT', '8003'))
    start_http_server(metrics_port)
    logger.info(f"Prometheus metrics server started on port {metrics_port}")
    
    # Initialize services in async context
    async def init_services(processor):
        await processor.initialize_services()
    
    # Create processor
    processor = WebSocketFrameProcessor(ws_url, config)
    
    # Initialize services
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(init_services(processor))
    loop.close()
    
    # Set up signal handlers
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        processor.stop()
        sys.exit(0)
        
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Run processing loop
        logger.info(f"Starting WebSocket frame processor, connecting to {ws_url}")
        processor.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        processor.stop()


if __name__ == "__main__":
    main()