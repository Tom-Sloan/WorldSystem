"""
Main entry point for the refactored frame processor service.

This module sets up the RabbitMQ consumer, metrics server, and
orchestrates the frame processing pipeline with async support.
"""

import asyncio
import signal
import sys
import json
import time
import cv2
import numpy as np
import aio_pika
from aio_pika import Message, ExchangeType
from prometheus_client import start_http_server, Counter, Gauge, Histogram
from typing import Optional
import psutil
import torch
import os
import threading
import queue
from datetime import datetime

from core.config import Config
from core.utils import get_logger, sync_ntp_time, get_ntp_time_ns, async_sync_ntp_time
from core.performance_monitor import get_performance_monitor, DetailedTimer
from core.h264_decoder import H264StreamDecoder
from pipeline.publisher import RabbitMQPublisher
from pipeline.output_manager import LocalOutputManager, MaskData
from pipeline.visualization_output_manager import VisualizationOutputManager
from video.processor import VideoProcessor
from external.lens_identifier import LensIdentifier

# Add common to path for WebSocket consumer
sys.path.append('/app/common')
from websocket_video_consumer import WebSocketVideoConsumer


logger = get_logger(__name__)


class WebSocketVideoAdapter(WebSocketVideoConsumer):
    """
    Adapter class to integrate WebSocket video consumption with the existing
    FrameProcessorService pipeline. This bridges the WebSocketVideoConsumer
    base class with our async processing architecture.
    """
    
    def __init__(self, frame_processor_service, ws_url: str):
        # Initialize with frame skip from config
        frame_skip = int(os.getenv('FRAME_PROCESSOR_SKIP', '1'))
        super().__init__(ws_url, "FrameProcessor", frame_skip)
        self.service = frame_processor_service
        self._frame_timestamps = {}  # Store timestamps for frames
        
    def process_frame(self, frame: np.ndarray, frame_number: int):
        """Process frame from WebSocket - called by base class in sync context"""
        # Store timestamp for this frame
        timestamp = time.time()
        self._frame_timestamps[frame_number] = timestamp
        
        # Run async processing in a thread-safe way
        websocket_id = "websocket_stream"
        
        # Schedule the coroutine to run on the main event loop
        future = asyncio.run_coroutine_threadsafe(
            self.service._process_websocket_frame(
                websocket_id, frame, frame_number, timestamp
            ),
            self.service._event_loop
        )
        
        # Don't wait for result to avoid blocking
        # Log any exceptions that occur
        def handle_result(fut):
            try:
                fut.result()
            except Exception as e:
                logger.error(f"Error in async frame processing: {e}")
        
        future.add_done_callback(handle_result)


# Prometheus metrics
frames_processed = Counter('frame_processor_frames_processed_total', 
                         'Total number of frames processed')
video_tracks = Counter('frame_processor_video_tracks_total', 
                        'Total number of video tracks')
processing_time = Histogram('frame_processor_processing_time_seconds', 
                          'Frame processing time in seconds')
frame_size = Histogram('frame_processor_frame_size_bytes', 
                     'Size of processed frames in bytes')
active_tracks = Gauge('frame_processor_active_tracks', 
                    'Number of active tracks')
api_calls = Counter('frame_processor_api_calls_total',
                  'Total number of API calls made')
connection_status = Gauge('frame_processor_connection_status', 
                        'RabbitMQ connection status (1=connected, 0=disconnected)')
ntp_offset = Gauge('frame_processor_ntp_time_offset_seconds',
                 'NTP time offset in seconds')


class FrameProcessorService:
    """
    Main service class that handles RabbitMQ integration and processing.
    """
    
    def __init__(self):
        """Initialize the frame processor service."""
        self.config = Config()
        self.publisher = None
        self.connection = None
        self.channel = None
        self.running = False
        self.h264_decoder = H264StreamDecoder()
        
        # Video processing components
        self.video_processor = None
        self.lens_identifier = None
        self.lens_batch_processor = None
        self.output_manager = None  # Replaces rerun_client
        
        # WebSocket video consumer
        self.websocket_consumer = None
        self._event_loop = None  # Store event loop for WebSocket adapter
        
        # Initialize performance monitor
        self.monitor = get_performance_monitor()
        self.monitor.start()  # Start the dashboard
        
        logger.info("Initializing Frame Processor Service...")
        logger.info("Video processing mode enabled")
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Start metrics server
        logger.info(f"Starting metrics server on port {self.config.metrics_port}")
        start_http_server(self.config.metrics_port)
        
        # Initial NTP sync will be done in run() method to avoid blocking
        # Set initial offset to 0
        ntp_offset.set(0.0)
        
        # NTP sync task will be started in run() method when event loop is running
        self._ntp_task = None
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        # Connection will be closed in the cleanup section of run()
    
    async def _ntp_sync_task(self):
        """Periodically sync NTP time."""
        while True:
            await asyncio.sleep(60)  # Sync every minute
            try:
                offset = await async_sync_ntp_time(self.config.ntp_server)
                ntp_offset.set(offset)
            except Exception as e:
                logger.error(f"NTP sync failed: {e}")
    
    async def connect_rabbitmq(self):
        """Connect to RabbitMQ and setup exchanges/queues."""
        try:
            logger.info(f"Connecting to RabbitMQ at {self.config.rabbitmq_url}")
            
            # Create connection
            self.connection = await aio_pika.connect_robust(
                self.config.rabbitmq_url,
                heartbeat=3600,
                reconnect_interval=5.0
            )
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=1)
            
            # Declare exchanges
            # Note: video_stream_exchange removed - using WebSocket for video input
            
            self.analysis_mode_exchange = await self.channel.declare_exchange(
                self.config.analysis_mode_exchange,
                ExchangeType.FANOUT,
                durable=True
            )
            
            # Declare api_results_exchange for publishing API results
            self.api_results_exchange = await self.channel.declare_exchange(
                "api_results_exchange",
                ExchangeType.FANOUT,
                durable=True
            )
            
            # Declare queues
            # Note: stream_queue removed - using WebSocket for video input
            
            self.mode_queue = await self.channel.declare_queue(
                'frame_processor_mode_queue',
                durable=True
            )
            
            # Bind mode queue for analysis updates
            await self.mode_queue.bind(
                self.analysis_mode_exchange
            )
            
            # Update metric
            connection_status.set(1)
            
            logger.info("Connected to RabbitMQ successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            connection_status.set(0)
            raise
    
    
    # NOTE: This method is no longer used - kept for reference
    # Video streams now come through WebSocket instead of RabbitMQ
    '''async def process_stream_message(self, message: aio_pika.IncomingMessage):
        """
        Process incoming H.264 video stream chunks.
        
        This is called by RabbitMQ for each stream chunk.
        """
        try:
            async with message.process():
                with DetailedTimer("stream_chunk_processing"):
                    # Extract headers
                    headers = message.headers or {}
                    websocket_id = headers.get('websocket_id', 'unknown')
                    timestamp_ns = int(headers.get('timestamp_ns', get_ntp_time_ns()))
                    
                    # Process H.264 chunk
                    decode_start = time.time()
                    frames = await self.h264_decoder.process_stream_chunk(websocket_id, message.body)
                    decode_time = (time.time() - decode_start) * 1000
                    
                    if len(frames) > 0:
                        logger.info(f"Decoded {len(frames)} frames in {decode_time:.1f}ms from H.264 chunk")
                    
                    # Process each decoded frame
                    frame_count = 0
                    for frame in frames:
                        frame_count += 1
                        if frame is None:
                            continue
                            
                        # Process frame with video processor
                        if self.video_processor:
                            # Use video processor for stream-aware processing
                            frame_start = time.time()
                            video_result = await self.video_processor.process_stream_frame(
                                websocket_id, frame, timestamp_ns
                            )
                            frame_time = (time.time() - frame_start) * 1000
                            
                            if frame_count == 1 or frame_count % 10 == 0:
                                logger.info(f"Frame {frame_count}: {frame_time:.1f}ms, {video_result.tracking_result.object_count} objects")
                            
                            # Update metrics
                            frames_processed.inc()
                            processing_time.observe(video_result.processing_time_ms / 1000.0)
                            active_tracks.set(video_result.tracking_result.object_count)
                            
                            # Update performance monitor
                            self.monitor.update_metric('frames_processed', frames_processed._value.get())
                            self.monitor.update_metric('fps', video_result.fps)
                            self.monitor.update_metric('active_tracks', video_result.tracking_result.object_count)
                            self.monitor.update_metric('detections_per_frame', len(video_result.tracking_result.masks))
                            
                            # Update memory metrics
                            try:
                                # CPU memory
                                process = psutil.Process()
                                memory_info = process.memory_info()
                                self.monitor.update_metric('memory_mb', memory_info.rss / (1024 * 1024))
                                
                                # GPU memory if available
                                if torch.cuda.is_available():
                                    gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                                    self.monitor.update_metric('gpu_memory_mb', gpu_memory)
                            except Exception as e:
                                logger.debug(f"Failed to update memory metrics: {e}")
                            
                            # Process video tracking result
                            await self._process_video_result(video_result, frame, timestamp_ns, websocket_id)
                            
                            # Record frame breakdown for performance monitoring
                            frame_breakdown = {
                                'total': video_result.processing_time_ms,
                                'sam2_tracking': 0,  # Will be filled from timing data
                                'enhancement': 0,
                                'api_calls': 0,
                                'other': 0
                            }
                            
                            # Get timing data from monitor
                            if hasattr(self.monitor, 'timings'):
                                with self.monitor._timing_lock:
                                    if 'sam2_tracking' in self.monitor.timings:
                                        frame_breakdown['sam2_tracking'] = self.monitor.timings['sam2_tracking'].recent[-1] if self.monitor.timings['sam2_tracking'].recent else 0
                                    if 'object_enhancement_total' in self.monitor.timings:
                                        frame_breakdown['enhancement'] = self.monitor.timings['object_enhancement_total'].recent[-1] if self.monitor.timings['object_enhancement_total'].recent else 0
                                    if 'lens_api_call' in self.monitor.timings:
                                        frame_breakdown['api_calls'] = self.monitor.timings['lens_api_call'].recent[-1] if self.monitor.timings['lens_api_call'].recent else 0
                            
                            # Calculate other time
                            accounted_time = frame_breakdown['sam2_tracking'] + frame_breakdown['enhancement'] + frame_breakdown['api_calls']
                            frame_breakdown['other'] = max(0, frame_breakdown['total'] - accounted_time)
                            
                            # Record breakdown
                            self.monitor.record_frame_breakdown(frame_breakdown)
                            
                            # Handle identifications asynchronously
                            asyncio.create_task(
                                self._process_identifications(websocket_id)
                            )
                        else:
                            logger.error("Video processor not initialized")
                            continue
                            logger.debug(f"Analysis mode is 'none', passing through streamed frame")
                            frames_processed.inc()
                    
        except Exception as e:
            self.monitor.add_event(f"Stream processing error: {e}", "error")
            logger.error(f"Error processing video stream: {e}", exc_info=True)
            # Message will be rejected/requeued by the context manager
    '''
    
    async def process_mode_message(self, message: aio_pika.IncomingMessage):
        """Process analysis mode update message - kept for compatibility."""
        try:
            async with message.process():
                # Parse mode update
                mode_data = json.loads(message.body)
                new_mode = mode_data.get('mode', 'on')
                
                logger.info(f"Received mode update: {new_mode} (video processing always active)")
                
        except Exception as e:
            logger.error(f"Error processing mode update: {e}")
    
    async def _process_video_result(self, result, frame: np.ndarray, timestamp_ns: int, websocket_id: str):
        """Process results from video tracking."""
        # Update metrics
        frames_processed.inc()
        processing_time.observe(result.processing_time_ms / 1000.0)
        active_tracks.set(result.tracking_result.object_count)
        
        # Log FPS periodically
        if result.tracking_result.frame_number % 30 == 0:
            logger.info(f"Video processing FPS: {result.fps:.1f}")
        
        # Save outputs locally
        if self.output_manager:
            logger.info(f"Output manager type: {type(self.output_manager).__name__}")
            # Convert tracking results to MaskData format
            mask_data_list = []
            for mask_info in result.tracking_result.masks:
                if 'segmentation' in mask_info and mask_info['segmentation'] is not None:
                    seg = mask_info['segmentation']
                    if seg.sum() > 0:
                        # Get bounding box from segmentation
                        y_coords, x_coords = np.where(seg)
                        bbox = [
                            int(x_coords.min()),
                            int(y_coords.min()),
                            int(x_coords.max()),
                            int(y_coords.max())
                        ]
                        
                        mask_data = MaskData(
                            instance_id=mask_info.get('object_id', 0),
                            mask=seg,
                            class_name=f"Object_{mask_info.get('object_id', 0)}",
                            bbox=bbox,
                            confidence=mask_info.get('confidence', 0.9),
                            area=int(seg.sum())
                        )
                        mask_data_list.append(mask_data)
            
            logger.info(f"Converted {len(mask_data_list)} masks for frame {result.tracking_result.frame_number}")
            
            # Save/send frame outputs
            if isinstance(self.output_manager, VisualizationOutputManager):
                logger.info("Using VisualizationOutputManager")
                # Send visualization update with processing stats
                processing_stats = {
                    'processing_time_ms': result.processing_time_ms,
                    'fps': self.monitor.metrics.get('fps', 0.0)
                }
                await self.output_manager.send_visualization_update(
                    frame=frame,
                    masks=mask_data_list,
                    frame_number=result.tracking_result.frame_number,
                    timestamp=timestamp_ns / 1e9 if timestamp_ns else None,
                    processing_stats=processing_stats
                )
            else:
                # Save to disk
                self.output_manager.save_frame_outputs(
                    frame=frame,
                    masks=mask_data_list,
                    frame_number=result.tracking_result.frame_number,
                    timestamp=timestamp_ns / 1e9 if timestamp_ns else None
                )
        
        # The visualization_output_manager already publishes everything needed
        # No need to duplicate by calling publisher.publish_processed_frame
        # The visualization update includes:
        # - Original frame
        # - Visualization with colored masks  
        # - Individual masks
        # - Enhanced crops
        # - All metadata
    
    async def _process_identifications(self, stream_id: str):
        """Process pending object identifications."""
        if not self.video_processor or not self.lens_batch_processor:
            return
            
        try:
            # Get objects ready for identification
            pending = await self.video_processor.get_pending_identifications()
            
            if pending:
                # Add items to batch processor
                await self.lens_batch_processor.add_items(pending)
                
                # Note: Results will be published automatically by the batch processor
                        
        except Exception as e:
            logger.error(f"Error processing identifications: {e}", exc_info=True)
    
    def _draw_video_annotations(self, frame: np.ndarray, tracks) -> np.ndarray:
        """Draw bounding boxes and track IDs on frame."""
        annotated = frame.copy()
        
        for track in tracks:
            bbox = track.get('bbox', [])
            if len(bbox) != 4:
                continue
                
            x1, y1, x2, y2 = bbox
            track_id = track.get('object_id', 'unknown')
            confidence = track.get('confidence', 0.0)
            
            # Draw box with track-specific color
            color = self._get_track_color(track_id)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"ID:{track_id} {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Background for text
            cv2.rectangle(annotated, 
                         (x1, y1 - label_size[1] - 4),
                         (x1 + label_size[0], y1),
                         color, -1)
            
            # Text
            cv2.putText(annotated, label,
                       (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 1)
        
        return annotated
    
    def _get_track_color(self, track_id):
        """Get consistent color for a track ID."""
        # Simple hash-based color generation
        if isinstance(track_id, str):
            hash_val = hash(track_id)
        else:
            hash_val = int(track_id)
        
        # Generate color from hash
        r = (hash_val * 123) % 256
        g = (hash_val * 456) % 256
        b = (hash_val * 789) % 256
        
        return (b, g, r)  # BGR for OpenCV
    
    async def _process_websocket_frame(self, websocket_id: str, frame: np.ndarray, 
                                     frame_number: int, timestamp: float):
        """
        Process a frame received from WebSocket.
        This replaces the RabbitMQ stream processing for video frames.
        """
        try:
            with DetailedTimer("websocket_frame_processing"):
                # Convert timestamp to nanoseconds for consistency
                timestamp_ns = int(timestamp * 1e9)
                
                # Process frame with video processor
                if self.video_processor:
                    frame_start = time.time()
                    video_result = await self.video_processor.process_stream_frame(
                        websocket_id, frame, timestamp_ns
                    )
                    frame_time = (time.time() - frame_start) * 1000
                    
                    if frame_number % 10 == 0:
                        logger.info(f"Frame {frame_number}: {frame_time:.1f}ms, "
                                  f"{video_result.tracking_result.object_count} objects")
                    
                    # Update metrics
                    frames_processed.inc()
                    processing_time.observe(video_result.processing_time_ms / 1000.0)
                    active_tracks.set(video_result.tracking_result.object_count)
                    
                    # Update performance monitor
                    self.monitor.update_metric('frames_processed', frames_processed._value.get())
                    self.monitor.update_metric('fps', video_result.fps)
                    self.monitor.update_metric('active_tracks', video_result.tracking_result.object_count)
                    self.monitor.update_metric('detections_per_frame', len(video_result.tracking_result.masks))
                    
                    # Update memory metrics
                    try:
                        # CPU memory
                        process = psutil.Process()
                        memory_info = process.memory_info()
                        self.monitor.update_metric('memory_mb', memory_info.rss / (1024 * 1024))
                        
                        # GPU memory if available
                        if torch.cuda.is_available():
                            gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                            self.monitor.update_metric('gpu_memory_mb', gpu_memory)
                    except Exception as e:
                        logger.debug(f"Failed to update memory metrics: {e}")
                    
                    # Process video tracking result
                    await self._process_video_result(video_result, frame, timestamp_ns, websocket_id)
                    
        except Exception as e:
            logger.error(f"Error processing WebSocket frame: {e}")
            import traceback
            traceback.print_exc()
    
    # Removed _run_websocket_consumer method - no longer needed
    
    async def run(self):
        """Main run loop."""
        self.running = True
        self._event_loop = asyncio.get_event_loop()  # Store event loop
        
        try:
            # Update component status
            self.monitor.update_component_status('rabbitmq', status='🔄 Connecting...')
            
            # Do initial NTP sync now that event loop is running
            initial_offset = await async_sync_ntp_time(self.config.ntp_server)
            ntp_offset.set(initial_offset)
            logger.info(f"Initial NTP sync complete, offset: {initial_offset:.3f}s")
            self.monitor.add_event("NTP sync complete", "success")
            
            # Start NTP sync task now that event loop is running
            self._ntp_task = asyncio.create_task(self._ntp_sync_task())
            
            # Initialize components
            self.publisher = RabbitMQPublisher(self.config)
            
            # Initialize video processing components
            logger.info("Initializing video processing components...")
            self.video_processor = VideoProcessor(self.config)
            
            # Initialize output manager based on environment
            use_visualization = os.getenv('USE_VISUALIZATION_OUTPUT', 'true').lower() == 'true'
            session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if use_visualization:
                # Use visualization output manager for real-time Rerun visualization
                self.output_manager = VisualizationOutputManager(session_id=session_id)
                logger.info("Visualization output manager initialized for real-time streaming")
            else:
                # Use local output manager for file-based output
                self.output_manager = LocalOutputManager(
                    output_dir=os.getenv('OUTPUT_DIR', '/app/outputs'),
                    session_id=session_id
                )
                logger.info("Local output manager initialized for file-based output")
            
            # Initialize API client for lens identifier
            from external.api_client import APIClient
            self.api_client = APIClient(self.config)
            self.api_client.load_caches()
            
            # Initialize Lens identifier if API is enabled
            if self.config.use_serpapi:
                try:
                    from external.lens_identifier import LensIdentifier
                    from external.lens_batch_processor import LensBatchProcessor
                    
                    # Initialize with API client
                    self.lens_identifier = LensIdentifier(self.config, self.api_client)
                    
                    # Initialize batch processor with publisher
                    self.lens_batch_processor = LensBatchProcessor(
                        self.config, 
                        self.lens_identifier,
                        publisher=self.publisher
                    )
                    
                    # Start batch processor
                    await self.lens_batch_processor.start()
                    
                    logger.info("Google Lens integration initialized with batch processing")
                except Exception as e:
                    logger.warning(f"Failed to initialize Lens identifier: {e}")
                    self.lens_identifier = None
                    self.lens_batch_processor = None
            
            # Update status
            self.monitor.update_component_status(
                'sam2',
                name='SAM2 Video',
                status='✅ Ready'
            )
            
            # Update enhancement status
            if hasattr(self.config, 'enhancement_enabled') and self.config.enhancement_enabled:
                self.monitor.update_component_status('enhancement', status='✅ Enabled')
            else:
                self.monitor.update_component_status('enhancement', status='⏸️ Disabled')
            
            # Connect to RabbitMQ
            await self.connect_rabbitmq()
            self.monitor.update_component_status('rabbitmq', status='✅ Connected')
            
            # Connect publisher (creates its own connection)
            await self.publisher.connect()
            
            # Update API status
            if self.config.use_serpapi or self.config.use_perplexity:
                self.monitor.update_component_status('api', status='✅ Enabled')
            else:
                self.monitor.update_component_status('api', status='⏸️ Disabled')
            
            # Update output status
            self.monitor.update_component_status('output', name='Local Output', status='✅ Ready')
            
            # Initialize WebSocket video consumer
            # Use VIDEO_STREAM_URL if set, otherwise construct from host/port
            ws_url = os.getenv('VIDEO_STREAM_URL')
            if not ws_url:
                ws_host = os.getenv('WS_HOST', '127.0.0.1')
                ws_port = os.getenv('WS_PORT', '5001')
                ws_url = f"ws://{ws_host}:{ws_port}/ws/video/consume"
            
            logger.info(f"Connecting to WebSocket video stream at {ws_url}")
            self.websocket_consumer = WebSocketVideoAdapter(self, ws_url)
            
            # Start WebSocket consumer in background thread (it runs its own loop)
            websocket_thread = threading.Thread(
                target=self.websocket_consumer.run,
                daemon=True
            )
            websocket_thread.start()
            
            # Setup RabbitMQ consumers (only for mode messages now)
            await self.mode_queue.consume(self.process_mode_message)
            
            logger.info("Frame processor service started with WebSocket video input...")
            
            # Keep the service running
            try:
                await asyncio.Future()  # Run forever until cancelled
            except asyncio.CancelledError:
                logger.info("Service cancelled, shutting down...")
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Fatal error in service: {e}", exc_info=True)
        finally:
            # Cleanup
            logger.info("Shutting down frame processor service...")
            
            # Stop WebSocket consumer
            if self.websocket_consumer:
                logger.info("Stopping WebSocket consumer...")
                self.websocket_consumer.stop()
            
            # Cleanup video processor streams
            if self.video_processor:
                logger.info("Cleaning up video processor streams...")
                for stream_id in list(self.video_processor.active_streams.keys()):
                    await self.video_processor.cleanup_stream(stream_id)
            
            # Cleanup output manager
            if self.output_manager:
                logger.info("Finalizing local outputs...")
                self.output_manager.cleanup()
            
            # Stop batch processor
            if hasattr(self, 'lens_batch_processor') and self.lens_batch_processor:
                logger.info("Stopping lens batch processor...")
                await self.lens_batch_processor.stop()
            
            # Cancel NTP sync task
            if self._ntp_task and not self._ntp_task.done():
                self._ntp_task.cancel()
                try:
                    await self._ntp_task
                except asyncio.CancelledError:
                    pass
            
            if hasattr(self, 'api_client') and self.api_client:
                if hasattr(self.api_client, 'close'):
                    await self.api_client.close()
            
            if self.h264_decoder:
                self.h264_decoder.cleanup_all()
            
            if self.publisher:
                await self.publisher.close()
            
            if self.connection and not self.connection.is_closed:
                await self.connection.close()
            
            # Stop monitor on shutdown
            self.monitor.stop()
            
            logger.info("Frame processor service stopped")


async def main():
    """Main entry point."""
    service = FrameProcessorService()
    await service.run()


if __name__ == "__main__":
    # Setup logging
    from core.utils import setup_logging
    from pathlib import Path
    
    # Initialize logging based on config
    config = Config()
    setup_logging(
        log_level=config.log_level,
        log_dir=config.log_dir if config.log_file else None
    )
    
    # Run the async main function
    asyncio.run(main())