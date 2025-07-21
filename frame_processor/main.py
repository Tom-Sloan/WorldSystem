"""
Main entry point for the refactored frame processor service.

This module sets up the RabbitMQ consumer, metrics server, and
orchestrates the frame processing pipeline with async support.
"""

import asyncio
import signal
import sys
import json
import cv2
import numpy as np
import aio_pika
from aio_pika import Message, ExchangeType
from prometheus_client import start_http_server, Counter, Gauge, Histogram
from typing import Optional

from core.config import Config
from core.utils import get_logger, sync_ntp_time, get_ntp_time_ns, async_sync_ntp_time
from core.performance_monitor import get_performance_monitor, DetailedTimer
from core.h264_decoder import H264StreamDecoder
from pipeline.publisher import RabbitMQPublisher
from video.processor import VideoProcessor
from external.lens_identifier import LensIdentifier


logger = get_logger(__name__)


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
        self.rerun_client = None
        
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
            self.video_stream_exchange = await self.channel.declare_exchange(
                self.config.video_stream_exchange,
                ExchangeType.FANOUT,
                durable=True
            )
            
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
            self.stream_queue = await self.channel.declare_queue(
                'frame_processor_stream_queue',
                durable=True
            )
            
            self.mode_queue = await self.channel.declare_queue(
                'frame_processor_mode_queue',
                durable=True
            )
            
            # Bind queues (fanout exchanges don't use routing keys)
            await self.stream_queue.bind(
                self.video_stream_exchange
            )
            
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
    
    
    async def process_stream_message(self, message: aio_pika.IncomingMessage):
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
                    frames = await self.h264_decoder.process_stream_chunk(websocket_id, message.body)
                    
                    # Process each decoded frame
                    for frame in frames:
                        if frame is None:
                            continue
                            
                        # Process frame with video processor
                        if self.video_processor:
                            # Use video processor for stream-aware processing
                            video_result = await self.video_processor.process_stream_frame(
                                websocket_id, frame, timestamp_ns
                            )
                            
                            # Update metrics
                            frames_processed.inc()
                            processing_time.observe(video_result.processing_time_ms / 1000.0)
                            active_tracks.set(video_result.tracking_result.object_count)
                            
                            # Process video tracking result
                            await self._process_video_result(video_result, frame, timestamp_ns, websocket_id)
                            
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
        
        # Publish to Rerun if enabled
        if self.config.rerun_enabled and self.rerun_client:
            # Convert masks to Detection format for visualization
            from visualization.rerun_client import Detection
            detections = []
            for mask_data in result.tracking_result.masks:
                if 'segmentation' in mask_data and mask_data['segmentation'] is not None:
                    # Get bounding box from segmentation
                    seg = mask_data['segmentation']
                    if seg.sum() > 0:
                        y_coords, x_coords = np.where(seg)
                        bbox = [
                            int(x_coords.min()),
                            int(y_coords.min()),
                            int(x_coords.max()),
                            int(y_coords.max())
                        ]
                        detection = Detection(
                            bbox=bbox,
                            class_name=f"Object_{mask_data.get('object_id', 0)}",
                            confidence=mask_data.get('confidence', 0.9),
                            mask=seg
                        )
                        detections.append(detection)
            
            # Convert tracks to TrackedObject format
            from visualization.rerun_client import TrackedObject
            tracked_objects = []
            for track in result.tracking_result.tracks:
                tracked = TrackedObject(
                    id=track.get('object_id', 0),
                    bbox=track.get('bbox', [0, 0, 0, 0]),
                    class_name=f"Track_{track.get('object_id', 0)}"
                )
                tracked_objects.append(tracked)
            
            # Log the frame with segmentation overlay
            self.rerun_client.log_frame(
                frame=frame,
                detections=detections,
                active_tracks=tracked_objects,
                frame_number=result.tracking_result.frame_number,
                timestamp_ns=timestamp_ns,
                websocket_id=websocket_id
            )
            
            # Log enhanced objects to Rerun grid if available
            if hasattr(result, 'enhanced_crops') and result.enhanced_crops:
                for crop_data in result.enhanced_crops:
                    # Create a TrackedObject-like structure for the enhanced object
                    tracked = TrackedObject(
                        id=crop_data.get('object_id', 0),
                        bbox=crop_data.get('bbox', [0, 0, 0, 0]),
                        class_name=f"Object_{crop_data.get('object_id', 0)}"
                    )
                    # Add the enhanced image
                    tracked.best_frame = crop_data.get('enhanced_crop', crop_data.get('original_crop'))
                    tracked.created_at = timestamp_ns
                    tracked.confidence = crop_data.get('confidence', 0.9)
                    
                    # Log to Rerun enhanced objects grid
                    self.rerun_client.log_enhanced_object(tracked)
        
        # Convert video tracks to detection format for publishing
        if result.tracking_result.tracks:
            detection_data = []
            for track in result.tracking_result.tracks:
                detection_data.append({
                    'bbox': track['bbox'],
                    'class_name': f"Track_{track['object_id']}",
                    'confidence': track.get('confidence', 0.0),
                    'track_id': track['track_id']
                })
            
            # Draw annotations
            annotated_frame = self._draw_video_annotations(frame, result.tracking_result.tracks)
            
            # Publish
            await self.publisher.publish_processed_frame(
                annotated_frame,
                detection_data,
                {
                    'timestamp_ns': timestamp_ns,
                    'frame_number': result.tracking_result.frame_number,
                    'processing_time_ms': result.processing_time_ms,
                    'ntp_time': get_ntp_time_ns(),
                    'ntp_offset': ntp_offset._value.get(),
                    'source': 'h264_video_stream',
                    'fps': result.fps,
                    'resolution': f"{result.original_resolution[0]}x{result.original_resolution[1]}"
                }
            )
    
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
    
    async def run(self):
        """Main run loop."""
        self.running = True
        
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
            
            # Initialize Rerun client if enabled
            if self.config.rerun_enabled:
                try:
                    from visualization.rerun_client import RerunClient
                    self.rerun_client = RerunClient(self.config)
                    logger.info("Rerun client initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize Rerun client: {e}")
                    self.rerun_client = None
            
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
                'video_tracker',
                name=self.video_processor.video_tracker.name,
                status='✅ Ready'
            )
            
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
            
            # Update Rerun status
            if self.config.rerun_enabled:
                self.monitor.update_component_status('rerun', status='✅ Connected')
            else:
                self.monitor.update_component_status('rerun', status='⏸️ Disabled')
            
            # Setup consumers
            await self.stream_queue.consume(self.process_stream_message)
            await self.mode_queue.consume(self.process_mode_message)
            
            logger.info("Frame processor service started, waiting for messages...")
            
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
            
            # Cleanup video processor streams
            if self.video_processor:
                logger.info("Cleaning up video processor streams...")
                for stream_id in list(self.video_processor.active_streams.keys()):
                    await self.video_processor.cleanup_stream(stream_id)
            
            # Cleanup Rerun video streams
            if self.rerun_client:
                logger.info("Cleaning up Rerun video streams...")
                for websocket_id in list(self.rerun_client.video_streams.keys()):
                    self.rerun_client.cleanup_video_stream(websocket_id)
            
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