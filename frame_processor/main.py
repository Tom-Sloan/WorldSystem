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
from pipeline.processor import FrameProcessor
from pipeline.publisher import RabbitMQPublisher


logger = get_logger(__name__)


# Prometheus metrics
frames_processed = Counter('frame_processor_frames_processed_total', 
                         'Total number of frames processed')
yolo_detections = Counter('frame_processor_yolo_detections_total', 
                        'Total number of YOLO detections')
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
        self.processor = None
        self.publisher = None
        self.connection = None
        self.channel = None
        self.running = False
        
        # Initialize performance monitor
        self.monitor = get_performance_monitor()
        self.monitor.start()  # Start the dashboard
        
        logger.info("Initializing Frame Processor Service...")
        
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
            self.video_frames_exchange = await self.channel.declare_exchange(
                self.config.video_frames_exchange,
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
            self.frame_queue = await self.channel.declare_queue(
                'frame_processor_queue',
                durable=True
            )
            
            self.mode_queue = await self.channel.declare_queue(
                'frame_processor_mode_queue',
                durable=True
            )
            
            # Bind queues (fanout exchanges don't use routing keys)
            await self.frame_queue.bind(
                self.video_frames_exchange
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
    
    async def process_frame_message(self, message: aio_pika.IncomingMessage):
        """
        Process incoming frame message.
        
        This is called by RabbitMQ for each frame.
        """
        try:
            async with message.process():
                with DetailedTimer("rabbitmq_message_processing"):
                    # Extract headers
                    headers = message.headers or {}
                    # Ensure timestamp_ns is an integer
                    timestamp_ns_raw = headers.get('timestamp_ns', get_ntp_time_ns())
                    timestamp_ns = int(timestamp_ns_raw) if timestamp_ns_raw else get_ntp_time_ns()
                    frame_number = headers.get('frame_number', 0)
                    
                    # Decode frame with timing
                    with DetailedTimer("frame_decode"):
                        nparr = np.frombuffer(message.body, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                    if frame is None:
                        self.monitor.add_event("Failed to decode frame", "error")
                        return
                
                # Process frame based on detection enabled
                if self.config.detection_enabled:
                    # Run processing pipeline
                    result = await self.processor.process_frame(frame, timestamp_ns)
                    
                    # Update metrics
                    frames_processed.inc()
                    yolo_detections.inc(result.detection_count)
                    processing_time.observe(result.processing_time_ms / 1000.0)
                    active_tracks.set(result.active_track_count)
                    
                    # Publish processed frame
                    if result.detections:
                        with DetailedTimer("result_publishing"):
                            # Draw bounding boxes
                            annotated_frame = self._draw_annotations(frame, result.detections)
                            
                            # Prepare detection data for publishing
                            detection_data = [
                                {
                                    'bbox': det.bbox,
                                    'class_name': det.class_name,
                                    'confidence': det.confidence
                                }
                                for det in result.detections
                            ]
                            
                            # Publish
                            await self.publisher.publish_processed_frame(
                            annotated_frame,
                            detection_data,
                            {
                                'timestamp_ns': timestamp_ns,
                                'frame_number': result.frame_number,
                                'processing_time_ms': result.processing_time_ms,
                                'ntp_time': get_ntp_time_ns(),
                                'ntp_offset': ntp_offset._value.get()
                            }
                        )
                    
                    # Publish API results for processed tracks
                    if result.tracks_for_api:
                        for track in result.tracks_for_api:
                            if track.api_processed and track.api_result:
                                await self.publisher.publish_api_result(track)
                                api_calls.inc()
                else:
                    # Pass through without processing
                    logger.debug(f"Analysis mode is 'none', passing through frame {frame_number}")
                    frames_processed.inc()
                
        except Exception as e:
            self.monitor.add_event(f"Frame processing error: {e}", "error")
            logger.error(f"Error processing frame: {e}", exc_info=True)
            # Message will be rejected/requeued by the context manager
    
    async def process_mode_message(self, message: aio_pika.IncomingMessage):
        """Process analysis mode update message."""
        try:
            async with message.process():
                # Parse mode update
                mode_data = json.loads(message.body)
                new_mode = mode_data.get('mode', 'on')
                
                logger.info(f"Received mode update: {new_mode}")
                
                # Update detection enabled based on mode
                # Accept various formats: "none"/"off"/"false" = disabled, anything else = enabled
                detection_enabled = new_mode.lower() not in ['none', 'off', 'false', '0']
                self.config.detection_enabled = detection_enabled
                
                if self.processor:
                    self.processor.config.detection_enabled = detection_enabled
                    logger.info(f"Detection {'enabled' if detection_enabled else 'disabled'}")
                
        except Exception as e:
            logger.error(f"Error processing mode update: {e}")
            # Message will be rejected/requeued by the context manager
    
    def _draw_annotations(self, frame: np.ndarray, detections) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{det.class_name} {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Background for text
            cv2.rectangle(annotated, 
                         (x1, y1 - label_size[1] - 4),
                         (x1 + label_size[0], y1),
                         (0, 255, 0), -1)
            
            # Text
            cv2.putText(annotated, label,
                       (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 0, 0), 1)
        
        return annotated
    
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
            self.processor = FrameProcessor(self.config)
            self.publisher = RabbitMQPublisher(self.config)
            
            # Update detector/tracker info
            self.monitor.update_component_status(
                'detector', 
                name=self.processor.detector.name,
                status='✅ Ready'
            )
            self.monitor.update_component_status(
                'tracker',
                name=self.processor.tracker.name,
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
            await self.frame_queue.consume(self.process_frame_message)
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
            
            # Cancel NTP sync task
            if self._ntp_task and not self._ntp_task.done():
                self._ntp_task.cancel()
                try:
                    await self._ntp_task
                except asyncio.CancelledError:
                    pass
            
            if self.processor:
                await self.processor.cleanup()
            
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