"""
RabbitMQ publisher for processed frames and results.

This module handles publishing processed frames and API results
to RabbitMQ exchanges for downstream services.
"""

import json
import cv2
import numpy as np
import aio_pika
from aio_pika import Message, ExchangeType
from typing import Dict, Any, Optional, List
from datetime import datetime

from core.utils import get_logger, PerformanceTimer
from core.config import Config
from tracking.base import TrackedObject


logger = get_logger(__name__)


class RabbitMQPublisher:
    """
    Publishes processed frames and results to RabbitMQ.
    
    This preserves the publishing logic from the original frame_processor.py
    while providing a cleaner interface for the refactored architecture.
    """
    
    def __init__(self, config: Config):
        """
        Initialize RabbitMQ publisher.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.connection = None
        self.channel = None
        
        # Exchange names
        self.processed_frames_exchange = config.processed_frames_exchange
        self.api_results_exchange = "api_results_exchange"  # New exchange for API results
        
        # Connection parameters
        self.rabbitmq_url = config.rabbitmq_url
        
        # Publishing statistics
        self.stats = {
            'frames_published': 0,
            'api_results_published': 0,
            'identifications_published': 0,
            'publish_errors': 0,
            'total_bytes_published': 0
        }
        
        # Connection will be established via connect() method
    
    async def connect(self):
        """Establish connection to RabbitMQ."""
        try:
            logger.info(f"Connecting to RabbitMQ at {self.rabbitmq_url}")
            
            # Create connection
            self.connection = await aio_pika.connect_robust(
                self.rabbitmq_url,
                heartbeat=3600,
                reconnect_interval=5.0
            )
            self.channel = await self.connection.channel()
            
            # Declare exchanges
            self.processed_frames_exchange_obj = await self.channel.declare_exchange(
                self.processed_frames_exchange,
                ExchangeType.FANOUT,
                durable=True
            )
            
            self.api_results_exchange_obj = await self.channel.declare_exchange(
                self.api_results_exchange,
                ExchangeType.FANOUT,
                durable=True
            )
            
            logger.info("Connected to RabbitMQ successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise
    
    async def publish_processed_frame(self, frame: np.ndarray, 
                                    detections: List[Dict[str, Any]],
                                    metadata: Dict[str, Any]):
        """
        Publish processed frame with detections.
        
        Args:
            frame: Processed frame with annotations
            detections: List of detection dictionaries
            metadata: Frame metadata including timestamps
        """
        with PerformanceTimer("rabbitmq_publish_frame", logger):
            try:
                # Encode frame as JPEG
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
                _, buffer = cv2.imencode('.jpg', frame, encode_params)
                frame_bytes = buffer.tobytes()
                
                # Prepare message headers
                headers = {
                    'timestamp_ns': metadata.get('timestamp_ns', 0),
                    'frame_number': metadata.get('frame_number', 0),
                    'processing_time_ms': metadata.get('processing_time_ms', 0),
                    'detection_count': len(detections),
                    'width': frame.shape[1],
                    'height': frame.shape[0],
                    'ntp_time': metadata.get('ntp_time', 0),
                    'ntp_offset': metadata.get('ntp_offset', 0)
                }
                
                # Add detection summary to headers
                if detections:
                    class_counts = {}
                    for det in detections:
                        class_name = det.get('class_name', 'unknown')
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    headers['class_summary'] = json.dumps(class_counts)
                
                # Create message
                message = Message(
                    body=frame_bytes,
                    headers=headers,
                    content_type='image/jpeg',
                    timestamp=int(datetime.now().timestamp())
                )
                
                # Publish to exchange
                await self.processed_frames_exchange_obj.publish(
                    message,
                    routing_key=''  # Fanout exchanges ignore routing key
                )
                
                # Update statistics
                self.stats['frames_published'] += 1
                self.stats['total_bytes_published'] += len(frame_bytes)
                
                logger.debug(
                    f"Published frame #{metadata.get('frame_number', 0)} "
                    f"({len(frame_bytes)} bytes, {len(detections)} detections)"
                )
                
            except Exception as e:
                logger.error(f"Failed to publish frame: {e}")
                self.stats['publish_errors'] += 1
                self._handle_connection_error()
    
    async def publish_api_result(self, track: TrackedObject):
        """
        Publish API processing result for a track.
        
        Args:
            track: TrackedObject with api_result populated
        """
        if not track.api_result:
            return
        
        with PerformanceTimer("rabbitmq_publish_api_result", logger):
            try:
                # Prepare message payload
                payload = {
                    'track_id': track.id,
                    'class_name': track.class_name,
                    'confidence': track.confidence,
                    'created_at': track.created_at,
                    'processed_at': datetime.now().isoformat(),
                    'api_result': track.api_result,
                    'bbox': track.bbox,
                    'best_score': track.best_score
                }
                
                # Include dimensions if available
                if track.api_result.get('dimensions'):
                    payload['dimensions_m'] = track.api_result['dimensions']
                
                # Encode as JSON
                message_body = json.dumps(payload)
                
                # Create message
                message = Message(
                    body=message_body.encode(),
                    content_type='application/json',
                    timestamp=int(datetime.now().timestamp())
                )
                
                # Publish to exchange
                await self.api_results_exchange_obj.publish(
                    message,
                    routing_key=''  # Fanout exchanges ignore routing key
                )
                
                # Update statistics
                self.stats['api_results_published'] += 1
                
                logger.debug(
                    f"Published API result for track #{track.id} ({track.class_name})"
                )
                
            except Exception as e:
                logger.error(f"Failed to publish API result: {e}")
                self.stats['publish_errors'] += 1
                self._handle_connection_error()
    
    async def publish_batch_results(self, tracks: List[TrackedObject]):
        """
        Publish multiple API results as a batch.
        
        Args:
            tracks: List of TrackedObjects with api_results
        """
        valid_tracks = [t for t in tracks if t.api_result]
        if not valid_tracks:
            return
        
        logger.info(f"Publishing batch of {len(valid_tracks)} API results")
        
        for track in valid_tracks:
            await self.publish_api_result(track)
    
    async def _handle_connection_error(self):
        """Handle RabbitMQ connection errors."""
        try:
            if self.connection and not self.connection.is_closed:
                return
            
            logger.warning("RabbitMQ connection lost, attempting to reconnect...")
            await self.connect()
            
        except Exception as e:
            logger.error(f"Failed to reconnect to RabbitMQ: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get publishing statistics.
        
        Returns:
            Dictionary with publishing stats
        """
        stats = self.stats.copy()
        
        # Add connection status
        stats['connected'] = (
            self.connection is not None and 
            not self.connection.is_closed
        )
        
        # Calculate average message size
        if stats['frames_published'] > 0:
            stats['avg_frame_size_kb'] = (
                stats['total_bytes_published'] / 
                stats['frames_published'] / 1024
            )
        else:
            stats['avg_frame_size_kb'] = 0
        
        return stats
    
    async def publish_identification(self, identification_data: Dict[str, Any]):
        """
        Publish object identification results from Google Lens.
        
        Args:
            identification_data: Dictionary containing:
                - object_id: Unique object identifier
                - identification: Lens API results
                - timestamp: Processing timestamp
                - batch_id: Optional batch identifier
                - from_cache: Whether result was from cache
        """
        if not self.api_results_exchange_obj:
            logger.error("RabbitMQ not connected")
            return
        
        with PerformanceTimer("publish_identification", logger):
            try:
                # Prepare message
                message_data = {
                    'type': 'object_identification',
                    'object_id': identification_data.get('object_id'),
                    'identification': identification_data.get('identification'),
                    'timestamp': identification_data.get('timestamp', datetime.now().timestamp()),
                    'batch_id': identification_data.get('batch_id'),
                    'from_cache': identification_data.get('from_cache', False)
                }
                
                # Publish
                message_body = json.dumps(message_data).encode()
                
                await self.api_results_exchange_obj.publish(
                    Message(
                        body=message_body,
                        content_type='application/json'
                    ),
                    routing_key=''
                )
                
                # Update statistics
                self.stats['identifications_published'] += 1
                self.stats['total_bytes_published'] += len(message_body)
                
                logger.debug(
                    f"Published identification for object {identification_data.get('object_id')}"
                )
                
            except Exception as e:
                logger.error(f"Failed to publish identification: {e}")
                self.stats['publish_errors'] += 1
                await self._handle_connection_error()
    
    async def close(self):
        """Close RabbitMQ connection."""
        try:
            if self.connection and not self.connection.is_closed:
                logger.info("Closing RabbitMQ connection...")
                await self.connection.close()
                logger.info("RabbitMQ connection closed")
        except Exception as e:
            logger.error(f"Error closing RabbitMQ connection: {e}")
    
