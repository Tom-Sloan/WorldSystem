"""
Enhanced output manager for real-time visualization through RabbitMQ.
Sends base64-encoded images and masks directly through messages for low latency.

Note: For production use with high resolutions, consider:
- H.264 video streaming to reduce bandwidth
- Compression optimizations
- Chunked message sending for very large frames
"""

import os
import cv2
import json
import numpy as np
import base64
from typing import List, Dict, Any, Optional
import time
import asyncio
import aio_pika
from dataclasses import dataclass
from io import BytesIO
from core.utils import get_logger

logger = get_logger(__name__)


@dataclass
class MaskData:
    """Data structure for mask information."""
    instance_id: int
    mask: np.ndarray
    class_name: str
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float = 0.0
    area: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'instance_id': self.instance_id,
            'class_name': self.class_name,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'area': self.area
        }


class VisualizationOutputManager:
    """
    Manages real-time visualization outputs for segmentation results.
    Sends base64-encoded data directly through RabbitMQ for low latency.
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize visualization output manager.
        
        Args:
            session_id: Unique session identifier
        """
        self.session_id = session_id or time.strftime('%Y%m%d_%H%M%S')
        
        # Tracking data
        self.frame_count = 0
        self.object_history = {}  # object_id -> list of frame appearances
        self.last_enhanced_objects = []  # Track enhanced object crops
        
        # Color palette for consistent visualization
        self.color_palette = [
            (255, 0, 0),   # Red
            (0, 255, 0),   # Green
            (0, 0, 255),   # Blue
            (255, 255, 0), # Yellow
            (255, 0, 255), # Magenta
            (0, 255, 255), # Cyan
            (255, 128, 0), # Orange
            (128, 255, 0), # Lime
            (255, 0, 128), # Rose
            (0, 128, 255), # Sky Blue
            (128, 0, 255), # Purple
            (255, 128, 128), # Light Red
            (128, 255, 128), # Light Green
            (128, 128, 255)  # Light Blue
        ]
        
        # RabbitMQ for publishing outputs
        self.rabbit_connection = None
        self.output_exchange = None
        self._setup_rabbitmq()
        
        logger.info(f"VisualizationOutputManager initialized. Session: {self.session_id}")
    
    def _setup_rabbitmq(self):
        """Setup RabbitMQ for publishing output events."""
        # Don't set up RabbitMQ in init - it will be set up when needed
        self.rabbit_connection = None
        self.output_exchange = None
        self.channel = None
        self._setup_complete = False
    
    async def setup_rabbitmq_async(self):
        """Async setup of RabbitMQ connection."""
        if self._setup_complete:
            return
            
        try:
            self.rabbit_connection = await aio_pika.connect_robust(
                os.getenv('RABBITMQ_URL', 'amqp://127.0.0.1:5672')
            )
            self.channel = await self.rabbit_connection.channel()
            self.output_exchange = await self.channel.declare_exchange(
                os.getenv('PROCESSED_FRAMES_EXCHANGE', 'processed_frames_exchange'),
                aio_pika.ExchangeType.FANOUT,
                durable=True
            )
            self._setup_complete = True
            logger.info("RabbitMQ connection established for visualization output")
        except Exception as e:
            logger.error(f"Failed to setup RabbitMQ: {e}")
            self._setup_complete = False
    
    def _encode_image_base64(self, image: np.ndarray, format: str = 'jpg', 
                            quality: int = 85) -> str:
        """
        Encode image to base64 string.
        
        Args:
            image: OpenCV image array
            format: Image format ('jpg' or 'png')
            quality: JPEG quality (1-100)
        
        Returns:
            Base64 encoded string
        """
        if format == 'jpg':
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, buffer = cv2.imencode('.jpg', image, encode_param)
        else:
            _, buffer = cv2.imencode('.png', image)
        
        return base64.b64encode(buffer).decode('utf-8')
    
    def _encode_masks_base64(self, masks: List[MaskData]) -> str:
        """
        Encode multiple masks into a single base64 string.
        Packs masks efficiently using a custom format.
        
        Returns:
            Base64 encoded string containing all masks
        """
        if not masks:
            return ""
        
        # Get dimensions from first mask
        h, w = masks[0].mask.shape
        
        # Create a composite mask image with all masks overlaid
        # Each mask gets a unique ID value (1, 2, 3, etc.)
        mask_composite = np.zeros((h, w), dtype=np.uint8)
        
        for i, mask_data in enumerate(masks):
            # Assign each mask a unique ID (starting from 1)
            mask_composite[mask_data.mask > 0] = i + 1
        
        # Encode as PNG for lossless compression
        _, buffer = cv2.imencode('.png', mask_composite)
        return base64.b64encode(buffer).decode('utf-8')
    
    def _create_visualization(self, frame: np.ndarray, masks: List[MaskData]) -> np.ndarray:
        """
        Create visualization with colored masks and labels.
        """
        vis_frame = frame.copy()
        
        for idx, mask_data in enumerate(masks):
            color = self.color_palette[idx % len(self.color_palette)]
            
            # Apply colored mask overlay with 50% opacity
            mask_indices = mask_data.mask > 0
            overlay = vis_frame.copy()
            overlay[mask_indices] = color
            vis_frame = cv2.addWeighted(vis_frame, 0.5, overlay, 0.5, 0)
            
            # Draw bounding box
            x1, y1, x2, y2 = mask_data.bbox
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label with background
            label = f"ID:{mask_data.instance_id} {mask_data.class_name}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 4), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_frame
    
    def _extract_enhanced_crops(self, frame: np.ndarray, masks: List[MaskData], 
                               max_crops: int = 12) -> List[Dict[str, Any]]:
        """
        Extract enhanced object crops for gallery view.
        
        Returns:
            List of dictionaries with base64 encoded crops and metadata
        """
        enhanced_crops = []
        
        # Sort by area to get largest objects
        sorted_masks = sorted(masks, key=lambda m: m.area, reverse=True)[:max_crops]
        
        for mask_data in sorted_masks:
            x1, y1, x2, y2 = mask_data.bbox
            
            # Ensure valid crop bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                # Extract crop
                crop = frame[y1:y2, x1:x2]
                
                # Resize to standard size for gallery (256x256)
                crop_resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
                
                # Encode crop
                crop_base64 = self._encode_image_base64(crop_resized, quality=90)
                
                enhanced_crops.append({
                    'data': crop_base64,
                    'metadata': mask_data.to_dict()
                })
        
        return enhanced_crops
    
    async def send_visualization_update(self, frame: np.ndarray, masks: List[MaskData], 
                                       frame_number: int, timestamp: Optional[float] = None,
                                       processing_stats: Optional[Dict[str, Any]] = None):
        """
        Send complete visualization update through RabbitMQ.
        
        Args:
            frame: Original frame
            masks: List of detected masks
            frame_number: Frame sequence number
            timestamp: Optional timestamp
            processing_stats: Optional processing statistics
        """
        logger.info(f"send_visualization_update called for frame {frame_number} with {len(masks)} masks")
        timestamp = timestamp or time.time()
        
        # Create visualization frame
        vis_frame = self._create_visualization(frame, masks)
        
        # Extract enhanced crops (update every 10 frames for performance)
        if frame_number % 10 == 0 and masks:
            self.last_enhanced_objects = self._extract_enhanced_crops(frame, masks)
        
        # Prepare message data
        message_data = {
            'type': 'visualization_update',
            'session_id': self.session_id,
            'frame_number': frame_number,
            'timestamp': timestamp,
            'data': {
                # Original frame
                'frame': {
                    'data_type': 'frame',
                    'content': self._encode_image_base64(frame, quality=85),
                    'metadata': {
                        'width': frame.shape[1],
                        'height': frame.shape[0],
                        'channels': frame.shape[2] if len(frame.shape) > 2 else 1
                    }
                },
                # Visualization with overlays
                'visualization': {
                    'data_type': 'visualization',
                    'content': self._encode_image_base64(vis_frame, quality=90),
                    'metadata': {
                        'num_objects': len(masks),
                        'object_ids': [m.instance_id for m in masks]
                    }
                },
                # Masks (send every 5 frames to reduce bandwidth)
                'masks': {
                    'data_type': 'masks',
                    'content': self._encode_masks_base64(masks) if frame_number % 5 == 0 else None,
                    'metadata': {
                        'mask_info': [m.to_dict() for m in masks]
                    }
                },
                # Enhanced object crops
                'enhanced': {
                    'data_type': 'enhanced',
                    'content': self.last_enhanced_objects,
                    'metadata': {
                        'count': len(self.last_enhanced_objects)
                    }
                },
                # Statistics
                'statistics': {
                    'data_type': 'statistics',
                    'content': {
                        'frame_count': self.frame_count,
                        'unique_objects': len(self.object_history),
                        'active_tracks': len(masks),
                        'processing_time_ms': processing_stats.get('processing_time_ms', 0) if processing_stats else 0,
                        'fps': processing_stats.get('fps', 0) if processing_stats else 0
                    }
                }
            }
        }
        
        # Update tracking
        self.frame_count += 1
        for mask in masks:
            if mask.instance_id not in self.object_history:
                self.object_history[mask.instance_id] = []
            self.object_history[mask.instance_id].append(frame_number)
        
        # Publish to RabbitMQ
        await self._publish_message(message_data)
    
    async def _publish_message(self, message_data: Dict[str, Any]):
        """Publish message to RabbitMQ."""
        # Ensure RabbitMQ connection is established
        if not self._setup_complete:
            await self.setup_rabbitmq_async()
            
        if not self._setup_complete or not self.output_exchange:
            logger.warning(f"Cannot publish frame {message_data['frame_number']} - RabbitMQ not connected (setup_complete={self._setup_complete}, exchange={self.output_exchange is not None})")
            return
        
        try:
            # Convert to JSON and encode
            message_json = json.dumps(message_data)
            
            # Log size for monitoring
            message_size = len(message_json.encode('utf-8'))
            if message_size > 10 * 1024 * 1024:  # Warn if > 10MB
                logger.warning(f"Large message size: {message_size / 1024 / 1024:.2f}MB")
            
            message = aio_pika.Message(
                body=message_json.encode('utf-8'),
                content_type='application/json'
            )
            
            # Publish directly since we're already in async context
            await self.output_exchange.publish(message, routing_key='')
            
            logger.debug(f"Published visualization update for frame {message_data['frame_number']}")
            
        except Exception as e:
            logger.error(f"Failed to publish visualization update: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        # Send final statistics
        final_stats = {
            'type': 'session_complete',
            'session_id': self.session_id,
            'total_frames': self.frame_count,
            'unique_objects': len(self.object_history),
            'timestamp': time.time()
        }
        self._publish_message(final_stats)
        
        # Close RabbitMQ connection
        if self.rabbit_connection and self.async_loop:
            self.async_loop.run_until_complete(self.rabbit_connection.close())
            self.async_loop.close()
        
        logger.info(f"Visualization output manager cleanup complete. Session: {self.session_id}")