"""
Local output manager for saving segmentation results.
Replaces Rerun visualization with local file saving and RabbitMQ publishing.
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
import asyncio
import aio_pika
from dataclasses import dataclass, asdict

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


class LocalOutputManager:
    """
    Manages local file outputs for segmentation results.
    Saves frames, masks, and metadata to disk.
    """
    
    def __init__(self, output_dir: str = '/app/outputs', session_id: Optional[str] = None):
        """
        Initialize output manager.
        
        Args:
            output_dir: Base directory for outputs
            session_id: Unique session identifier
        """
        self.output_dir = Path(output_dir)
        self.session_id = session_id or datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create directory structure
        self.session_dir = self.output_dir / self.session_id
        self.frames_dir = self.session_dir / 'frames'
        self.masks_dir = self.session_dir / 'masks'
        self.visualizations_dir = self.session_dir / 'visualizations'
        self.metadata_dir = self.session_dir / 'metadata'
        
        self._create_directories()
        
        # Tracking data
        self.frame_count = 0
        self.object_history = {}  # object_id -> list of frame appearances
        self.session_metadata = {
            'session_id': self.session_id,
            'start_time': time.time(),
            'frames_processed': 0,
            'unique_objects': 0
        }
        
        # RabbitMQ for publishing outputs
        self.rabbit_connection = None
        self.output_exchange = None
        self._setup_rabbitmq()
        
        logger.info(f"LocalOutputManager initialized. Session: {self.session_id}")
    
    def _create_directories(self):
        """Create necessary directory structure."""
        for directory in [self.frames_dir, self.masks_dir, 
                         self.visualizations_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_rabbitmq(self):
        """Setup RabbitMQ for publishing output events."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def connect():
                self.rabbit_connection = await aio_pika.connect_robust(
                    os.getenv('RABBITMQ_URL', 'amqp://127.0.0.1:5672')
                )
                channel = await self.rabbit_connection.channel()
                self.output_exchange = await channel.declare_exchange(
                    'frame_processor_outputs',
                    aio_pika.ExchangeType.FANOUT,
                    durable=True
                )
            
            loop.run_until_complete(connect())
            self.async_loop = loop
        except Exception as e:
            logger.warning(f"Failed to setup RabbitMQ: {e}")
            self.rabbit_connection = None
    
    def save_frame_outputs(self, frame: np.ndarray, masks: List[MaskData], 
                          frame_number: int, timestamp: Optional[float] = None):
        """
        Save frame, masks, and visualization.
        
        Args:
            frame: Original frame
            masks: List of detected masks
            frame_number: Frame sequence number
            timestamp: Optional timestamp
        """
        timestamp = timestamp or time.time()
        
        # Save original frame
        frame_path = self.frames_dir / f'frame_{frame_number:06d}.jpg'
        cv2.imwrite(str(frame_path), frame)
        
        # Save masks and metadata
        frame_metadata = {
            'frame_number': frame_number,
            'timestamp': timestamp,
            'objects': []
        }
        
        for mask_data in masks:
            # Save individual mask
            mask_filename = f'mask_f{frame_number:06d}_obj{mask_data.instance_id:04d}.png'
            mask_path = self.masks_dir / mask_filename
            cv2.imwrite(str(mask_path), (mask_data.mask * 255).astype(np.uint8))
            
            # Add to metadata
            obj_meta = mask_data.to_dict()
            obj_meta['mask_file'] = mask_filename
            frame_metadata['objects'].append(obj_meta)
            
            # Update object history
            if mask_data.instance_id not in self.object_history:
                self.object_history[mask_data.instance_id] = []
            self.object_history[mask_data.instance_id].append(frame_number)
        
        # Save frame metadata
        metadata_path = self.metadata_dir / f'frame_{frame_number:06d}.json'
        with open(metadata_path, 'w') as f:
            json.dump(frame_metadata, f, indent=2)
        
        # Create and save visualization
        vis_frame = self._create_visualization(frame, masks)
        vis_path = self.visualizations_dir / f'vis_{frame_number:06d}.jpg'
        cv2.imwrite(str(vis_path), vis_frame)
        
        # Update session metadata
        self.frame_count += 1
        self.session_metadata['frames_processed'] = self.frame_count
        self.session_metadata['unique_objects'] = len(self.object_history)
        
        # Publish output event
        self._publish_output_event({
            'type': 'frame_processed',
            'session_id': self.session_id,
            'frame_number': frame_number,
            'timestamp': timestamp,
            'num_objects': len(masks),
            'paths': {
                'frame': str(frame_path),
                'visualization': str(vis_path),
                'metadata': str(metadata_path)
            }
        })
        
        # Save session metadata periodically
        if frame_number % 30 == 0:
            self.save_session_metadata()
    
    def _create_visualization(self, frame: np.ndarray, masks: List[MaskData]) -> np.ndarray:
        """
        Create visualization with colored masks and labels.
        
        Args:
            frame: Original frame
            masks: List of masks to visualize
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        
        # Color palette
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (255, 128, 0), (128, 255, 0), (255, 0, 128),
            (0, 128, 255), (128, 0, 255), (255, 128, 128)
        ]
        
        for idx, mask_data in enumerate(masks):
            color = colors[idx % len(colors)]
            
            # Apply colored mask overlay
            mask_indices = mask_data.mask > 0
            overlay = vis_frame.copy()
            overlay[mask_indices] = color
            vis_frame = cv2.addWeighted(vis_frame, 0.7, overlay, 0.3, 0)
            
            # Draw bounding box
            x1, y1, x2, y2 = mask_data.bbox
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"ID:{mask_data.instance_id} {mask_data.class_name}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 4), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_frame
    
    def save_session_metadata(self):
        """Save session metadata to disk."""
        metadata_path = self.session_dir / 'session_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.session_metadata, f, indent=2)
        
        # Save object tracking history
        history_path = self.session_dir / 'object_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.object_history, f, indent=2)
    
    def _publish_output_event(self, event_data: Dict[str, Any]):
        """Publish output event to RabbitMQ."""
        if not self.rabbit_connection:
            return
        
        try:
            message = aio_pika.Message(
                body=json.dumps(event_data).encode(),
                content_type='application/json'
            )
            
            asyncio.run_coroutine_threadsafe(
                self.output_exchange.publish(message, routing_key=''),
                self.async_loop
            )
        except Exception as e:
            logger.debug(f"Failed to publish output event: {e}")
    
    def create_video_summary(self, fps: int = 15):
        """
        Create a video summary from saved visualizations.
        
        Args:
            fps: Frames per second for output video
        """
        vis_files = sorted(self.visualizations_dir.glob('vis_*.jpg'))
        if not vis_files:
            logger.warning("No visualization files found for video creation")
            return
        
        # Get frame dimensions
        sample_frame = cv2.imread(str(vis_files[0]))
        height, width = sample_frame.shape[:2]
        
        # Create video writer
        video_path = self.session_dir / f'summary_{self.session_id}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        # Write frames
        for vis_file in vis_files:
            frame = cv2.imread(str(vis_file))
            writer.write(frame)
        
        writer.release()
        logger.info(f"Created video summary: {video_path}")
        
        # Publish completion event
        self._publish_output_event({
            'type': 'video_summary_created',
            'session_id': self.session_id,
            'video_path': str(video_path),
            'num_frames': len(vis_files),
            'fps': fps
        })
    
    def cleanup(self):
        """Clean up resources."""
        # Save final metadata
        self.save_session_metadata()
        
        # Create video summary
        self.create_video_summary()
        
        # Close RabbitMQ connection
        if self.rabbit_connection and self.async_loop:
            self.async_loop.run_until_complete(self.rabbit_connection.close())
            self.async_loop.close()
        
        logger.info(f"Output manager cleanup complete. Session: {self.session_id}")