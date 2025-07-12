#!/usr/bin/env python3
"""
Streamlined SLAM3R Processor - Pure SLAM functionality without visualization.

This processor:
- Consumes RGB frames from RabbitMQ
- Runs SLAM3R pipeline (I2P → L2W)
- Publishes keyframes to mesh_service via shared memory
- Uses StreamingSLAM3R wrapper for clean architecture
"""

import asyncio
import logging
import os
import time
from typing import Dict

import aio_pika
import cv2
import numpy as np
import torch
import yaml
import msgpack

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import StreamingSLAM3R wrapper
from streaming_slam3r import StreamingSLAM3R

# Import shared memory for keyframe publishing
try:
    from shared_memory import StreamingKeyframePublisher
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    logger.warning("Shared memory streaming not available")

# Configure PyTorch CUDA allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.getenv(
    "PYTORCH_CUDA_ALLOC_CONF", 
    "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7"
)


class SLAM3RProcessor:
    """Streamlined SLAM3R processor using StreamingSLAM3R wrapper."""
    
    def __init__(self, config_path: str = "./configs/wild.yaml"):
        """Initialize the SLAM3R processor."""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize StreamingSLAM3R
        self.slam3r = None
        self._initialize_models()
        
        # RabbitMQ connection
        self.rabbitmq_connection = None
        self.rabbitmq_channel = None
        
        # Shared memory publisher for keyframes
        self.keyframe_publisher = None
        if STREAMING_AVAILABLE:
            try:
                self.keyframe_publisher = StreamingKeyframePublisher()
                logger.info("Initialized shared memory keyframe publisher")
            except Exception as e:
                logger.error(f"Failed to initialize keyframe publisher: {e}")
        
        # Processing statistics
        self.frame_count = 0
        self.keyframe_count = 0
        self.last_fps_time = time.time()
        self.last_fps_frame_count = 0
        
        # Video segment handling
        self.current_video_id = None
        self.segment_frame_count = 0
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load camera intrinsics
        intrinsics_path = config.get('wild_cam_intri', './configs/camera_intrinsics.yaml')
        with open(intrinsics_path, 'r') as f:
            camera_config = yaml.safe_load(f)
            config['camera_intrinsics'] = camera_config
        
        return config
    
    def _initialize_models(self):
        """Initialize the StreamingSLAM3R wrapper with models."""
        try:
            # Import model classes
            from slam3r.models import Image2PointsModel, Local2WorldModel
            
            # Model paths
            i2p_ckpt = self.config.get('I2P_ckpt_path', './checkpoints/slam3r_I2P.ckpt')
            l2w_ckpt = self.config.get('L2W_ckpt_path', './checkpoints/slam3r_L2W.ckpt')
            
            # Load I2P model
            i2p_model = Image2PointsModel(
                device=self.device,
                ckpt_path=i2p_ckpt,
                head='dpt',
                use_conf=True
            )
            
            # Load L2W model
            l2w_model = Local2WorldModel(
                device=self.device,
                ckpt_path=l2w_ckpt,
                use_conf=True
            )
            
            # Initialize StreamingSLAM3R with loaded models
            self.slam3r = StreamingSLAM3R(
                i2p_model=i2p_model,
                l2w_model=l2w_model,
                config=self.config,
                device=str(self.device)
            )
            
            logger.info("Successfully initialized StreamingSLAM3R models")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    async def connect_rabbitmq(self):
        """Establish RabbitMQ connection."""
        rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
        rabbitmq_port = int(os.getenv("RABBITMQ_PORT", "5672"))
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.rabbitmq_connection = await aio_pika.connect_robust(
                    f"amqp://guest:guest@{rabbitmq_host}:{rabbitmq_port}/"
                )
                self.rabbitmq_channel = await self.rabbitmq_connection.channel()
                await self.rabbitmq_channel.set_qos(prefetch_count=10)
                
                logger.info(f"Connected to RabbitMQ at {rabbitmq_host}:{rabbitmq_port}")
                return
                
            except Exception as e:
                logger.error(f"RabbitMQ connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(5)
                else:
                    raise
    
    async def process_frame(self, message: aio_pika.IncomingMessage):
        """Process a single frame from RabbitMQ."""
        async with message.process():
            try:
                # Deserialize message
                data = msgpack.unpackb(message.body, raw=False)
                
                timestamp = data['timestamp']
                frame_data = data['frame']
                video_id = data.get('video_id', 'default')
                
                # Handle video segment changes
                if video_id != self.current_video_id:
                    logger.info(f"New video segment detected: {video_id}")
                    self.current_video_id = video_id
                    self.segment_frame_count = 0
                    # Reset SLAM3R for new segment
                    self.slam3r.reset()
                
                # Decode image
                img_bytes = np.frombuffer(frame_data, dtype=np.uint8)
                img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    logger.error("Failed to decode image")
                    return
                
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                # Process frame through SLAM3R
                result = self.slam3r.process_frame(
                    image=img_rgb,
                    timestamp=timestamp
                )
                
                # Update counters
                self.frame_count += 1
                self.segment_frame_count += 1
                
                # Handle keyframe
                if result and result.get('is_keyframe', False):
                    self.keyframe_count += 1
                    await self._publish_keyframe(result, timestamp)
                
                # Log FPS periodically
                if self.frame_count % 30 == 0:
                    self._log_fps()
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}", exc_info=True)
    
    async def _publish_keyframe(self, keyframe_data: Dict, timestamp: int):
        """Publish keyframe to mesh_service via shared memory."""
        try:
            if self.keyframe_publisher and STREAMING_AVAILABLE:
                # Prepare keyframe data for mesh_service
                keyframe = {
                    'timestamp': timestamp,
                    'frame_id': keyframe_data['frame_id'],
                    'keyframe_id': self.keyframe_count,
                    'pts3d_world': keyframe_data['pts3d_world'].cpu().numpy(),
                    'conf_world': keyframe_data['conf_world'].cpu().numpy(),
                    'pose': keyframe_data.get('pose', np.eye(4)),
                }
                
                # Publish via shared memory
                self.keyframe_publisher.publish_keyframe(
                    keyframe_id=self.keyframe_count,
                    timestamp=timestamp,
                    pts3d=keyframe['pts3d_world'],
                    confidence=keyframe['conf_world'],
                    pose=keyframe['pose']
                )
                
                logger.debug(f"Published keyframe {self.keyframe_count} to mesh_service")
            
            # Also publish to RabbitMQ for compatibility
            if self.rabbitmq_channel:
                exchange_name = "slam3r_keyframe_exchange"
                try:
                    exchange = await self.rabbitmq_channel.get_exchange(exchange_name)
                except aio_pika.exceptions.ChannelNotFoundEntity:
                    exchange = await self.rabbitmq_channel.declare_exchange(
                        exchange_name, 
                        aio_pika.ExchangeType.FANOUT
                    )
                
                message_body = msgpack.packb({
                    'timestamp': timestamp,
                    'keyframe_id': self.keyframe_count,
                    'frame_id': keyframe_data['frame_id'],
                    'pts3d_world': keyframe_data['pts3d_world'].cpu().numpy().tolist(),
                    'conf_world': keyframe_data['conf_world'].cpu().numpy().tolist(),
                })
                
                await exchange.publish(
                    aio_pika.Message(body=message_body),
                    routing_key=""
                )
                
        except Exception as e:
            logger.error(f"Failed to publish keyframe: {e}")
    
    def _log_fps(self):
        """Log current FPS."""
        current_time = time.time()
        time_diff = current_time - self.last_fps_time
        frame_diff = self.frame_count - self.last_fps_frame_count
        
        if time_diff > 0:
            fps = frame_diff / time_diff
            logger.info(
                f"FPS: {fps:.2f} | "
                f"Frames: {self.frame_count} | "
                f"Keyframes: {self.keyframe_count} | "
                f"Segment frames: {self.segment_frame_count}"
            )
        
        self.last_fps_time = current_time
        self.last_fps_frame_count = self.frame_count
    
    async def run(self):
        """Main processing loop."""
        # Connect to RabbitMQ
        await self.connect_rabbitmq()
        
        # Declare queue
        queue_name = os.getenv("SLAM3R_QUEUE", "slam3r_frames")
        queue = await self.rabbitmq_channel.declare_queue(
            queue_name, 
            durable=True
        )
        
        # Bind to frame processor exchange
        exchange_name = "frame_processor_exchange"
        try:
            exchange = await self.rabbitmq_channel.get_exchange(exchange_name)
            await queue.bind(exchange, routing_key="slam3r")
        except Exception as e:
            logger.warning(f"Could not bind to exchange {exchange_name}: {e}")
        
        logger.info(f"Listening for frames on queue: {queue_name}")
        
        # Start consuming messages
        await queue.consume(self.process_frame)
        
        # Keep running
        try:
            await asyncio.Future()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            if self.rabbitmq_connection:
                await self.rabbitmq_connection.close()
            if self.keyframe_publisher:
                self.keyframe_publisher.close()


async def main():
    """Main entry point."""
    processor = SLAM3RProcessor()
    await processor.run()


if __name__ == "__main__":
    asyncio.run(main())