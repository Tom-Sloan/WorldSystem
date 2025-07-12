#!/usr/bin/env python3
"""
SLAM3R Processor V2: RabbitMQ integration using StreamingSLAM3R architecture.

This is a cleaner implementation that uses the StreamingSLAM3R wrapper
for better state management and dimension handling.
"""

import os
import sys
import json
import logging
import asyncio
import numpy as np
import msgpack
import torch
import cv2
import aio_pika
from typing import Optional, Dict, Any

# Add SLAM3R_engine to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from streaming_slam3r import StreamingSLAM3R, AsyncStreamingSLAM3R
from slam3r.models import Image2PointsModel, Local2WorldModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Global variables
slam3r_processor: Optional[AsyncStreamingSLAM3R] = None
i2p_model: Optional[Image2PointsModel] = None
l2w_model: Optional[Local2WorldModel] = None


async def initialize_models():
    """Initialize SLAM3R models"""
    global i2p_model, l2w_model, slam3r_processor
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load models
        logger.info("Loading I2P model...")
        i2p_model = Image2PointsModel.from_pretrained("siyan824/slam3r_i2p").to(device).eval()
        
        logger.info("Loading L2W model...")
        l2w_model = Local2WorldModel.from_pretrained("siyan824/slam3r_l2w").to(device).eval()
        
        # Configuration for StreamingSLAM3R
        config = {
            'batch_size': int(os.getenv('SLAM3R_BATCH_SIZE', '5')),
            'window_size': int(os.getenv('SLAM3R_WINDOW_SIZE', '20')),
            'initial_keyframe_stride': int(os.getenv('SLAM3R_INIT_KF_STRIDE', '5')),
            'init_frames': int(os.getenv('SLAM3R_INIT_FRAMES', '5')),
            'initial_winsize': int(os.getenv('SLAM3R_INIT_WINSIZE', '5')),
            'conf_thres_i2p': float(os.getenv('SLAM3R_CONF_THRES', '5.0')),
            'num_scene_frame': int(os.getenv('SLAM3R_NUM_SCENE_FRAME', '5')),
            'norm_input_l2w': os.getenv('SLAM3R_NORM_L2W', 'false').lower() == 'true'
        }
        
        # Create streaming processor
        streaming_slam3r = StreamingSLAM3R(
            i2p_model=i2p_model,
            l2w_model=l2w_model,
            config=config,
            device=str(device)
        )
        
        # Wrap for async operation
        slam3r_processor = AsyncStreamingSLAM3R(streaming_slam3r)
        
        logger.info("SLAM3R models initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise


async def on_video_frame_message(message: aio_pika.IncomingMessage):
    """Handle incoming video frame messages"""
    async with message.process():
        try:
            # Check if message is msgpack or raw image
            if message.headers and 'timestamp_ns' in message.headers:
                # Raw JPEG format (from test)
                img_bytes = message.body
                timestamp = int(message.headers.get('timestamp_ns', 0))
                
                # Decode JPEG
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is None:
                    logger.warning("Failed to decode JPEG image")
                    return
                    
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            else:
                # Msgpack format
                data = msgpack.unpackb(message.body, raw=False)
                
                # Extract frame data
                frame_data = data.get('frame')
                if not frame_data:
                    logger.warning("No frame data in message")
                    return
                
                # Decode image
                img_bytes = frame_data.get('data')
                shape = frame_data.get('shape')
                dtype = frame_data.get('dtype', 'uint8')
                
                if not img_bytes or not shape:
                    logger.warning("Invalid frame data")
                    return
                
                # Reconstruct image
                img = np.frombuffer(img_bytes, dtype=np.dtype(dtype)).reshape(shape)
                timestamp = data.get('timestamp', 0)
            
            # Log frame reception
            logger.info(f"Received frame with shape {img.shape}, timestamp {timestamp}")
            
            # Process frame
            result = await slam3r_processor.process_frame(img, timestamp)
            
            if result is not None:
                logger.info(f"Got result from SLAM3R: {result.keys()}")
                # Publish results
                await publish_results(result)
            else:
                logger.debug("No result from SLAM3R yet")
                
        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)


async def publish_results(result: Dict[str, Any]):
    """Publish SLAM results to RabbitMQ"""
    try:
        # Get connection and channel
        connection = await aio_pika.connect_robust(
            os.getenv('RABBITMQ_URL', 'amqp://guest:guest@localhost/')
        )
        
        async with connection:
            channel = await connection.channel()
            
            # Prepare messages
            messages = []
            
            # Pose message
            if result.get('pose') is not None:
                pose_msg = {
                    'type': 'pose_update',
                    'pose': result['pose'].tolist(),
                    'timestamp': result['timestamp'],
                    'frame_id': result['frame_id']
                }
                messages.append(('slam.pose', pose_msg))
            
            # Point cloud message
            if result.get('points') is not None and len(result['points']) > 0:
                pc_msg = {
                    'type': 'pointcloud_update',
                    'points': result['points'].tolist(),
                    'confidence': result['confidence'].tolist(),
                    'timestamp': result['timestamp'],
                    'frame_id': result['frame_id'],
                    'is_keyframe': result.get('is_keyframe', False)
                }
                messages.append(('slam.pointcloud', pc_msg))
            
            # Publish messages
            for routing_key, msg in messages:
                await channel.default_exchange.publish(
                    aio_pika.Message(
                        body=msgpack.packb(msg, use_bin_type=True),
                        content_type='application/msgpack'
                    ),
                    routing_key=routing_key
                )
            
            logger.info(f"Published results for frame {result['frame_id']}")
            
    except Exception as e:
        logger.error(f"Failed to publish results: {e}")


async def main():
    """Main entry point"""
    # Initialize models
    await initialize_models()
    
    # Connect to RabbitMQ
    connection = await aio_pika.connect_robust(
        os.getenv('RABBITMQ_URL', 'amqp://guest:guest@localhost/')
    )
    
    async with connection:
        # Create channel
        channel = await connection.channel()
        
        # Set prefetch count for better performance
        await channel.set_qos(prefetch_count=10)
        
        # Declare exchange
        exchange = await channel.declare_exchange(
            os.getenv('VIDEO_FRAMES_EXCHANGE', 'video_frames_exchange'),
            aio_pika.ExchangeType.FANOUT,
            durable=True
        )
        
        # Declare and bind queue
        queue = await channel.declare_queue(
            'slam3r_queue',
            durable=True
        )
        
        await queue.bind(exchange)
        
        logger.info("Connected to RabbitMQ and waiting for messages...")
        
        # Start consuming
        await queue.consume(on_video_frame_message)
        
        # Keep running
        try:
            await asyncio.Future()
        except KeyboardInterrupt:
            logger.info("Shutting down...")


if __name__ == '__main__':
    asyncio.run(main())