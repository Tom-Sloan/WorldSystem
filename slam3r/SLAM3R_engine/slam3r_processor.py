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
    
    def __init__(self, config_path: str = "./SLAM3R_engine/configs/wild.yaml"):
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
        self.keyframe_exchange = None
        
        # Shared memory publisher for keyframes
        self.keyframe_publisher = None
        
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
        intrinsics_path = config.get('wild_cam_intri', './SLAM3R_engine/configs/camera_intrinsics.yaml')
        with open(intrinsics_path, 'r') as f:
            camera_config = yaml.safe_load(f)
            config['camera_intrinsics'] = camera_config
        
        return config
    
    def _initialize_models(self):
        """Initialize the StreamingSLAM3R wrapper with models."""
        try:
            # Import model classes
            from slam3r.models import Image2PointsModel, Local2WorldModel
            
            # Load models from HuggingFace (as done in Dockerfile and v2)
            logger.info("Loading I2P model from HuggingFace...")
            i2p_model = Image2PointsModel.from_pretrained("siyan824/slam3r_i2p").to(self.device).eval()
            
            logger.info("Loading L2W model from HuggingFace...")
            l2w_model = Local2WorldModel.from_pretrained("siyan824/slam3r_l2w").to(self.device).eval()
            
            # Debug: Check which patch embedding the models use
            logger.info(f"I2P model patch_embed type: {type(i2p_model.patch_embed).__name__}")
            logger.info(f"L2W model patch_embed type: {type(l2w_model.patch_embed).__name__}")
            
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
                
                # Create keyframe exchange
                exchange_name = os.getenv("SLAM3R_KEYFRAME_EXCHANGE", "slam3r_keyframe_exchange")
                self.keyframe_exchange = await self.rabbitmq_channel.declare_exchange(
                    exchange_name,
                    aio_pika.ExchangeType.TOPIC,
                    durable=True
                )
                
                # Initialize keyframe publisher with exchange
                if STREAMING_AVAILABLE and self.keyframe_exchange:
                    try:
                        self.keyframe_publisher = StreamingKeyframePublisher(self.keyframe_exchange)
                        logger.info("Initialized shared memory keyframe publisher with exchange")
                    except Exception as e:
                        logger.error(f"Failed to initialize keyframe publisher: {e}")
                
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
                # Get timestamp from headers
                timestamp = int(message.headers.get("timestamp_ns", "0"))
                
                # Get video_id from headers if available
                video_id = message.headers.get("video_id", "default")
                
                # Handle video segment changes
                if video_id != self.current_video_id:
                    logger.info(f"New video segment detected: {video_id}")
                    self.current_video_id = video_id
                    self.segment_frame_count = 0
                    # Reset SLAM3R for new segment
                    self.slam3r.reset()
                
                # Message body contains the image directly
                img_bgr = cv2.imdecode(
                    np.frombuffer(message.body, dtype=np.uint8), 
                    cv2.IMREAD_COLOR
                )
                if img_bgr is None:
                    logger.error("Failed to decode image")
                    return
                
                # Convert BGR to RGB and store for color extraction
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
                    # Add RGB data to result for color extraction
                    result['rgb_image'] = img_rgb
                    await self._publish_keyframe(result, timestamp)
                
                # Log FPS periodically
                if self.frame_count % 30 == 0:
                    self._log_fps()
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}", exc_info=True)
    
    async def _publish_keyframe(self, keyframe_data: Dict, timestamp: int):
        """Publish keyframe to mesh_service via shared memory."""
        try:
            logger.info(f"Publishing keyframe {self.keyframe_count} with frame_id {keyframe_data['frame_id']}")
            
            # Debug the incoming keyframe data
            pts3d_world = keyframe_data.get('pts3d_world')
            conf_world = keyframe_data.get('conf_world')
            
            if pts3d_world is None:
                logger.error("pts3d_world is None in keyframe data!")
                return
            if conf_world is None:
                logger.error("conf_world is None in keyframe data!")
                return
                
            logger.info(f"pts3d_world shape: {pts3d_world.shape}, dtype: {pts3d_world.dtype}")
            logger.info(f"conf_world shape: {conf_world.shape}, dtype: {conf_world.dtype}")
            
            if self.keyframe_publisher and STREAMING_AVAILABLE:
                # Prepare keyframe data for mesh_service
                keyframe = {
                    'timestamp': timestamp,
                    'frame_id': keyframe_data['frame_id'],
                    'keyframe_id': self.keyframe_count,
                    'pts3d_world': pts3d_world.cpu().numpy(),
                    'conf_world': conf_world.cpu().numpy(),
                    'pose': keyframe_data.get('pose', np.eye(4)),
                }
                
                # Extract points and colors from pts3d_world
                # Handle different tensor shapes - matching original implementation
                pts3d_np = keyframe['pts3d_world']
                conf_np = keyframe['conf_world']
                
                # Log shapes before reshaping
                logger.info(f"Before reshape - pts3d_np shape: {pts3d_np.shape}, conf_np shape: {conf_np.shape}")
                
                # Handle tensor dimensions - matching original: squeeze(0).cpu().reshape(-1, 3)
                if pts3d_np.ndim == 4 and pts3d_np.shape[0] == 1:
                    pts3d_np = pts3d_np.squeeze(0)  # Remove batch dim
                if conf_np.ndim == 3 and conf_np.shape[0] == 1:
                    conf_np = conf_np.squeeze(0)  # Remove batch dim
                
                # Reshape to flat arrays
                pts3d = pts3d_np.reshape(-1, 3)
                conf = conf_np.squeeze().reshape(-1)
                
                # Use L2W confidence threshold - matching original
                conf_thresh = self.config.get('recon_pipeline', {}).get('conf_thres_l2w', 12.0)
                mask = conf > conf_thresh
                
                # Fallback to lower threshold if too few points - matching original
                if mask.sum() < 3:
                    conf_thresh_fallback = 0.5 * conf_thresh
                    mask = conf > conf_thresh_fallback
                    logger.info(f"Too few points with conf > {conf_thresh}, using fallback threshold {conf_thresh_fallback}")
                
                filtered_pts = pts3d[mask]
                
                # Check if we have any valid points
                if len(filtered_pts) == 0:
                    logger.warning(f"No valid points for keyframe {self.keyframe_count} after confidence filtering. "
                                 f"Max conf: {conf.max():.3f}, min: {conf.min():.3f}, threshold: {conf_thresh}")
                    return
                
                # Extract RGB colors if available
                if 'rgb_image' in keyframe_data and keyframe_data['rgb_image'] is not None:
                    rgb_image = keyframe_data['rgb_image']
                    # Get target shape from confidence tensor
                    H, W = conf_np.shape[-2:] if conf_np.ndim >= 2 else (224, 224)
                    
                    # Resize RGB to match point cloud resolution
                    if rgb_image.shape[:2] != (H, W):
                        rgb_resized = cv2.resize(rgb_image, (W, H), interpolation=cv2.INTER_LINEAR)
                    else:
                        rgb_resized = rgb_image
                    
                    # Extract colors for valid points
                    rgb_flat = rgb_resized.reshape(-1, 3)
                    colors = rgb_flat[mask].astype(np.uint8)
                else:
                    # Default gray colors
                    colors = np.ones((len(filtered_pts), 3), dtype=np.uint8) * 200
                
                logger.info(f"Publishing keyframe {self.keyframe_count}: {len(filtered_pts)} valid points "
                           f"(from {len(pts3d)} total, {mask.sum()} passed confidence)")
                
                # Get pose matrix
                pose = keyframe_data.get('pose', np.eye(4))
                if isinstance(pose, list):
                    pose = np.array(pose).reshape(4, 4)
                
                # Publish via shared memory
                await self.keyframe_publisher.publish_keyframe(
                    keyframe_id=str(self.keyframe_count),
                    pose=pose,
                    points=filtered_pts.astype(np.float32),
                    colors=colors
                )
                
                logger.info(f"Successfully published keyframe {self.keyframe_count} with {len(filtered_pts)} points")
            
            # Also publish to RabbitMQ for compatibility
            if self.keyframe_exchange:
                # DEBUG: Log what we're sending
                logger.info(f"[DEBUG] Publishing direct RabbitMQ message for keyframe {self.keyframe_count}")
                logger.info(f"[DEBUG] Message fields: timestamp, keyframe_id, frame_id, pts3d_world, conf_world")
                logger.info(f"[DEBUG] Note: NO 'type' field in message")
                
                message_body = msgpack.packb({
                    'timestamp': timestamp,
                    'keyframe_id': self.keyframe_count,
                    'frame_id': keyframe_data['frame_id'],
                    'pts3d_world': keyframe_data['pts3d_world'].cpu().numpy().tolist(),
                    'conf_world': keyframe_data['conf_world'].cpu().numpy().tolist(),
                })
                
                logger.info(f"[DEBUG] Message size: {len(message_body)} bytes")
                
                await self.keyframe_exchange.publish(
                    aio_pika.Message(body=message_body),
                    routing_key="keyframe.new"
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
        
        # Bind to video frames exchange
        exchange_name = os.getenv("VIDEO_FRAMES_EXCHANGE", "video_frames_exchange")
        try:
            # Declare exchange if it doesn't exist
            exchange = await self.rabbitmq_channel.declare_exchange(
                exchange_name,
                aio_pika.ExchangeType.FANOUT,
                durable=True
            )
            await queue.bind(exchange)
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
            # Clean up shared memory
            if self.keyframe_publisher:
                logger.info("Cleaning up shared memory segments...")
                self.keyframe_publisher.cleanup()
            
            # Close RabbitMQ connection
            if self.rabbitmq_connection:
                await self.rabbitmq_connection.close()


async def main():
    """Main entry point."""
    processor = SLAM3RProcessor()
    await processor.run()


if __name__ == "__main__":
    asyncio.run(main())