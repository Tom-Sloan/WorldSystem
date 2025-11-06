#!/usr/bin/env python3
"""
Streamlined SLAM3R Processor - Uses WebSocketVideoConsumer base class.

This processor:
- Inherits from WebSocketVideoConsumer for H.264 decoding
- Connects to server's WebSocket consumer (/ws/video/consume)
- Runs SLAM3R pipeline (I2P → L2W) on decoded frames
- Publishes keyframes to mesh_service via shared memory
- Uses StreamingSLAM3R wrapper for clean architecture
"""

import asyncio
import logging
import os
import sys
import time
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import torch
import yaml
import trimesh

# Add common directory to path for WebSocketVideoConsumer
# In Docker: /app/common, locally: ../../common from SLAM3R_engine
common_path = os.path.join(os.path.dirname(__file__), '../../common')
if not os.path.exists(common_path):
    common_path = '/app/common'  # Docker path
sys.path.insert(0, common_path)
from websocket_video_consumer import WebSocketVideoConsumer

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


def estimate_rigid_transform_svd(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate rigid transform (R, t) that aligns P to Q using SVD.
    P: source points (N x 3)
    Q: target points (N x 3)
    Returns: R (3x3 rotation), t (3x1 translation)
    """
    # Validate inputs
    if P.shape[0] < 3 or Q.shape[0] < 3:
        logger.warning(f"Insufficient points for SVD: P={P.shape[0]}, Q={Q.shape[0]}")
        return np.eye(3), np.zeros((3, 1))
    
    if P.shape[0] != Q.shape[0]:
        logger.error(f"Point cloud size mismatch: P={P.shape}, Q={Q.shape}")
        return np.eye(3), np.zeros((3, 1))
    
    try:
        # Center the point clouds
        P_mean = P.mean(axis=0)
        Q_mean = Q.mean(axis=0)
        Pc = P - P_mean
        Qc = Q - Q_mean
        
        # Compute cross-covariance matrix
        H = Pc.T @ Qc
        
        # SVD decomposition
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = Q_mean.reshape(3, 1) - R @ P_mean.reshape(3, 1)
        
        # Validate the result
        det_R = np.linalg.det(R)
        if abs(det_R - 1.0) > 1e-6:
            logger.warning(f"Invalid rotation matrix determinant: {det_R}")
            return np.eye(3), np.zeros((3, 1))
        
        return R, t
        
    except Exception as e:
        logger.error(f"SVD failed: {e}")
        return np.eye(3), np.zeros((3, 1))

# Configure PyTorch CUDA allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.getenv(
    "PYTORCH_CUDA_ALLOC_CONF", 
    "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7"
)


class SLAM3RProcessor(WebSocketVideoConsumer):
    """Streamlined SLAM3R processor using StreamingSLAM3R wrapper and WebSocketVideoConsumer base."""

    def __init__(self, config_path: str = "./SLAM3R_engine/configs/wild.yaml"):
        """Initialize the SLAM3R processor."""
        # Get WebSocket URL from environment
        ws_url = os.getenv("VIDEO_STREAM_URL", "ws://127.0.0.1:5001/ws/video/consume")

        # Initialize parent WebSocketVideoConsumer
        super().__init__(ws_url=ws_url, service_name="SLAM3R", frame_skip=1)

        self.config_path = config_path
        self.config = self._load_config()

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize StreamingSLAM3R
        self.slam3r = None
        self._initialize_models()

        # RabbitMQ connection (for keyframe publishing)
        self.rabbitmq_connection = None
        self.rabbitmq_channel = None
        self.keyframe_exchange = None

        # Shared memory publisher for keyframes
        self.keyframe_publisher = None

        # Keyframe tracking
        self.keyframe_count = 0

        # Video segment handling
        self.current_video_id = None
        self.segment_frame_count = 0

        # PCD saving configuration
        self.save_pcd = os.getenv("SLAM3R_SAVE_PCD", "false").lower() == "true"
        self.pcd_output_dir = os.getenv("SLAM3R_PCD_OUTPUT_DIR", "/debug_output/pcd")
        if self.save_pcd:
            os.makedirs(self.pcd_output_dir, exist_ok=True)
            logger.info(f"PCD saving enabled. Output directory: {self.pcd_output_dir}")

        # Event loop reference for async operations from sync thread
        self._event_loop = None
    
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
    
    def process_frame(self, frame: np.ndarray, frame_number: int):
        """Process a single decoded frame through SLAM3R pipeline.

        This method is called by WebSocketVideoConsumer for each decoded frame.
        Frame is already in BGR format from the parent class.

        Note: This runs in a separate thread, so we use run_coroutine_threadsafe
        to schedule async tasks on the main event loop.
        """
        try:
            # Convert BGR to RGB for SLAM3R
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process through SLAM3R pipeline
            timestamp = int(time.time_ns())
            result = self.slam3r.process_frame(img_rgb, timestamp)

            # Publish keyframe if detected
            if result and result.get('is_keyframe', False):
                result['rgb_image'] = img_rgb

                # Schedule async keyframe publishing on main event loop
                # This is thread-safe and doesn't block the processing thread
                if self._event_loop is not None:
                    future = asyncio.run_coroutine_threadsafe(
                        self._publish_keyframe(result, timestamp),
                        self._event_loop
                    )

                    # Add callback to log any errors
                    def handle_publish_error(fut):
                        try:
                            fut.result()
                        except Exception as e:
                            logger.error(f"Error publishing keyframe: {e}", exc_info=True)

                    future.add_done_callback(handle_publish_error)
                else:
                    logger.warning("Event loop not available, skipping keyframe publish")

            # Update segment counter
            self.segment_frame_count += 1

            # Log FPS every 30 frames
            if frame_number % 30 == 0:
                avg_fps = sum(self.fps_samples) / len(self.fps_samples) if self.fps_samples else 0
                logger.info(f"SLAM3R FPS: {avg_fps:.1f} | Frames: {frame_number} | "
                          f"Keyframes: {self.keyframe_count} | Segment frames: {self.segment_frame_count}")

        except Exception as e:
            logger.error(f"Error processing frame {frame_number}: {e}", exc_info=True)

    async def connect_rabbitmq(self):
        """Establish RabbitMQ connection for keyframe publishing only."""
        rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
        rabbitmq_port = int(os.getenv("RABBITMQ_PORT", "5672"))

        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Import aio_pika here since we still need it for keyframe publishing
                import aio_pika

                self.rabbitmq_connection = await aio_pika.connect_robust(
                    f"amqp://guest:guest@{rabbitmq_host}:{rabbitmq_port}/"
                )
                self.rabbitmq_channel = await self.rabbitmq_connection.channel()

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

                logger.info(f"Connected to RabbitMQ at {rabbitmq_host}:{rabbitmq_port} for keyframe publishing")
                return

            except Exception as e:
                logger.error(f"RabbitMQ connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(5)
                else:
                    logger.warning("Failed to connect to RabbitMQ - keyframe publishing will be disabled")
    
    async def _publish_keyframe(self, keyframe_data: Dict, timestamp: int):
        """Publish keyframe to mesh_service via shared memory."""
        try:
            logger.info(f"Publishing keyframe {self.keyframe_count} with frame_id {keyframe_data['frame_id']}")
            
            # Debug the incoming keyframe data
            pts3d_world = keyframe_data.get('pts3d_world')
            conf_world = keyframe_data.get('conf_world')
            pts3d_cam = keyframe_data.get('pts3d_cam')
            conf_cam = keyframe_data.get('conf_cam')
            
            if pts3d_world is None:
                logger.error("pts3d_world is None in keyframe data!")
                return
            if conf_world is None:
                logger.error("conf_world is None in keyframe data!")
                return
                
            logger.info(f"pts3d_world shape: {pts3d_world.shape}, dtype: {pts3d_world.dtype}")
            logger.info(f"conf_world shape: {conf_world.shape}, dtype: {conf_world.dtype}")
            
            if self.keyframe_publisher and STREAMING_AVAILABLE:
                # Estimate camera pose using SVD alignment between camera and world points
                pose = self._estimate_camera_pose(keyframe_data)
                
                # DEBUG: Log the estimated pose
                logger.info(f"[SLAM3R POSE DEBUG] Keyframe data keys: {list(keyframe_data.keys())}")
                logger.info(f"[SLAM3R POSE DEBUG] Using SVD-estimated pose")
                logger.info(f"[SLAM3R POSE DEBUG] Pose shape: {pose.shape}")
                logger.info(f"[SLAM3R POSE DEBUG] Pose translation: [{pose[0,3]:.3f}, {pose[1,3]:.3f}, {pose[2,3]:.3f}]")
                is_identity = np.allclose(pose, np.eye(4), atol=1e-6)
                logger.info(f"[SLAM3R POSE DEBUG] Is identity matrix: {is_identity}")
                logger.info(f"[SLAM3R POSE DEBUG] Pose value:\n{pose}")
                
                keyframe = {
                    'timestamp': timestamp,
                    'frame_id': keyframe_data['frame_id'],
                    'keyframe_id': self.keyframe_count,
                    'pts3d_world': pts3d_world.cpu().numpy(),
                    'conf_world': conf_world.cpu().numpy(),
                    'pose': pose,
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
                
                # Additional validation: filter corrupted points
                max_coord = 100.0  # 100m scene bounds
                valid_mask = np.all(np.isfinite(filtered_pts), axis=1)
                valid_mask &= np.all(np.abs(filtered_pts) < max_coord, axis=1)
                
                corrupted_count = (~valid_mask).sum()
                if corrupted_count > 0:
                    logger.warning(f"Filtering {corrupted_count} corrupted points with extreme values")
                    filtered_pts = filtered_pts[valid_mask]
                    # Need to filter corresponding confidence values for color mapping
                    if 'rgb_image' in keyframe_data and keyframe_data['rgb_image'] is not None:
                        # Update mask to reflect additional filtering
                        original_indices = np.where(mask)[0]
                        mask[original_indices[~valid_mask]] = False
                
                # Check if we have any valid points
                if len(filtered_pts) == 0:
                    logger.warning(f"No valid points for keyframe {self.keyframe_count} after filtering. "
                                 f"Max conf: {conf.max():.3f}, min: {conf.min():.3f}, threshold: {conf_thresh}")
                    return
                
                # Save point cloud for debugging
                if self.save_pcd and self.keyframe_count % 10 == 0:
                    # Get colors if available
                    debug_colors = None
                    if 'rgb_image' in keyframe_data and keyframe_data['rgb_image'] is not None:
                        rgb_image = keyframe_data['rgb_image']
                        H, W = conf_np.shape[-2:] if conf_np.ndim >= 2 else (224, 224)
                        if rgb_image.shape[:2] != (H, W):
                            rgb_resized = cv2.resize(rgb_image, (W, H), interpolation=cv2.INTER_LINEAR)
                        else:
                            rgb_resized = rgb_image
                        rgb_flat = rgb_resized.reshape(-1, 3)
                        debug_colors = rgb_flat[mask]
                    self._save_debug_pointcloud(filtered_pts, conf[mask], self.keyframe_count, debug_colors)
                
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
                self.keyframe_count += 1
                
        except Exception as e:
            logger.error(f"Failed to publish keyframe: {e}")
    
    
    def _save_debug_pointcloud(self, points: np.ndarray, confidence: np.ndarray, keyframe_id: int, colors: np.ndarray = None):
        """Save point cloud to PLY file for debugging using trimesh."""
        try:
            debug_dir = "/debug_output"
            os.makedirs(debug_dir, exist_ok=True)
            
            filename = os.path.join(debug_dir, f"slam3r_keyframe_{keyframe_id:06d}.ply")
            
            # Create metadata dict with confidence values
            metadata = {
                'confidence_min': float(confidence.min()),
                'confidence_max': float(confidence.max()),
                'confidence_mean': float(confidence.mean()),
                'keyframe_id': keyframe_id,
                'point_count': len(points)
            }
            
            # If no colors provided, use confidence-based coloring
            if colors is None:
                # Normalize confidence to 0-1 range for coloring
                conf_norm = (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-6)
                # Create red-to-green gradient based on confidence
                colors = np.zeros((len(points), 3))
                colors[:, 0] = (1 - conf_norm) * 255  # Red for low confidence
                colors[:, 1] = conf_norm * 255        # Green for high confidence
                colors = colors.astype(np.uint8)
            
            # Ensure colors are in 0-255 range for trimesh
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            
            # Create point cloud with trimesh
            pcd = trimesh.points.PointCloud(points, colors=colors)
            
            # Add metadata
            for key, value in metadata.items():
                pcd.metadata[key] = value
            
            # Export to PLY
            pcd.export(filename)
            
            logger.info(f"[PCD] Saved point cloud to {filename} ({len(points)} points)")
            logger.info(f"[PCD] Point cloud bounds - Min: {points.min(axis=0)}, Max: {points.max(axis=0)}")
            logger.info(f"[PCD] Confidence stats - Min: {confidence.min():.3f}, Max: {confidence.max():.3f}, Mean: {confidence.mean():.3f}")
            
        except Exception as e:
            logger.error(f"Failed to save debug point cloud: {e}", exc_info=True)
    
    def _estimate_camera_pose(self, keyframe_data: Dict) -> np.ndarray:
        """Estimate camera pose using SVD alignment between camera and world points."""
        pts3d_cam = keyframe_data.get('pts3d_cam')
        pts3d_world = keyframe_data.get('pts3d_world')
        conf_cam = keyframe_data.get('conf_cam')
        conf_world = keyframe_data.get('conf_world')
        
        if pts3d_cam is None or pts3d_world is None:
            logger.warning("Missing pts3d_cam or pts3d_world, returning identity pose")
            return np.eye(4)
        
        # Convert tensors to numpy and handle dimensions
        if torch.is_tensor(pts3d_cam):
            P_cam = pts3d_cam.squeeze(0).cpu().numpy()
        else:
            P_cam = np.asarray(pts3d_cam)
            if P_cam.ndim > 3:
                P_cam = P_cam.squeeze(0)
                
        if torch.is_tensor(pts3d_world):
            P_world = pts3d_world.squeeze(0).cpu().numpy()
        else:
            P_world = np.asarray(pts3d_world)
            if P_world.ndim > 3:
                P_world = P_world.squeeze(0)
        
        # Ensure proper shape (-1, 3)
        P_cam = P_cam.reshape(-1, 3)
        P_world = P_world.reshape(-1, 3)
        
        # Get confidence masks
        if conf_cam is not None and torch.is_tensor(conf_cam):
            conf_cam_flat = conf_cam.squeeze().cpu().numpy().reshape(-1)
        else:
            conf_cam_flat = np.ones(len(P_cam))
            
        if conf_world is not None and torch.is_tensor(conf_world):
            conf_world_flat = conf_world.squeeze().cpu().numpy().reshape(-1)
        else:
            conf_world_flat = np.ones(len(P_world))
        
        # Use confidence thresholds from config
        conf_thres_i2p = self.config.get('recon_pipeline', {}).get('conf_thres_i2p', 10.0)
        conf_thres_l2w = self.config.get('recon_pipeline', {}).get('conf_thres_l2w', 12.0)
        
        # Create mask for valid points
        mask = ((conf_cam_flat > conf_thres_i2p) & 
                (conf_world_flat > conf_thres_l2w))
        
        # Fallback to lower threshold if too few points
        if mask.sum() < 3:
            mask = ((conf_cam_flat > conf_thres_i2p) & 
                    (conf_world_flat > 0.5 * conf_thres_l2w))
        
        # Estimate rigid transform using SVD
        if mask.sum() >= 3:
            R, t = estimate_rigid_transform_svd(P_cam[mask], P_world[mask])
        else:
            logger.warning(f"Too few valid points for SVD ({mask.sum()}), using identity")
            R, t = np.eye(3), np.zeros((3, 1))
        
        # Build 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.squeeze()
        
        # Additional debug information
        logger.info(f"[SLAM3R POSE DEBUG] Estimated camera pose (translation): [{T[0,3]:.4f}, {T[1,3]:.4f}, {T[2,3]:.4f}]")
        logger.info(f"[SLAM3R POSE DEBUG] Valid points for SVD: {mask.sum()} out of {len(P_cam)}")
        
        # Log point cloud statistics for debugging
        if mask.sum() > 0:
            cam_center = P_cam[mask].mean(axis=0)
            world_center = P_world[mask].mean(axis=0)
            logger.info(f"[SLAM3R POSE DEBUG] Camera space center: [{cam_center[0]:.3f}, {cam_center[1]:.3f}, {cam_center[2]:.3f}]")
            logger.info(f"[SLAM3R POSE DEBUG] World space center: [{world_center[0]:.3f}, {world_center[1]:.3f}, {world_center[2]:.3f}]")
            
            # Check if the transformation makes sense
            transformed_center = (R @ cam_center.reshape(3, 1) + t).flatten()
            error = np.linalg.norm(transformed_center - world_center)
            logger.info(f"[SLAM3R POSE DEBUG] Transform validation error: {error:.4f}")
        
        return T
    
    async def run_async(self):
        """Override parent's run_async to connect RabbitMQ before starting."""
        # Store reference to event loop for thread-safe async operations
        self._event_loop = asyncio.get_running_loop()
        logger.info("Event loop captured for thread-safe async operations")

        # Connect to RabbitMQ for keyframe publishing
        await self.connect_rabbitmq()

        # Call parent's run_async to handle WebSocket connection and frame processing
        await super().run_async()

    def run(self):
        """Override parent's run to add cleanup logic."""
        self.is_running = True
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            logger.info("Shutting down SLAM3R processor...")
        finally:
            self.stop()

            # Clean up shared memory
            if self.keyframe_publisher:
                logger.info("Cleaning up shared memory segments...")
                try:
                    self.keyframe_publisher.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up keyframe publisher: {e}")

            # Close RabbitMQ connection
            if self.rabbitmq_connection:
                logger.info("Closing RabbitMQ connection...")
                try:
                    # Create new event loop for cleanup if needed
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.rabbitmq_connection.close())
                except Exception as e:
                    logger.error(f"Error closing RabbitMQ connection: {e}")


if __name__ == "__main__":
    processor = SLAM3RProcessor()
    processor.run()