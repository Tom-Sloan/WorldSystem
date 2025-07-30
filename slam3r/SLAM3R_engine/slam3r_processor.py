#!/usr/bin/env python3
"""
Streamlined SLAM3R Processor - Reads from WebSocket instead of RabbitMQ.

This processor:
- Connects to server's WebSocket consumer (/ws/video/consume)
- Receives H.264 binary packets
- Accumulates and parses bytes for robust decoding
- Decodes to RGB frames using PyAV
- Runs SLAM3R pipeline (I2P → L2W)
- Publishes keyframes to mesh_service via shared memory
- Uses StreamingSLAM3R wrapper for clean architecture
"""

import asyncio
import logging
import os
import time
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import torch
import yaml
import trimesh
import websockets
import av  # For H.264 decoding

# Max buffer size to prevent memory issues
MAX_BYTE_BUFFER_SIZE = 1 << 20  # 1MB

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
        
        # WebSocket configuration
        self.ws_url = os.getenv("VIDEO_STREAM_URL", "ws://127.0.0.1:5001/ws/video/consume")
        self.codec = None  # PyAV codec context for H.264 decoding
        self.reconnect_delay = 5  # seconds
        self.byte_buffer = bytearray()  # Accumulate raw H.264 bytes
        
        # RabbitMQ connection (for keyframe publishing)
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
        
        # PCD saving configuration
        self.save_pcd = os.getenv("SLAM3R_SAVE_PCD", "false").lower() == "true"
        self.pcd_output_dir = os.getenv("SLAM3R_PCD_OUTPUT_DIR", "/debug_output/pcd")
        if self.save_pcd:
            os.makedirs(self.pcd_output_dir, exist_ok=True)
            logger.info(f"PCD saving enabled. Output directory: {self.pcd_output_dir}")
    
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
    
    async def _process_h264_packet(self, data: bytes):
        """Process H.264 packet and decode to RGB frames."""
        try:
            # Append data to buffer
            self.byte_buffer.extend(data)
            logger.debug(f"Received {len(data)} bytes, buffer size: {len(self.byte_buffer)}")
            
            # Prevent buffer overflow
            if len(self.byte_buffer) > MAX_BYTE_BUFFER_SIZE:
                logger.warning("Byte buffer overflow - clearing oldest data")
                self.byte_buffer = self.byte_buffer[-MAX_BYTE_BUFFER_SIZE:]
            
            # Parse buffer into valid packets
            packets = self.codec.parse(self.byte_buffer)
            if not packets:
                logger.debug("No valid packets parsed yet")
                return
            
            # Process each packet
            for packet in packets:
                try:
                    frames = self.codec.decode(packet)
                    for frame in frames:
                        img_rgb = frame.to_ndarray(format='rgb24')
                        timestamp = int(time.time_ns())  # TODO: Extract from stream if available
                        result = self.slam3r.process_frame(img_rgb, timestamp)
                        if result and result.get('is_keyframe', False):
                            result['rgb_image'] = img_rgb
                            await self._publish_keyframe(result, timestamp)
                        self.frame_count += 1
                        self.segment_frame_count += 1
                        if self.frame_count % 30 == 0:
                            self._log_fps()
                except av.AVError as e:
                    logger.debug(f"Skipping invalid packet: {e}")
            
            # Update buffer (remove parsed data)
            consumed = sum(len(p) for p in packets)
            self.byte_buffer = self.byte_buffer[consumed:]
            
            # Flush decoder for any remaining frames
            try:
                flushed_frames = self.codec.decode()  # Empty packet flushes
                for frame in flushed_frames:
                    img_rgb = frame.to_ndarray(format='rgb24')
                    timestamp = int(time.time_ns())  # TODO: Extract from stream if available
                    result = self.slam3r.process_frame(img_rgb, timestamp)
                    if result and result.get('is_keyframe', False):
                        result['rgb_image'] = img_rgb
                        await self._publish_keyframe(result, timestamp)
                    self.frame_count += 1
                    self.segment_frame_count += 1
                    if self.frame_count % 30 == 0:
                        self._log_fps()
            except av.AVError:
                pass
                    
        except av.error.InvalidDataError:
            # This is normal for partial packets, skip
            pass
        except Exception as e:
            logger.error(f"Error processing H.264 packet: {e}")
    
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
    
    async def run(self):
        """Main processing loop with WebSocket connection."""
        # Connect to RabbitMQ for keyframe publishing
        await self.connect_rabbitmq()
        
        logger.info(f"Starting WebSocket consumer, connecting to: {self.ws_url}")
        
        while True:
            try:
                # Connect to WebSocket
                async with websockets.connect(self.ws_url) as websocket:
                    logger.info(f"Connected to WebSocket: {self.ws_url}")
                    
                    # Initialize H.264 decoder
                    self.codec = av.CodecContext.create('h264', 'r')
                    self.codec.thread_type = 'AUTO'  # Enable multi-threading
                    self.codec.extradata = None  # Reset extradata for new stream
                    
                    # Process incoming messages
                    async for message in websocket:
                        if isinstance(message, bytes):
                            await self._process_h264_packet(message)
                        else:
                            logger.debug(f"Non-binary WebSocket message received: {message}")
                            
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed. Reconnecting...")
            except Exception as e:
                logger.error(f"WebSocket error: {e}. Reconnecting in {self.reconnect_delay}s...")
                await asyncio.sleep(self.reconnect_delay)
            finally:
                # Clean up codec
                if self.codec:
                    self.codec = None


async def main():
    """Main entry point."""
    processor = SLAM3RProcessor()
    try:
        await processor.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        # Clean up shared memory
        if processor.keyframe_publisher:
            logger.info("Cleaning up shared memory segments...")
            processor.keyframe_publisher.cleanup()
        
        # Close RabbitMQ connection
        if processor.rabbitmq_connection:
            await processor.rabbitmq_connection.close()


if __name__ == "__main__":
    asyncio.run(main())