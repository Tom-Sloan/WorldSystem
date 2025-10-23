"""
Mesh Service - Main entry point.

Real-time SLAM point cloud and video streaming service with efficient
voxel downsampling and Rerun visualization.
"""

import asyncio
import os
import sys
import msgpack
import cv2
import numpy as np
import rerun as rr
from colorama import Fore, Style

from config import config
from utils.logger import setup_logging, get_logger
from io_handlers.shared_memory import SharedMemoryReader
from io_handlers.rabbitmq_consumer import RabbitMQConsumer
from io_handlers.websocket_consumer import WebSocketVideoConsumer
from processing.point_cloud_manager import PointCloudManager
from visualization.rerun_publisher import RerunPublisher
from visualization.blueprint_manager import (
    create_slam_visualization_blueprint,
    create_point_cloud_only_blueprint
)

logger = get_logger(__name__)


class MeshService:
    """
    Main mesh service orchestrator.

    Consumes SLAM keyframes and video frames, performs efficient point cloud
    accumulation with voxel downsampling, and visualizes in Rerun.
    """

    def __init__(self):
        """Initialize mesh service components"""
        logger.info("Initializing Mesh Service...")

        # Components
        self.shm_reader = SharedMemoryReader(config.shm_path)
        self.point_cloud_mgr = PointCloudManager(
            voxel_size=config.voxel_size,
            max_points=config.max_points,
            max_raw_points=config.max_raw_points
        )
        self.rabbitmq = RabbitMQConsumer(config.rabbitmq_url)
        self.rerun_publisher = RerunPublisher(
            log_interval_seconds=config.log_interval_seconds,
            video_log_interval_seconds=config.video_log_interval_seconds
        )

        # WebSocket video consumer (for camera feed)
        ws_url = os.getenv("VIDEO_STREAM_URL", "ws://127.0.0.1:5001/ws/video/consume")
        self.websocket_consumer = WebSocketVideoConsumer(ws_url)
        self.websocket_consumer.set_frame_callback(self.handle_websocket_video_frame)

        # State
        self.last_stats_log = asyncio.get_event_loop().time()
        self.stats_log_interval = 5.0  # Log stats every 5 seconds
        self.running = False

        logger.info("✅ Mesh Service initialized")

    async def handle_slam_keyframe(self, message) -> None:
        """
        Handle SLAM3R keyframe message.

        Args:
            message: RabbitMQ message with keyframe metadata
        """
        async with message.process():
            try:
                # Parse msgpack message
                data = msgpack.unpackb(message.body)

                message_type = data.get('type')
                if message_type != 'keyframe.new':
                    return

                shm_key = data.get('shm_key')
                keyframe_id = data.get('keyframe_id')
                timestamp_ns = data.get('timestamp_ns', 0)

                logger.debug(f"Received keyframe: {keyframe_id} (shm: {shm_key})")

                # Read from shared memory
                keyframe = self.shm_reader.read_keyframe(shm_key)

                if keyframe is None:
                    logger.warning(f"Failed to read keyframe from shared memory: {shm_key}")
                    return

                # Add to point cloud manager
                added = self.point_cloud_mgr.add_keyframe(
                    keyframe_id=keyframe_id,
                    points=keyframe.points,
                    colors=keyframe.colors,
                    pose=keyframe.pose_matrix
                )

                if added:
                    logger.info(
                        f"✅ Added keyframe {keyframe_id}: {keyframe.point_count:,} points"
                    )

                    # Log camera pose
                    self.rerun_publisher.log_camera_pose(
                        pose=keyframe.pose_matrix,
                        keyframe_id=keyframe_id,
                        timestamp_ns=timestamp_ns
                    )

                    # Log point cloud if interval elapsed
                    if self.rerun_publisher.should_log_point_cloud():
                        await self.log_point_cloud_update(timestamp_ns)

                # Optionally unlink shared memory
                if config.unlink_shm:
                    self.shm_reader.unlink(shm_key)

            except Exception as e:
                logger.error(f"Error handling SLAM keyframe: {e}", exc_info=True)

    async def handle_websocket_video_frame(self, frame_rgb: np.ndarray, timestamp_ns: int) -> None:
        """
        Handle video frame from WebSocket (already decoded to RGB).

        Args:
            frame_rgb: RGB frame as numpy array
            timestamp_ns: Frame timestamp in nanoseconds
        """
        try:
            # Check if video logging is enabled
            if not config.enable_video:
                return

            # Check if it's time to log video
            if not self.rerun_publisher.should_log_video():
                return

            # Log to Rerun (frame is already in RGB format)
            self.rerun_publisher.log_video_frame(frame_rgb, timestamp_ns)

            logger.debug(f"Logged video frame: {frame_rgb.shape}")

        except Exception as e:
            logger.error(f"Error handling WebSocket video frame: {e}", exc_info=True)

    async def handle_video_frame(self, message) -> None:
        """
        Handle video frame message from RabbitMQ (legacy, kept for compatibility).

        Args:
            message: RabbitMQ message with video frame
        """
        async with message.process():
            try:
                # Check if video logging is enabled
                if not config.enable_video:
                    return

                # Check if it's time to log video
                if not self.rerun_publisher.should_log_video():
                    return

                # Get timestamp from headers
                timestamp_ns = message.headers.get('timestamp_ns', 0) if message.headers else 0

                # Decode frame based on content type
                content_type = message.content_type or 'application/octet-stream'

                # Server sends JPEG data with content_type="application/octet-stream"
                if content_type in ('image/jpeg', 'application/octet-stream'):
                    # Decode JPEG
                    frame_array = np.frombuffer(message.body, np.uint8)
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

                    if frame is not None:
                        # Convert BGR to RGB for Rerun
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Log to Rerun
                        self.rerun_publisher.log_video_frame(frame_rgb, timestamp_ns)

                        logger.debug(f"Logged video frame: {frame.shape}")
                    else:
                        logger.warning(f"Failed to decode video frame (content_type: {content_type})")

            except Exception as e:
                logger.error(f"Error handling video frame: {e}", exc_info=True)

    async def log_point_cloud_update(self, timestamp_ns: int = None) -> None:
        """
        Log point cloud update to Rerun.

        Args:
            timestamp_ns: Optional timestamp
        """
        try:
            # Get downsampled point cloud
            points, colors = self.point_cloud_mgr.get_downsampled_cloud()

            if len(points) > 0:
                # Log to Rerun
                self.rerun_publisher.log_point_cloud(points, colors, timestamp_ns)

                logger.debug(f"Logged point cloud update: {len(points):,} points")

        except Exception as e:
            logger.error(f"Error logging point cloud: {e}", exc_info=True)

    async def log_statistics(self) -> None:
        """Log statistics to Rerun"""
        try:
            # Get statistics
            stats = self.point_cloud_mgr.get_statistics()
            stats.update(self.rerun_publisher.get_statistics())

            # Log to Rerun
            self.rerun_publisher.log_statistics(stats)

            # Console output
            logger.info(
                f"📊 Stats: keyframes={stats['total_keyframes']}, "
                f"points={stats['display_point_count']:,}/{stats['raw_point_count']:,}, "
                f"mem={stats['memory_estimate_mb']:.1f}MB"
            )

        except Exception as e:
            logger.error(f"Error logging statistics: {e}", exc_info=True)

    async def periodic_stats_logger(self) -> None:
        """Periodically log statistics"""
        while self.running:
            try:
                await asyncio.sleep(self.stats_log_interval)
                await self.log_statistics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stats logger: {e}")

    async def run(self) -> None:
        """Main service loop"""
        try:
            self.running = True

            # Print configuration
            logger.info(f"\n{config}")

            # Initialize Rerun
            logger.info("Initializing Rerun...")
            rr.init("mesh_service", spawn=False)

            if config.rerun_connect:
                try:
                    rr.connect_grpc(url=config.rerun_url)
                    logger.info(f"✅ Connected to Rerun at {config.rerun_url}")
                except Exception as e:
                    logger.warning(f"Could not connect to Rerun: {e}. Logging to memory.")

            # Send blueprint
            if config.enable_video:
                blueprint = create_slam_visualization_blueprint()
            else:
                blueprint = create_point_cloud_only_blueprint()

            rr.send_blueprint(blueprint)
            logger.info("✅ Rerun blueprint sent")

            # Connect to RabbitMQ
            await self.rabbitmq.connect()

            # Start consumers
            await self.rabbitmq.consume_slam_keyframes(self.handle_slam_keyframe)

            # Start WebSocket video consumer if video is enabled
            websocket_task = None
            if config.enable_video:
                websocket_task = asyncio.create_task(self.websocket_consumer.run())
                logger.info("🎥 Started WebSocket video consumer")

            # Start periodic stats logger
            stats_task = asyncio.create_task(self.periodic_stats_logger())

            logger.info(f"\n{Fore.GREEN}{'='*60}")
            logger.info(f"🚀 Mesh Service Running")
            logger.info(f"{'='*60}{Style.RESET_ALL}\n")

            # Keep running
            try:
                await asyncio.Future()  # Run forever
            except KeyboardInterrupt:
                logger.info("Received shutdown signal")

            # Cleanup
            stats_task.cancel()
            await stats_task

            if websocket_task:
                await self.websocket_consumer.stop()
                websocket_task.cancel()
                try:
                    await websocket_task
                except asyncio.CancelledError:
                    pass

        except Exception as e:
            logger.error(f"Fatal error in service: {e}", exc_info=True)
            raise

        finally:
            self.running = False
            await self.rabbitmq.close()
            logger.info("Mesh Service stopped")


async def main():
    """Main entry point"""
    # Setup logging
    setup_logging(config.log_level)

    # Print banner
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"  Mesh Service v2.0.0")
    print(f"  Real-time SLAM Point Cloud Visualization")
    print(f"{'='*60}{Style.RESET_ALL}\n")

    # Create and run service
    service = MeshService()

    try:
        await service.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Service failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
