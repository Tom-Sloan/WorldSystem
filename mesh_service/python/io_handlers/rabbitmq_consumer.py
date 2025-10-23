"""
RabbitMQ consumer for SLAM keyframes and video frames.
"""

import aio_pika
import asyncio
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class RabbitMQConsumer:
    """Async RabbitMQ consumer for mesh service"""

    def __init__(self, rabbitmq_url: str):
        """
        Initialize RabbitMQ consumer.

        Args:
            rabbitmq_url: RabbitMQ connection URL (e.g., amqp://guest:guest@localhost:5672/)
        """
        self.url = rabbitmq_url
        self.connection: Optional[aio_pika.RobustConnection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.queues = {}
        logger.info(f"RabbitMQ consumer initialized: {rabbitmq_url}")

    async def connect(self):
        """Establish connection to RabbitMQ"""
        logger.info("Connecting to RabbitMQ...")
        self.connection = await aio_pika.connect_robust(
            self.url,
            timeout=10
        )
        self.channel = await self.connection.channel()
        await self.channel.set_qos(prefetch_count=1)  # Backpressure control
        logger.info("✅ Connected to RabbitMQ")

    async def consume_slam_keyframes(self, callback: Callable) -> None:
        """
        Consume SLAM3R keyframe notifications.

        Args:
            callback: Async callback function(message: aio_pika.IncomingMessage)
        """
        if not self.channel:
            raise RuntimeError("Not connected to RabbitMQ")

        # Declare exchange
        exchange = await self.channel.declare_exchange(
            'slam3r_keyframe_exchange',
            aio_pika.ExchangeType.TOPIC,
            durable=True
        )

        # Declare exclusive queue
        queue = await self.channel.declare_queue(
            'mesh_service_slam_keyframes',
            exclusive=True,
            auto_delete=True
        )

        # Bind to all keyframe messages
        await queue.bind(exchange, routing_key='#')
        self.queues['slam'] = queue

        # Start consuming
        await queue.consume(callback)
        logger.info("📡 Consuming SLAM keyframe messages")

    async def consume_video_frames(self, callback: Callable) -> None:
        """
        Consume video frame messages.

        Args:
            callback: Async callback function(message: aio_pika.IncomingMessage)
        """
        if not self.channel:
            raise RuntimeError("Not connected to RabbitMQ")

        # Declare exchange
        exchange = await self.channel.declare_exchange(
            'video_frames_exchange',
            aio_pika.ExchangeType.FANOUT,
            durable=True
        )

        # Declare exclusive queue
        queue = await self.channel.declare_queue(
            'mesh_service_video_frames',
            exclusive=True,
            auto_delete=True
        )

        # Bind queue
        await queue.bind(exchange)
        self.queues['video'] = queue

        # Start consuming
        await queue.consume(callback)
        logger.info("🎥 Consuming video frame messages")

    async def close(self):
        """Close RabbitMQ connection"""
        if self.connection:
            await self.connection.close()
            logger.info("RabbitMQ connection closed")
