"""
Visualization service main entry point.
Receives segmentation results and provides Rerun visualization.
"""

import asyncio
import os
import json
import aio_pika
import rerun as rr
from typing import Dict, Any
import numpy as np
import cv2

from core.utils import get_logger

logger = get_logger(__name__)


class VisualizationService:
    """Service for visualizing frame processing results with Rerun."""
    
    def __init__(self):
        self.rabbitmq_url = os.getenv('RABBITMQ_URL', 'amqp://127.0.0.1:5672')
        self.rerun_port = int(os.getenv('RERUN_PORT', '9876'))
        
        # Initialize Rerun
        rr.init("worldsystem_visualization", spawn=False)
        rr.serve(bind=f"0.0.0.0:{self.rerun_port}")
        logger.info(f"Rerun server started on port {self.rerun_port}")
        
        self.connection = None
        self.channel = None
    
    async def connect_rabbitmq(self):
        """Connect to RabbitMQ and setup consumers."""
        self.connection = await aio_pika.connect_robust(self.rabbitmq_url)
        self.channel = await self.connection.channel()
        
        # Declare exchange for receiving outputs
        self.output_exchange = await self.channel.declare_exchange(
            'frame_processor_outputs',
            aio_pika.ExchangeType.FANOUT,
            durable=True
        )
        
        # Create queue
        self.output_queue = await self.channel.declare_queue(
            'visualization_outputs',
            durable=True
        )
        
        # Bind queue to exchange
        await self.output_queue.bind(self.output_exchange)
        
        # Start consuming
        await self.output_queue.consume(self.process_output)
        
        logger.info("Connected to RabbitMQ and started consuming")
    
    async def process_output(self, message: aio_pika.IncomingMessage):
        """Process output messages from frame processor."""
        async with message.process():
            try:
                data = json.loads(message.body)
                
                if data['type'] == 'frame_processed':
                    await self.visualize_frame(data)
                elif data['type'] == 'video_summary_created':
                    await self.log_summary(data)
                    
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def visualize_frame(self, data: Dict[str, Any]):
        """Visualize a processed frame in Rerun."""
        frame_number = data['frame_number']
        
        # Set time context
        rr.set_time_sequence("frame", frame_number)
        rr.set_time_seconds("time", data['timestamp'])
        
        # Load and log visualization if available
        vis_path = data['paths'].get('visualization')
        if vis_path and os.path.exists(vis_path):
            vis_image = cv2.imread(vis_path)
            vis_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
            rr.log("visualization", rr.Image(vis_rgb))
        
        # Log metadata
        metadata_path = data['paths'].get('metadata')
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            # Log object count
            rr.log("metrics/object_count", rr.Scalar(len(metadata['objects'])))
            
            # Log text summary
            summary = f"Frame {frame_number}: {len(metadata['objects'])} objects detected"
            rr.log("summary", rr.TextLog(summary))
    
    async def log_summary(self, data: Dict[str, Any]):
        """Log session summary."""
        summary_text = f"""
Session: {data['session_id']}
Total frames: {data['num_frames']}
Video saved: {data['video_path']}
        """
        rr.log("session_summary", rr.TextDocument(summary_text))
    
    async def run(self):
        """Main run loop."""
        await self.connect_rabbitmq()
        
        logger.info("Visualization service running...")
        
        # Keep running
        try:
            await asyncio.Future()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            if self.connection:
                await self.connection.close()


async def main():
    """Main entry point."""
    service = VisualizationService()
    await service.run()


if __name__ == "__main__":
    # Setup logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run service
    asyncio.run(main())