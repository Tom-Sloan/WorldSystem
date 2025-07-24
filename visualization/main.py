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
import logging

from visualization_handler import VisualizationHandler
from blueprint_manager import ViewMode

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VisualizationService:
    """Service for visualizing frame processing results with Rerun."""
    
    def __init__(self):
        self.rabbitmq_url = os.getenv('RABBITMQ_URL', 'amqp://127.0.0.1:5672')
        self.rerun_port = int(os.getenv('RERUN_PORT', '9876'))
        
        # Initialize Rerun and connect to viewer
        rr.init("worldsystem_visualization", spawn=False)
        try:
            # Connect to existing Rerun viewer
            rr.connect_grpc(url=f"rerun+http://127.0.0.1:{self.rerun_port}/proxy")
            logger.info(f"Connected to Rerun viewer on port {self.rerun_port}")
        except Exception as e:
            logger.warning(f"Could not connect to Rerun viewer: {e}. Will log to memory instead.")
        
        # Initialize visualization handler
        self.handler = VisualizationHandler()
        
        self.connection = None
        self.channel = None
    
    async def connect_rabbitmq(self):
        """Connect to RabbitMQ and setup consumers."""
        self.connection = await aio_pika.connect_robust(self.rabbitmq_url)
        self.channel = await self.connection.channel()
        
        # Declare exchange for receiving outputs
        exchange_name = os.getenv('FRAME_PROCESSOR_OUTPUTS_EXCHANGE', 'processed_frames_exchange')
        self.output_exchange = await self.channel.declare_exchange(
            exchange_name,
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
                # Check content type to determine how to process the message
                content_type = message.content_type or 'application/json'
                
                if content_type == 'image/jpeg':
                    # Handle binary JPEG data with metadata in headers
                    import cv2
                    import numpy as np
                    
                    # Decode JPEG image
                    frame_array = np.frombuffer(message.body, np.uint8)
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                    
                    # Extract metadata from headers
                    headers = message.headers or {}
                    data = {
                        'type': 'processed_frame',
                        'frame': frame,
                        'timestamp_ns': headers.get('timestamp_ns', 0),
                        'frame_number': headers.get('frame_number', 0),
                        'processing_time_ms': headers.get('processing_time_ms', 0),
                        'detection_count': headers.get('detection_count', 0),
                        'width': headers.get('width', frame.shape[1] if frame is not None else 0),
                        'height': headers.get('height', frame.shape[0] if frame is not None else 0),
                        'ntp_time': headers.get('ntp_time', 0),
                        'ntp_offset': headers.get('ntp_offset', 0),
                        'class_summary': json.loads(headers.get('class_summary', '{}'))
                    }
                else:
                    # Handle JSON messages
                    data = json.loads(message.body)
                
                # Pass message to handler
                self.handler.process_message(data)
                    
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
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