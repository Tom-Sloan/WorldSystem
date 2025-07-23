#!/usr/bin/env python3
"""
Grounded-SAM-2 Frame Processor with Rerun and API Integration
Consumes RTSP stream and performs open-vocabulary object detection and tracking
Maintains all existing Rerun visualization and API functionality
"""

import os
import cv2
import torch
import numpy as np
import time
import logging
import asyncio
import aio_pika
import json
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Add common to path
sys.path.append('/app/common')
from websocket_video_consumer import WebSocketVideoConsumer

# Grounded-SAM-2 imports
sys.path.append('./Grounded-SAM-2')
sys.path.append('./Grounded-SAM-2/grounding_dino')
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict

# Your existing imports for Rerun and API
import rerun as rr
from core.config import Config
from external.api_client import APIClient
from pipeline.enhancer import ImageEnhancer
from visualization.rerun_client import RerunClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroundedSAM2Processor(WebSocketVideoConsumer):
    def __init__(self, ws_url: str):
        # Frame processor: process every frame for best tracking
        frame_skip = int(os.getenv('FRAME_PROCESSOR_SKIP', '1'))
        super().__init__(ws_url, "FrameProcessor", frame_skip)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.setup_models()
        
        # Initialize Rerun
        self.setup_rerun()
        
        # Initialize configuration and API clients
        self.config = Config()
        self.api_client = APIClient(self.config)
        self.image_enhancer = ImageEnhancer(self.config)
        self.rerun_viz = RerunClient(self.config)
        
        # Tracking state
        self.if_init = False
        self.object_tracks = defaultdict(list)  # obj_id -> [frames]
        
        # RabbitMQ setup
        self.setup_rabbitmq()
    
    def setup_rerun(self):
        """Initialize Rerun connection"""
        try:
            rr.init("frame_processor", spawn=False)
            rr.connect("localhost:9876")
            logger.info("Connected to Rerun viewer")
        except Exception as e:
            logger.warning(f"Could not connect to Rerun: {e}")
    
    def setup_rabbitmq(self):
        """Setup RabbitMQ connections"""
        # Create and run all async operations in a single coroutine
        async def _async_setup():
            self.rabbit_connection = await aio_pika.connect_robust(
                os.getenv('RABBITMQ_URL', 'amqp://127.0.0.1:5672')
            )
            self.rabbit_channel = await self.rabbit_connection.channel()
            
            # Declare exchanges
            self.processed_exchange = await self.rabbit_channel.declare_exchange(
                'processed_frames_exchange',
                aio_pika.ExchangeType.FANOUT,
                durable=True
            )
            self.api_results_exchange = await self.rabbit_channel.declare_exchange(
                'api_results_exchange',
                aio_pika.ExchangeType.FANOUT,
                durable=True
            )
        
        # Run the async setup in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_async_setup())
        finally:
            # Keep the loop for later async operations
            self.async_loop = loop
    
    def setup_models(self):
        """Initialize Grounded-SAM-2 models"""
        logger.info(f"Setting up models on {self.device}")
        
        # SAM2 setup
        sam2_checkpoint = "/app/Grounded-SAM-2/checkpoints/sam2_hiera_small.pt"
        model_cfg = "sam2_hiera_s.yaml"
        self.sam2_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        
        # Grounding DINO setup
        grounding_dino_config = "/app/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        grounding_dino_checkpoint = "/app/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
        self.grounding_model = load_model(grounding_dino_config, grounding_dino_checkpoint)
        
        logger.info("Models loaded successfully")
    
    def detect_objects(self, image: np.ndarray, text_prompt: str, 
                      box_threshold: float = 0.25, text_threshold: float = 0.2):
        """Detect objects using Grounding DINO"""
        # Convert to PIL Image for Grounding DINO
        from PIL import Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Run detection
        boxes, logits, phrases = predict(
            model=self.grounding_model,
            image=pil_image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        # Convert to numpy for SAM2
        h, w = image.shape[:2]
        boxes_scaled = boxes * torch.tensor([w, h, w, h])
        boxes_xyxy = boxes_scaled.cpu().numpy()
        
        return boxes_xyxy
    
    def process_frame(self, frame: np.ndarray, frame_number: int):
        """Process frame through Grounded-SAM-2 pipeline"""
        
        # Text prompt for open-vocabulary detection - detect everything
        text_prompt = "all objects. item. thing. stuff."
        
        try:
            # First frame initialization
            if not self.if_init:
                logger.info("Initializing SAM2 with first frame")
                self.sam2_predictor.load_first_frame(frame)
                self.if_init = True
                
                # Detect objects with Grounding DINO
                boxes = self.detect_objects(frame, text_prompt)
                logger.info(f"Detected {len(boxes)} objects")
                
                if len(boxes) > 0:
                    # Initialize tracking with detected boxes
                    _, out_obj_ids, out_mask_logits = self.sam2_predictor.add_new_prompt(
                        frame_idx=0,
                        obj_id=0,
                        bbox=boxes
                    )
            else:
                # Track objects in subsequent frames
                out_obj_ids, out_mask_logits = self.sam2_predictor.track(frame)
            
            # Visualize in Rerun
            self.visualize_frame_in_rerun(frame, out_mask_logits, out_obj_ids, frame_number)
            
            # Process tracked objects
            if out_obj_ids is not None and len(out_obj_ids) > 0:
                masks = (out_mask_logits > 0.0).cpu().numpy()
                
                # Publish tracking results to RabbitMQ
                self.publish_tracking_results(out_obj_ids, masks, frame_number)
                
                for obj_id, mask in zip(out_obj_ids, masks):
                    # Crop object with padding
                    cropped = self.crop_object_with_padding(frame, mask, padding=0.2)
                    
                    if cropped is not None:
                        # Store frames for each object
                        self.object_tracks[obj_id].append({
                            'frame': cropped,
                            'frame_num': frame_number,
                            'timestamp': time.time(),
                            'mask': mask
                        })
                        
                        # Process with enhancement and API after collecting enough frames
                        if len(self.object_tracks[obj_id]) >= 30:  # ~1 second at 30fps
                            # Schedule the coroutine on the existing event loop
                            asyncio.run_coroutine_threadsafe(
                                self.process_object_with_api(obj_id),
                                self.async_loop
                            )
                            # Keep only recent frames
                            self.object_tracks[obj_id] = self.object_tracks[obj_id][-30:]
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            import traceback
            traceback.print_exc()
    
    def crop_object_with_padding(self, frame: np.ndarray, mask: np.ndarray, 
                                padding: float = 0.2) -> np.ndarray:
        """Crop object from frame with padding"""
        if mask.sum() == 0:  # Empty mask
            return None
        
        # Find bounding box of mask
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None
        
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        
        # Add padding
        width = x_max - x_min
        height = y_max - y_min
        
        pad_x = int(width * padding / 2)
        pad_y = int(height * padding / 2)
        
        x_min = max(0, x_min - pad_x)
        x_max = min(frame.shape[1], x_max + pad_x)
        y_min = max(0, y_min - pad_y)
        y_max = min(frame.shape[0], y_max + pad_y)
        
        return frame[y_min:y_max, x_min:x_max].copy()
    
    def visualize_frame_in_rerun(self, frame: np.ndarray, mask_logits, obj_ids, frame_number: int):
        """Visualize frame and detections in Rerun"""
        try:
            # Log the frame
            rr.log("frame", rr.Image(frame))
            
            # Log masks if available
            if mask_logits is not None:
                masks = (mask_logits > 0.0).cpu().numpy()
                for i, (obj_id, mask) in enumerate(zip(obj_ids, masks)):
                    rr.log(f"masks/object_{obj_id}", rr.SegmentationImage(mask.astype(np.uint8) * 255))
            
            # Log frame number
            rr.log("frame_number", rr.TextLog(f"Frame {frame_number}"))
            
        except Exception as e:
            logger.debug(f"Rerun visualization error: {e}")
    
    def publish_tracking_results(self, obj_ids, masks, frame_number: int):
        """Publish tracking results to RabbitMQ"""
        message = {
            "frame_number": frame_number,
            "timestamp": time.time(),
            "objects": [
                {
                    "id": int(obj_id),
                    "mask_shape": mask.shape,
                    "area": int(mask.sum())
                }
                for obj_id, mask in zip(obj_ids, masks)
            ]
        }
        
        # Use the existing event loop
        asyncio.set_event_loop(self.async_loop)
        
        try:
            self.async_loop.run_until_complete(
                self.processed_exchange.publish(
                    aio_pika.Message(
                        body=json.dumps(message).encode(),
                        content_type="application/json"
                    ),
                    routing_key=''
                )
            )
        except Exception as e:
            logger.error(f"Error publishing tracking results: {e}")
    
    async def process_object_with_api(self, obj_id: int):
        """Process object with enhancement and API calls"""
        frames_data = self.object_tracks[obj_id]
        frames = [fd['frame'] for fd in frames_data]
        
        logger.info(f"Processing object {obj_id} with {len(frames)} frames")
        
        # Apply enhancement
        enhanced = self.image_enhancer.enhance_from_multiple_frames(frames)
        
        # Call APIs (Google Lens, Perplexity, etc.)
        api_results = await self.api_client.identify_object(enhanced)
        
        # Publish results
        await self.publish_api_results(obj_id, api_results)
        
        # Log to Rerun
        rr.log(f"api_results/object_{obj_id}", rr.TextLog(json.dumps(api_results)))
    
    async def publish_api_results(self, obj_id: int, results: dict):
        """Publish API results to RabbitMQ"""
        message = {
            "object_id": obj_id,
            "timestamp": time.time(),
            "results": results
        }
        
        await self.api_results_exchange.publish(
            aio_pika.Message(body=json.dumps(message).encode()),
            routing_key=''
        )

    def cleanup(self):
        """Clean up async resources"""
        if hasattr(self, 'async_loop') and self.async_loop:
            # Close RabbitMQ connections
            async def _async_cleanup():
                try:
                    if hasattr(self, 'rabbit_channel') and self.rabbit_channel:
                        await self.rabbit_channel.close()
                    if hasattr(self, 'rabbit_connection') and self.rabbit_connection:
                        await self.rabbit_connection.close()
                except Exception as e:
                    logger.error(f"Error during cleanup: {e}")
            
            self.async_loop.run_until_complete(_async_cleanup())
            self.async_loop.close()
    
    def stop(self):
        """Stop processing and clean up"""
        super().stop()
        self.cleanup()


def main():
    """Main entry point"""
    # WebSocket URL
    ws_url = os.getenv('VIDEO_STREAM_URL', 'ws://127.0.0.1:5001/ws/video/consume')
    
    logger.info(f"Connecting to WebSocket stream at {ws_url}")
    
    # Create processor
    processor = GroundedSAM2Processor(ws_url)
    
    try:
        # Run processing loop
        processor.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        processor.stop()

if __name__ == "__main__":
    main()