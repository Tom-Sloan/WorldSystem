#!/usr/bin/env python3
"""
WebSocket Video Consumer Base Class
Base class for services that consume H.264 video via WebSocket
"""

import asyncio
import websockets
import numpy as np
import av
import logging
from abc import ABC, abstractmethod
from io import BytesIO
import threading
import queue
from typing import Optional
import time

logger = logging.getLogger(__name__)

class WebSocketVideoConsumer(ABC):
    """Base class for services that consume H.264 video streams via WebSocket"""
    
    def __init__(self, ws_url: str, service_name: str = "Consumer", frame_skip: int = 1):
        self.ws_url = ws_url
        self.service_name = service_name
        self.frame_skip = frame_skip  # Process every Nth frame (1 = no skip)
        self.is_running = False
        self.frame_count = 0
        self.frames_processed = 0
        self.logger = logging.getLogger(service_name)
        
        # H.264 decoder
        self.codec = av.CodecContext.create('h264', 'r')
        self.codec.thread_type = 'AUTO'  # Use threading for better performance
        
        # Frame queue for processing
        self.frame_queue = queue.Queue(maxsize=30)
        
        # WebSocket connection
        self.websocket = None
        
        # Performance tracking
        self.last_frame_time = time.time()
        self.fps_samples = []
        
    async def connect(self) -> bool:
        """Connect to WebSocket video stream"""
        try:
            self.logger.info(f"[{self.service_name}] Connecting to {self.ws_url}")
            self.websocket = await websockets.connect(
                self.ws_url,
                max_size=10 * 1024 * 1024,  # 10MB max message size
                ping_interval=20,
                ping_timeout=10
            )
            self.logger.info(f"✓ [{self.service_name}] Connected to WebSocket stream")
            return True
        except Exception as e:
            self.logger.error(f"[{self.service_name}] Failed to connect: {e}")
            return False
            
    def decode_h264_frame(self, h264_data: bytes) -> Optional[np.ndarray]:
        """Decode H.264 NAL unit to numpy array"""
        try:
            # Create packet from H.264 data
            packet = av.Packet(h264_data)
            
            # Decode frames
            frames = self.codec.decode(packet)
            
            for frame in frames:
                # Convert to numpy array (BGR for OpenCV compatibility)
                img = frame.to_ndarray(format='bgr24')
                return img
                
        except av.AVError as e:
            # Normal for partial frames or when waiting for keyframe
            self.logger.debug(f"[{self.service_name}] Decode error (normal during init): {e}")
        except Exception as e:
            self.logger.error(f"[{self.service_name}] Unexpected decode error: {e}")
        return None
            
    @abstractmethod
    def process_frame(self, frame: np.ndarray, frame_number: int):
        """Process a single frame - implement in subclass"""
        pass
        
    async def receive_loop(self):
        """Receive H.264 data from WebSocket"""
        sps_pps_received = False
        consecutive_decode_failures = 0
        max_failures = 30  # Wait for up to 30 frames for SPS/PPS
        
        while self.is_running:
            try:
                # Receive H.264 NAL unit
                h264_data = await self.websocket.recv()
                
                if isinstance(h264_data, bytes):
                    # Check for SPS/PPS (usually starts with 0x00000001 followed by 0x67/0x68)
                    if len(h264_data) > 4:
                        nal_type = h264_data[4] & 0x1F
                        if nal_type == 7:  # SPS
                            self.logger.info(f"[{self.service_name}] Received SPS ({len(h264_data)} bytes)")
                            sps_pps_received = True
                        elif nal_type == 8:  # PPS
                            self.logger.info(f"[{self.service_name}] Received PPS ({len(h264_data)} bytes)")
                    
                    # Decode H.264 to frame
                    frame = self.decode_h264_frame(h264_data)
                    
                    if frame is not None:
                        consecutive_decode_failures = 0
                        self.frame_count += 1
                        
                        # Calculate FPS
                        current_time = time.time()
                        if self.last_frame_time > 0:
                            fps = 1.0 / (current_time - self.last_frame_time)
                            self.fps_samples.append(fps)
                            if len(self.fps_samples) > 30:
                                self.fps_samples.pop(0)
                        self.last_frame_time = current_time
                        
                        # Skip frames if needed
                        if self.frame_skip <= 1 or self.frame_count % self.frame_skip == 0:
                            try:
                                self.frame_queue.put_nowait((frame, self.frame_count))
                            except queue.Full:
                                # Drop oldest frame
                                try:
                                    self.frame_queue.get_nowait()
                                    self.frame_queue.put_nowait((frame, self.frame_count))
                                except:
                                    pass
                    else:
                        consecutive_decode_failures += 1
                        if consecutive_decode_failures > max_failures and not sps_pps_received:
                            self.logger.warning(f"[{self.service_name}] No valid frames after {max_failures} attempts, may need SPS/PPS")
                            
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning(f"[{self.service_name}] WebSocket connection closed")
                break
            except Exception as e:
                self.logger.error(f"[{self.service_name}] Error receiving data: {e}")
                await asyncio.sleep(0.1)  # Brief pause before retry
                
    def process_loop(self):
        """Process frames from queue in separate thread"""
        while self.is_running:
            try:
                frame, frame_number = self.frame_queue.get(timeout=1)
                self.process_frame(frame, frame_number)
                self.frames_processed += 1
                
                # Log progress
                if self.frames_processed % 100 == 0:
                    avg_fps = sum(self.fps_samples) / len(self.fps_samples) if self.fps_samples else 0
                    skip_info = f" (skipping {self.frame_skip-1} of {self.frame_skip})" if self.frame_skip > 1 else ""
                    self.logger.info(f"[{self.service_name}] Processed {self.frames_processed} frames{skip_info}, "
                                   f"avg FPS: {avg_fps:.1f}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"[{self.service_name}] Error processing frame: {e}")
                
    async def run_async(self):
        """Main async run method"""
        reconnect_delay = 2
        max_reconnect_delay = 30
        
        while self.is_running:
            if not await self.connect():
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                continue
                
            reconnect_delay = 2  # Reset on successful connection
            
            # Start processing thread
            process_thread = threading.Thread(target=self.process_loop, daemon=True)
            process_thread.start()
            
            try:
                # Run receive loop
                await self.receive_loop()
            except Exception as e:
                self.logger.error(f"[{self.service_name}] Error in receive loop: {e}")
            finally:
                # Wait for queue to drain
                timeout = 5
                start_time = time.time()
                while not self.frame_queue.empty() and time.time() - start_time < timeout:
                    await asyncio.sleep(0.1)
                    
                # Stop processing thread
                self.is_running = False
                process_thread.join(timeout=2)
                
                # Close WebSocket
                if self.websocket:
                    await self.websocket.close()
                    self.websocket = None
                    
            if self.is_running:
                self.logger.info(f"[{self.service_name}] Reconnecting in {reconnect_delay} seconds...")
                await asyncio.sleep(reconnect_delay)
                
    def run(self):
        """Synchronous run method"""
        self.is_running = True
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            self.logger.info(f"[{self.service_name}] Interrupted by user")
        finally:
            self.stop()
            
    def stop(self):
        """Stop processing"""
        self.logger.info(f"[{self.service_name}] Stopping...")
        self.is_running = False
        
        # Close codec if it exists and is open
        if hasattr(self, 'codec') and self.codec is not None:
            try:
                if hasattr(self.codec, 'is_open') and self.codec.is_open:
                    self.codec.close()
                elif not hasattr(self.codec, 'is_open'):
                    # Some PyAV versions don't have is_open, try to close anyway
                    self.codec.close()
            except ValueError as e:
                # Already closed, that's fine
                self.logger.debug(f"Codec already closed: {e}")
            except Exception as e:
                self.logger.error(f"Error closing codec: {e}")
            
        self.logger.info(f"[{self.service_name}] Stopped. Total frames: {self.frame_count}, "
                        f"Processed: {self.frames_processed}")


class WebSocketVideoCapture:
    """OpenCV VideoCapture-compatible WebSocket video stream for drop-in replacement"""
    
    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self.consumer = _WebSocketVideoCaptureConsumer(ws_url)
        self.latest_frame = None
        self._is_opened = False
        
        # Start consumer in background
        self.thread = threading.Thread(target=self._run_consumer, daemon=True)
        self.thread.start()
        
        # Wait for first frame or timeout
        timeout = 10  # seconds
        start_time = time.time()
        while self.consumer.latest_frame is None and time.time() - start_time < timeout:
            time.sleep(0.1)
            
        if self.consumer.latest_frame is not None:
            self._is_opened = True
            logger.info(f"WebSocketVideoCapture ready, connected to {ws_url}")
        else:
            logger.error(f"WebSocketVideoCapture failed to receive frames from {ws_url}")
        
    def _run_consumer(self):
        self.consumer.run()
        
    def read(self):
        """OpenCV-compatible read method"""
        if self.consumer.latest_frame is not None:
            frame = self.consumer.latest_frame.copy()
            return True, frame
        return False, None
        
    def isOpened(self):
        """Check if stream is opened"""
        return self._is_opened and self.consumer.is_running
        
    def release(self):
        """Release resources"""
        self.consumer.stop()
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        self._is_opened = False
        
    def get(self, prop_id):
        """Get video property (minimal implementation)"""
        if prop_id == cv2.CAP_PROP_FPS:
            if self.consumer.fps_samples:
                return sum(self.consumer.fps_samples) / len(self.consumer.fps_samples)
            return 30.0  # Default
        elif prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            if self.consumer.latest_frame is not None:
                return self.consumer.latest_frame.shape[1]
            return 0
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            if self.consumer.latest_frame is not None:
                return self.consumer.latest_frame.shape[0]
            return 0
        return 0
        
        
class _WebSocketVideoCaptureConsumer(WebSocketVideoConsumer):
    """Internal consumer for WebSocketVideoCapture"""
    
    def __init__(self, ws_url: str):
        super().__init__(ws_url, "VideoCapture", frame_skip=1)
        self.latest_frame = None
        
    def process_frame(self, frame: np.ndarray, frame_number: int):
        self.latest_frame = frame


# Import cv2 only if needed for VideoCapture compatibility
try:
    import cv2
except ImportError:
    logger.warning("OpenCV (cv2) not available, WebSocketVideoCapture will not work")
    cv2 = None