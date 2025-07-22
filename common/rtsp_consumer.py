#!/usr/bin/env python3
"""
Base RTSP consumer class for all services
"""
import cv2
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)

class RTSPConsumer(ABC):
    """Base class for services that consume RTSP streams with configurable frame skipping"""
    
    def __init__(self, rtsp_url: str, service_name: str, frame_skip: int = 0):
        self.rtsp_url = rtsp_url
        self.service_name = service_name
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.reconnect_delay = 2
        self.frame_count = 0
        self.frame_skip = frame_skip  # Process every Nth frame (0 = no skip)
        self.frames_processed = 0
        
    def connect(self) -> bool:
        """Connect to RTSP stream"""
        try:
            logger.info(f"[{self.service_name}] Connecting to {self.rtsp_url}")
            
            # Explicitly use FFmpeg backend for RTSP
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            
            # Test connection
            ret, _ = self.cap.read()
            if ret:
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(f"[{self.service_name}] Connected! {width}x{height} @ {fps} FPS")
                return True
            
            self.cap.release()
            return False
            
        except Exception as e:
            logger.error(f"[{self.service_name}] Connection failed: {e}")
            return False
    
    def run(self):
        """Main processing loop"""
        self.is_running = True
        
        while self.is_running:
            if not self.cap or not self.cap.isOpened():
                if not self.connect():
                    time.sleep(self.reconnect_delay)
                    self.reconnect_delay = min(self.reconnect_delay * 2, 30)
                    continue
                self.reconnect_delay = 2
            
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning(f"[{self.service_name}] Failed to read frame")
                    self.cap.release()
                    self.cap = None
                    continue
                
                self.frame_count += 1
                
                # Apply frame skipping
                if self.frame_skip > 0 and self.frame_count % self.frame_skip != 0:
                    continue
                
                # Process frame
                self.process_frame(frame, self.frame_count)
                self.frames_processed += 1
                
                # Log progress
                if self.frames_processed % 100 == 0:
                    skip_ratio = f" (skipping {self.frame_skip-1} of {self.frame_skip})" if self.frame_skip > 1 else ""
                    logger.info(f"[{self.service_name}] Processed {self.frames_processed} frames{skip_ratio}")
                    
            except Exception as e:
                logger.error(f"[{self.service_name}] Error processing frame: {e}")
    
    @abstractmethod
    def process_frame(self, frame, frame_number: int):
        """Process a single frame - implement in subclass"""
        pass
    
    def stop(self):
        """Stop processing"""
        self.is_running = False
        if self.cap:
            self.cap.release()