#!/usr/bin/env python3
"""
video_storage.py

Handles video stream storage for the WorldSystem. This module:
- Receives H.264 video chunks from RabbitMQ
- Assembles chunks into video segments
- Saves video segments to disk
- Optionally skips saving simulation data based on environment flag
"""

import os
import time
import json
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import numpy as np
from collections import defaultdict
import threading

# Prometheus metrics
from prometheus_client import Counter, Histogram

# Define metrics for video storage
video_segments_saved_counter = Counter(
    "video_storage_segments_saved_total",
    "Total number of video segments saved"
)
video_chunks_received_counter = Counter(
    "video_storage_chunks_received_total",
    "Total number of video chunks received"
)
save_video_segment_hist = Histogram(
    "video_storage_save_segment_seconds",
    "Time spent saving a video segment",
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5]
)


class VideoSegmentWriter:
    """Handles writing video segments from chunks"""
    
    def __init__(self, output_path: Path, width: int, height: int, fps: float = 30.0):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.writer = None
        self.frame_count = 0
        self.start_time = time.time()
        self._lock = threading.Lock()
        
    def initialize(self):
        """Initialize the video writer"""
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create video writer with H.264 codec
        # Try H264 first, fall back to mp4v if not available
        try:
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            self.writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                (self.width, self.height)
            )
            if not self.writer.isOpened():
                raise RuntimeError("H264 codec not available")
        except:
            # Fallback to mp4v
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                (self.width, self.height)
            )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to create video writer for {self.output_path}")
            
        print(f"[VideoWriter] Initialized: {self.output_path} ({self.width}x{self.height}@{self.fps}fps)")
        
    def write_frame(self, frame: np.ndarray):
        """Write a frame to the video segment"""
        with self._lock:
            if self.writer is None:
                self.initialize()
                
            if frame.shape[:2] != (self.height, self.width):
                # Resize frame if dimensions don't match
                frame = cv2.resize(frame, (self.width, self.height))
                
            self.writer.write(frame)
            self.frame_count += 1
            
    def finalize(self):
        """Finalize and close the video segment"""
        with self._lock:
            if self.writer is not None:
                self.writer.release()
                self.writer = None
                
                duration = time.time() - self.start_time
                print(f"[VideoWriter] Finalized: {self.output_path} "
                      f"({self.frame_count} frames, {duration:.2f}s)")
                
                return True
        return False
        
    def __del__(self):
        """Ensure writer is released"""
        self.finalize()


class VideoStorage:
    """Main video storage handler"""
    
    def __init__(self):
        self.recording_path = None
        self.video_writers: Dict[str, VideoSegmentWriter] = {}
        self.chunk_buffers: Dict[str, Dict[int, bytes]] = defaultdict(dict)
        self.segment_duration = 10.0  # seconds
        self.segment_counters: Dict[str, int] = defaultdict(int)
        self.last_segment_times: Dict[str, float] = {}
        
        # Environment flags
        self.save_simulation_data = os.getenv("SAVE_SIMULATION_DATA", "true").lower() == "true"
        
        print(f"[VideoStorage] Initialized with SAVE_SIMULATION_DATA={self.save_simulation_data}")
        
    def set_recording_path(self, path: Path):
        """Set the base recording path"""
        self.recording_path = path
        
    def should_save(self, headers: dict) -> bool:
        """Determine if this video data should be saved"""
        # Check if this is simulation data
        source = headers.get("source", "").lower()
        websocket_id = headers.get("websocket_id", "")
        
        # Skip simulation data if flag is false
        if not self.save_simulation_data:
            if source in ["simulation", "simulator"] or "simulator" in websocket_id:
                print(f"[VideoStorage] Skipping simulation data (SAVE_SIMULATION_DATA=false)")
                return False
                
        return True
        
    def get_segment_path(self, stream_id: str, segment_index: int) -> Path:
        """Generate path for a video segment"""
        if self.recording_path is None:
            raise RuntimeError("Recording path not set")
            
        # Create video segments directory
        segments_dir = self.recording_path / "video_segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate segment filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_stream_{stream_id}_segment_{segment_index:03d}.mp4"
        
        return segments_dir / filename
        
    def process_video_chunk(self, body: bytes, headers: dict):
        """Process a video chunk from RabbitMQ"""
        start_time = time.time()
        
        try:
            # Check if we should save this data
            if not self.should_save(headers):
                return
                
            video_chunks_received_counter.inc()
            
            # Extract metadata
            stream_id = headers.get("websocket_id", "unknown")
            chunk_index = int(headers.get("chunk_index", 0))
            total_chunks = int(headers.get("total_chunks", 1))
            width = int(headers.get("width", 1920))
            height = int(headers.get("height", 1080))
            fps = float(headers.get("fps", 30.0))
            
            # Handle complete chunks (non-fragmented data)
            if total_chunks == 1:
                # This is a complete frame or small video
                self.process_complete_data(body, stream_id, width, height, fps)
            else:
                # This is a chunk of a larger video
                self.process_video_fragment(body, stream_id, chunk_index, total_chunks, 
                                          width, height, fps)
                
        except Exception as e:
            print(f"[VideoStorage] Error processing video chunk: {e}")
        finally:
            elapsed = time.time() - start_time
            save_video_segment_hist.observe(elapsed)
            
    def process_complete_data(self, data: bytes, stream_id: str, 
                            width: int, height: int, fps: float):
        """Process complete video data (single frame or small video)"""
        # Check if this is a decoded frame (JPEG)
        if data[:2] == b'\xff\xd8':  # JPEG magic number
            # Decode JPEG frame
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                self.write_frame_to_segment(frame, stream_id, width, height, fps)
        else:
            # This might be raw H.264 data - for now, skip
            print(f"[VideoStorage] Received non-JPEG data for stream {stream_id}")
            
    def write_frame_to_segment(self, frame: np.ndarray, stream_id: str,
                              width: int, height: int, fps: float):
        """Write a frame to the current video segment"""
        current_time = time.time()
        
        # Check if we need to start a new segment
        if stream_id not in self.video_writers:
            # Create new segment
            self.start_new_segment(stream_id, width, height, fps)
        else:
            # Check if segment duration exceeded
            last_time = self.last_segment_times.get(stream_id, current_time)
            if current_time - last_time > self.segment_duration:
                # Finalize current segment and start new one
                self.finalize_segment(stream_id)
                self.start_new_segment(stream_id, width, height, fps)
                
        # Write frame to current segment
        writer = self.video_writers.get(stream_id)
        if writer:
            writer.write_frame(frame)
            self.last_segment_times[stream_id] = current_time
            
    def start_new_segment(self, stream_id: str, width: int, height: int, fps: float):
        """Start a new video segment"""
        # Increment segment counter
        self.segment_counters[stream_id] += 1
        segment_index = self.segment_counters[stream_id]
        
        # Generate segment path
        segment_path = self.get_segment_path(stream_id, segment_index)
        
        # Create new writer
        writer = VideoSegmentWriter(segment_path, width, height, fps)
        self.video_writers[stream_id] = writer
        self.last_segment_times[stream_id] = time.time()
        
        print(f"[VideoStorage] Started new segment for stream {stream_id}: {segment_path}")
        
    def finalize_segment(self, stream_id: str):
        """Finalize and save a video segment"""
        writer = self.video_writers.get(stream_id)
        if writer:
            if writer.finalize():
                video_segments_saved_counter.inc()
            del self.video_writers[stream_id]
            
    def process_video_fragment(self, fragment: bytes, stream_id: str, 
                             chunk_index: int, total_chunks: int,
                             width: int, height: int, fps: float):
        """Process a fragment of a larger video"""
        # Store fragment in buffer
        buffer_key = f"{stream_id}_video"
        self.chunk_buffers[buffer_key][chunk_index] = fragment
        
        # Check if all chunks received
        if len(self.chunk_buffers[buffer_key]) == total_chunks:
            # Reassemble video
            complete_video = b''.join(
                self.chunk_buffers[buffer_key][i] 
                for i in range(total_chunks)
            )
            
            # Clear buffer
            del self.chunk_buffers[buffer_key]
            
            # Save complete video
            self.save_complete_video(complete_video, stream_id, width, height, fps)
            
    def save_complete_video(self, video_data: bytes, stream_id: str,
                          width: int, height: int, fps: float):
        """Save a complete video file"""
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_stream_{stream_id}_complete.mp4"
            video_path = self.recording_path / "video_segments" / filename
            video_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write video data
            with open(video_path, 'wb') as f:
                f.write(video_data)
                
            print(f"[VideoStorage] Saved complete video: {video_path} ({len(video_data)/1024/1024:.2f} MB)")
            video_segments_saved_counter.inc()
            
        except Exception as e:
            print(f"[VideoStorage] Error saving complete video: {e}")
            
    def cleanup(self):
        """Clean up all active video writers"""
        for stream_id in list(self.video_writers.keys()):
            self.finalize_segment(stream_id)
            
        print("[VideoStorage] Cleanup completed")
        

# Integration with existing data_storage.py
class VideoStorageIntegration:
    """Integration layer for video storage with existing data storage"""
    
    def __init__(self, recording_path: Path):
        self.video_storage = VideoStorage()
        self.video_storage.set_recording_path(recording_path)
        
    def handle_video_message(self, ch, method, properties, body):
        """Callback for video messages from RabbitMQ"""
        if properties and properties.headers:
            # Check if this is video data
            source_format = properties.headers.get("source_format", "")
            if source_format in ["h264_decoded", "h264", "video"]:
                self.video_storage.process_video_chunk(body, properties.headers)
            elif properties.headers.get("source") == "h264_decoded":
                # Decoded frames from H.264
                self.video_storage.process_video_chunk(body, properties.headers)
                
    def cleanup(self):
        """Clean up video storage"""
        self.video_storage.cleanup()