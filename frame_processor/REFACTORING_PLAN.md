# Frame Processor Refactoring Plan

## Executive Summary

This document outlines a pragmatic refactoring plan for the frame_processor service to transform it from a monolithic 1261-line script into a modular, maintainable system with swappable detection and tracking components. The refactoring prioritizes modularity for core algorithms while avoiding over-engineering, enabling reliable API integration (GCS, Google Lens, Perplexity) and future algorithm upgrades.

## Current State Analysis

### Problems Identified
- **Monolithic Architecture**: Single 1261-line file handling multiple responsibilities
- **Technical Debt**: 91+ debug print statements, commented dead code, incomplete features
- **Poor Error Handling**: Generic exception catches, no circuit breakers for external APIs
- **Memory Issues**: Full frame storage in tracking history, synchronous processing
- **Configuration Chaos**: 20+ environment variables scattered throughout code
- **No Testing**: Zero unit tests, no integration tests
- **API Integration Disabled**: Core features (dimension estimation) turned off by default

### Current Architecture
```
frame_processor.py (1261 lines)
├── Imports and setup (lines 1-99)
├── Prometheus metrics (lines 100-127)
├── EnhancedFrameProcessor class (lines 130-838)
│   ├── __init__ (lines 134-216)
│   ├── process_frame (lines 217-436)
│   ├── log_tracked_objects_to_rerun (lines 438-570)
│   ├── update_track_quality (lines 571-590)
│   ├── enhancement_worker (lines 592-676)
│   ├── log_dimension_results (lines 677-714)
│   ├── update_and_publish_scene_scale (lines 715-751)
│   ├── publish_scene_scale (lines 752-778)
│   ├── log_tracking_status (lines 779-813)
│   └── annotate_frame (lines 814-838)
├── NTP synchronization (lines 841-865)
├── Main execution (lines 866-1261)
```

## Target Architecture

### Pragmatic Modular Structure

```
frame_processor/
├── core/
│   ├── __init__.py
│   ├── config.py            # Centralized configuration with Pydantic
│   ├── interfaces.py        # Abstract base classes for modularity
│   └── utils.py            # Shared utilities, logging setup
│
├── detection/
│   ├── __init__.py
│   ├── base.py             # Abstract detector interface
│   ├── yolo.py             # YOLO implementation (current)
│   ├── detectron2.py       # Future: Detectron2 implementation
│   └── grounding_dino.py   # Future: Grounding DINO implementation
│
├── tracking/
│   ├── __init__.py
│   ├── base.py             # Abstract tracker interface
│   ├── iou_tracker.py      # Current IOU tracker
│   ├── sort.py             # Future: SORT tracker
│   ├── deep_sort.py        # Future: DeepSORT with ReID
│   └── bytetrack.py        # Future: ByteTrack
│
├── external/
│   ├── __init__.py
│   ├── api_client.py       # Unified API client (GCS, Lens, Perplexity)
│   ├── storage.py          # Storage abstraction (GCS with local fallback)
│   └── cache.py            # Simple caching layer
│
├── pipeline/
│   ├── __init__.py
│   ├── processor.py        # Main orchestrator with factory pattern
│   ├── enhancer.py         # Image enhancement (modular)
│   ├── scorer.py           # Frame quality scoring
│   └── publisher.py        # RabbitMQ result publishing
│
├── visualization/
│   ├── __init__.py
│   └── rerun_client.py     # Simplified Rerun integration
│
├── tests/
│   ├── __init__.py
│   ├── test_detectors.py   # Test all detector implementations
│   ├── test_trackers.py    # Test all tracker implementations
│   ├── test_pipeline.py    # Integration tests
│   └── fixtures/           # Test data and mocks
│
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── Dockerfile             # Docker build
├── docker-compose.yml     # Service definition
└── README.md              # Documentation
```

**Total: ~25 files instead of 70+, but with full modularity where it matters**

## Detailed Implementation

### 1. Core Interfaces for Modularity

#### Detection Interface (`detection/base.py`)

```python
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class Detection:
    """Universal detection format regardless of detector."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    embedding: Optional[np.ndarray] = None  # For ReID trackers
    mask: Optional[np.ndarray] = None      # For segmentation

class Detector(ABC):
    """Abstract base class for all detectors."""
    
    @abstractmethod
    async def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a single frame."""
        pass
    
    @abstractmethod
    async def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Detect objects in multiple frames."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Detector name for logging/metrics."""
        pass
    
    @property
    @abstractmethod
    def supported_classes(self) -> List[str]:
        """List of classes this detector can identify."""
        pass
```

#### Tracking Interface (`tracking/base.py`)

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import time

@dataclass
class TrackedObject:
    """Universal tracked object format."""
    id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    
    # Tracking state
    age: int = 0  # Frames since first seen
    time_since_update: int = 0
    hits: int = 1
    
    # For API processing - store only ROI, not full frame
    best_frame: Optional[np.ndarray] = None  # Just the ROI
    best_score: float = 0.0
    created_at: float = field(default_factory=time.time)
    
    # Results
    api_result: Optional[Dict[str, Any]] = None
    
    def should_process(self, current_time: float, 
                      process_after: float = 1.5,
                      reprocess_interval: float = 3.0) -> bool:
        """Check if object should be processed."""
        time_since_creation = current_time - self.created_at
        
        if self.api_result is None:
            return time_since_creation >= process_after
        else:
            # Reprocessing logic if needed
            return False

class Tracker(ABC):
    """Abstract base class for all trackers."""
    
    @abstractmethod
    def update(self, detections: List[Detection], frame: np.ndarray, 
               frame_number: int) -> List[TrackedObject]:
        """Update tracker with new detections."""
        pass
    
    @abstractmethod
    def get_active_tracks(self) -> List[TrackedObject]:
        """Get all active tracks."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tracker name for logging/metrics."""
        pass
```

### 2. Configuration Management

#### Centralized Config (`core/config.py`)

```python
from pydantic import BaseSettings, validator
from typing import Literal, Optional
import os

class Config(BaseSettings):
    """Centralized configuration with modular component selection."""
    
    # Component Selection
    detector_type: Literal["yolo", "detectron2", "grounding_dino"] = "yolo"
    tracker_type: Literal["iou", "sort", "deep_sort", "bytetrack"] = "iou"
    
    # Detector Configuration
    detector_model: str = "yolov11l.pt"
    detector_confidence: float = 0.5
    detector_device: str = "cuda"
    detector_batch_size: int = 1
    
    # Tracker Configuration
    tracker_iou_threshold: float = 0.3
    tracker_max_lost: int = 10
    tracker_max_tracks: int = 100
    process_after_seconds: float = 1.5
    reprocess_interval_seconds: float = 3.0
    
    # API Configuration (Feature Flags)
    use_gcs: bool = False
    use_serpapi: bool = False
    use_perplexity: bool = False
    
    # API Keys
    serpapi_key: Optional[str] = None
    perplexity_key: Optional[str] = None
    gcs_bucket_name: str = "worldsystem-frame-processor"
    gcs_credentials_path: Optional[str] = None
    
    # Enhancement
    enhancement_enabled: bool = True
    enhancement_gamma: float = 1.2
    enhancement_alpha: float = 1.3
    enhancement_beta: int = 20
    
    # System Configuration
    rabbitmq_url: str = "amqp://rabbitmq"
    metrics_port: int = 8003
    log_level: str = "INFO"
    rerun_enabled: bool = True
    
    # Performance
    queue_size: int = 100
    worker_threads: int = 4
    batch_timeout: float = 0.1
    
    @validator('detector_type')
    def validate_detector(cls, v):
        supported = ["yolo", "detectron2", "grounding_dino"]
        if v not in supported:
            raise ValueError(f"Detector must be one of {supported}")
        return v
    
    @validator('tracker_type')
    def validate_tracker(cls, v):
        supported = ["iou", "sort", "deep_sort", "bytetrack"]
        if v not in supported:
            raise ValueError(f"Tracker must be one of {supported}")
        return v
    
    @validator('serpapi_key')
    def validate_serpapi(cls, v, values):
        if values.get('use_serpapi') and not v:
            raise ValueError("SERPAPI_API_KEY required when USE_SERPAPI=true")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

### 3. Detector Implementations

#### YOLO Detector (`detection/yolo.py`)

```python
from ultralytics import YOLO
import torch
from typing import List
import numpy as np
import logging
from .base import Detector, Detection

logger = logging.getLogger(__name__)

class YOLODetector(Detector):
    """YOLO v8/v11 detector implementation."""
    
    def __init__(self, model_path: str, confidence: float, device: str):
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        
        logger.info(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)
        logger.info(f"YOLO model loaded on {device}")
        
    async def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLO detection on single frame."""
        try:
            results = self.model(frame, conf=self.confidence)
            detections = []
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        detections.append(Detection(
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            confidence=float(box.conf[0]),
                            class_id=int(box.cls[0]),
                            class_name=self.model.names[int(box.cls[0])]
                        ))
            
            return detections
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []
    
    async def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Batch detection for efficiency."""
        try:
            results = self.model(frames, conf=self.confidence)
            batch_detections = []
            
            for r in results:
                frame_detections = []
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        frame_detections.append(Detection(
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            confidence=float(box.conf[0]),
                            class_id=int(box.cls[0]),
                            class_name=self.model.names[int(box.cls[0])]
                        ))
                batch_detections.append(frame_detections)
            
            return batch_detections
            
        except Exception as e:
            logger.error(f"YOLO batch detection failed: {e}")
            return [[] for _ in frames]
    
    @property
    def name(self) -> str:
        return f"YOLO-{self.model_path}"
    
    @property
    def supported_classes(self) -> List[str]:
        return list(self.model.names.values())
```

### 4. Tracker Implementations

#### IOU Tracker (`tracking/iou_tracker.py`)

```python
from typing import List, Dict, Tuple
import numpy as np
import logging
from .base import Tracker, TrackedObject
from ..detection.base import Detection
from ..pipeline.scorer import FrameScorer

logger = logging.getLogger(__name__)

class IOUTracker(Tracker):
    """Simple IOU-based tracker with optimized memory usage."""
    
    def __init__(self, iou_threshold: float, max_lost: int, 
                 process_after_seconds: float):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.process_after_seconds = process_after_seconds
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_id = 1
        self.scorer = FrameScorer()
        
    def update(self, detections: List[Detection], frame: np.ndarray, 
               frame_number: int) -> List[TrackedObject]:
        """Update tracks with new detections."""
        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_trks = self._match_detections(detections)
        
        # Update matched tracks
        for det_idx, trk_id in matched:
            detection = detections[det_idx]
            track = self.tracks[trk_id]
            
            track.bbox = detection.bbox
            track.confidence = detection.confidence
            track.time_since_update = 0
            track.hits += 1
            track.age += 1
            
            # Update best frame if needed (store only ROI)
            score = self.scorer.score_frame(frame, detection.bbox)
            if score > track.best_score:
                x1, y1, x2, y2 = detection.bbox
                # Add padding for context
                pad = 10
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(frame.shape[1], x2 + pad)
                y2 = min(frame.shape[0], y2 + pad)
                
                track.best_frame = frame[y1:y2, x1:x2].copy()
                track.best_score = score
        
        # Create new tracks
        for i in unmatched_dets:
            detection = detections[i]
            track = TrackedObject(
                id=self.next_id,
                class_name=detection.class_name,
                bbox=detection.bbox,
                confidence=detection.confidence
            )
            self.tracks[self.next_id] = track
            self.next_id += 1
        
        # Update lost tracks
        for trk_id in unmatched_trks:
            track = self.tracks[trk_id]
            track.time_since_update += 1
            if track.time_since_update > self.max_lost:
                del self.tracks[trk_id]
        
        # Return tracks ready for processing
        ready_tracks = []
        current_time = time.time()
        for track in self.tracks.values():
            if track.should_process(current_time, self.process_after_seconds):
                ready_tracks.append(track)
        
        return ready_tracks
    
    def get_active_tracks(self) -> List[TrackedObject]:
        """Get all active tracks."""
        return list(self.tracks.values())
    
    @property
    def name(self) -> str:
        return "IOU-Tracker"
    
    def _match_detections(self, detections: List[Detection]) -> Tuple:
        """Match detections to existing tracks using IOU."""
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(self.tracks.keys())
        
        # Calculate IOU matrix
        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        track_ids = list(self.tracks.keys())
        
        for d_idx, det in enumerate(detections):
            for t_idx, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                iou_matrix[d_idx, t_idx] = self._calculate_iou(det.bbox, track.bbox)
        
        # Simple greedy matching
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = track_ids.copy()
        
        while True:
            # Find best match
            if iou_matrix.size == 0:
                break
                
            max_iou_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[max_iou_idx]
            
            if max_iou < self.iou_threshold:
                break
            
            d_idx, t_idx = max_iou_idx
            matched.append((unmatched_dets[d_idx], unmatched_trks[t_idx]))
            
            # Remove matched items
            unmatched_dets.pop(d_idx)
            unmatched_trks.pop(t_idx)
            
            # Remove from matrix
            iou_matrix = np.delete(iou_matrix, d_idx, axis=0)
            iou_matrix = np.delete(iou_matrix, t_idx, axis=1)
        
        return matched, unmatched_dets, unmatched_trks
    
    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate Intersection over Union."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
```

### 5. API Client Implementation

#### Simplified API Client (`external/api_client.py`)

```python
import logging
import asyncio
from typing import Optional, Dict, Any
import numpy as np
import cv2
import json
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

class APIClient:
    """Unified API client for GCS, Google Lens, and Perplexity."""
    
    def __init__(self, config):
        self.config = config
        self.cache_dir = Path("/app/cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize clients based on config
        self._init_clients()
    
    def _init_clients(self):
        """Initialize only enabled APIs."""
        self.gcs_client = None
        self.lens_client = None
        
        if self.config.use_gcs and self.config.gcs_credentials_path:
            try:
                from google.cloud import storage
                self.gcs_client = storage.Client()
                self.bucket = self.gcs_client.bucket(self.config.gcs_bucket_name)
                logger.info(f"GCS initialized with bucket: {self.config.gcs_bucket_name}")
            except Exception as e:
                logger.error(f"Failed to initialize GCS: {e}")
                self.config.use_gcs = False
    
    async def identify_object(self, image: np.ndarray, obj_id: int, 
                            class_name: str) -> Optional[Dict[str, Any]]:
        """Complete pipeline: upload → identify → get dimensions."""
        try:
            result = {
                "object_id": obj_id,
                "class_name": class_name,
                "image_url": None,
                "product_name": None,
                "dimensions": None
            }
            
            # Step 1: Upload to GCS (if enabled)
            if self.config.use_gcs:
                image_url = await self._upload_to_gcs(image, obj_id)
                result["image_url"] = image_url
            else:
                # Save locally for testing
                local_path = self.cache_dir / f"obj_{obj_id}.jpg"
                cv2.imwrite(str(local_path), image)
                image_url = str(local_path)
            
            # Step 2: Identify with Google Lens (if enabled)
            if self.config.use_serpapi and self.config.serpapi_key and image_url:
                product_info = await self._identify_with_lens(image_url)
                if product_info:
                    result["product_name"] = product_info.get("name")
            
            # Step 3: Get dimensions with Perplexity (if enabled)
            if (self.config.use_perplexity and self.config.perplexity_key and 
                result["product_name"]):
                dimensions = await self._get_dimensions(result["product_name"])
                if dimensions:
                    result["dimensions"] = dimensions
            
            return result if any([result["product_name"], result["dimensions"]]) else None
            
        except Exception as e:
            logger.error(f"API pipeline failed for object {obj_id}: {e}")
            return None
    
    async def _upload_to_gcs(self, image: np.ndarray, obj_id: int) -> Optional[str]:
        """Upload image to GCS and return signed URL."""
        if not self.gcs_client:
            return None
            
        try:
            # Generate unique blob name
            timestamp = int(time.time())
            blob_name = f"objects/{timestamp}_{obj_id}.jpg"
            
            # Encode image
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Upload
            blob = self.bucket.blob(blob_name)
            blob.upload_from_string(buffer.tobytes(), content_type="image/jpeg")
            
            # Generate signed URL (15 minutes)
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=15),
                method="GET"
            )
            
            logger.info(f"Uploaded object {obj_id} to GCS: {blob_name}")
            return url
            
        except Exception as e:
            logger.error(f"GCS upload failed: {e}")
            return None
    
    async def _identify_with_lens(self, image_url: str) -> Optional[Dict]:
        """Use Google Lens via SerpAPI."""
        cache_key = hashlib.md5(image_url.encode()).hexdigest()
        cache_file = self.cache_dir / f"lens_{cache_key}.json"
        
        # Check cache
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        
        try:
            from serpapi import GoogleSearch
            
            params = {
                "api_key": self.config.serpapi_key,
                "engine": "google_lens",
                "url": image_url,
                "hl": "en"
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            # Parse results
            product_info = self._parse_lens_results(results)
            
            # Cache results
            if product_info:
                with open(cache_file, 'w') as f:
                    json.dump(product_info, f)
            
            return product_info
            
        except Exception as e:
            logger.error(f"Google Lens API failed: {e}")
            return None
    
    def _parse_lens_results(self, results: Dict) -> Optional[Dict]:
        """Extract product information from Lens results."""
        if "knowledge_graph" in results and results["knowledge_graph"]:
            kg = results["knowledge_graph"][0]
            return {
                "name": kg.get("title", ""),
                "type": kg.get("type", ""),
                "description": kg.get("description", "")
            }
        
        if "visual_matches" in results and results["visual_matches"]:
            # Use first visual match as fallback
            match = results["visual_matches"][0]
            return {
                "name": match.get("title", ""),
                "source": match.get("source", "")
            }
        
        return None
    
    async def _get_dimensions(self, product_name: str) -> Optional[Dict]:
        """Get dimensions using Perplexity AI."""
        cache_key = hashlib.md5(product_name.lower().encode()).hexdigest()
        cache_file = self.cache_dir / f"dims_{cache_key}.json"
        
        # Check cache
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        
        try:
            import aiohttp
            
            headers = {
                "Authorization": f"Bearer {self.config.perplexity_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""Find the exact physical dimensions for: {product_name}

Please provide the dimensions in this exact JSON format:
{{
    "width": <number>,
    "height": <number>, 
    "depth": <number>,
    "unit": "cm",
    "width_m": <width in meters>,
    "height_m": <height in meters>,
    "depth_m": <depth in meters>,
    "confidence": <0.0-1.0>
}}"""
            
            payload = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a precise assistant that provides accurate product dimensions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 500
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        dimensions = self._parse_dimensions(data)
                        
                        # Cache if valid
                        if dimensions:
                            with open(cache_file, 'w') as f:
                                json.dump(dimensions, f)
                        
                        return dimensions
                    else:
                        logger.error(f"Perplexity API returned {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Perplexity API failed: {e}")
            return None
    
    def _parse_dimensions(self, response: Dict) -> Optional[Dict]:
        """Parse dimensions from Perplexity response."""
        try:
            import re
            content = response["choices"][0]["message"]["content"]
            
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if json_match:
                dimensions = json.loads(json_match.group())
                
                # Ensure metric conversions
                if "width" in dimensions and "width_m" not in dimensions:
                    dimensions["width_m"] = dimensions["width"] / 100
                if "height" in dimensions and "height_m" not in dimensions:
                    dimensions["height_m"] = dimensions["height"] / 100
                if "depth" in dimensions and "depth_m" not in dimensions:
                    dimensions["depth_m"] = dimensions["depth"] / 100
                
                return dimensions
        except Exception as e:
            logger.error(f"Failed to parse dimensions: {e}")
        
        return None
```

### 6. Main Processing Pipeline

#### Factory Pattern Processor (`pipeline/processor.py`)

```python
import asyncio
import logging
from typing import Dict, Type, List, Any
from dataclasses import dataclass
import time

from ..core.config import Config
from ..detection.base import Detector, Detection
from ..detection.yolo import YOLODetector
from ..tracking.base import Tracker, TrackedObject  
from ..tracking.iou_tracker import IOUTracker
from ..external.api_client import APIClient
from .enhancer import ImageEnhancer
from ..visualization.rerun_client import RerunClient

logger = logging.getLogger(__name__)

# Registry of available components
DETECTORS: Dict[str, Type[Detector]] = {
    "yolo": YOLODetector,
    # Future additions:
    # "detectron2": Detectron2Detector,
    # "grounding_dino": GroundingDINODetector,
}

TRACKERS: Dict[str, Type[Tracker]] = {
    "iou": IOUTracker,
    # Future additions:
    # "sort": SORTTracker,
    # "deep_sort": DeepSORTTracker,
    # "bytetrack": ByteTracker,
}

@dataclass
class ProcessingResult:
    """Result of frame processing."""
    frame_number: int
    detections: List[Detection]
    active_tracks: List[TrackedObject]
    processing_time_ms: float
    api_queue_size: int

class FrameProcessor:
    """Main processor with swappable detection and tracking components."""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Create detector
        detector_class = DETECTORS.get(config.detector_type)
        if not detector_class:
            raise ValueError(f"Unknown detector type: {config.detector_type}")
            
        self.detector = detector_class(
            model_path=config.detector_model,
            confidence=config.detector_confidence,
            device=config.detector_device
        )
        
        # Create tracker
        tracker_class = TRACKERS.get(config.tracker_type)
        if not tracker_class:
            raise ValueError(f"Unknown tracker type: {config.tracker_type}")
            
        self.tracker = tracker_class(
            iou_threshold=config.tracker_iou_threshold,
            max_lost=config.tracker_max_lost,
            process_after_seconds=config.process_after_seconds
        )
        
        # Other components
        self.enhancer = ImageEnhancer(config) if config.enhancement_enabled else None
        self.api_client = APIClient(config)
        self.rerun_client = RerunClient(config) if config.rerun_enabled else None
        
        # Processing queue
        self.api_queue = asyncio.Queue(maxsize=config.queue_size)
        self.api_workers = []
        
        logger.info(f"Initialized processor with {self.detector.name} "
                   f"detector and {self.tracker.name} tracker")
    
    async def start(self):
        """Start background workers."""
        # Start API workers
        for i in range(self.config.worker_threads):
            worker = asyncio.create_task(self._api_worker(i))
            self.api_workers.append(worker)
        
        logger.info(f"Started {len(self.api_workers)} API workers")
    
    async def stop(self):
        """Stop background workers."""
        # Cancel all workers
        for worker in self.api_workers:
            worker.cancel()
        
        # Wait for cancellation
        await asyncio.gather(*self.api_workers, return_exceptions=True)
        logger.info("All workers stopped")
    
    async def process_frame(self, frame: np.ndarray, metadata: Dict[str, Any]) -> ProcessingResult:
        """Process a single frame through the pipeline."""
        start_time = time.time()
        frame_number = metadata.get("frame_number", 0)
        
        # Step 1: Detection
        detections = await self.detector.detect(frame)
        
        # Step 2: Tracking
        ready_for_api = self.tracker.update(detections, frame, frame_number)
        
        # Step 3: Queue for API processing
        for track in ready_for_api:
            if track.best_frame is not None and not track.api_result:
                try:
                    await asyncio.wait_for(
                        self.api_queue.put(track),
                        timeout=0.1  # Don't block
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"API queue full, skipping track {track.id}")
        
        # Step 4: Visualization (if enabled)
        if self.rerun_client:
            self.rerun_client.log_frame(frame, detections, 
                                       self.tracker.get_active_tracks(),
                                       frame_number)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return ProcessingResult(
            frame_number=frame_number,
            detections=detections,
            active_tracks=self.tracker.get_active_tracks(),
            processing_time_ms=processing_time,
            api_queue_size=self.api_queue.qsize()
        )
    
    async def _api_worker(self, worker_id: int):
        """Worker for processing API calls."""
        logger.info(f"API worker {worker_id} started")
        
        while True:
            try:
                # Get track from queue
                track = await self.api_queue.get()
                
                # Enhance image if enabled
                image = track.best_frame
                if self.enhancer:
                    image = self.enhancer.enhance(image)
                
                # Call API pipeline
                result = await self.api_client.identify_object(
                    image, track.id, track.class_name
                )
                
                if result:
                    track.api_result = result
                    logger.info(f"Worker {worker_id}: Identified track {track.id} "
                               f"as {result.get('product_name')}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"API worker {worker_id} error: {e}")
```

### 7. Simple Main Entry Point

#### Clean Entry (`main.py`)

```python
import asyncio
import signal
import logging
import json
import cv2
import numpy as np
import pika

from core.config import Config
from core.utils import setup_logging, get_ntp_time
from pipeline.processor import FrameProcessor
from pipeline.publisher import ResultPublisher

logger = logging.getLogger(__name__)

async def main():
    """Main application entry point."""
    # Load configuration
    config = Config()
    
    # Setup logging
    setup_logging(config.log_level)
    logger.info("Frame Processor starting...")
    
    # Initialize processor
    processor = FrameProcessor(config)
    await processor.start()
    
    # Initialize RabbitMQ
    connection = pika.BlockingConnection(
        pika.URLParameters(config.rabbitmq_url)
    )
    channel = connection.channel()
    
    # Declare exchanges and queues
    channel.exchange_declare(exchange='video_frames_exchange', 
                           exchange_type='fanout', durable=True)
    result = channel.queue_declare(queue='', exclusive=True)
    queue_name = result.method.queue
    channel.queue_bind(exchange='video_frames_exchange', queue=queue_name)
    
    # Result publisher
    publisher = ResultPublisher(channel)
    
    # Shutdown handling
    shutdown_event = asyncio.Event()
    
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        shutdown_event.set()
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Message processing
    async def process_message(ch, method, properties, body):
        try:
            # Decode frame
            np_arr = np.frombuffer(body, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.error("Failed to decode frame")
                return
            
            # Extract metadata
            metadata = {
                "frame_number": properties.headers.get("frame_number", 0),
                "timestamp_ns": properties.headers.get("timestamp_ns"),
            }
            
            # Process frame
            result = await processor.process_frame(frame, metadata)
            
            # Publish results
            await publisher.publish_result(result, properties)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    # Start consuming
    channel.basic_consume(
        queue=queue_name,
        on_message_callback=lambda *args: asyncio.create_task(process_message(*args)),
        auto_ack=True
    )
    
    logger.info("Starting message consumption...")
    
    # Run until shutdown
    try:
        await shutdown_event.wait()
    finally:
        logger.info("Shutting down...")
        await processor.stop()
        channel.stop_consuming()
        connection.close()
    
    logger.info("Frame Processor stopped")

if __name__ == "__main__":
    asyncio.run(main())
```

### 8. Testing Strategy

#### Test Examples (`tests/test_detectors.py`)

```python
import pytest
import numpy as np
from detection.base import Detection
from detection.yolo import YOLODetector

class MockDetector:
    """Mock detector for testing."""
    
    async def detect(self, frame: np.ndarray) -> List[Detection]:
        # Return predictable detections
        return [
            Detection(
                bbox=(100, 100, 200, 200),
                confidence=0.9,
                class_id=0,
                class_name="person"
            )
        ]
    
    @property
    def name(self):
        return "MockDetector"

@pytest.mark.asyncio
async def test_detector_interface():
    """Test detector implements interface correctly."""
    detector = MockDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    detections = await detector.detect(frame)
    assert len(detections) == 1
    assert detections[0].class_name == "person"

@pytest.mark.asyncio 
async def test_tracker_interface():
    """Test tracker with mock detector."""
    from tracking.iou_tracker import IOUTracker
    
    tracker = IOUTracker(iou_threshold=0.3, max_lost=10, 
                        process_after_seconds=1.5)
    
    # Create mock detections
    detections = [
        Detection(bbox=(100, 100, 200, 200), confidence=0.9,
                 class_id=0, class_name="person")
    ]
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ready = tracker.update(detections, frame, 0)
    
    # Should not be ready immediately
    assert len(ready) == 0
    
    # Check active tracks
    active = tracker.get_active_tracks()
    assert len(active) == 1
    assert active[0].class_name == "person"
```

## Implementation Timeline

### Week 1: Core Refactoring
- **Day 1-2**: Set up new structure, implement interfaces and config
- **Day 3**: Implement YOLO detector and IOU tracker with new interfaces
- **Day 4**: Implement API client with proper error handling
- **Day 5**: Create main processor with factory pattern

### Week 2: Testing and Migration
- **Day 1-2**: Write comprehensive tests
- **Day 3**: Run parallel with old system
- **Day 4**: Performance testing and optimization
- **Day 5**: Documentation and deployment

## Benefits of This Approach

1. **Modular Where It Matters**: Detection and tracking are fully swappable
2. **Simple Architecture**: Only 25 files vs 70+
3. **Easy to Understand**: Clear separation of concerns
4. **Future-Proof**: Easy to add new detectors/trackers
5. **Testable**: Clean interfaces make testing straightforward
6. **Maintainable**: Each file has a single, clear purpose

## Migration Strategy

1. **Parallel Development**: Build alongside existing code
2. **Component Testing**: Test each module independently
3. **Integration Testing**: Test full pipeline with real data
4. **Gradual Rollout**: Use feature flags to switch between old/new
5. **Monitoring**: Watch metrics during transition

## Conclusion

This pragmatic refactoring provides the modularity needed for swapping detection and tracking algorithms while avoiding unnecessary complexity. The architecture is clean, testable, and maintainable - ready for both current needs and future enhancements.