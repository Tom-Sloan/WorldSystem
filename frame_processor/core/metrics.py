"""
Comprehensive metrics collection for video processing.

This module provides detailed Prometheus metrics for monitoring
all aspects of the video processing pipeline.
"""

from prometheus_client import Counter, Gauge, Histogram, Summary, Info
from typing import Dict, Any, Optional
import time
import psutil
import GPUtil
from dataclasses import dataclass
import asyncio

from .utils import get_logger
from .config import Config

logger = get_logger(__name__)


# ========== Core Processing Metrics ==========

# Frame processing
frames_received = Counter(
    'frame_processor_frames_received_total',
    'Total number of frames received',
    ['stream_id', 'source']
)

frames_processed = Counter(
    'frame_processor_frames_processed_total',
    'Total number of frames processed',
    ['stream_id', 'tracker']
)

frames_dropped = Counter(
    'frame_processor_frames_dropped_total',
    'Total number of frames dropped',
    ['stream_id', 'reason']
)

processing_time = Histogram(
    'frame_processor_processing_time_seconds',
    'Frame processing time in seconds',
    ['stream_id', 'stage'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)

frame_resolution = Histogram(
    'frame_processor_frame_resolution_pixels',
    'Frame resolution in pixels',
    ['stream_id', 'dimension'],
    buckets=(240, 360, 480, 720, 1080, 1440, 2160, 4320)
)

# ========== Video Tracking Metrics ==========

active_tracks = Gauge(
    'frame_processor_active_tracks',
    'Number of active tracks',
    ['stream_id', 'tracker']
)

track_lifetime = Histogram(
    'frame_processor_track_lifetime_seconds',
    'Lifetime of tracks in seconds',
    ['stream_id'],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600)
)

track_confidence = Histogram(
    'frame_processor_track_confidence',
    'Track confidence scores',
    ['stream_id'],
    buckets=(0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
)

prompts_generated = Counter(
    'frame_processor_prompts_generated_total',
    'Total prompts generated',
    ['stream_id', 'strategy']
)

masks_generated = Counter(
    'frame_processor_masks_generated_total',
    'Total masks generated',
    ['stream_id']
)

# ========== Memory Tree Metrics ==========

memory_tree_branches = Gauge(
    'frame_processor_memory_tree_branches',
    'Number of active memory tree branches',
    ['stream_id']
)

memory_tree_depth = Gauge(
    'frame_processor_memory_tree_depth',
    'Current memory tree depth',
    ['stream_id']
)

memory_tree_pruning = Counter(
    'frame_processor_memory_tree_pruning_total',
    'Number of branches pruned',
    ['stream_id', 'reason']
)

# ========== Performance Metrics ==========

current_fps = Gauge(
    'frame_processor_current_fps',
    'Current frames per second',
    ['stream_id']
)

target_fps_achievement = Gauge(
    'frame_processor_target_fps_achievement_ratio',
    'Ratio of current to target FPS',
    ['stream_id']
)

quality_adjustments = Counter(
    'frame_processor_quality_adjustments_total',
    'Number of quality adjustments made',
    ['stream_id', 'adjustment_type']
)

# ========== API Metrics ==========

api_calls = Counter(
    'frame_processor_api_calls_total',
    'Total API calls made',
    ['api_type', 'status']
)

api_response_time = Histogram(
    'frame_processor_api_response_time_seconds',
    'API response time',
    ['api_type'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
)

api_cache_hits = Counter(
    'frame_processor_api_cache_hits_total',
    'API cache hits',
    ['cache_type']
)

api_batch_size = Histogram(
    'frame_processor_api_batch_size',
    'API batch sizes',
    ['api_type'],
    buckets=(1, 5, 10, 20, 50, 100)
)

# ========== Stream Management Metrics ==========

active_streams = Gauge(
    'frame_processor_active_streams',
    'Number of active streams'
)

stale_streams = Gauge(
    'frame_processor_stale_streams',
    'Number of stale streams'
)

stream_lifetime = Histogram(
    'frame_processor_stream_lifetime_seconds',
    'Stream lifetime in seconds',
    buckets=(10, 60, 300, 600, 1800, 3600, 7200)
)

stream_events = Counter(
    'frame_processor_stream_events_total',
    'Stream lifecycle events',
    ['event_type']
)

# ========== GPU Metrics ==========

gpu_memory_used = Gauge(
    'frame_processor_gpu_memory_used_mb',
    'GPU memory used in MB',
    ['device_id']
)

gpu_memory_free = Gauge(
    'frame_processor_gpu_memory_free_mb',
    'GPU memory free in MB',
    ['device_id']
)

gpu_utilization = Gauge(
    'frame_processor_gpu_utilization_percent',
    'GPU utilization percentage',
    ['device_id']
)

gpu_temperature = Gauge(
    'frame_processor_gpu_temperature_celsius',
    'GPU temperature in Celsius',
    ['device_id']
)

model_switches = Counter(
    'frame_processor_model_switches_total',
    'Number of model switches',
    ['reason']
)

# ========== System Metrics ==========

cpu_usage = Gauge(
    'frame_processor_cpu_usage_percent',
    'CPU usage percentage'
)

memory_usage = Gauge(
    'frame_processor_memory_usage_mb',
    'Memory usage in MB'
)

# ========== Error Metrics ==========

errors = Counter(
    'frame_processor_errors_total',
    'Total errors',
    ['error_type', 'component']
)

oom_recoveries = Counter(
    'frame_processor_oom_recoveries_total',
    'GPU OOM recovery attempts',
    ['recovery_strategy']
)

# ========== Enhancement Metrics ==========

enhancement_time = Histogram(
    'frame_processor_enhancement_time_seconds',
    'Image enhancement time',
    ['enhancement_type'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1)
)

enhancement_quality = Histogram(
    'frame_processor_enhancement_quality_score',
    'Enhancement quality scores',
    buckets=(0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99)
)


class MetricsCollector:
    """
    Centralized metrics collection and export.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.enabled = config.metrics_enabled
        
        # Component info
        self.component_info = Info(
            'frame_processor_component_info',
            'Component version and configuration'
        )
        
        # Set initial info
        self.component_info.info({
            'version': '2.0.0',
            'video_mode': str(config.video_mode),
            'tracker_type': config.video_tracker_type if config.video_mode else config.tracker_type,
            'profile': config.config_profile
        })
        
        # Background monitoring
        self._monitor_task: Optional[asyncio.Task] = None
        self._monitor_interval = 5.0  # seconds
        
        logger.info(f"Initialized MetricsCollector (enabled={self.enabled})")
    
    async def start_monitoring(self):
        """Start background system monitoring."""
        if not self.enabled or self._monitor_task:
            return
        
        self._monitor_task = asyncio.create_task(self._monitor_system())
        logger.info("Started metrics monitoring")
    
    async def stop_monitoring(self):
        """Stop background monitoring."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_system(self):
        """Monitor system resources."""
        while True:
            try:
                # CPU and memory
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                cpu_usage.set(cpu_percent)
                memory_usage.set(memory.used / (1024 * 1024))  # MB
                
                # GPU metrics
                try:
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        gpu_memory_used.labels(device_id=str(gpu.id)).set(gpu.memoryUsed)
                        gpu_memory_free.labels(device_id=str(gpu.id)).set(gpu.memoryFree)
                        gpu_utilization.labels(device_id=str(gpu.id)).set(gpu.load * 100)
                        gpu_temperature.labels(device_id=str(gpu.id)).set(gpu.temperature)
                except Exception as e:
                    logger.debug(f"GPU metrics collection failed: {e}")
                
                await asyncio.sleep(self._monitor_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                await asyncio.sleep(self._monitor_interval * 2)
    
    # ========== Convenience Methods ==========
    
    def record_frame_received(self, stream_id: str, source: str = "websocket"):
        """Record frame received."""
        if self.enabled:
            frames_received.labels(stream_id=stream_id, source=source).inc()
    
    def record_frame_processed(self, stream_id: str, tracker: str, processing_time_ms: float):
        """Record frame processed."""
        if self.enabled:
            frames_processed.labels(stream_id=stream_id, tracker=tracker).inc()
            processing_time.labels(stream_id=stream_id, stage="total").observe(processing_time_ms / 1000.0)
    
    def record_frame_dropped(self, stream_id: str, reason: str):
        """Record frame dropped."""
        if self.enabled:
            frames_dropped.labels(stream_id=stream_id, reason=reason).inc()
    
    def update_active_tracks(self, stream_id: str, tracker: str, count: int):
        """Update active track count."""
        if self.enabled:
            active_tracks.labels(stream_id=stream_id, tracker=tracker).set(count)
    
    def record_api_call(self, api_type: str, status: str, response_time_s: float):
        """Record API call."""
        if self.enabled:
            api_calls.labels(api_type=api_type, status=status).inc()
            api_response_time.labels(api_type=api_type).observe(response_time_s)
    
    def record_error(self, error_type: str, component: str):
        """Record error."""
        if self.enabled:
            errors.labels(error_type=error_type, component=component).inc()
    
    def update_stream_count(self, active: int, stale: int):
        """Update stream counts."""
        if self.enabled:
            active_streams.set(active)
            stale_streams.set(stale)
    
    def record_model_switch(self, reason: str):
        """Record model switch."""
        if self.enabled:
            model_switches.labels(reason=reason).inc()
    
    def record_oom_recovery(self, strategy: str):
        """Record OOM recovery."""
        if self.enabled:
            oom_recoveries.labels(recovery_strategy=strategy).inc()
    
    def update_fps(self, stream_id: str, current: float, target: float):
        """Update FPS metrics."""
        if self.enabled:
            current_fps.labels(stream_id=stream_id).set(current)
            achievement = current / target if target > 0 else 0
            target_fps_achievement.labels(stream_id=stream_id).set(achievement)
    
    def record_memory_tree_update(self, stream_id: str, branches: int, depth: int):
        """Update memory tree metrics."""
        if self.enabled:
            memory_tree_branches.labels(stream_id=stream_id).set(branches)
            memory_tree_depth.labels(stream_id=stream_id).set(depth)
    
    def record_prompts_generated(self, stream_id: str, strategy: str, count: int):
        """Record prompts generated."""
        if self.enabled:
            prompts_generated.labels(stream_id=stream_id, strategy=strategy).inc(count)
    
    def record_batch_processed(self, api_type: str, batch_size: int):
        """Record API batch processed."""
        if self.enabled:
            api_batch_size.labels(api_type=api_type).observe(batch_size)


# Singleton instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector(config: Optional[Config] = None) -> MetricsCollector:
    """
    Get or create the metrics collector singleton.
    
    Args:
        config: Configuration (required for first initialization)
        
    Returns:
        MetricsCollector instance
    """
    global _metrics_collector
    
    if _metrics_collector is None:
        if config is None:
            raise ValueError("Config required for first metrics collector initialization")
        _metrics_collector = MetricsCollector(config)
    
    return _metrics_collector


@dataclass
class TimedMetric:
    """
    Context manager for timing operations with metrics.
    """
    metric_name: str
    labels: Dict[str, str]
    collector: Optional[MetricsCollector] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if self.collector and self.collector.enabled:
            if self.metric_name == "processing":
                processing_time.labels(**self.labels).observe(duration)
            elif self.metric_name == "enhancement":
                enhancement_time.labels(**self.labels).observe(duration)
            elif self.metric_name == "api":
                api_response_time.labels(**self.labels).observe(duration)
