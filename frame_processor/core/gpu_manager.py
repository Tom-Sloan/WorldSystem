"""
GPU memory management and dynamic model switching.

This module provides GPU memory monitoring and automatic model switching
to handle memory pressure and prevent OOM errors.
"""

import torch
import psutil
import GPUtil
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import asyncio
import time
from collections import deque
import threading

from .utils import get_logger
from .config import Config

logger = get_logger(__name__)


@dataclass
class GPUStatus:
    """Current GPU status information."""
    device_id: int
    memory_used_mb: float
    memory_total_mb: float
    memory_free_mb: float
    utilization_percent: float
    temperature: float
    timestamp: float
    
    @property
    def memory_percent(self) -> float:
        """Memory usage percentage."""
        return (self.memory_used_mb / self.memory_total_mb) * 100 if self.memory_total_mb > 0 else 0
    
    @property
    def is_under_pressure(self) -> bool:
        """Check if GPU is under memory pressure."""
        return self.memory_free_mb < 1000 or self.memory_percent > 85


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    size: str  # tiny, small, base, large
    memory_mb: float
    device: str
    handle: Any  # Model object reference
    last_used: float
    usage_count: int = 0


class GPUManager:
    """
    Manages GPU memory and coordinates model switching.
    
    Features:
    - Real-time GPU memory monitoring
    - Automatic model size switching on memory pressure
    - Model caching and lifecycle management
    - OOM prevention strategies
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device_id = self._get_device_id()
        
        # Model size memory requirements (approximate)
        self.model_memory_map = {
            "tiny": 500,    # ~500MB
            "small": 1000,  # ~1GB
            "base": 2000,   # ~2GB
            "large": 4000   # ~4GB
        }
        
        # Loaded models cache
        self.loaded_models: Dict[str, ModelInfo] = {}
        self.current_model_size = config.sam2_model_size
        
        # Memory monitoring
        self.memory_history: deque = deque(maxlen=60)  # 1 minute history
        self.monitoring_enabled = True
        self._monitor_thread = None
        self._monitor_lock = threading.Lock()
        
        # Thresholds
        self.switch_threshold_mb = config.model_switch_threshold_mb
        self.critical_threshold_mb = 500  # Critical threshold
        
        # Model switching state
        self.switch_cooldown = 30.0  # Seconds between switches
        self.last_switch_time = 0.0
        self.switch_count = 0
        
        logger.info(f"Initialized GPUManager for device {self.device_id}")
        logger.info(f"Model switching enabled: {config.enable_dynamic_model_switching}")
        
        # Start monitoring if enabled
        if config.enable_dynamic_model_switching:
            self.start_monitoring()
    
    def _get_device_id(self) -> int:
        """Get CUDA device ID from config or environment."""
        if self.config.detector_device == "cpu":
            return -1
        
        # Extract device ID from cuda:X format
        if self.config.detector_device.startswith("cuda:"):
            try:
                return int(self.config.detector_device.split(":")[1])
            except:
                return 0
        
        return 0  # Default to first GPU
    
    def start_monitoring(self):
        """Start GPU monitoring thread."""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self.monitoring_enabled = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop, 
                daemon=True
            )
            self._monitor_thread.start()
            logger.info("Started GPU monitoring thread")
    
    def stop_monitoring(self):
        """Stop GPU monitoring."""
        self.monitoring_enabled = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Stopped GPU monitoring")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring_enabled:
            try:
                status = self.get_gpu_status()
                if status:
                    with self._monitor_lock:
                        self.memory_history.append(status)
                    
                    # Check if we need to switch models
                    if self.config.enable_dynamic_model_switching:
                        self._check_model_switch(status)
                
                time.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Error in GPU monitoring: {e}")
                time.sleep(5.0)  # Back off on error
    
    def get_gpu_status(self) -> Optional[GPUStatus]:
        """Get current GPU status."""
        if self.device_id < 0:  # CPU mode
            return None
        
        try:
            gpus = GPUtil.getGPUs()
            if self.device_id >= len(gpus):
                logger.error(f"GPU {self.device_id} not found")
                return None
            
            gpu = gpus[self.device_id]
            
            # Also get PyTorch's view of memory
            if torch.cuda.is_available():
                torch_allocated = torch.cuda.memory_allocated(self.device_id) / (1024**2)
                torch_reserved = torch.cuda.memory_reserved(self.device_id) / (1024**2)
            else:
                torch_allocated = 0
                torch_reserved = 0
            
            return GPUStatus(
                device_id=self.device_id,
                memory_used_mb=gpu.memoryUsed,
                memory_total_mb=gpu.memoryTotal,
                memory_free_mb=gpu.memoryFree,
                utilization_percent=gpu.load * 100,
                temperature=gpu.temperature,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Failed to get GPU status: {e}")
            return None
    
    def _check_model_switch(self, status: GPUStatus):
        """Check if we should switch to a different model size."""
        # Check cooldown
        if time.time() - self.last_switch_time < self.switch_cooldown:
            return
        
        # Determine if switch is needed
        if status.memory_free_mb < self.critical_threshold_mb:
            # Critical - switch to smaller model immediately
            target_size = self._get_smaller_model_size(self.current_model_size)
            if target_size:
                logger.warning(f"Critical GPU memory ({status.memory_free_mb}MB free), "
                             f"switching to {target_size} model")
                self._request_model_switch(target_size, priority="critical")
        
        elif status.memory_free_mb < self.switch_threshold_mb:
            # Under pressure - consider switching
            avg_free = self._get_average_free_memory()
            if avg_free < self.switch_threshold_mb:
                target_size = self._get_smaller_model_size(self.current_model_size)
                if target_size:
                    logger.info(f"GPU memory pressure detected ({avg_free:.0f}MB avg free), "
                               f"switching to {target_size} model")
                    self._request_model_switch(target_size, priority="normal")
        
        elif status.memory_free_mb > self.switch_threshold_mb * 2:
            # Plenty of memory - consider upgrading
            if self.switch_count > 0:  # Only upgrade if we've downgraded before
                target_size = self._get_larger_model_size(self.current_model_size)
                if target_size and self._can_fit_model(target_size, status):
                    logger.info(f"GPU memory available ({status.memory_free_mb}MB free), "
                               f"upgrading to {target_size} model")
                    self._request_model_switch(target_size, priority="upgrade")
    
    def _get_smaller_model_size(self, current: str) -> Optional[str]:
        """Get next smaller model size."""
        sizes = ["tiny", "small", "base", "large"]
        try:
            idx = sizes.index(current)
            if idx > 0:
                return sizes[idx - 1]
        except ValueError:
            pass
        return None
    
    def _get_larger_model_size(self, current: str) -> Optional[str]:
        """Get next larger model size."""
        sizes = ["tiny", "small", "base", "large"]
        try:
            idx = sizes.index(current)
            if idx < len(sizes) - 1:
                return sizes[idx + 1]
        except ValueError:
            pass
        return None
    
    def _can_fit_model(self, model_size: str, status: GPUStatus) -> bool:
        """Check if a model size can fit in current memory."""
        required_mb = self.model_memory_map.get(model_size, 2000)
        # Add safety margin
        return status.memory_free_mb > required_mb + 1000
    
    def _get_average_free_memory(self) -> float:
        """Get average free memory over recent history."""
        with self._monitor_lock:
            if not self.memory_history:
                return float('inf')
            
            recent = list(self.memory_history)[-10:]  # Last 10 seconds
            return sum(s.memory_free_mb for s in recent) / len(recent)
    
    def _request_model_switch(self, target_size: str, priority: str = "normal"):
        """Request a model switch (to be handled by the tracker)."""
        self.last_switch_time = time.time()
        self.switch_count += 1
        self.current_model_size = target_size
        
        # This would typically emit an event or call a callback
        # For now, we'll store the request
        switch_request = {
            "target_size": target_size,
            "priority": priority,
            "timestamp": time.time(),
            "reason": f"GPU memory pressure"
        }
        
        # The actual switch would be handled by the video tracker
        logger.info(f"Model switch requested: {switch_request}")
    
    async def allocate_memory(self, size_mb: float, purpose: str = "general") -> bool:
        """
        Try to allocate GPU memory, potentially triggering cleanup.
        
        Args:
            size_mb: Memory size needed in MB
            purpose: What the memory is for (logging)
            
        Returns:
            True if memory is available, False otherwise
        """
        status = self.get_gpu_status()
        if not status:
            return True  # CPU mode
        
        if status.memory_free_mb >= size_mb:
            return True
        
        logger.warning(f"Insufficient GPU memory for {purpose}: "
                      f"need {size_mb}MB, have {status.memory_free_mb}MB free")
        
        # Try to free memory
        freed = await self.free_memory(size_mb)
        
        # Check again
        status = self.get_gpu_status()
        return status and status.memory_free_mb >= size_mb
    
    async def free_memory(self, target_mb: float) -> float:
        """
        Try to free GPU memory.
        
        Args:
            target_mb: Target amount to free
            
        Returns:
            Amount of memory freed in MB
        """
        initial_status = self.get_gpu_status()
        if not initial_status:
            return 0.0
        
        initial_free = initial_status.memory_free_mb
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Get new status
        final_status = self.get_gpu_status()
        if not final_status:
            return 0.0
        
        freed = final_status.memory_free_mb - initial_free
        if freed > 0:
            logger.info(f"Freed {freed:.0f}MB of GPU memory")
        
        return freed
    
    def register_model(self, name: str, size: str, handle: Any, 
                      memory_mb: Optional[float] = None):
        """Register a loaded model."""
        if memory_mb is None:
            memory_mb = self.model_memory_map.get(size, 1000)
        
        self.loaded_models[name] = ModelInfo(
            name=name,
            size=size,
            memory_mb=memory_mb,
            device=self.config.detector_device,
            handle=handle,
            last_used=time.time()
        )
        
        logger.info(f"Registered model {name} (size={size}, memory={memory_mb}MB)")
    
    def unregister_model(self, name: str):
        """Unregister a model."""
        if name in self.loaded_models:
            del self.loaded_models[name]
            logger.info(f"Unregistered model {name}")
    
    def get_model_info(self, name: str) -> Optional[ModelInfo]:
        """Get information about a loaded model."""
        return self.loaded_models.get(name)
    
    def get_recommended_model_size(self) -> str:
        """
        Get recommended model size based on current GPU status.
        
        Returns:
            Recommended model size (tiny, small, base, large)
        """
        status = self.get_gpu_status()
        if not status:
            return self.config.sam2_model_size  # Use configured size for CPU
        
        # Determine based on available memory
        free_mb = status.memory_free_mb
        
        if free_mb < 1500:
            return "tiny"
        elif free_mb < 2500:
            return "small"
        elif free_mb < 4500:
            return "base"
        else:
            return "large"
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics for monitoring."""
        status = self.get_gpu_status()
        
        if not status:
            return {
                "device": "cpu",
                "memory_used_mb": 0,
                "memory_free_mb": 0,
                "memory_percent": 0,
                "models_loaded": len(self.loaded_models),
                "switch_count": self.switch_count
            }
        
        with self._monitor_lock:
            history = list(self.memory_history)
        
        return {
            "device": f"cuda:{status.device_id}",
            "memory_used_mb": status.memory_used_mb,
            "memory_free_mb": status.memory_free_mb,
            "memory_percent": status.memory_percent,
            "utilization_percent": status.utilization_percent,
            "temperature": status.temperature,
            "models_loaded": len(self.loaded_models),
            "current_model_size": self.current_model_size,
            "switch_count": self.switch_count,
            "avg_free_mb_1min": sum(s.memory_free_mb for s in history) / len(history) if history else 0,
            "min_free_mb_1min": min(s.memory_free_mb for s in history) if history else 0,
            "under_pressure": status.is_under_pressure
        }
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_monitoring()
        
        # Clear model references
        self.loaded_models.clear()
        
        # Final cache clear
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("GPU manager cleaned up")


# Singleton instance
_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager(config: Optional[Config] = None) -> GPUManager:
    """
    Get or create the GPU manager singleton.
    
    Args:
        config: Configuration (required for first call)
        
    Returns:
        GPUManager instance
    """
    global _gpu_manager
    
    if _gpu_manager is None:
        if config is None:
            raise ValueError("Config required for first GPU manager initialization")
        _gpu_manager = GPUManager(config)
    
    return _gpu_manager
