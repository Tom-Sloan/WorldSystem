"""
Stream lifecycle management.

This module manages the lifecycle of video streams, including:
- Stream registration and cleanup
- Staleness detection
- Resource management
- Stream health monitoring
"""

import asyncio
from typing import Dict, Optional, List, Set, Any, Callable
from dataclasses import dataclass, field
import time
from collections import defaultdict
import weakref

from .utils import get_logger
from .config import Config

logger = get_logger(__name__)


@dataclass
class StreamState:
    """State information for a video stream."""
    stream_id: str
    created_at: float
    last_frame_time: float
    frame_count: int = 0
    error_count: int = 0
    is_active: bool = True
    is_stale: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age_seconds(self) -> float:
        """Stream age in seconds."""
        return time.time() - self.created_at
    
    @property
    def idle_seconds(self) -> float:
        """Time since last frame in seconds."""
        return time.time() - self.last_frame_time
    
    @property
    def average_fps(self) -> float:
        """Average FPS over stream lifetime."""
        if self.age_seconds > 0:
            return self.frame_count / self.age_seconds
        return 0.0


@dataclass
class StreamEvent:
    """Event related to stream lifecycle."""
    stream_id: str
    event_type: str  # created, frame, error, stale, cleanup
    timestamp: float
    data: Optional[Dict[str, Any]] = None


class StreamManager:
    """
    Manages lifecycle of video streams.
    
    Features:
    - Automatic stream registration
    - Staleness detection
    - Resource cleanup
    - Event notification
    - Health monitoring
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Stream tracking
        self.streams: Dict[str, StreamState] = {}
        self._stream_locks: Dict[str, asyncio.Lock] = {}
        
        # Cleanup configuration
        self.stale_timeout = config.stream_stale_timeout_seconds
        self.cleanup_timeout = config.stream_cleanup_timeout_seconds
        
        # Event handling
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_history: List[StreamEvent] = []
        self.max_event_history = 1000
        
        # Resource tracking
        self.resource_refs: Dict[str, List[weakref.ref]] = defaultdict(list)
        
        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self.running = False
        
        logger.info(f"Initialized StreamManager with stale_timeout={self.stale_timeout}s, "
                   f"cleanup_timeout={self.cleanup_timeout}s")
    
    async def start(self):
        """Start stream management tasks."""
        if self.running:
            return
        
        self.running = True
        self._monitor_task = asyncio.create_task(self._monitor_streams())
        self._cleanup_task = asyncio.create_task(self._cleanup_streams())
        
        logger.info("Started stream management tasks")
    
    async def stop(self):
        """Stop stream management tasks."""
        self.running = False
        
        # Cancel tasks
        for task in [self._monitor_task, self._cleanup_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Stopped stream management tasks")
    
    async def register_stream(self, stream_id: str, metadata: Optional[Dict[str, Any]] = None) -> StreamState:
        """
        Register a new stream.
        
        Args:
            stream_id: Unique stream identifier
            metadata: Optional stream metadata
            
        Returns:
            StreamState object
        """
        async with self._get_lock(stream_id):
            if stream_id in self.streams:
                # Reactivate existing stream
                stream = self.streams[stream_id]
                stream.is_active = True
                stream.is_stale = False
                logger.info(f"Reactivated existing stream {stream_id}")
            else:
                # Create new stream
                stream = StreamState(
                    stream_id=stream_id,
                    created_at=time.time(),
                    last_frame_time=time.time(),
                    metadata=metadata or {}
                )
                self.streams[stream_id] = stream
                logger.info(f"Registered new stream {stream_id}")
            
            # Emit event
            await self._emit_event(StreamEvent(
                stream_id=stream_id,
                event_type="created",
                timestamp=time.time(),
                data=metadata
            ))
            
            return stream
    
    async def update_stream(self, stream_id: str, frame_received: bool = True, 
                           error: Optional[Exception] = None) -> Optional[StreamState]:
        """
        Update stream state.
        
        Args:
            stream_id: Stream identifier
            frame_received: Whether a frame was received
            error: Optional error that occurred
            
        Returns:
            Updated StreamState or None if stream not found
        """
        async with self._get_lock(stream_id):
            stream = self.streams.get(stream_id)
            if not stream:
                # Auto-register unknown streams
                stream = await self.register_stream(stream_id)
            
            if frame_received:
                stream.last_frame_time = time.time()
                stream.frame_count += 1
                stream.is_stale = False
                
                # Emit frame event periodically
                if stream.frame_count % 100 == 0:
                    await self._emit_event(StreamEvent(
                        stream_id=stream_id,
                        event_type="frame",
                        timestamp=time.time(),
                        data={"frame_count": stream.frame_count}
                    ))
            
            if error:
                stream.error_count += 1
                await self._emit_event(StreamEvent(
                    stream_id=stream_id,
                    event_type="error",
                    timestamp=time.time(),
                    data={"error": str(error), "error_count": stream.error_count}
                ))
            
            return stream
    
    async def mark_stream_inactive(self, stream_id: str):
        """
        Mark a stream as inactive.
        
        Args:
            stream_id: Stream identifier
        """
        async with self._get_lock(stream_id):
            stream = self.streams.get(stream_id)
            if stream:
                stream.is_active = False
                logger.info(f"Marked stream {stream_id} as inactive")
    
    async def cleanup_stream(self, stream_id: str, force: bool = False) -> bool:
        """
        Clean up a stream and its resources.
        
        Args:
            stream_id: Stream identifier
            force: Force cleanup even if stream is active
            
        Returns:
            True if cleanup was performed
        """
        async with self._get_lock(stream_id):
            stream = self.streams.get(stream_id)
            if not stream:
                return False
            
            if stream.is_active and not force:
                logger.warning(f"Cannot cleanup active stream {stream_id} without force=True")
                return False
            
            # Clean up resources
            await self._cleanup_resources(stream_id)
            
            # Remove stream
            del self.streams[stream_id]
            
            # Remove lock
            if stream_id in self._stream_locks:
                del self._stream_locks[stream_id]
            
            # Emit cleanup event
            await self._emit_event(StreamEvent(
                stream_id=stream_id,
                event_type="cleanup",
                timestamp=time.time(),
                data={
                    "age_seconds": stream.age_seconds,
                    "frame_count": stream.frame_count,
                    "error_count": stream.error_count
                }
            ))
            
            logger.info(f"Cleaned up stream {stream_id} "
                       f"(age={stream.age_seconds:.1f}s, frames={stream.frame_count})")
            
            return True
    
    def register_resource(self, stream_id: str, resource: Any):
        """
        Register a resource associated with a stream.
        Uses weak references to avoid preventing garbage collection.
        
        Args:
            stream_id: Stream identifier
            resource: Resource object to track
        """
        try:
            ref = weakref.ref(resource)
            self.resource_refs[stream_id].append(ref)
        except TypeError:
            # Object doesn't support weak references
            logger.debug(f"Resource {type(resource).__name__} doesn't support weak refs")
    
    async def _cleanup_resources(self, stream_id: str):
        """
        Clean up resources associated with a stream.
        
        Args:
            stream_id: Stream identifier
        """
        if stream_id not in self.resource_refs:
            return
        
        # Get live references
        live_refs = []
        for ref in self.resource_refs[stream_id]:
            obj = ref()
            if obj is not None:
                live_refs.append(obj)
        
        # Clean up each resource
        for obj in live_refs:
            try:
                # Try common cleanup methods
                if hasattr(obj, 'cleanup'):
                    if asyncio.iscoroutinefunction(obj.cleanup):
                        await obj.cleanup()
                    else:
                        obj.cleanup()
                elif hasattr(obj, 'close'):
                    if asyncio.iscoroutinefunction(obj.close):
                        await obj.close()
                    else:
                        obj.close()
            except Exception as e:
                logger.error(f"Error cleaning up resource for {stream_id}: {e}")
        
        # Remove references
        del self.resource_refs[stream_id]
    
    async def _monitor_streams(self):
        """Background task to monitor stream health."""
        while self.running:
            try:
                current_time = time.time()
                
                for stream_id, stream in list(self.streams.items()):
                    if not stream.is_active:
                        continue
                    
                    # Check for staleness
                    if stream.idle_seconds > self.stale_timeout and not stream.is_stale:
                        stream.is_stale = True
                        await self._emit_event(StreamEvent(
                            stream_id=stream_id,
                            event_type="stale",
                            timestamp=current_time,
                            data={"idle_seconds": stream.idle_seconds}
                        ))
                        logger.warning(f"Stream {stream_id} marked as stale "
                                     f"(idle for {stream.idle_seconds:.1f}s)")
                
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in stream monitor: {e}")
                await asyncio.sleep(10.0)
    
    async def _cleanup_streams(self):
        """Background task to cleanup inactive streams."""
        while self.running:
            try:
                cleanup_candidates = []
                
                for stream_id, stream in list(self.streams.items()):
                    # Cleanup criteria
                    should_cleanup = (
                        (not stream.is_active and stream.idle_seconds > 60) or
                        (stream.is_stale and stream.idle_seconds > self.cleanup_timeout) or
                        (stream.error_count > 100 and stream.idle_seconds > 300)
                    )
                    
                    if should_cleanup:
                        cleanup_candidates.append(stream_id)
                
                # Perform cleanup
                for stream_id in cleanup_candidates:
                    try:
                        await self.cleanup_stream(stream_id, force=True)
                    except Exception as e:
                        logger.error(f"Error cleaning up stream {stream_id}: {e}")
                
                await asyncio.sleep(30.0)  # Run every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in stream cleanup: {e}")
                await asyncio.sleep(60.0)
    
    async def _emit_event(self, event: StreamEvent):
        """Emit a stream event to registered handlers."""
        # Store in history
        self.event_history.append(event)
        if len(self.event_history) > self.max_event_history:
            self.event_history.pop(0)
        
        # Call handlers
        handlers = self.event_handlers.get(event.event_type, [])
        handlers.extend(self.event_handlers.get("*", []))  # Wildcard handlers
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """
        Add an event handler.
        
        Args:
            event_type: Event type to handle (or "*" for all)
            handler: Callback function
        """
        self.event_handlers[event_type].append(handler)
    
    def remove_event_handler(self, event_type: str, handler: Callable):
        """
        Remove an event handler.
        
        Args:
            event_type: Event type
            handler: Handler to remove
        """
        if event_type in self.event_handlers:
            self.event_handlers[event_type].remove(handler)
    
    def _get_lock(self, stream_id: str) -> asyncio.Lock:
        """Get or create a lock for a stream."""
        if stream_id not in self._stream_locks:
            self._stream_locks[stream_id] = asyncio.Lock()
        return self._stream_locks[stream_id]
    
    def get_stream_state(self, stream_id: str) -> Optional[StreamState]:
        """Get current state of a stream."""
        return self.streams.get(stream_id)
    
    def get_all_streams(self) -> Dict[str, StreamState]:
        """Get all stream states."""
        return self.streams.copy()
    
    def get_active_streams(self) -> List[str]:
        """Get list of active stream IDs."""
        return [sid for sid, state in self.streams.items() if state.is_active]
    
    def get_stale_streams(self) -> List[str]:
        """Get list of stale stream IDs."""
        return [sid for sid, state in self.streams.items() if state.is_stale]
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get overall stream statistics."""
        total_streams = len(self.streams)
        active_streams = sum(1 for s in self.streams.values() if s.is_active)
        stale_streams = sum(1 for s in self.streams.values() if s.is_stale)
        
        total_frames = sum(s.frame_count for s in self.streams.values())
        total_errors = sum(s.error_count for s in self.streams.values())
        
        # Calculate average metrics
        if self.streams:
            avg_fps = sum(s.average_fps for s in self.streams.values()) / len(self.streams)
            avg_age = sum(s.age_seconds for s in self.streams.values()) / len(self.streams)
        else:
            avg_fps = 0.0
            avg_age = 0.0
        
        return {
            "total_streams": total_streams,
            "active_streams": active_streams,
            "stale_streams": stale_streams,
            "total_frames": total_frames,
            "total_errors": total_errors,
            "average_fps": avg_fps,
            "average_age_seconds": avg_age,
            "event_history_size": len(self.event_history)
        }
    
    def get_recent_events(self, stream_id: Optional[str] = None, 
                         event_type: Optional[str] = None,
                         limit: int = 100) -> List[StreamEvent]:
        """
        Get recent events.
        
        Args:
            stream_id: Filter by stream ID
            event_type: Filter by event type
            limit: Maximum number of events
            
        Returns:
            List of recent events
        """
        events = self.event_history
        
        if stream_id:
            events = [e for e in events if e.stream_id == stream_id]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:]


# Example usage with event handlers
def example_event_handler(event: StreamEvent):
    """Example event handler."""
    if event.event_type == "stale":
        logger.warning(f"Stream {event.stream_id} went stale!")
    elif event.event_type == "cleanup":
        logger.info(f"Stream {event.stream_id} was cleaned up")
