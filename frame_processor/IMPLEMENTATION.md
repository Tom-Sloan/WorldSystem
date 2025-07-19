Looking at your code, you already have a `PerformanceTimer` utility, but the terminal output could definitely be more informative and visually appealing. Tools like `pspy` and `tqdm` use ANSI escape codes to create auto-updating displays. The best modern approach is to use the `rich` library, which provides beautiful terminal dashboards.

Here's a comprehensive solution to add detailed timing and create an auto-updating terminal dashboard:

## 1. First, add `rich` to requirements.txt:

```python
# frame_processor/requirements.txt
# Add this:
rich>=13.7.0  # For beautiful terminal output
```

## 2. Create a Performance Monitor Module:

```python
# frame_processor/core/performance_monitor.py
"""
Performance monitoring and terminal dashboard for frame processor.

This module provides real-time performance metrics and a beautiful
terminal interface using the rich library.
"""

import time
import threading
from typing import Dict, List, Optional, Any
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.text import Text
from rich.align import Align
from rich.box import ROUNDED

from core.utils import get_logger


logger = get_logger(__name__)


@dataclass
class TimingStats:
    """Statistics for a timed operation."""
    name: str
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    recent: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def avg_ms(self) -> float:
        """Average time in milliseconds."""
        return self.total_ms / self.count if self.count > 0 else 0.0
    
    @property
    def recent_avg_ms(self) -> float:
        """Recent average (last 100 samples)."""
        return sum(self.recent) / len(self.recent) if self.recent else 0.0
    
    def add_sample(self, duration_ms: float):
        """Add a timing sample."""
        self.count += 1
        self.total_ms += duration_ms
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)
        self.recent.append(duration_ms)


class PerformanceMonitor:
    """
    Central performance monitoring with rich terminal dashboard.
    
    This replaces scattered print statements with a beautiful,
    auto-updating terminal interface.
    """
    
    def __init__(self, update_interval: float = 0.5):
        """
        Initialize performance monitor.
        
        Args:
            update_interval: Dashboard update interval in seconds
        """
        self.update_interval = update_interval
        self.console = Console()
        self.live = None
        self._running = False
        self._update_thread = None
        
        # Timing statistics
        self.timings: Dict[str, TimingStats] = {}
        self._timing_lock = threading.Lock()
        
        # General metrics
        self.metrics = {
            'fps': 0.0,
            'frames_processed': 0,
            'active_tracks': 0,
            'detections_per_frame': 0.0,
            'api_calls': 0,
            'api_cache_hits': 0,
            'memory_mb': 0.0,
            'gpu_memory_mb': 0.0,
        }
        self._metrics_lock = threading.Lock()
        
        # Recent events log (for status messages)
        self.events = deque(maxlen=10)
        self._events_lock = threading.Lock()
        
        # Component status
        self.component_status = {
            'detector': {'name': 'Unknown', 'status': '⏸️ Waiting'},
            'tracker': {'name': 'Unknown', 'status': '⏸️ Waiting'},
            'api': {'status': '⏸️ Waiting'},
            'rabbitmq': {'status': '❌ Disconnected'},
            'rerun': {'status': '❌ Disconnected'},
        }
        self._status_lock = threading.Lock()
        
        # Frame timing breakdown
        self.frame_breakdown = deque(maxlen=60)  # Last 60 frames
        
        # Start time
        self.start_time = time.time()
    
    def start(self):
        """Start the performance monitor dashboard."""
        if self._running:
            return
        
        self._running = True
        self._update_thread = threading.Thread(target=self._run_dashboard, daemon=True)
        self._update_thread.start()
        
        self.add_event("🚀 Performance monitor started", "info")
    
    def stop(self):
        """Stop the performance monitor."""
        self._running = False
        if self.live:
            self.live.stop()
        if self._update_thread:
            self._update_thread.join(timeout=1.0)
        
        self.add_event("🛑 Performance monitor stopped", "info")
    
    def record_timing(self, operation: str, duration_ms: float):
        """
        Record timing for an operation.
        
        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
        """
        with self._timing_lock:
            if operation not in self.timings:
                self.timings[operation] = TimingStats(name=operation)
            self.timings[operation].add_sample(duration_ms)
    
    def update_metric(self, name: str, value: Any):
        """Update a general metric."""
        with self._metrics_lock:
            self.metrics[name] = value
    
    def add_event(self, message: str, level: str = "info"):
        """Add an event to the log."""
        with self._events_lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.events.append({
                'timestamp': timestamp,
                'message': message,
                'level': level
            })
    
    def update_component_status(self, component: str, **kwargs):
        """Update component status."""
        with self._status_lock:
            if component in self.component_status:
                self.component_status[component].update(kwargs)
    
    def record_frame_breakdown(self, breakdown: Dict[str, float]):
        """Record timing breakdown for a frame."""
        self.frame_breakdown.append(breakdown)
    
    def _create_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()
        
        # Main layout structure
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=12)
        )
        
        # Split main area
        layout["main"].split_row(
            Layout(name="stats", ratio=2),
            Layout(name="timing", ratio=3)
        )
        
        # Split stats area
        layout["stats"].split_column(
            Layout(name="overview", size=10),
            Layout(name="components", size=8)
        )
        
        # Split timing area
        layout["timing"].split_column(
            Layout(name="operations", ratio=2),
            Layout(name="breakdown", ratio=1)
        )
        
        return layout
    
    def _make_header(self) -> Panel:
        """Create header panel."""
        runtime = time.time() - self.start_time
        hours, remainder = divmod(runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        header_text = Text()
        header_text.append("🎥 Frame Processor Performance Monitor", style="bold cyan")
        header_text.append(f"  |  Runtime: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}", style="dim")
        
        return Panel(Align.center(header_text), box=ROUNDED, style="cyan")
    
    def _make_overview_panel(self) -> Panel:
        """Create overview metrics panel."""
        with self._metrics_lock:
            metrics = self.metrics.copy()
        
        # Create metrics display
        content = Text()
        
        # FPS with color coding
        fps = metrics['fps']
        fps_style = "green" if fps >= 25 else "yellow" if fps >= 15 else "red"
        content.append(f"📊 FPS: {fps:.1f}\n", style=f"bold {fps_style}")
        
        content.append(f"🎯 Frames: {metrics['frames_processed']:,}\n")
        content.append(f"👥 Active Tracks: {metrics['active_tracks']}\n")
        content.append(f"🔍 Avg Detections: {metrics['detections_per_frame']:.1f}\n")
        
        # API metrics
        total_api = metrics['api_calls'] + metrics['api_cache_hits']
        cache_rate = (metrics['api_cache_hits'] / total_api * 100) if total_api > 0 else 0
        content.append(f"🌐 API Calls: {metrics['api_calls']:,} ")
        content.append(f"(Cache: {cache_rate:.0f}%)\n", style="dim")
        
        # Memory
        content.append(f"💾 Memory: {metrics['memory_mb']:.0f} MB\n")
        content.append(f"🎮 GPU: {metrics['gpu_memory_mb']:.0f} MB", style="yellow")
        
        return Panel(content, title="Overview", box=ROUNDED)
    
    def _make_components_panel(self) -> Panel:
        """Create component status panel."""
        with self._status_lock:
            status = self.component_status.copy()
        
        table = Table(show_header=False, box=None, padding=(0, 1))
        
        # Detector
        detector = status['detector']
        table.add_row("🤖 Detector:", f"{detector['name']} {detector['status']}")
        
        # Tracker
        tracker = status['tracker']
        table.add_row("🎯 Tracker:", f"{tracker['name']} {tracker['status']}")
        
        # API
        table.add_row("🌐 API:", status['api']['status'])
        
        # RabbitMQ
        table.add_row("🐰 RabbitMQ:", status['rabbitmq']['status'])
        
        # Rerun
        table.add_row("👁️ Rerun:", status['rerun']['status'])
        
        return Panel(table, title="Components", box=ROUNDED)
    
    def _make_operations_panel(self) -> Panel:
        """Create operations timing panel."""
        with self._timing_lock:
            timings = list(self.timings.values())
        
        if not timings:
            return Panel("No timing data yet...", title="Operation Timings", box=ROUNDED)
        
        # Sort by total time
        timings.sort(key=lambda x: x.total_ms, reverse=True)
        
        # Create table
        table = Table(box=None)
        table.add_column("Operation", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Avg", justify="right", style="yellow")
        table.add_column("Min", justify="right", style="green")
        table.add_column("Max", justify="right", style="red")
        table.add_column("Recent", justify="right", style="bright_yellow")
        
        # Add top operations
        for timing in timings[:10]:
            # Color code average time
            avg_style = "green" if timing.avg_ms < 10 else "yellow" if timing.avg_ms < 50 else "red"
            
            table.add_row(
                timing.name[:30],
                f"{timing.count:,}",
                Text(f"{timing.avg_ms:.1f}ms", style=avg_style),
                f"{timing.min_ms:.1f}ms",
                f"{timing.max_ms:.1f}ms",
                f"{timing.recent_avg_ms:.1f}ms"
            )
        
        return Panel(table, title="Operation Timings", box=ROUNDED)
    
    def _make_breakdown_panel(self) -> Panel:
        """Create frame breakdown panel."""
        if not self.frame_breakdown:
            return Panel("No frame breakdown data yet...", title="Frame Breakdown", box=ROUNDED)
        
        # Average breakdown over recent frames
        breakdown_avg = defaultdict(float)
        for frame in self.frame_breakdown:
            for op, duration in frame.items():
                breakdown_avg[op] += duration
        
        # Calculate percentages
        total = sum(breakdown_avg.values())
        if total == 0:
            return Panel("No timing data", title="Frame Breakdown", box=ROUNDED)
        
        # Create bar chart
        content = Text()
        sorted_ops = sorted(breakdown_avg.items(), key=lambda x: x[1], reverse=True)
        
        for op, duration in sorted_ops[:5]:
            avg_ms = duration / len(self.frame_breakdown)
            percentage = (duration / total) * 100
            
            # Create bar
            bar_width = int(percentage / 2)  # Scale to fit
            bar = "█" * bar_width
            
            # Color based on duration
            if avg_ms < 5:
                style = "green"
            elif avg_ms < 20:
                style = "yellow"
            else:
                style = "red"
            
            content.append(f"{op:15} ", style="cyan")
            content.append(f"{bar:20} ", style=style)
            content.append(f"{avg_ms:5.1f}ms ({percentage:4.1f}%)\n", style="dim")
        
        return Panel(content, title="Frame Breakdown (Avg)", box=ROUNDED)
    
    def _make_events_panel(self) -> Panel:
        """Create events log panel."""
        with self._events_lock:
            events = list(self.events)
        
        if not events:
            return Panel("No events yet...", title="Recent Events", box=ROUNDED)
        
        # Create event display
        content = Text()
        
        for event in reversed(events):  # Show newest first
            # Style based on level
            if event['level'] == 'error':
                style = "red"
                icon = "❌"
            elif event['level'] == 'warning':
                style = "yellow"
                icon = "⚠️"
            elif event['level'] == 'success':
                style = "green"
                icon = "✅"
            else:
                style = "cyan"
                icon = "ℹ️"
            
            content.append(f"[{event['timestamp']}] ", style="dim")
            content.append(f"{icon} ", style=style)
            content.append(f"{event['message']}\n", style=style)
        
        return Panel(content, title="Recent Events", box=ROUNDED)
    
    def _create_dashboard(self) -> Layout:
        """Create complete dashboard."""
        layout = self._create_layout()
        
        # Fill layout
        layout["header"].update(self._make_header())
        layout["overview"].update(self._make_overview_panel())
        layout["components"].update(self._make_components_panel())
        layout["operations"].update(self._make_operations_panel())
        layout["breakdown"].update(self._make_breakdown_panel())
        layout["footer"].update(self._make_events_panel())
        
        return layout
    
    def _run_dashboard(self):
        """Run the dashboard update loop."""
        try:
            with Live(
                self._create_dashboard(),
                console=self.console,
                refresh_per_second=1.0 / self.update_interval,
                screen=True
            ) as live:
                self.live = live
                
                while self._running:
                    # Update dashboard
                    live.update(self._create_dashboard())
                    time.sleep(self.update_interval)
                    
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            self.add_event(f"Dashboard error: {e}", "error")


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create the global performance monitor."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


class DetailedTimer:
    """
    Enhanced context manager for detailed timing with automatic dashboard updates.
    
    Use this instead of PerformanceTimer for operations you want to track.
    """
    
    def __init__(self, operation: str, log_threshold_ms: float = 100.0):
        """
        Initialize timer.
        
        Args:
            operation: Operation name to track
            log_threshold_ms: Only log if duration exceeds this threshold
        """
        self.operation = operation
        self.log_threshold_ms = log_threshold_ms
        self.start_time = None
        self.monitor = get_performance_monitor()
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        
        # Record to monitor
        self.monitor.record_timing(self.operation, duration_ms)
        
        # Log if exceeds threshold
        if duration_ms > self.log_threshold_ms:
            logger.warning(f"{self.operation} took {duration_ms:.1f}ms (threshold: {self.log_threshold_ms}ms)")
```

## 3. Update the Frame Processor Pipeline:

```python
# frame_processor/pipeline/processor.py
# Add these imports at the top:
from core.performance_monitor import DetailedTimer, get_performance_monitor
import psutil
import torch

# In the FrameProcessor class, update process_frame method:

async def process_frame(self, frame: np.ndarray, 
                      timestamp_ns: Optional[int] = None) -> ProcessingResult:
    """
    Process a single frame through the entire pipeline with detailed timing.
    """
    monitor = get_performance_monitor()
    frame_breakdown = {}
    
    with DetailedTimer("total_frame_processing"):
        start_time = time.time()
        self.frame_number += 1
        
        if timestamp_ns is None:
            timestamp_ns = get_ntp_time_ns()
        
        # Step 1: Run detection with timing
        if self.config.detection_enabled:
            with DetailedTimer("detection") as timer:
                detections = await self.detector.detect(frame)
                self.stats['total_detections'] += len(detections)
            frame_breakdown['detection'] = timer.elapsed_ms
            
            # Log detection details
            monitor.add_event(
                f"Detected {len(detections)} objects ({timer.elapsed_ms:.1f}ms)",
                "info" if timer.elapsed_ms < 50 else "warning"
            )
        else:
            detections = []
            frame_breakdown['detection'] = 0.0
        
        # Step 2: Update tracking with timing
        with DetailedTimer("tracking") as timer:
            tracks_ready = self.tracker.update(detections, frame, self.frame_number)
        frame_breakdown['tracking'] = timer.elapsed_ms
        
        # Step 3: Process tracks ready for API
        api_tasks = []
        api_start = time.perf_counter()
        
        for track in tracks_ready:
            if track.best_frame is not None and not track.api_processed:
                api_tasks.append(self._process_track_for_api(track))
        
        # Run API processing concurrently
        if api_tasks:
            with DetailedTimer(f"api_processing_{len(api_tasks)}_tracks"):
                await asyncio.gather(*api_tasks)
                monitor.add_event(
                    f"Processed {len(api_tasks)} tracks through API",
                    "success"
                )
        
        frame_breakdown['api_processing'] = (time.perf_counter() - api_start) * 1000
        
        # Step 4: Update visualization with timing
        if self.rerun_client:
            with DetailedTimer("visualization") as timer:
                active_tracks = self.tracker.get_active_tracks()
                self.rerun_client.log_frame(
                    frame, detections, active_tracks, 
                    self.frame_number, timestamp_ns
                )
            frame_breakdown['visualization'] = timer.elapsed_ms
        else:
            frame_breakdown['visualization'] = 0.0
        
        # Calculate total processing time
        processing_time_ms = (time.time() - start_time) * 1000
        frame_breakdown['total'] = processing_time_ms
        
        # Update monitor metrics
        monitor.record_frame_breakdown(frame_breakdown)
        monitor.update_metric('frames_processed', self.stats['frames_processed'])
        monitor.update_metric('active_tracks', len(self.tracker.get_active_tracks()))
        monitor.update_metric('detections_per_frame', 
                            self.stats['total_detections'] / max(1, self.stats['frames_processed']))
        
        # Update memory usage
        process = psutil.Process()
        monitor.update_metric('memory_mb', process.memory_info().rss / 1024 / 1024)
        
        if torch.cuda.is_available():
            monitor.update_metric('gpu_memory_mb', 
                                torch.cuda.memory_allocated() / 1024 / 1024)
        
        # Calculate FPS
        self.stats['frames_processed'] += 1
        self.stats['total_processing_time_ms'] += processing_time_ms
        
        if self.frame_number % 30 == 0:  # Update FPS every 30 frames
            avg_time = self.stats['total_processing_time_ms'] / self.stats['frames_processed']
            fps = 1000.0 / avg_time if avg_time > 0 else 0
            monitor.update_metric('fps', fps)
        
        # Log performance periodically with better formatting
        if self.frame_number % 100 == 0:
            monitor.add_event(
                f"Milestone: {self.frame_number} frames processed",
                "success"
            )
        
        return ProcessingResult(
            frame_number=self.frame_number,
            timestamp_ns=timestamp_ns,
            detections=detections,
            tracks_for_api=tracks_ready,
            processing_time_ms=processing_time_ms,
            detection_count=len(detections),
            active_track_count=len(self.tracker.get_active_tracks())
        )

# Also update _process_track_for_api to add timing:

async def _process_track_for_api(self, track: TrackedObject):
    """Process a track through the API pipeline with detailed timing."""
    monitor = get_performance_monitor()
    
    try:
        track.is_being_processed = True
        start_time = time.time()
        
        # Enhance image with timing
        with DetailedTimer(f"enhancement_track_{track.id}"):
            enhanced_image = self.enhancer.enhance_roi(track.best_frame)
        
        # Process through API client with timing
        with DetailedTimer(f"api_call_track_{track.id}") as timer:
            api_result = await self.api_client.process_object_for_dimensions(
                enhanced_image, track.class_name
            )
        
        # Check if it was a cache hit
        if timer.elapsed_ms < 10:  # Likely a cache hit if very fast
            monitor.update_metric('api_cache_hits', 
                                self.stats.get('api_cache_hits', 0) + 1)
        
        # Store result
        track.api_result = api_result
        track.api_processed = True
        track.is_being_processed = False
        track.processing_time = time.time() - start_time
        
        # Update stats
        self.stats['api_calls_made'] += 1
        monitor.update_metric('api_calls', self.stats['api_calls_made'])
        
        # Log success/failure
        if api_result.get('dimensions'):
            monitor.add_event(
                f"✅ Track #{track.id}: {api_result.get('product_name', 'Unknown')}",
                "success"
            )
        else:
            monitor.add_event(
                f"❌ Track #{track.id}: Failed to get dimensions",
                "warning"
            )
        
    except Exception as e:
        logger.error(f"API processing failed for track #{track.id}: {e}")
        monitor.add_event(f"API error for track #{track.id}: {str(e)}", "error")
        track.api_processed = True
        track.api_result = {"error": str(e)}
        track.is_being_processed = False
```

## 4. Update the Main Entry Point:

```python
# frame_processor/main.py
# Add imports:
from core.performance_monitor import get_performance_monitor, DetailedTimer

# In the FrameProcessorService class:

def __init__(self):
    """Initialize the frame processor service."""
    self.config = Config()
    self.processor = None
    self.publisher = None
    self.connection = None
    self.channel = None
    self.running = False
    
    # Initialize performance monitor
    self.monitor = get_performance_monitor()
    self.monitor.start()  # Start the dashboard
    
    logger.info("Initializing Frame Processor Service...")
    
    # ... rest of init code ...

# Update process_frame_message:

async def process_frame_message(self, message: aio_pika.IncomingMessage):
    """Process incoming frame message with detailed timing."""
    try:
        async with message.process():
            with DetailedTimer("rabbitmq_message_processing"):
                # Extract headers
                headers = message.headers or {}
                timestamp_ns_raw = headers.get('timestamp_ns', get_ntp_time_ns())
                timestamp_ns = int(timestamp_ns_raw) if timestamp_ns_raw else get_ntp_time_ns()
                frame_number = headers.get('frame_number', 0)
                
                # Decode frame with timing
                with DetailedTimer("frame_decode"):
                    nparr = np.frombuffer(message.body, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    self.monitor.add_event("Failed to decode frame", "error")
                    return
                
                # Process frame
                if self.config.detection_enabled:
                    result = await self.processor.process_frame(frame, timestamp_ns)
                    
                    # Update metrics
                    frames_processed.inc()
                    yolo_detections.inc(result.detection_count)
                    processing_time.observe(result.processing_time_ms / 1000.0)
                    active_tracks.set(result.active_track_count)
                    
                    # Publish if needed
                    if result.detections:
                        with DetailedTimer("result_publishing"):
                            # ... existing publishing code ...
                            pass
                else:
                    frames_processed.inc()
                
    except Exception as e:
        self.monitor.add_event(f"Frame processing error: {e}", "error")
        logger.error(f"Error processing frame: {e}", exc_info=True)

# Update run method to show component status:

async def run(self):
    """Main run loop."""
    self.running = True
    
    try:
        # Update component status
        self.monitor.update_component_status('rabbitmq', status='🔄 Connecting...')
        
        # ... NTP sync code ...
        self.monitor.add_event("NTP sync complete", "success")
        
        # Initialize components
        self.processor = FrameProcessor(self.config)
        self.publisher = RabbitMQPublisher(self.config)
        
        # Update detector/tracker info
        self.monitor.update_component_status(
            'detector', 
            name=self.processor.detector.name,
            status='✅ Ready'
        )
        self.monitor.update_component_status(
            'tracker',
            name=self.processor.tracker.name,
            status='✅ Ready'
        )
        
        # Connect to RabbitMQ
        await self.connect_rabbitmq()
        self.monitor.update_component_status('rabbitmq', status='✅ Connected')
        
        # Update API status
        if self.config.use_serpapi or self.config.use_perplexity:
            self.monitor.update_component_status('api', status='✅ Enabled')
        else:
            self.monitor.update_component_status('api', status='⏸️ Disabled')
        
        # Update Rerun status
        if self.config.rerun_enabled:
            self.monitor.update_component_status('rerun', status='✅ Connected')
        else:
            self.monitor.update_component_status('rerun', status='⏸️ Disabled')
        
        # ... rest of run code ...
        
    finally:
        # Stop monitor on shutdown
        self.monitor.stop()
```

## 5. Usage and Benefits:

Now when you run the frame processor, you'll see a beautiful auto-updating dashboard like this:

```
╭─────────────────────────────────────────────────────────────────╮
│      🎥 Frame Processor Performance Monitor  |  Runtime: 00:05:23 │
╰─────────────────────────────────────────────────────────────────╯

╭─ Overview ──────────╮  ╭─ Operation Timings ─────────────────────────╮
│ 📊 FPS: 28.3       │  │ Operation              Count  Avg    Min    │
│ 🎯 Frames: 8,523   │  │ detection              8523   32.1ms 28.3ms │
│ 👥 Active Tracks: 5 │  │ tracking               8523   5.2ms  3.1ms  │
│ 🔍 Avg Detections: 3│  │ visualization          8523   12.3ms 8.5ms  │
│ 🌐 API Calls: 142   │  │ api_call_track_1       5      1230ms 980ms  │
│    (Cache: 67%)     │  │ enhancement_track_1    5      185ms  150ms  │
│ 💾 Memory: 512 MB   │  │ frame_decode           8523   0.8ms  0.5ms  │
│ 🎮 GPU: 1,248 MB    │  ╰─────────────────────────────────────────────╯
╰─────────────────────╯
                         ╭─ Frame Breakdown (Avg) ─────────────────────╮
╭─ Components ────────╮  │ detection      ████████████ 32.1ms (65.2%) │
│ 🤖 Detector: SAM ✅  │  │ visualization  ████         12.3ms (25.0%) │
│ 🎯 Tracker: IOU ✅   │  │ tracking       ██           5.2ms  (10.6%) │
│ 🌐 API: ✅ Enabled   │  │ api_processing █            2.1ms  (4.3%)  │
│ 🐰 RabbitMQ: ✅      │  │ frame_decode   ▌            0.8ms  (1.6%)  │
│ 👁️ Rerun: ✅        │  ╰─────────────────────────────────────────────╯
╰─────────────────────╯

╭─ Recent Events ──────────────────────────────────────────────────╮
│ [14:23:45] ✅ Track #5: Apple iPhone 13                          │
│ [14:23:43] ℹ️ Detected 4 objects (31.2ms)                        │
│ [14:23:42] ✅ Processed 2 tracks through API                     │
│ [14:23:40] ⚠️ detection took 125.3ms (threshold: 100.0ms)        │
│ [14:23:38] ✅ Milestone: 8500 frames processed                   │
│ [14:23:35] ❌ Track #3: Failed to get dimensions                 │
│ [14:23:32] ✅ NTP sync complete                                  │
│ [14:23:30] 🚀 Performance monitor started                        │
╰──────────────────────────────────────────────────────────────────╯
```

## Key Features:

1. **Auto-updating Dashboard**: Updates every 0.5 seconds with no flicker
2. **Detailed Timing Breakdown**: See exactly where time is spent
3. **Component Status**: Know what's working and what's not at a glance
4. **Performance Trends**: Track FPS, memory usage, cache hit rates
5. **Event Log**: See important events with timestamps
6. **Color Coding**: Green = good, Yellow = warning, Red = problem
7. **No Console Spam**: All information organized in one clean view

## Additional Timing Points You Can Add:

```python
# In detection/sam.py or detection/yolo.py:
with DetailedTimer("yolo_preprocessing"):
    # preprocessing code

with DetailedTimer("yolo_inference"):
    # actual model inference

with DetailedTimer("yolo_postprocessing"):
    # postprocessing code

# In external/api_client.py:
with DetailedTimer("gcs_upload"):
    # upload code

with DetailedTimer("serpapi_request"):
    # API request

with DetailedTimer("perplexity_request"):
    # API request

# In pipeline/enhancer.py:
with DetailedTimer("gamma_correction"):
    # gamma code

with DetailedTimer("contrast_adjustment"):
    # contrast code
```

This gives you comprehensive performance monitoring with a beautiful, professional terminal interface that auto-updates like `htop` or `pspy`!