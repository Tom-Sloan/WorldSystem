"""
Performance monitoring and terminal dashboard for frame processor.

This module provides real-time performance metrics and a beautiful
terminal interface using the rich library.
"""

import os
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
        
        # Check if we're in Docker or non-TTY environment
        self.is_docker = os.environ.get('IS_DOCKER', 'false').lower() == 'true'
        self.enable_rich = os.environ.get('ENABLE_RICH_TERMINAL', 'false').lower() == 'true'
        
        # Use simple mode if:
        # 1. We're in Docker AND rich terminal is not explicitly enabled
        # 2. OR we don't have a TTY available
        self.simple_mode = (self.is_docker and not self.enable_rich) or not os.isatty(0)
        
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
        
        # Log startup mode
        if self.simple_mode:
            logger.info("Performance monitor starting in simple mode (periodic logging)")
            if self.is_docker and not self.enable_rich:
                logger.info("To enable rich terminal in Docker, set ENABLE_RICH_TERMINAL=true")
        else:
            logger.info("Performance monitor starting with rich terminal display")
        
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
        if self.simple_mode:
            # Simple mode for Docker/non-TTY - just log periodically
            self._run_simple_dashboard()
        else:
            # Full rich dashboard for interactive terminals
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
                # Fall back to simple mode
                self.simple_mode = True
                if self._running:
                    self._run_simple_dashboard()
    
    def _run_simple_dashboard(self):
        """Simple dashboard for Docker/non-TTY environments."""
        last_log_time = 0
        log_interval = 5.0  # Log every 5 seconds
        
        while self._running:
            current_time = time.time()
            
            if current_time - last_log_time >= log_interval:
                # Create a simple status summary
                with self._metrics_lock:
                    metrics = self.metrics.copy()
                
                with self._timing_lock:
                    top_timings = sorted(
                        self.timings.values(), 
                        key=lambda x: x.total_ms, 
                        reverse=True
                    )[:3]
                
                # Log performance summary
                summary_parts = [
                    f"FPS: {metrics['fps']:.1f}",
                    f"Frames: {metrics['frames_processed']}",
                    f"Tracks: {metrics['active_tracks']}",
                    f"Detections/frame: {metrics['detections_per_frame']:.1f}",
                    f"Memory: {metrics['memory_mb']:.0f}MB",
                ]
                
                if metrics['gpu_memory_mb'] > 0:
                    summary_parts.append(f"GPU: {metrics['gpu_memory_mb']:.0f}MB")
                
                logger.info(f"[PERF] {' | '.join(summary_parts)}")
                
                # Log top operations
                if top_timings:
                    timing_info = ", ".join([
                        f"{t.name}: {t.avg_ms:.1f}ms" 
                        for t in top_timings
                    ])
                    logger.info(f"[TIMING] Top operations: {timing_info}")
                
                last_log_time = current_time
            
            time.sleep(self.update_interval)


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
        self.elapsed_ms = 0.0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        
        # Record to monitor
        self.monitor.record_timing(self.operation, self.elapsed_ms)
        
        # Log if exceeds threshold
        if self.elapsed_ms > self.log_threshold_ms:
            logger.warning(f"{self.operation} took {self.elapsed_ms:.1f}ms (threshold: {self.log_threshold_ms}ms)")