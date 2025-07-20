#!/usr/bin/env python3
"""
Test script for the performance monitor implementation.
Run this to see the terminal dashboard in action.
"""

import asyncio
import time
import random
from core.performance_monitor import get_performance_monitor, DetailedTimer


async def simulate_frame_processing():
    """Simulate frame processing with various operations."""
    monitor = get_performance_monitor()
    monitor.start()
    
    # Update component status
    monitor.update_component_status('detector', name='YOLO', status='✅ Ready')
    monitor.update_component_status('tracker', name='IOU', status='✅ Ready')
    monitor.update_component_status('rabbitmq', status='✅ Connected')
    monitor.update_component_status('api', status='✅ Enabled')
    monitor.update_component_status('rerun', status='✅ Connected')
    
    monitor.add_event("Starting simulation...", "info")
    
    try:
        for frame_num in range(200):
            # Simulate frame processing
            frame_breakdown = {}
            
            # Detection
            with DetailedTimer("detection") as timer:
                await asyncio.sleep(random.uniform(0.025, 0.045))  # 25-45ms
            frame_breakdown['detection'] = timer.elapsed_ms
            
            detections = random.randint(1, 5)
            monitor.add_event(f"Detected {detections} objects", "info")
            
            # Tracking
            with DetailedTimer("tracking") as timer:
                await asyncio.sleep(random.uniform(0.003, 0.007))  # 3-7ms
            frame_breakdown['tracking'] = timer.elapsed_ms
            
            # API processing (occasionally)
            if random.random() < 0.1:  # 10% chance
                with DetailedTimer("api_processing") as timer:
                    await asyncio.sleep(random.uniform(0.8, 1.5))  # 800-1500ms
                frame_breakdown['api_processing'] = timer.elapsed_ms
                monitor.add_event(f"✅ Track #{random.randint(1, 10)}: iPhone 13", "success")
                monitor.update_metric('api_calls', monitor.metrics.get('api_calls', 0) + 1)
            else:
                frame_breakdown['api_processing'] = 0.0
            
            # Visualization
            with DetailedTimer("visualization") as timer:
                await asyncio.sleep(random.uniform(0.008, 0.015))  # 8-15ms
            frame_breakdown['visualization'] = timer.elapsed_ms
            
            # Record total time
            total_time = sum(frame_breakdown.values())
            frame_breakdown['total'] = total_time
            
            # Update metrics
            monitor.record_frame_breakdown(frame_breakdown)
            monitor.update_metric('frames_processed', frame_num + 1)
            monitor.update_metric('active_tracks', random.randint(2, 8))
            monitor.update_metric('detections_per_frame', detections)
            monitor.update_metric('memory_mb', 512 + random.randint(-50, 50))
            monitor.update_metric('gpu_memory_mb', 1248 + random.randint(-100, 100))
            
            # Calculate FPS every 30 frames
            if frame_num % 30 == 0 and frame_num > 0:
                fps = 1000.0 / (total_time + random.uniform(-5, 5))
                monitor.update_metric('fps', fps)
            
            # Milestone events
            if frame_num % 50 == 0 and frame_num > 0:
                monitor.add_event(f"Milestone: {frame_num} frames processed", "success")
            
            # Occasional warnings/errors
            if random.random() < 0.05:  # 5% chance
                monitor.add_event("Detection took 125ms (threshold: 100ms)", "warning")
            
            # Small delay between frames
            await asyncio.sleep(0.033)  # ~30 FPS
            
    except KeyboardInterrupt:
        monitor.add_event("Simulation interrupted", "warning")
    finally:
        monitor.stop()
        print("\nPerformance monitor stopped.")


if __name__ == "__main__":
    print("Starting performance monitor test...")
    print("Press Ctrl+C to stop\n")
    
    try:
        asyncio.run(simulate_frame_processing())
    except KeyboardInterrupt:
        print("\nTest completed.")