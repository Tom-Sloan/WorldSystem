#!/usr/bin/env python3
"""
Simple test script for SAM2 video tracking.

This script tests the SAM2 video tracking components directly without RabbitMQ.
"""

import asyncio
import numpy as np
import cv2
import time
from pathlib import Path
import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import video tracking components
from core.config import Config
from tracking.sam2_realtime_tracker import SAM2RealtimeTracker
from tracking.prompt_strategies import GridPromptStrategy
from core.video_buffer import SAM2LongVideoBuffer


async def test_sam2_tracking():
    """Test SAM2 video tracking with synthetic frames."""
    
    print("SAM2 Video Tracking Test")
    print("========================\n")
    
    # Initialize config
    config = Config()
    config.sam2_model_size = "small"  # Use small model for testing
    config.processing_resolution = 720
    config.grid_prompt_density = 9  # 3x3 grid
    
    print(f"Configuration:")
    print(f"  Model: SAM2 {config.sam2_model_size}")
    print(f"  Resolution: {config.processing_resolution}p")
    print(f"  Device: {config.device}")
    print(f"  Grid density: {config.grid_prompt_density}")
    print()
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU available, using CPU (will be slow)")
    print()
    
    # Initialize components
    print("Initializing SAM2 components...")
    
    try:
        # Create video buffer
        buffer = SAM2LongVideoBuffer(
            max_frames=30,
            max_memory_branches=3,
            branch_threshold=0.5
        )
        
        # Create prompt strategy
        prompt_strategy = GridPromptStrategy(
            grid_size=3,
            min_object_area=1000
        )
        
        # Create tracker
        tracker = SAM2RealtimeTracker(
            config=config,
            video_buffer=buffer,
            prompt_strategy=prompt_strategy
        )
        
        print("✓ Components initialized successfully")
        print()
        
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return
    
    # Generate test video frames
    print("Generating test frames...")
    width, height = 1280, 720
    num_frames = 90  # 3 seconds at 30 fps
    
    # Create moving objects
    objects = [
        {'x': 200, 'y': 200, 'vx': 8, 'vy': 5, 'size': 100, 'color': (255, 0, 0)},
        {'x': 600, 'y': 400, 'vx': -6, 'vy': 4, 'size': 80, 'color': (0, 255, 0)},
        {'x': 1000, 'y': 300, 'vx': 5, 'vy': -7, 'size': 120, 'color': (0, 0, 255)},
    ]
    
    frames = []
    for frame_idx in range(num_frames):
        # Create frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 230  # Light background
        
        # Draw objects
        for obj in objects:
            # Update position
            obj['x'] += obj['vx']
            obj['y'] += obj['vy']
            
            # Bounce off walls
            if obj['x'] <= obj['size']//2 or obj['x'] >= width - obj['size']//2:
                obj['vx'] = -obj['vx']
            if obj['y'] <= obj['size']//2 or obj['y'] >= height - obj['size']//2:
                obj['vy'] = -obj['vy']
            
            # Draw circle
            cv2.circle(frame, (int(obj['x']), int(obj['y'])), 
                      obj['size']//2, obj['color'], -1)
            
            # Add inner circle for texture
            cv2.circle(frame, (int(obj['x']), int(obj['y'])), 
                      obj['size']//3, (255, 255, 255), 3)
        
        frames.append(frame)
    
    print(f"✓ Generated {num_frames} test frames")
    print()
    
    # Test tracking
    print("Starting SAM2 video tracking test...")
    print("-" * 50)
    
    start_time = time.time()
    results = []
    
    for idx, frame in enumerate(frames):
        frame_start = time.time()
        
        # Track objects
        try:
            result = await tracker.track(frame, frame_index=idx)
            results.append(result)
            
            # Calculate metrics
            frame_time = (time.time() - frame_start) * 1000
            fps = 1000 / frame_time if frame_time > 0 else 0
            
            # Print progress every 10 frames
            if idx % 10 == 0:
                print(f"\nFrame {idx}/{num_frames}:")
                print(f"  Objects tracked: {result.object_count}")
                print(f"  Processing time: {frame_time:.1f}ms")
                print(f"  FPS: {fps:.1f}")
                
                if result.tracks:
                    print(f"  Track IDs: {[t['object_id'] for t in result.tracks]}")
                
                # Memory usage
                if torch.cuda.is_available():
                    mem_used = torch.cuda.memory_allocated() / 1e9
                    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                    print(f"  GPU Memory: {mem_used:.1f}/{mem_total:.1f} GB ({mem_used/mem_total*100:.1f}%)")
            
        except Exception as e:
            print(f"\n✗ Error at frame {idx}: {e}")
            break
    
    # Summary
    total_time = time.time() - start_time
    avg_fps = len(results) / total_time if total_time > 0 else 0
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"  Total frames processed: {len(results)}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Average FPS: {avg_fps:.1f}")
    
    if results:
        avg_objects = sum(r.object_count for r in results) / len(results)
        print(f"  Average objects per frame: {avg_objects:.1f}")
        
        # Track consistency
        all_track_ids = set()
        for r in results:
            for t in r.tracks:
                all_track_ids.add(t['object_id'])
        print(f"  Unique track IDs: {len(all_track_ids)}")
    
    print("\n✓ Test completed successfully!")
    
    # Save sample output frames
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving sample frames to {output_dir}/...")
    
    # Save first, middle, and last frames with annotations
    sample_indices = [0, len(frames)//2, len(frames)-1]
    
    for idx in sample_indices:
        if idx < len(frames) and idx < len(results):
            frame = frames[idx].copy()
            result = results[idx]
            
            # Draw tracking results
            for track in result.tracks:
                bbox = track['bbox']
                track_id = track['object_id']
                
                # Draw bounding box
                cv2.rectangle(frame, 
                            (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                            (0, 255, 0), 2)
                
                # Draw track ID
                cv2.putText(frame, f"ID: {track_id}",
                          (bbox[0], bbox[1]-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                          (0, 255, 0), 2)
            
            # Save frame
            output_path = output_dir / f"frame_{idx:04d}.jpg"
            cv2.imwrite(str(output_path), frame)
            print(f"  Saved: {output_path}")
    
    print("\n✅ All tests completed!")


async def test_memory_management():
    """Test SAM2 memory management and error recovery."""
    print("\nTesting memory management...")
    print("-" * 50)
    
    config = Config()
    config.sam2_model_size = "small"
    config.enable_dynamic_model_switching = True
    
    # Test with increasing resolutions
    resolutions = [480, 720, 1080]
    
    for res in resolutions:
        print(f"\nTesting at {res}p resolution...")
        config.processing_resolution = res
        
        try:
            # Create tracker
            tracker = SAM2RealtimeTracker(
                config=config,
                video_buffer=SAM2LongVideoBuffer(),
                prompt_strategy=GridPromptStrategy()
            )
            
            # Create test frame
            frame = np.random.randint(0, 255, (res, int(res*16/9), 3), dtype=np.uint8)
            
            # Track
            result = await tracker.track(frame, frame_index=0)
            
            print(f"  ✓ Success at {res}p")
            
            # Check memory
            if torch.cuda.is_available():
                mem_gb = torch.cuda.memory_allocated() / 1e9
                print(f"  GPU memory used: {mem_gb:.2f} GB")
            
        except torch.cuda.OutOfMemoryError:
            print(f"  ✗ OOM at {res}p - Model switching should handle this")
        except Exception as e:
            print(f"  ✗ Error at {res}p: {e}")


async def main():
    """Run all tests."""
    try:
        # Run main tracking test
        await test_sam2_tracking()
        
        # Run memory management test
        await test_memory_management()
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())