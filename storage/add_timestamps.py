"""
add_timestamps.py

This script processes image files in mav0/cam0/data directories to extract timestamps from JPG filenames.
It calculates timing statistics (average, max, min, std dev, FPS) between consecutive frames and writes:
1. A timestamps.txt file containing all timestamps in nanoseconds
2. A timestamp_stats.txt file with timing statistics in milliseconds
"""

import os
from pathlib import Path
import numpy as np

def extract_timestamps():
    # Get the current working directory
    data_dir = Path('data')
    
    # Walk through all mav0/cam0/data directories
    matching_dirs = []
    for root, dirs, files in os.walk(data_dir):
        if root.endswith('mav0/cam0/data'):
            matching_dirs.append((root, files))

    # Process each directory
    for root, files in matching_dirs:
        timestamps = []  # Reset timestamps for each directory
        
        # Extract timestamps from image filenames
        for file in files:
            if file.endswith('.jpg'):
                # Remove the file extension to get the timestamp
                timestamp = file.replace('.jpg', '')
                timestamps.append(int(timestamp))  # Store original nanosecond timestamps

        # Only process if we found any timestamps
        if timestamps:
            # Sort timestamps numerically
            timestamps.sort()

            # Calculate time differences between consecutive timestamps (in ns)
            time_diffs = np.diff(timestamps)
            
            # Convert time differences to milliseconds for stats
            time_diffs_ms = time_diffs / 1_000_000

            # Calculate statistics (in milliseconds)
            stats = {
                'average_diff': np.mean(time_diffs_ms),
                'max_diff': np.max(time_diffs_ms),
                'min_diff': np.min(time_diffs_ms),
                'std_diff': np.std(time_diffs_ms),
                'fps': 1000 / np.mean(time_diffs_ms)  # Convert ms to fps
            }

            # Create timestamps.txt in the mav0 directory
            mav0_dir = Path(root).parents[2]  # Go up 3 levels: data -> cam0 -> mav0
            output_file = mav0_dir / 'timestamps.txt'
            stats_file = mav0_dir / 'timestamp_stats.txt'
            
            # Write timestamps to file (in original nanoseconds)
            with open(output_file, 'w') as f:
                for timestamp in timestamps:
                    f.write(f'{timestamp}\n')

            # Write statistics to file (in milliseconds)
            with open(stats_file, 'w') as f:
                f.write(f'Average time between frames: {stats["average_diff"]:,.0f} ms\n')
                f.write(f'Maximum time between frames: {stats["max_diff"]:,.0f} ms\n')
                f.write(f'Minimum time between frames: {stats["min_diff"]:,.0f} ms\n')
                f.write(f'Standard deviation: {stats["std_diff"]:,.0f} ms\n')
                f.write(f'Average FPS: {stats["fps"]:.1f}\n')
        else:
            print(f"No jpg files found in directory: {root}")

if __name__ == '__main__':
    extract_timestamps()
