import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def analyze_imu_files(directory_path):
    # Convert string path to Path object and resolve to absolute path
    path = Path(directory_path).resolve()
    
    # Get all CSV files in the specified directory
    csv_files = list(path.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in directory: {path}")
        return

    for file_path in tqdm(csv_files, desc="Processing files"):
        print(f"\nAnalyzing {file_path.name}:")
        
        # Read timestamps from the first column
        timestamps = []
        with open(file_path, 'r') as f:
            # Skip header line (starts with #)
            header = next(f)
            if not header.startswith('#'):
                f.seek(0)  # If no header, go back to start
                
            # Count lines for progress bar
            num_lines = sum(1 for _ in f)
            f.seek(0)  # Reset file pointer
            if header.startswith('#'):
                next(f)  # Skip header again if it exists
            
            # Process each line with progress bar
            for line in tqdm(f, total=num_lines-1, desc="Reading timestamps"):
                if line.strip():  # Skip empty lines
                    try:
                        timestamp = int(line.split(',')[0])
                        timestamps.append(timestamp)
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line: {line.strip()}")
                        continue

        if not timestamps:
            print(f"No valid timestamps found in {file_path.name}")
            continue

        # Calculate time differences between consecutive timestamps
        time_diffs = np.diff(timestamps)  # Already in milliseconds
        
        # Calculate statistics
        mean_diff = np.mean(time_diffs)
        min_time = np.min(time_diffs)
        max_time = np.max(time_diffs)
        
        # Calculate FPS from milliseconds
        avg_fps = 1000.0 / mean_diff if mean_diff > 0 else 0

        # Print results with better formatting
        print("\nResults:")
        print(f"Number of timestamps: {len(timestamps)}")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Minimum time between frames: {min_time:.1f} ms")
        print(f"Maximum time between frames: {max_time:.1f} ms")
        print(f"Average time between frames: {mean_diff:.1f} ms")
        
        # Print some sample time differences for verification
        print("\nSample time differences (first 5):")
        for i in range(min(5, len(time_diffs))):
            print(f"  {time_diffs[i]:.1f} ms")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze IMU data files for timing statistics.')
    parser.add_argument('--path', type=str, default='.',
                      help='Path to directory containing CSV files (default: current directory)')
    
    args = parser.parse_args()
    analyze_imu_files(args.path)
