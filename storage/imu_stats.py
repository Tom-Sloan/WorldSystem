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
                        # Convert timestamp to nanoseconds if needed
                        timestamp_str = str(timestamp)
                        if len(timestamp_str) == 10:  # seconds precision
                            timestamp = int(timestamp_str + "000000000")
                        elif len(timestamp_str) == 13:  # milliseconds precision
                            timestamp = int(timestamp_str + "000000")
                        elif len(timestamp_str) == 16:  # microseconds precision
                            timestamp = int(timestamp_str + "000")
                        # else assume it's already in nanoseconds (19 digits)
                        timestamps.append(timestamp)
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line: {line.strip()}")
                        continue

        if not timestamps:
            print(f"No valid timestamps found in {file_path.name}")
            continue

        # Calculate time differences between consecutive timestamps in milliseconds
        time_diffs = np.diff(timestamps) / 1_000_000  # Convert nanoseconds to milliseconds
        
        # Calculate statistics
        mean_diff = np.mean(time_diffs)
        min_time = np.min(time_diffs)
        max_time = np.max(time_diffs)
        
        # Calculate FPS from milliseconds
        avg_fps = 1000.0 / mean_diff if mean_diff > 0 else 0

        # Print results with better formatting
        print("\nResults:")
        print(f"First timestamp: {timestamps[0]}")
        print(f"Last timestamp: {timestamps[-1]}")
        print(f"Number of timestamps: {len(timestamps)}")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Minimum time between frames: {min_time:.1f} ms")
        print(f"Maximum time between frames: {max_time:.1f} ms")
        print(f"Average time between frames: {mean_diff:.1f} ms")
        
        # Print some sample time differences for verification
        print("\nSample time differences (first 5):")
        for i in range(min(5, len(time_diffs))):
            print(f"  {time_diffs[i]:.1f} ms")

def convert_timestamps_to_ns(directory_path):
    """Convert timestamps in CSV files to nanosecond precision."""
    path = Path(directory_path).resolve()
    csv_files = list(path.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in directory: {path}")
        return

    for file_path in tqdm(csv_files, desc="Processing files"):
        print(f"\nConverting timestamps in {file_path.name}")
        
        # Create a temporary file
        temp_file = file_path.with_suffix('.tmp')
        
        try:
            with open(file_path, 'r') as input_file, open(temp_file, 'w') as output_file:
                # Handle header
                header = next(input_file)
                output_file.write(header)
                
                # Process each line with progress bar
                # First count lines for the progress bar
                input_file.seek(0)
                num_lines = sum(1 for _ in input_file) - 1  # subtract 1 for header
                input_file.seek(0)
                next(input_file)  # skip header again
                
                for line in tqdm(input_file, total=num_lines, desc="Converting timestamps"):
                    if not line.strip():  # Skip empty lines
                        continue
                        
                    parts = line.strip().split(',')
                    try:
                        timestamp = int(parts[0])
                        # Convert timestamp to nanoseconds if needed
                        timestamp_str = str(timestamp)
                        if len(timestamp_str) == 10:  # seconds precision
                            timestamp = int(timestamp_str + "000000000")
                        elif len(timestamp_str) == 13:  # milliseconds precision
                            timestamp = int(timestamp_str + "000000")
                        elif len(timestamp_str) == 16:  # microseconds precision
                            timestamp = int(timestamp_str + "000")
                        # else assume it's already in nanoseconds (19 digits)
                        
                        # Write the updated line
                        parts[0] = str(timestamp)
                        output_file.write(','.join(parts) + '\n')
                    except (ValueError, IndexError) as e:
                        print(f"Error processing line: {line.strip()}")
                        output_file.write(line)  # Write original line if there's an error
                        continue
                        
            # Replace original file with updated file
            temp_file.replace(file_path)
            print(f"Successfully updated timestamps in {file_path.name}")
            
        except Exception as e:
            print(f"Error processing file {file_path.name}: {e}")
            if temp_file.exists():
                temp_file.unlink()  # Delete temp file if it exists

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze and convert IMU data files.')
    parser.add_argument('--path', type=str, default='.',
                      help='Path to directory containing CSV files (default: current directory)')
    parser.add_argument('--convert', action='store_true',
                      help='Convert timestamps to nanoseconds')
    
    args = parser.parse_args()
    
    if args.convert:
        convert_timestamps_to_ns(args.path)
    else:
        analyze_imu_files(args.path)
