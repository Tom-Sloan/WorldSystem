import os
from pathlib import Path

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
                timestamps.append(timestamp)

        # Sort timestamps numerically
        timestamps.sort(key=lambda x: int(x))

        # Create timestamps.txt in the mav0 directory
        mav0_dir = Path(root).parents[2]  # Go up 3 levels: data -> cam0 -> mav0
        output_file = mav0_dir / 'timestamps.txt'
        
        # Write timestamps to file
        with open(output_file, 'w') as f:
            for timestamp in timestamps:
                f.write(f'{timestamp}\n')

if __name__ == '__main__':
    extract_timestamps()
