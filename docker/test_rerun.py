#!/usr/bin/env python3
"""Minimal example of using the Rerun SDK to log and serve data over HTTP in an NVIDIA container."""

from __future__ import annotations

import numpy as np
import rerun as rr  # pip install rerun-sdk
import time
import math
import os

def main() -> None:
    # Get environment variables for Rerun configuration
    viewer_address = os.environ.get("RERUN_VIEWER_ADDRESS", "127.0.0.1:9090")
    print(f"Viewer address: {viewer_address}")
    
    # Properly initialize Rerun with application name
    rr.init("rerun_docker_test_minimal")
    
    print(f"Initialized Rerun with viewer address: {viewer_address}")

    # Log some static 3D points with explicit types to avoid conversion issues
    num_points = 15
    
    # Create arrays directly with the correct types (no conversions)
    points = np.random.rand(num_points, 3).astype(np.float32) * 10
    colors = np.random.randint(0, 255, size=(num_points, 3), dtype=np.uint8)
    radii = np.random.rand(num_points).astype(np.float32) * 0.5

    # Try-except block to handle potential errors with more diagnostic information
    try:
        # Log the points as static
        rr.log("my_docker_points", rr.Points3D(points, colors=colors, radii=radii))
        print(f"Successfully logged {num_points} static points to Rerun.")
    except Exception as e:
        print(f"Error logging points: {e}")
        print(f"Points shape: {points.shape}, dtype: {points.dtype}")
        print(f"Colors shape: {colors.shape}, dtype: {colors.dtype}")
        print(f"Radii shape: {radii.shape}, dtype: {radii.dtype}")

    # Log a scalar over time to create a graph
    print("Logging time-series data for a graph...")
    for i in range(200):  # Log 200 data points for the sine wave
        # Set the current time for this data point
        rr.set_time_sequence("plot_time", i)  # Updated according to docs

        # Calculate a sine wave value
        value = math.sin(i * 0.1)

        # Log the scalar value
        rr.log("my_graph/sine_wave", rr.Scalar(value))  # Using Scalar instead of Scalars

        time.sleep(0.05)  # Sleep for 50ms to make the updates visible

    print("Finished logging time-series data.")
    print("Attempting to serve a Rerun web viewer...")

    try:
        # Configure the web viewer for Docker with host networking
        # Using the address format according to the Rerun docs
        rr.serve_web_viewer(open_browser=False)
        print(f"Rerun web viewer is being served at http://{viewer_address}")
        print("Select 'my_graph/sine_wave' in the Rerun viewer to see the plot.")
        print("The script will keep running to serve the viewer. Press Ctrl+C to stop.")
        
        # Keep the main thread alive to serve the web viewer
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Ctrl+C received, shutting down viewer and script.")
    except Exception as e:
        print(f"Error serving Rerun web viewer: {e}")
        print(f"Viewer address: {viewer_address}")
        exit(1)

if __name__ == "__main__":
    main()