import rerun as rr
import numpy as np
import time
import os

def main() -> None:
    print(f"NumPy version: {np.__version__}")
    # Get environment variables for Rerun configuration
    viewer_address = os.environ.get("RERUN_VIEWER_ADDRESS", "0.0.0.0:9090")
    print(f"Viewer address: {viewer_address}")
    
    # Initialize Rerun with application name - don't automatically spawn a viewer
    rr.init("rerun_docker_test_minimal", spawn=False)
    
    print(f"Initialized Rerun with viewer address: {viewer_address}")

    # Log some static 3D points with explicit types to avoid conversion issues
    SIZE = 10

    # Create a simple range of positions instead of using meshgrid
    x = np.linspace(-10, 10, SIZE)
    y = np.linspace(-10, 10, SIZE)
    z = np.linspace(-10, 10, SIZE)
    
    # Manually create points to avoid the numpy.dtype size issue
    positions = []
    colors = []
    
    for i in range(SIZE):
        for j in range(SIZE):
            for k in range(SIZE):
                positions.append([float(x[i]), float(y[j]), float(z[k])])
                colors.append([int(i * 255/SIZE), int(j * 255/SIZE), int(k * 255/SIZE)])
    
    # Convert to NumPy arrays with explicit dtypes after creation
    positions = np.array(positions, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)
    
    # Log using the component API instead of the Points3D class
    rr.log("my_points", {
        "positions": positions,
        "colors": colors,
        "radii": 0.5
    })
    
    print("Finished logging time-series data.")
    print("Attempting to serve a Rerun web viewer...")

    try:
        # Serving the web viewer with the correct API call
        rr.serve_web_viewer(open_browser=False)
        print(f"Rerun web viewer is being served at http://{viewer_address}")
        print("Select 'my_points' in the Rerun viewer to see the visualizations.")
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