import rerun as rr
import numpy as np
import os

def main() -> None:
    print(f"NumPy version: {np.__version__}")
    # Get environment variables for Rerun configuration
    viewer_address = os.environ.get("RERUN_VIEWER_ADDRESS", "0.0.0.0:9090")
    print(f"Viewer address: {viewer_address}")
    
    # Initialize Rerun with application name - don't automatically spawn a viewer
    rr.init("rerun_docker_test_minimal", spawn=False)
    
    print(f"Initialized Rerun with viewer address: {viewer_address}")

    print("Finished logging time-series data.")
    print("Attempting to connect to a Rerun viewer via gRPC...")

    try:
        # Get the recording stream
        positions = np.zeros((10, 3))
        positions[:,0] = np.linspace(-10,10,10)

        colors = np.zeros((10,3), dtype=np.uint8)
        colors[:,0] = np.linspace(0,255,10)

        rr.log(
            "my_points",
            rr.Points3D(positions, colors=colors, radii=0.5)
        )
                
        # Connect to the viewer using gRPC instead of serving the web viewer
        url = os.getenv(
            "RERUN_CONNECT_URL",
            "rerun+http://localhost:9876/proxy"  # sensible fallback
        )
        print(f"Connecting to Rerun viewer via gRPC at {url}")
        rr.connect_grpc(url)
        
        print(f"Connected to Rerun viewer via gRPC at {viewer_address}")
        print("Select 'my_points' in the Rerun viewer to see the visualizations.")
        print("Data has been sent to the viewer. Script will now exit.")
        
        # Note: No need to keep the main thread alive with connect_grpc unless
        # you're continuously sending data
        
    except Exception as e:
        print(f"Error connecting to Rerun viewer via gRPC: {e}")
        print(f"Viewer address: {viewer_address}")
        exit(1)

if __name__ == "__main__":
    main()