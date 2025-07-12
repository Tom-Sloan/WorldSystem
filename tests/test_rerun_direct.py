#!/usr/bin/env python3
"""
Direct test of Rerun connectivity - bypasses mesh_service entirely.
This helps verify that Rerun viewer is accessible and working.
"""

import rerun as rr
import numpy as np
import time


def create_test_mesh():
    """Create a simple triangle mesh for testing."""
    # Simple pyramid mesh
    vertices = np.array([
        [0, 0, 0],      # base center
        [1, 0, 0],      # base right
        [0, 1, 0],      # base back
        [-1, 0, 0],     # base left
        [0, -1, 0],     # base front
        [0, 0, 2],      # apex
    ], dtype=np.float32)
    
    # Faces (triangles)
    faces = np.array([
        # Base (as two triangles)
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 1],
        # Sides
        [1, 5, 2],
        [2, 5, 3],
        [3, 5, 4],
        [4, 5, 1],
    ], dtype=np.uint32)
    
    # Vertex colors
    colors = np.array([
        [255, 0, 0],    # red
        [0, 255, 0],    # green
        [0, 0, 255],    # blue
        [255, 255, 0],  # yellow
        [255, 0, 255],  # magenta
        [0, 255, 255],  # cyan
    ], dtype=np.uint8)
    
    return vertices, faces, colors


def main():
    """Test Rerun connectivity directly."""
    print("Direct Rerun Connectivity Test")
    print("==============================")
    
    # Get Rerun address from environment or use default
    import os
    rerun_address = os.environ.get('RERUN_VIEWER_ADDRESS', '127.0.0.1:9876')
    
    try:
        # Initialize Rerun
        print(f"Connecting to Rerun at {rerun_address}...")
        rr.init("rerun_connectivity_test", spawn=True)
        # The init with spawn=True will automatically connect to a viewer
        print("Rerun viewer spawned successfully!")
        
        # Create test mesh
        vertices, faces, colors = create_test_mesh()
        
        # Log the mesh
        print("Sending test mesh to Rerun...")
        rr.log(
            "/test/mesh",
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=faces,
                vertex_colors=colors,
            )
        )
        
        # Also log some points
        rr.log(
            "/test/points", 
            rr.Points3D(
                vertices,
                colors=colors,
                radii=0.05
            )
        )
        
        # Log camera
        rr.log(
            "/test/camera",
            rr.Transform3D(
                translation=[0, -3, 1],
                rotation=rr.RotationAxisAngle(axis=[1, 0, 0], angle=0.2)
            )
        )
        
        print("\nSuccess! Check Rerun viewer - you should see:")
        print("- A colored pyramid mesh at /test/mesh")
        print("- Colored points at /test/points")
        print("- A camera transform at /test/camera")
        
        # Keep updating for a bit
        for i in range(10):
            # Rotate the mesh
            angle = i * 0.1
            rotation = rr.RotationAxisAngle(axis=[0, 0, 1], angle=angle)
            rr.log("/test/mesh", rr.Transform3D(rotation=rotation))
            time.sleep(0.5)
            
    except Exception as e:
        print(f"\nError connecting to Rerun: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Rerun viewer is running:")
        print("   rerun --port 9876")
        print("2. Check if the port is correct")
        print("3. Try opening http://localhost:9876 in a browser")
        

if __name__ == "__main__":
    main()