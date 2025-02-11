import pickle
import numpy as np

# Load the camera calibration results from the pickle files
with open("cameraMatrix.pkl", "rb") as f:
    cameraMatrix = pickle.load(f)

with open("dist.pkl", "rb") as f:
    dist = pickle.load(f)

# Extract intrinsic parameters from cameraMatrix
fx = cameraMatrix[0, 0]
fy = cameraMatrix[1, 1]
cx = cameraMatrix[0, 2]
cy = cameraMatrix[1, 2]

# Extract distortion coefficients from dist
# The format of 'dist' can be either a flat array or a 2D array.
if dist.ndim == 2:
    k1 = dist[0, 0]
    k2 = dist[0, 1]
    p1 = dist[0, 2]
    p2 = dist[0, 3]
else:
    k1 = dist[0]
    k2 = dist[1]
    p1 = dist[2]
    p2 = dist[3]

# Print out the extracted parameters
print("Camera1.fx:", fx)
print("Camera1.fy:", fy)
print("Camera1.cx:", cx)
print("Camera1.cy:", cy)
print("Camera1.k1:", k1)
print("Camera1.k2:", k2)
print("Camera1.p1:", p1)
print("Camera1.p2:", p2)

# Optionally, write the parameters into a configuration file format
config_template = f"""#--------------------------------------------------------------------------------------------
# System config
#--------------------------------------------------------------------------------------------

File.version: "1.0"
Camera.type: "PinHole"

# Camera calibration and distortion parameters (from OpenCV calibration)
Camera1.fx: {fx}
Camera1.fy: {fy}
Camera1.cx: {cx}
Camera1.cy: {cy}

Camera1.k1: {k1}
Camera1.k2: {k2}
Camera1.p1: {p1}
Camera1.p2: {p2}

# Adjust image dimensions (update as needed)
Camera.width: 1280
Camera.height: 720

Camera.newWidth: 1280
Camera.newHeight: 720

# Camera frames per second
Camera.fps: 30

# Color order of the images (0: BGR, 1: RGB)
Camera.RGB: 0

#--------------------------------------------------------------------------------------------
# ORB Parameters
#--------------------------------------------------------------------------------------------
ORBextractor.nFeatures: 1000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
"""

# Write the configuration to a file (optional)
with open("camera_config.txt", "w") as config_file:
    config_file.write(config_template)

print("\nConfiguration file 'camera_config.txt' created successfully.")
