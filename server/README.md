# Drone Server

A FastAPI-based server for drone navigation and video processing.

## Project Structure

drone_server/
├── src/
│ ├── init.py
│ ├── main.py # FastAPI app initialization
│ ├── config/
│ │ ├── init.py
│ │ └── settings.py # Configuration and environment settings
│ └── core/
│ ├── init.py
│ └── frame_processor.py # Video processing and YOLO integration
├── Mac/
│ ├── Dockerfile # macOS specific Dockerfile
│ ├── environment.yml # macOS conda environment
│ └── Old_Files/
│ └── combined_backend.py # Legacy combined server implementation
├── models/
│ └── OBJ/ # 3D model files
├── Dockerfile # GPU-enabled Dockerfile
└── environment.yml # Main conda environment with CUDA support

## Setup

1. Build the Docker image:

```
docker build --no-cache -t drone_server:latest .
```

2. Run the Docker container:

```
docker run --gpus all --name drone_server_container -it \
 -p 5001:5001 \
 -v "$(pwd)":/app \
 drone_server:latest
```

3. Start the container:

```
docker start -i drone_server_container
```

## Codebase Overview

### Key Components

1. **Core Video Processing**

    - Centralized frame processing in `src/core/frame_processor.py`
    - YOLO model integration with GPU support
    - Asynchronous frame queue management

2. **Configuration Management**

    - Dedicated settings module in `src/config/settings.py`
    - GPU/CUDA configuration
    - Performance tuning parameters

3. **Platform-Specific Builds**

    - Separate Dockerfile for macOS builds
    - GPU-enabled Dockerfile for CUDA support
    - Platform-specific environment configurations

4. **WebSocket Communication**
    - Video frame streaming
    - Telemetry data handling (attitude, velocity)
    - Connection management

For reference to the implementation details, see:

### Environment Requirements

The project uses Conda for environment management with the following key dependencies:

-   Python 3.11
-   FastAPI
-   Ultralytics (YOLO)
-   OpenCV
-   PyTorch
-   NumPy
-   Trimesh
-   Pathfinding3D

### API Endpoints

1. **Model Management**
    - `/obj` - Serve and load 3D model files
2. **Navigation**

    - `/route_projection` - Calculate pathfinding routes
    - `/current_location` - Get/set drone position
    - `/route_random` - Generate test points
    - `/torus` - Generate torus path points
    - `/destination` - Get current destination

3. **WebSocket**
    - `/ws/video` - Handle video streaming and commands

### Debug Mode

The server includes comprehensive debugging features:

-   System resource logging
-   Performance metrics tracking
-   Detailed error reporting
-   Frame processing statistics

## Data Output
recording_dir/
  └── 20250128_193743/           # Timestamp-based recording folder
      └── mav0/
          ├── EuRoC.yaml         # ORB_SLAM3 config file
          ├── cam0/              # Camera data
          │   ├── data/          # Image files
          │   │   ├── 1738111168152999936.jpg
          │   │   ├── 1738111168184999936.jpg
          │   │   └── ...
          │   ├── data.csv       # Image timestamps and filenames
          │   └── timestamps.txt # Raw timestamps list
          │
          └── imu0/             # IMU data
              └── data/         # Individual IMU readings
                  ├── 1738111168152999936.txt
                  ├── 1738111168184999936.txt
                  └── ...

### Message Formats

To summarize the message types:
"velocity_data" - for velocity information
"attitude_data" - for aircraft attitude
"location_data" - for GPS coordinates
"status_message" - for general status updates (converted from plain text)
"error_message" - for error messages
"button_event" - for button interactions
"battery_status" - for battery updates
"video_frame" - for video frames (JPEG format)
"is_local" - for local connection status updates
{
"type": "video_frame",
"frame_data": "base64_encoded_frame_data",
"timestamp": 1234567890,
"imu_data": {
    "attitude": {
        "pitch": -4.5,
        "roll": -0.2, 
        "yaw": 146.5
    },
    "gimbal_attitude": {
        "pitch": 0,
        "roll": 0,
        "yaw": 147.1,
        "yaw_relative_to_aircraft": 0
    },
    "gimbal_calibration": {
        "state": "UNKNOWN",
        "progress": 0,
        "message": ""
    },
    "imu_calibration": {
        "state": "NONE", 
        "required_orientations": [
            "TAIL_DOWN",
            "RIGHT_DOWN", 
            "LEFT_DOWN",
            "BOTTOM_DOWN",
            "TOP_DOWN"
        ],
        "message": ""
    },
    "location": {
        "latitude": 0,
        "longitude": 0
    },
    "velocity": {
        "x": 0,
        "y": 0,
        "z": 0
    },
"timestamp": 1234567890
}
}

#### IMU Data

IMU data is written to a CSV file with the following format:

```
timestamp [ns], w_RS_S_x [rad s^-1], w_RS_S_y [rad s^-1], w_RS_S_z [rad s^-1], a_RS_S_x [m s^-2], a_RS_S_y [m s^-2], a_RS_S_z [m s^-2]
```