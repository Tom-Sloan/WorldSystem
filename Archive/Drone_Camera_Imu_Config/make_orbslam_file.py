#!/usr/bin/env python3
"""
Generate an ORB-SLAM configuration (mono or mono-inertial) from a YAML calibration file.
Preserve `!!opencv-matrix` for IMU.T_b_c1, using a custom Loader/Dumper.
And produce a flow-style list for `data`.
"""

import argparse
import yaml
import numpy as np

#
# 1) Define a small class + custom Loader/Dumper for !!opencv-matrix
#

class OpenCVMatrix:
    def __init__(self, rows=4, cols=4, dt="f", data=None):
        self.rows = rows
        self.cols = cols
        self.dt = dt
        self.data = data if data is not None else []

class MyLoader(yaml.SafeLoader):
    pass

class MyDumper(yaml.SafeDumper):
    pass

def opencv_matrix_constructor(loader, node):
    """
    Parse a YAML map with tag !opencv-matrix into an OpenCVMatrix object.
    E.g.:
      !!opencv-matrix
        rows: 4
        cols: 4
        dt: f
        data: [0,1,2,3,...]
    """
    mapping = loader.construct_mapping(node, deep=True)
    return OpenCVMatrix(**mapping)

def opencv_matrix_representer(dumper, matrix):
    """
    Dump an OpenCVMatrix object as !!opencv-matrix in block style,
    but with the 'data' array in flow style.
    """
    # Build a list of (key_node, value_node) pairs for the map
    rows_node = dumper.represent_data(matrix.rows)
    cols_node = dumper.represent_data(matrix.cols)
    dt_node   = dumper.represent_data(matrix.dt)

    # Represent the data array. We force "flow_style=True" so it appears in brackets.
    data_node = dumper.represent_data(matrix.data)
    data_node.flow_style = True

    # Create the mapping node for !!opencv-matrix
    mapping_node = yaml.representer.MappingNode(
        tag='tag:yaml.org,2002:opencv-matrix',
        value=[
            (dumper.represent_data('rows'), rows_node),
            (dumper.represent_data('cols'), cols_node),
            (dumper.represent_data('dt'),   dt_node),
            (dumper.represent_data('data'), data_node),
        ],
        flow_style=False  # block style for the map itself
    )
    return mapping_node

# Register the constructor & representer
MyLoader.add_constructor(u'tag:yaml.org,2002:opencv-matrix', opencv_matrix_constructor)
MyDumper.add_representer(OpenCVMatrix, opencv_matrix_representer)

#
# 2) Templates with !!opencv-matrix in the inertial version
#   (No %YAML line at top, we'll prepend it ourselves on output)
#

EUROC_MONO_TEMPLATE = """File.version: "1.0"
Camera.type: "PinHole"

Camera1.fx: 905.9805764723004
Camera1.fy: 903.8697216493965
Camera1.cx: 628.6583531774853
Camera1.cy: 348.90016625665316

Camera1.k1: 0.20796653247283758
Camera1.k2: -1.0209783326144832
Camera1.p1: -0.0005810334199116348
Camera1.p2: -0.0033936448334717616

Camera.width: 1280
Camera.height: 720
Camera.newWidth: 1280
Camera.newHeight: 720
Camera.fps: 30
Camera.RGB: 0

ORBextractor.nFeatures: 1000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

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

EUROC_MONO_INERTIAL_TEMPLATE = """File.version: "1.0"
Camera.type: "PinHole"

Camera1.fx: 905.9805764723004
Camera1.fy: 903.8697216493965
Camera1.cx: 628.6583531774853
Camera1.cy: 348.90016625665316

Camera1.k1: 0.20796653247283758
Camera1.k2: -1.0209783326144832
Camera1.p1: -0.0005810334199116348
Camera1.p2: -0.0033936448334717616

Camera.width: 1280
Camera.height: 720
Camera.newWidth: 1280
Camera.newHeight: 720
Camera.fps: 30
Camera.RGB: 0

IMU.T_b_c1: !!opencv-matrix
   rows: 4
   cols: 4
   dt: f
   data: [0, 0, -1, 0,
          1, 0,  0, 0,
          0, 1,  0, 0,
          0, 0,  0, 1]

IMU.NoiseGyro: 9.343406e-04
IMU.NoiseAcc: 1.517782e-02
IMU.GyroWalk: 9.382950e+00
IMU.AccWalk: 1.528946e+02
IMU.Frequency: 200.0

ORBextractor.nFeatures: 1000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -3.5
Viewer.ViewpointF: 500.0
"""

#
# 3) Utility to load the new calibration data
#

def load_yaml_file(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_template(is_inertial=False):
    """
    Load the default template with MyLoader so it can parse !!opencv-matrix.
    """
    if is_inertial:
        return yaml.load(EUROC_MONO_INERTIAL_TEMPLATE, Loader=MyLoader)
    else:
        return yaml.load(EUROC_MONO_TEMPLATE, Loader=MyLoader)

def invert_4x4(matrix):
    arr = np.array(matrix, dtype=float)
    return np.linalg.inv(arr).tolist()

#
# 4) Main logic: override the template, then dump with MyDumper
#

def main():
    parser = argparse.ArgumentParser(
        description="Generate an ORB-SLAM config from camera and IMU YAML files, preserving !!opencv-matrix."
    )
    parser.add_argument("--cam-file", required=True, help="Path to camera calibration YAML (with T_cam_imu).")
    parser.add_argument("--imu-file", required=True, help="Path to IMU calibration YAML.")
    parser.add_argument("--output", required=True, help="Path to output ORB-SLAM config.")
    args = parser.parse_args()

    # Load new calibration data
    cam_data = load_yaml_file(args.cam_file)
    imu_data = load_yaml_file(args.imu_file)
    if not cam_data or not imu_data:
        raise ValueError(f"Input files are empty or invalid YAML.")

    # Always use inertial template since we have two input files
    template = load_template(is_inertial=True)

    # Override camera fields from cam_data
    if "cam0" in cam_data:
        cam = cam_data["cam0"]
        intr = cam.get("intrinsics", [])
        if len(intr) == 4:
            template["Camera1.fx"] = intr[0]
            template["Camera1.fy"] = intr[1]
            template["Camera1.cx"] = intr[2]
            template["Camera1.cy"] = intr[3]

        dist = cam.get("distortion_coeffs", [])
        if len(dist) >= 4:
            template["Camera1.k1"] = dist[0]
            template["Camera1.k2"] = dist[1]
            template["Camera1.p1"] = dist[2]
            template["Camera1.p2"] = dist[3]

        res = cam.get("resolution", [])
        if len(res) == 2:
            template["Camera.width"] = res[0]
            template["Camera.height"] = res[1]
            template["Camera.newWidth"] = res[0]
            template["Camera.newHeight"] = res[1]

        # Get T_cam_imu and invert to get T_b_c1
        if "T_cam_imu" in cam:
            T_cam_imu = cam["T_cam_imu"]  # 4x4
            T_b_c1_mat = invert_4x4(T_cam_imu)
            # Store as OpenCVMatrix so the dumper re-tags it as !opencv-matrix
            ocv = OpenCVMatrix(rows=4, cols=4, dt="f", data=sum(T_b_c1_mat, []))
            template["IMU.T_b_c1"] = ocv

    # Override IMU noise from imu_data
    if "imu0" in imu_data:
        imu = imu_data["imu0"]
        if "gyroscope_noise_density" in imu:
            template["IMU.NoiseGyro"] = imu["gyroscope_noise_density"]
        if "accelerometer_noise_density" in imu:
            template["IMU.NoiseAcc"] = imu["accelerometer_noise_density"]
        if "gyroscope_random_walk" in imu:
            template["IMU.GyroWalk"] = imu["gyroscope_random_walk"]
        if "accelerometer_random_walk" in imu:
            template["IMU.AccWalk"] = imu["accelerometer_random_walk"]
        if "update_rate" in imu:
            template["IMU.Frequency"] = float(imu["update_rate"])

    # Write output
    with open(args.output, "w") as f:
        f.write("%YAML:1.0\n\n")
        yaml.dump(template, f, Dumper=MyDumper, sort_keys=False, width=9999)

    print(f"ORB-SLAM config generated at: {args.output}")

if __name__ == "__main__":
    main()
