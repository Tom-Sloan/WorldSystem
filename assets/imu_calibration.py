import numpy as np

# Transformation from phone to camera coordinates (provided)
R_phone_to_cam = np.array([
    [ 0,  1,  0 ],
    [ 0,  0,  1 ],
    [-1,  0,  0 ]
], dtype=float)

def transform_imu(phone_gyro, phone_accel):
    """
    Transform IMU data from phone coordinates to camera coordinates.
    
    Args:
        phone_gyro (np.ndarray): [gx, gy, gz] in phone coordinates.
        phone_accel (np.ndarray): [ax, ay, az] in phone coordinates.
    
    Returns:
        tuple: (gyro_cam, accel_cam) in camera coordinates.
    """
    gyro_cam = R_phone_to_cam @ phone_gyro
    accel_cam = R_phone_to_cam @ phone_accel
    return gyro_cam, accel_cam

def read_imu_file(filename):
    """
    Reads an IMU file in EuRoC format.
    
    Expected CSV line format:
      timestamp [ms], w_RS_S_x, w_RS_S_y, w_RS_S_z, a_RS_S_x, a_RS_S_y, a_RS_S_z
    
    Skips header and comment lines.
    
    Returns:
        List of tuples: Each tuple is (timestamp, gyro_cam, accel_cam), where
            - timestamp is a float (milliseconds),
            - gyro_cam and accel_cam are 3-element numpy arrays (in camera frame).
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(',')
            if len(parts) < 7:
                continue
            # Parse values; timestamp is in milliseconds.
            timestamp = float(parts[0])
            phone_gyro  = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            phone_accel = np.array([float(parts[4]), float(parts[5]), float(parts[6])])
            # Transform to camera coordinates.
            gyro_cam, accel_cam = transform_imu(phone_gyro, phone_accel)
            data.append((timestamp, gyro_cam, accel_cam))
    return data

def compute_calibration_parameters(data):
    """
    Computes IMU noise parameters and random walk estimates from the data.
    
    Returns:
        noise_gyro: Estimated white noise (std) for the gyroscope (average over axes).
        noise_accel: Estimated white noise (std) for the accelerometer (average over axes).
        gyro_walk: Estimated gyro random walk (rad/s/√s).
        accel_walk: Estimated accelerometer random walk (m/s²/√s).
        frequency: Sampling frequency in Hz.
    """
    if not data:
        return None, None, None, None, None

    # Stack measurements for easier processing.
    gyros = np.array([d[1] for d in data])    # shape: (N, 3)
    accels = np.array([d[2] for d in data])     # shape: (N, 3)
    timestamps = np.array([d[0] for d in data]) # in ms

    # Compute biases (means) and residuals.
    gyro_bias = np.mean(gyros, axis=0)
    accel_bias = np.mean(accels, axis=0)
    gyro_res = gyros - gyro_bias
    accel_res = accels - accel_bias

    # White noise estimates (standard deviation) averaged over axes.
    noise_gyro = np.mean(np.std(gyro_res, axis=0))
    noise_accel = np.mean(np.std(accel_res, axis=0))

    # Estimate sampling frequency (timestamps are in milliseconds)
    dt = np.diff(timestamps) * 1e-3  # convert ms to seconds
    dt_mean = np.mean(dt)
    frequency = 1.0 / dt_mean if dt_mean > 0 else 0.0

    # Estimate random walk parameters.
    # Approximate random walk using the standard deviation of consecutive differences.
    gyro_diff = np.diff(gyros, axis=0)
    accel_diff = np.diff(accels, axis=0)
    gyro_diff_std = np.mean(np.std(gyro_diff, axis=0))
    accel_diff_std = np.mean(np.std(accel_diff, axis=0))
    gyro_walk = gyro_diff_std / np.sqrt(2 * dt_mean) if dt_mean > 0 else 0.0
    accel_walk = accel_diff_std / np.sqrt(2 * dt_mean) if dt_mean > 0 else 0.0

    return noise_gyro, noise_accel, gyro_walk, accel_walk, frequency

def print_transformation_yaml(T):
    """
    Prints the 4x4 transformation matrix T in EuRoC YAML format
    using OpenCV's matrix notation.
    
    Args:
        T (np.ndarray): 4x4 transformation matrix.
    """
    flat_T = T.flatten()
    # Format each value with 12 significant digits.
    data_str = ", ".join("{:.12g}".format(val) for val in flat_T)
    print("# Transformation from camera to body-frame (imu)")
    print("IMU.T_b_c1: !!opencv-matrix")
    print("   rows: 4")
    print("   cols: 4")
    print("   dt: f")
    print("   data: [" + data_str + "]")

if __name__ == '__main__':
    # Path to your EuRoC-format IMU file (adjust as needed)
    imu_file = './assets/imu.csv'
    
    # Read the IMU data.
    data = read_imu_file(imu_file)
    if not data:
        print("No data read from file. Please check the file format and path.")
        exit(1)
    
    # Compute calibration parameters.
    noise_gyro, noise_accel, gyro_walk, accel_walk, frequency = compute_calibration_parameters(data)
    
    # Print out the discovered calibration constants.
    print("IMU.NoiseGyro: {:.6e}".format(noise_gyro))
    print("IMU.NoiseAcc: {:.6e}".format(noise_accel))
    print("IMU.GyroWalk: {:.6e}".format(gyro_walk))
    print("IMU.AccWalk: {:.6e}".format(accel_walk))
    print("IMU.Frequency: {:.1f}".format(frequency))
    print()
    
    # Compute the transformation from camera to body frame.
    # Here we assume the body (IMU) frame is the phone frame.
    # Given that R_phone_to_cam transforms phone -> camera, the inverse (transpose)
    # transforms camera -> phone (body).
    R_cam_to_body = R_phone_to_cam.T
    T_b_c1 = np.eye(4)
    T_b_c1[:3, :3] = R_cam_to_body

    # Print the transformation in YAML format.
    print_transformation_yaml(T_b_c1)
