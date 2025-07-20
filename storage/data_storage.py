import time
import pika
import os
import json
import cv2
import base64
import numpy as np
from pathlib import Path
from datetime import datetime
from video_storage import VideoStorageIntegration

# PROMETHEUS
from prometheus_client import start_http_server, Counter, Histogram

############
# Define metrics
############

# Counters: total number of each data type saved
images_saved_counter = Counter(
    "data_storage_images_saved_total",
    "Total number of images successfully saved by data_storage"
)
imu_saved_counter = Counter(
    "data_storage_imu_saved_total",
    "Total number of IMU records saved by data_storage"
)
processed_imu_saved_counter = Counter(
    "data_storage_processed_imu_saved_total",
    "Total number of processed IMU records saved by data_storage"
)
trajectories_saved_counter = Counter(
    "data_storage_trajectories_saved_total",
    "Total number of trajectory lines saved by data_storage"
)
ply_saved_counter = Counter(
    "data_storage_ply_saved_total",
    "Total number of .ply files saved by data_storage"
)

# Histograms: time taken for each save operation
save_image_hist = Histogram(
    "data_storage_save_image_seconds",
    "Time spent saving a single image file",
    buckets=[0.001, 0.01, 0.1, 0.5, 1, 2, 5]
)
save_imu_hist = Histogram(
    "data_storage_save_imu_seconds",
    "Time spent saving a single IMU record",
    buckets=[0.0005, 0.001, 0.005, 0.01, 0.1, 1]
)
save_processed_imu_hist = Histogram(
    "data_storage_save_processed_imu_seconds",
    "Time spent saving a single processed IMU record",
    buckets=[0.0005, 0.001, 0.005, 0.01, 0.1, 1]
)
save_trajectory_hist = Histogram(
    "data_storage_save_trajectory_seconds",
    "Time spent saving a single trajectory record",
    buckets=[0.0005, 0.001, 0.005, 0.01, 0.1, 1]
)
save_ply_hist = Histogram(
    "data_storage_save_ply_seconds",
    "Time spent saving a single PLY file",
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10]
)

class DataStorage:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.recording_path = None
        self.trajectory_file = None
        self.video_storage = None  # Will be initialized in start_recording
        self.save_individual_frames = os.getenv("SAVE_INDIVIDUAL_FRAMES", "false").lower() == "true"
        self._connect_with_retry()  # tries to connect until success

    def _connect_with_retry(self, max_retries=30, delay=2):
        """Keep trying to connect to RabbitMQ up to `max_retries` times."""
        amqp_url = os.getenv("RABBITMQ_URL", "amqp://rabbitmq")
        for attempt in range(1, max_retries + 1):
            try:
                params = pika.URLParameters(amqp_url)
                params.heartbeat = 3600  # 1 hour heartbeat

                self.connection = pika.BlockingConnection(params)
                print(f"Attempting to connect to RabbitMQ: {amqp_url}, attempt {attempt}")
                self.channel = self.connection.channel()
                print("Connected to RabbitMQ successfully!")
                break
            except pika.exceptions.AMQPConnectionError as e:
                print(f"Connection attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    raise e
                time.sleep(delay)

        # ------------------------------------------------------
        # Declare the fanout exchanges from FrameProcessor or SLAM:
        # ------------------------------------------------------
        VIDEO_FRAMES_EXCHANGE   = os.getenv("VIDEO_FRAMES_EXCHANGE", "video_frames_exchange")
        IMU_DATA_EXCHANGE       = os.getenv("IMU_DATA_EXCHANGE", "imu_data_exchange")
        TRAJECTORY_DATA_EXCHANGE = os.getenv("TRAJECTORY_DATA_EXCHANGE", "trajectory_data_exchange")
        PLY_FANOUT_EXCHANGE      = os.getenv("PLY_FANOUT_EXCHANGE", "ply_fanout_exchange")
        RESTART_EXCHANGE         = os.getenv("RESTART_EXCHANGE", "restart_exchange")
        PROCESSED_IMU_EXCHANGE   = os.getenv("PROCESSED_IMU_EXCHANGE", "processed_imu_exchange")

        self.channel.exchange_declare(exchange=VIDEO_FRAMES_EXCHANGE, exchange_type='fanout', durable=True)
        self.channel.exchange_declare(exchange=IMU_DATA_EXCHANGE, exchange_type='fanout', durable=True)
        self.channel.exchange_declare(exchange=TRAJECTORY_DATA_EXCHANGE, exchange_type='fanout', durable=True)
        self.channel.exchange_declare(exchange=PLY_FANOUT_EXCHANGE, exchange_type='fanout', durable=True)
        self.channel.exchange_declare(exchange=RESTART_EXCHANGE, exchange_type='fanout', durable=True)
        self.channel.exchange_declare(exchange=PROCESSED_IMU_EXCHANGE, exchange_type='fanout', durable=True)

        # ------------------------------------------------------
        # Create queues & bind them to the relevant exchange
        # ------------------------------------------------------
        # images -> image_data_storage
        self.channel.queue_declare(queue='image_data_storage', durable=True)
        self.channel.queue_bind(
            exchange=VIDEO_FRAMES_EXCHANGE,
            queue='image_data_storage',
            routing_key=''
        )

        # IMU -> imu_data_storage
        self.channel.queue_declare(queue='imu_data_storage', durable=True)
        self.channel.queue_bind(
            exchange=IMU_DATA_EXCHANGE,
            queue='imu_data_storage',
            routing_key=''
        )

        # trajectory -> trajectory_data_storage
        self.channel.queue_declare(queue='trajectory_data_storage', durable=True)
        self.channel.queue_bind(
            exchange=TRAJECTORY_DATA_EXCHANGE,
            queue='trajectory_data_storage',
            routing_key=''
        )

        # .ply -> ply_data_storage
        self.channel.queue_declare(queue='ply_data_storage', durable=True)
        self.channel.queue_bind(
            exchange=PLY_FANOUT_EXCHANGE,
            queue='ply_data_storage',
            routing_key=''
        )

        # restart -> restart_data_storage
        self.channel.queue_declare(queue='restart_data_storage', durable=True)
        self.channel.queue_bind(
            exchange=RESTART_EXCHANGE,
            queue='restart_data_storage',
            routing_key=''
        )
        self.channel.basic_consume(
            queue='restart_data_storage',
            on_message_callback=self._handle_restart,
            auto_ack=True
        )

        # Processed IMU -> processed_imu_data_storage
        self.channel.queue_declare(queue='processed_imu_data_storage', durable=True)
        self.channel.queue_bind(
            exchange=PROCESSED_IMU_EXCHANGE,
            queue='processed_imu_data_storage',
            routing_key=''
        )

    def start_recording(self):
        """Create the /data/<timestamp>/mav0 folder and open files for trajectory and IMU data."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_path = Path('/data') / timestamp / "mav0"
        
        # Initialize video storage
        self.video_storage = VideoStorageIntegration(self.recording_path)
        
        # Create directories for cameras, IMU and ply data.
        (self.recording_path / "cam0" / "data").mkdir(parents=True, exist_ok=True)
        (self.recording_path / "cam1" / "data").mkdir(parents=True, exist_ok=True)  # New PNG folder
        (self.recording_path / "imu0").mkdir(parents=True, exist_ok=True)
        (self.recording_path / "imu1").mkdir(parents=True, exist_ok=True)  # New folder for processed IMU data
        (self.recording_path / "ply").mkdir(parents=True, exist_ok=True)
        
        # Open trajectory file (unchanged).
        trajectory_file_path = self.recording_path / "trajectory.txt"
        self.trajectory_file = open(trajectory_file_path, 'a')
        print(f"Trajectory will be saved to: {trajectory_file_path}")
        
        # Open a persistent CSV file for IMU data.
        imu_csv_path = self.recording_path / "imu0" / "data.csv"
        self.imu_csv_file = open(imu_csv_path, 'w')
        # Write the EuRoC header.
        self.imu_csv_file.write(
            "#timestamp [ns],"
            "w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
            "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
        )
        
        # Open a persistent CSV file for processed IMU data
        processed_imu_csv_path = self.recording_path / "imu1" / "data.csv"
        self.processed_imu_csv_file = open(processed_imu_csv_path, 'w')
        # Write the EuRoC header.
        self.processed_imu_csv_file.write(
            "#timestamp [ns],"
            "w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
            "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
        )

    def _save_image(self, ch, method, properties, body):
        start_time = time.time()
        try:
            # First, try to handle as video data
            if self.video_storage:
                self.video_storage.handle_video_message(ch, method, properties, body)
                
            # Only save individual frames if enabled
            if not self.save_individual_frames:
                return
                
            # If there is no timestamp in the headers, discard the message.
            if not (properties and properties.headers and "timestamp_ns" in properties.headers):
                print("[!] No timestamp in image message; discarding message.")
                return

            timestamp_ns = int(properties.headers["timestamp_ns"])
            # Decode the raw JPEG bytes directly
            np_data = np.frombuffer(body, np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
            if frame is None:
                print(f"[!] Received invalid image data at timestamp {timestamp_ns}")
                return

            # Save as JPEG in cam0
            jpg_path = self.recording_path / "cam0" / "data" / f"{timestamp_ns}.jpg"
            cv2.imwrite(str(jpg_path), frame)

            # Save as PNG in cam1
            png_path = self.recording_path / "cam1" / "data" / f"{timestamp_ns}.png"
            cv2.imwrite(str(png_path), frame)

            images_saved_counter.inc(2)  # Increment by 2 since we're saving 2 images
        finally:
            elapsed = time.time() - start_time
            save_image_hist.observe(elapsed)

    def _save_imu(self, ch, method, properties, body):
        start_time = time.time()
        try:
            data = json.loads(body)

            # Ensure a timestamp is present.
            if 'timestamp' not in data:
                print("[!] IMU message missing 'timestamp'; discarding message.")
                return

            imu_timestamp = data['timestamp']
            
            # Convert timestamp to nanoseconds if needed
            timestamp_str = str(imu_timestamp)
            if len(timestamp_str) == 10:  # seconds precision
                imu_timestamp = int(timestamp_str + "000000000")
            elif len(timestamp_str) == 13:  # milliseconds precision
                imu_timestamp = int(timestamp_str + "000000")
            elif len(timestamp_str) == 16:  # microseconds precision
                imu_timestamp = int(timestamp_str + "000")
            # else assume it's already in nanoseconds (19 digits)

            # Use nested 'imu_data' if available; otherwise assume sensor data is at the top level.
            if 'imu_data' in data:
                sensor_data = data['imu_data']
            else:
                sensor_data = data

            # Check that both gyroscope and accelerometer data exist.
            if 'gyroscope' not in sensor_data or 'accelerometer' not in sensor_data:
                print("[!] IMU message missing sensor data; discarding message.")
                return

            gyroscope = sensor_data['gyroscope']
            accelerometer = sensor_data['accelerometer']

            # Ensure expected keys exist in both sensor dictionaries.
            if not all(k in gyroscope for k in ['x', 'y', 'z']):
                print("[!] IMU message missing gyroscope components; discarding message.")
                return
            if not all(k in accelerometer for k in ['x', 'y', 'z']):
                print("[!] IMU message missing accelerometer components; discarding message.")
                return

            # Create a CSV-formatted line (EuRoC format):
            # Format: timestamp [ns], w_RS_S_x, w_RS_S_y, w_RS_S_z, a_RS_S_x, a_RS_S_y, a_RS_S_z
            line = (
                f"{imu_timestamp},"
                f"{gyroscope['x']},{gyroscope['y']},{gyroscope['z']},"
                f"{accelerometer['x']},{accelerometer['y']},{accelerometer['z']}\n"
            )

            # Append the new line to the persistent CSV file.
            self.imu_csv_file.write(line)
            self.imu_csv_file.flush()  # Optional: flush after each write to persist data immediately.

            imu_saved_counter.inc()
        except Exception as e:
            print("[!] Exception in _save_imu:", e)
        finally:
            elapsed = time.time() - start_time
            save_imu_hist.observe(elapsed)

    def _save_trajectory(self, ch, method, properties, body):
        """Callback for storing trajectory data from 'trajectory_data_storage' queue."""
        start_time = time.time()
        try:
            if not self.trajectory_file:
                print("[!] Trajectory file not open! Did you call start_recording() first?")
                return

            data = json.loads(body)
            # Discard if no timestamp in trajectory message.
            if "timestamp_ns" not in data:
                print("[!] No timestamp in trajectory message; discarding message.")
                return
            ts_ns = data["timestamp_ns"]
            pose = data.get("pose")
            record = {
                "timestamp_ns": ts_ns,
                "pose": pose,
                "received_timestamp_ns": int(time.time() * 1e9)  # Add received timestamp
            }
            line = json.dumps(record)
            self.trajectory_file.write(line + "\n")
            self.trajectory_file.flush()

            trajectories_saved_counter.inc()
        finally:
            elapsed = time.time() - start_time
            save_trajectory_hist.observe(elapsed)

    def _save_ply(self, ch, method, properties, body):
        """Callback for storing .ply files from 'ply_data_storage' queue."""
        start_time = time.time()
        try:
            print(f"[+] Received PLY data message on {method.exchange}/{method.routing_key} queue")
            msg = json.loads(body)
            ply_filename = msg.get("ply_filename", "output.ply")
            ply_data_b64 = msg.get("ply_data_b64")
            if not ply_data_b64:
                print("[!] No base64 data in ply message.")
                return

            ply_data = base64.b64decode(ply_data_b64)
            out_path = self.recording_path / "ply" / ply_filename
            with open(out_path, "wb") as f:
                f.write(ply_data)

            print(f"[+] Successfully saved PLY file '{ply_filename}' ({len(ply_data)/1024:.1f} KB) to: {out_path}")

            ply_saved_counter.inc()
        finally:
            elapsed = time.time() - start_time
            save_ply_hist.observe(elapsed)

    def _save_processed_imu(self, ch, method, properties, body):
        start_time = time.time()
        try:
            data = json.loads(body)

            # Ensure a timestamp is present.
            if 'timestamp' not in data:
                print("[!] Processed IMU message missing 'timestamp'; discarding message.")
                return

            imu_timestamp = data['timestamp']
            
            # Convert timestamp to nanoseconds if needed
            timestamp_str = str(imu_timestamp)
            if len(timestamp_str) == 10:  # seconds precision
                imu_timestamp = int(timestamp_str + "000000000")
            elif len(timestamp_str) == 13:  # milliseconds precision
                imu_timestamp = int(timestamp_str + "000000")
            elif len(timestamp_str) == 16:  # microseconds precision
                imu_timestamp = int(timestamp_str + "000")
            # else assume it's already in nanoseconds (19 digits)

            # Check that both gyroscope and accelerometer data exist.
            if 'gyroscope' not in data or 'accelerometer' not in data:
                print("[!] Processed IMU message missing sensor data; discarding message.")
                return

            gyroscope = data['gyroscope']
            accelerometer = data['accelerometer']

            # Ensure expected keys exist in both sensor dictionaries.
            if not all(k in gyroscope for k in ['x', 'y', 'z']):
                print("[!] Processed IMU message missing gyroscope components; discarding message.")
                return
            if not all(k in accelerometer for k in ['x', 'y', 'z']):
                print("[!] Processed IMU message missing accelerometer components; discarding message.")
                return

            # Create a CSV-formatted line (EuRoC format):
            # Format: timestamp [ns], w_RS_S_x, w_RS_S_y, w_RS_S_z, a_RS_S_x, a_RS_S_y, a_RS_S_z
            line = (
                f"{imu_timestamp},"
                f"{gyroscope['x']},{gyroscope['y']},{gyroscope['z']},"
                f"{accelerometer['x']},{accelerometer['y']},{accelerometer['z']}\n"
            )

            # Append the new line to the persistent CSV file.
            self.processed_imu_csv_file.write(line)
            self.processed_imu_csv_file.flush()  # Optional: flush after each write to persist data immediately.

            processed_imu_saved_counter.inc()
        except Exception as e:
            print("[!] Exception in _save_processed_imu:", e)
        finally:
            elapsed = time.time() - start_time
            save_processed_imu_hist.observe(elapsed)

    def _handle_restart(self, ch, method, properties, body):
        try:
            msg = json.loads(body)
            if msg.get("type") == "restart":
                print("Restart command received in data_storage. Restarting recording session...")
                # Close current files if open.
                if self.trajectory_file:
                    self.trajectory_file.close()
                    self.trajectory_file = None
                if hasattr(self, "imu_csv_file") and not self.imu_csv_file.closed:
                    self.imu_csv_file.close()
                if hasattr(self, "processed_imu_csv_file") and not self.processed_imu_csv_file.closed:
                    self.processed_imu_csv_file.close()
                # Clean up video storage
                if self.video_storage:
                    self.video_storage.cleanup()
                self.start_recording()  # This creates a new timestamped folder and new files.
        except Exception as e:
            print("Error handling restart message:", e)

    def run(self):
        """Continuously consume from all relevant queues (image, imu, trajectory, ply)."""
        while True:
            try:
                # (Re-declare consumers each loop if we lose the connection.)
                self.channel.basic_consume(
                    queue='image_data_storage',
                    on_message_callback=self._save_image,
                    auto_ack=True
                )
                self.channel.basic_consume(
                    queue='imu_data_storage',
                    on_message_callback=self._save_imu,
                    auto_ack=True
                )
                self.channel.basic_consume(
                    queue='processed_imu_data_storage',
                    on_message_callback=self._save_processed_imu,
                    auto_ack=True
                )
                self.channel.basic_consume(
                    queue='trajectory_data_storage',
                    on_message_callback=self._save_trajectory,
                    auto_ack=True
                )
                self.channel.basic_consume(
                    queue='ply_data_storage',
                    on_message_callback=self._save_ply,
                    auto_ack=True
                )

                print(" [*] Waiting for messages. To exit press CTRL+C")
                self.channel.start_consuming()
            except pika.exceptions.AMQPConnectionError as err:
                print("Connection lost, reconnecting...", err)
                self._connect_with_retry()
                # Next loop iteration will re-declare consumers

if __name__ == "__main__":
    # Start the Prometheus metrics server on port 8002
    start_http_server(8002)

    ds = DataStorage()
    ds.start_recording()
    ds.run()
