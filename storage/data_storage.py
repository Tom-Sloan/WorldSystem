import time
import pika
import os
import json
import cv2
import base64
import numpy as np
from pathlib import Path
from datetime import datetime
from worldsystem_common import EXCHANGES, ROUTING_KEYS, declare_exchanges_sync

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
        # Declare the exchanges using shared function:
        # ------------------------------------------------------
        declare_exchanges_sync(self.channel)

        # ------------------------------------------------------
        # Create queues & bind them to the relevant exchange
        # ------------------------------------------------------
        # images -> image_data_storage
        self.channel.queue_declare(queue='image_data_storage', durable=True)
        self.channel.queue_bind(
            exchange='sensor_data',
            queue='image_data_storage',
            routing_key=ROUTING_KEYS['VIDEO_FRAMES']
        )

        # IMU data no longer needed - skip binding

        # trajectory -> trajectory_data_storage
        self.channel.queue_declare(queue='trajectory_data_storage', durable=True)
        self.channel.queue_bind(
            exchange='assets',
            queue='trajectory_data_storage',
            routing_key=ROUTING_KEYS['TRAJECTORY']
        )

        # .ply -> ply_data_storage
        self.channel.queue_declare(queue='ply_data_storage', durable=True)
        self.channel.queue_bind(
            exchange='assets',
            queue='ply_data_storage',
            routing_key=ROUTING_KEYS['PLY_FILE']
        )

        # restart -> restart_data_storage
        self.channel.queue_declare(queue='restart_data_storage', durable=True)
        self.channel.queue_bind(
            exchange='control_commands',
            queue='restart_data_storage',
            routing_key=ROUTING_KEYS['RESTART']
        )
        self.channel.basic_consume(
            queue='restart_data_storage',
            on_message_callback=self._handle_restart,
            auto_ack=True
        )

    def start_recording(self):
        """Create the /data/<timestamp>/mav0 folder and open files for trajectory and IMU data."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_path = Path('/data') / timestamp / "mav0"
        
        # Create directories for cameras and ply data.
        (self.recording_path / "cam0" / "data").mkdir(parents=True, exist_ok=True)
        (self.recording_path / "cam1" / "data").mkdir(parents=True, exist_ok=True)  # New PNG folder
        (self.recording_path / "ply").mkdir(parents=True, exist_ok=True)
        
        # Open trajectory file (unchanged).
        trajectory_file_path = self.recording_path / "trajectory.txt"
        self.trajectory_file = open(trajectory_file_path, 'a')
        print(f"Trajectory will be saved to: {trajectory_file_path}")
        
        # IMU file creation removed - no longer needed

    def _save_image(self, ch, method, properties, body):
        start_time = time.time()
        try:
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

    # IMU save methods removed - no longer needed

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


    def _handle_restart(self, ch, method, properties, body):
        try:
            msg = json.loads(body)
            if msg.get("type") == "restart":
                print("Restart command received in data_storage. Restarting recording session...")
                # Close current files if open.
                if self.trajectory_file:
                    self.trajectory_file.close()
                    self.trajectory_file = None
                # IMU file cleanup removed - no longer needed
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
                # IMU consumers removed - no longer needed
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
