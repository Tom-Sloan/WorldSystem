import time
import pika
import os
import json
import cv2
import base64
import numpy as np
from pathlib import Path
from datetime import datetime

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
        # Declare the fanout exchanges from FrameProcessor or SLAM:
        # ------------------------------------------------------
        self.channel.exchange_declare(exchange='image_data_exchange', exchange_type='fanout', durable=True)
        self.channel.exchange_declare(exchange='imu_data_exchange', exchange_type='fanout', durable=True)
        self.channel.exchange_declare(exchange='trajectory_data_exchange', exchange_type='fanout', durable=True)
        self.channel.exchange_declare(exchange='ply_fanout_exchange', exchange_type='fanout', durable=True)

        # ------------------------------------------------------
        # Create queues & bind them to the relevant exchange
        # ------------------------------------------------------
        # images -> image_data_storage
        self.channel.queue_declare(queue='image_data_storage', durable=True)
        self.channel.queue_bind(
            exchange='image_data_exchange',
            queue='image_data_storage',
            routing_key=''
        )

        # IMU -> imu_data_storage
        self.channel.queue_declare(queue='imu_data_storage', durable=True)
        self.channel.queue_bind(
            exchange='imu_data_exchange',
            queue='imu_data_storage',
            routing_key=''
        )

        # trajectory -> trajectory_data_storage
        self.channel.queue_declare(queue='trajectory_data_storage', durable=True)
        self.channel.queue_bind(
            exchange='trajectory_data_exchange',
            queue='trajectory_data_storage',
            routing_key=''
        )

        # .ply -> ply_data_storage
        self.channel.queue_declare(queue='ply_data_storage', durable=True)
        self.channel.queue_bind(
            exchange='ply_fanout_exchange',
            queue='ply_data_storage',
            routing_key=''
        )

    def start_recording(self):
        """Create the /data/<timestamp>/mav0 folder & open the trajectory file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_path = Path('/data') / timestamp / "mav0"
        (self.recording_path / "cam0" / "data").mkdir(parents=True, exist_ok=True)
        (self.recording_path / "imu0" / "data").mkdir(parents=True, exist_ok=True)
        (self.recording_path / "ply").mkdir(parents=True, exist_ok=True)

        # Open a single file for storing trajectory data in text/CSV-like format
        trajectory_file_path = self.recording_path / "trajectory.txt"
        self.trajectory_file = open(trajectory_file_path, 'a')
        print(f"Trajectory will be saved to: {trajectory_file_path}")

    def _save_image(self, ch, method, properties, body):
        """Callback for storing images from 'image_data_storage' queue."""
        start_time = time.time()
        try:
            data = json.loads(body)
            frame_data = data['frame_data']
            frame_timestamp = data['timestamp']
            np_data = np.frombuffer(base64.b64decode(frame_data), np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
            if frame is None:
                print(f"[!] Received invalid image data at timestamp {frame_timestamp}")
                return

            frame_path = self.recording_path / "cam0" / "data" / f"{frame_timestamp}.jpg"
            cv2.imwrite(str(frame_path), frame)

            # Increment counter if successfully saved
            images_saved_counter.inc()
        finally:
            elapsed = time.time() - start_time
            save_image_hist.observe(elapsed)

    def _save_imu(self, ch, method, properties, body):
        """Callback for storing IMU data from 'imu_data_storage' queue."""
        start_time = time.time()
        try:
            data = json.loads(body)
            imu_timestamp = data['timestamp']
            angular_vel = data['angular_velocity']
            linear_acc = data['linear_acceleration']

            imu_path = self.recording_path / "imu0" / "data" / f"{imu_timestamp}.txt"
            with open(imu_path, 'w') as f:
                f.write(f"angular_velocity: {angular_vel}\n")
                f.write(f"linear_acceleration: {linear_acc}\n")

            # Increment counter if successfully saved
            imu_saved_counter.inc()
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
            ts_ns = data.get("timestamp_ns")
            pose = data.get("pose")
            record = {
                "timestamp_ns": ts_ns,
                "pose": pose
            }
            line = json.dumps(record)
            self.trajectory_file.write(line + "\n")
            self.trajectory_file.flush()

            # Increment counter if successfully saved
            trajectories_saved_counter.inc()
        finally:
            elapsed = time.time() - start_time
            save_trajectory_hist.observe(elapsed)

    def _save_ply(self, ch, method, properties, body):
        """Callback for storing .ply files from 'ply_data_storage' queue."""
        start_time = time.time()
        try:
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

            print(f"Saved .ply file to: {out_path}")

            # Increment counter if successfully saved
            ply_saved_counter.inc()
        finally:
            elapsed = time.time() - start_time
            save_ply_hist.observe(elapsed)

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
