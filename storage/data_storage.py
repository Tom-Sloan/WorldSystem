import time
import pika
import os
import json
import cv2
import base64
import numpy as np
from pathlib import Path
from datetime import datetime

class DataStorage:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.recording_path = None
        self.trajectory_file = None  # We'll open this once we start recording
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
        # 1) image_data_exchange
        self.channel.exchange_declare(exchange='image_data_exchange', exchange_type='fanout', durable=True)
        # 2) imu_data_exchange
        self.channel.exchange_declare(exchange='imu_data_exchange', exchange_type='fanout', durable=True)
        # 3) trajectory_data_exchange
        self.channel.exchange_declare(exchange='trajectory_data_exchange', exchange_type='fanout', durable=True)
        # 4) ply_fanout_exchange - for .ply files from continuous_reconstruction.py
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

        # (NEW) .ply -> ply_data_storage
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
        data = json.loads(body)
        frame_data = data['frame_data']
        frame_timestamp = data['timestamp']
        # Decode base64 -> OpenCV image
        np_data = np.frombuffer(base64.b64decode(frame_data), np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[!] Received invalid image data at timestamp {frame_timestamp}")
            return

        frame_path = self.recording_path / "cam0" / "data" / f"{frame_timestamp}.jpg"
        cv2.imwrite(str(frame_path), frame)

    def _save_imu(self, ch, method, properties, body):
        """Callback for storing IMU data from 'imu_data_storage' queue."""
        data = json.loads(body)
        imu_timestamp = data['timestamp']
        angular_vel = data['angular_velocity']
        linear_acc = data['linear_acceleration']

        imu_path = self.recording_path / "imu0" / "data" / f"{imu_timestamp}.txt"
        with open(imu_path, 'w') as f:
            f.write(f"angular_velocity: {angular_vel}\n")
            f.write(f"linear_acceleration: {linear_acc}\n")

    def _save_trajectory(self, ch, method, properties, body):
        """
        Callback for storing trajectory data from 'trajectory_data_storage' queue.
        We'll write them all to a single file 'trajectory.txt'.
        """
        if not self.trajectory_file:
            print("[!] Trajectory file not open! Did you call start_recording() first?")
            return

        try:
            data = json.loads(body)
            # data might look like: {"timestamp_ns": 1234567890, "pose": [[...],[...]]}
            ts_ns = data.get("timestamp_ns")
            pose = data.get("pose")

            # We can store this as JSON or a simple line
            record = {
                "timestamp_ns": ts_ns,
                "pose": pose
            }
            line = json.dumps(record)
            self.trajectory_file.write(line + "\n")
            self.trajectory_file.flush()

        except Exception as e:
            print(f"[!] Error saving trajectory: {e}")

    def _save_ply(self, ch, method, properties, body):
        """
        Callback for storing .ply files from 'ply_data_storage' queue.
        The message format is typically:
          {
            "ply_filename": "<filename>.ply",
            "ply_data_b64": "<base64-encoded .ply file>"
          }
        """
        try:
            msg = json.loads(body)
            ply_filename = msg.get("ply_filename", "output.ply")
            ply_data_b64 = msg.get("ply_data_b64")
            if not ply_data_b64:
                print("[!] No base64 data in ply message.")
                return

            # Decode the .ply data from base64
            ply_data = base64.b64decode(ply_data_b64)

            # Save into the "ply" folder of our recording path
            # We'll just use the provided name or you can use a timestamp
            out_path = self.recording_path / "ply" / ply_filename
            with open(out_path, "wb") as f:
                f.write(ply_data)

            print(f"Saved .ply file to: {out_path}")

        except Exception as e:
            print(f"[!] Error saving .ply file: {e}")

    def run(self):
        """Continuously consume from all relevant queues (image, imu, trajectory, ply) in a loop."""
        while True:
            try:
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

                print(" [*] Waiting for messages (images, IMU, trajectory, ply). To exit press CTRL+C")
                self.channel.start_consuming()

            except pika.exceptions.AMQPConnectionError as err:
                print("Connection lost, reconnecting...", err)
                self._connect_with_retry()
                # We'll loop and re-declare the consumers in the next iteration


if __name__ == "__main__":
    ds = DataStorage()
    ds.start_recording()
    ds.run()
