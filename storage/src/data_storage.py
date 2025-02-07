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
        self.connection = pika.BlockingConnection(
            pika.URLParameters(os.getenv('RABBITMQ_URL')))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='image_data')
        self.channel.queue_declare(queue='imu_data')
        self.recording_path = None

    def start_recording(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_path = Path('/data') / timestamp / "mav0"
        (self.recording_path / "cam0/data").mkdir(parents=True, exist_ok=True)
        (self.recording_path / "imu0/data").mkdir(parents=True, exist_ok=True)

    def _save_image(self, ch, method, properties, body):
        data = json.loads(body)
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(data['frame_data']), np.uint8), cv2.IMREAD_COLOR)
        frame_path = self.recording_path / "cam0/data" / f"{data['timestamp']}.jpg"
        cv2.imwrite(str(frame_path), frame)

    def _save_imu(self, ch, method, properties, body):
        data = json.loads(body)
        imu_path = self.recording_path / "imu0/data" / f"{data['timestamp']}.txt"
        with open(imu_path, 'w') as f:
            f.write(f"{data['angular_velocity']} {data['linear_acceleration']}")

    def run(self):
        self.channel.basic_consume(queue='image_data', on_message_callback=self._save_image, auto_ack=True)
        self.channel.basic_consume(queue='imu_data', on_message_callback=self._save_imu, auto_ack=True)
        self.channel.start_consuming()

if __name__ == "__main__":
    ds = DataStorage()
    ds.start_recording()
    ds.run() 