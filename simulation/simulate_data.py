#!/usr/bin/env python3
"""
simulate_data_source_realtime.py

Simulate the data source by reading recorded data from the /data directory and publishing
messages to RabbitMQ exchanges in real time. The delays are computed from the recorded timestamps.
However, between recording folders a fixed 10-second gap is inserted instead of using the
recorded gap between folders.
"""

import os
import asyncio
import json
import base64
import time
from pathlib import Path

import aio_pika

# Configuration: Use the same environment variables and exchange names as your live system.
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://rabbitmq")
VIDEO_FRAMES_EXCHANGE = os.getenv("VIDEO_FRAMES_EXCHANGE", "video_frames_exchange")
IMU_DATA_EXCHANGE = os.getenv("IMU_DATA_EXCHANGE", "imu_data_exchange")
TRAJECTORY_DATA_EXCHANGE = os.getenv("TRAJECTORY_DATA_EXCHANGE", "trajectory_data_exchange")
PLY_FANOUT_EXCHANGE = os.getenv("PLY_FANOUT_EXCHANGE", "ply_fanout_exchange")

# The root of your recorded data.
DATA_ROOT = Path("/data")


async def replay_images(video_exchange, cam_data_path: Path):
    """Replay image messages in real time based on their filename timestamp (in nanoseconds)."""
    # Get sorted list of image files (filenames are expected to be <timestamp>.jpg)
    image_files = sorted(cam_data_path.glob("*.jpg"), key=lambda p: int(p.stem))
    if not image_files:
        return

    # Use the timestamp of the first image as the base.
    base_ts = int(image_files[0].stem)
    # Record the starting real time (seconds) for this folder.
    start_time = asyncio.get_event_loop().time()

    for img_file in image_files:
        try:
            ts = int(img_file.stem)
        except ValueError:
            print(f"[Images] Skipping {img_file.name}: invalid timestamp in filename")
            continue

        # Compute the delay (in seconds) relative to the base timestamp.
        delay = (ts - base_ts) / 1e9
        scheduled_time = start_time + delay
        now = asyncio.get_event_loop().time()
        if scheduled_time > now:
            await asyncio.sleep(scheduled_time - now)

        with open(img_file, "rb") as f:
            img_bytes = f.read()

        message = aio_pika.Message(
            body=img_bytes,
            content_type="application/octet-stream",
            headers={"timestamp_ns": str(ts)}
        )
        await video_exchange.publish(message, routing_key="")
        print(f"[Images] Published {img_file.name} (timestamp {ts})")


async def replay_imu(imu_exchange, imu_data_path: Path):
    """Replay IMU messages in real time based on their filename timestamp (in nanoseconds)."""
    imu_files = sorted(imu_data_path.glob("*.txt"), key=lambda p: int(p.stem))
    if not imu_files:
        return

    base_ts = int(imu_files[0].stem)
    start_time = asyncio.get_event_loop().time()

    for imu_file in imu_files:
        try:
            ts = int(imu_file.stem)
        except ValueError:
            ts = int(time.time() * 1e9)
        delay = (ts - base_ts) / 1e9
        scheduled_time = start_time + delay
        now = asyncio.get_event_loop().time()
        if scheduled_time > now:
            await asyncio.sleep(scheduled_time - now)

        try:
            with open(imu_file, "r") as f:
                imu_data = json.load(f)
        except Exception as e:
            print(f"[IMU] Failed to load {imu_file.name}: {e}")
            continue

        msg = {
            "type": "imu_data",
            "timestamp": ts,
            "imu_data": imu_data,
        }
        message = aio_pika.Message(
            body=json.dumps(msg).encode(),
            content_type="application/json"
        )
        await imu_exchange.publish(message, routing_key="")
        print(f"[IMU] Published {imu_file.name} (timestamp {ts})")


async def replay_trajectory(trajectory_exchange, trajectory_file: Path):
    """
    Replay trajectory records in real time.
    Each line in trajectory.txt is expected to be a JSON record with a "timestamp_ns" key.
    """
    if not trajectory_file.is_file():
        return

    # Read and parse each line.
    lines = trajectory_file.read_text().splitlines()
    records = []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            records.append(record)
        except Exception as e:
            print(f"[Trajectory] Failed to parse a line: {e}")
    if not records:
        return

    # Sort records by their recorded timestamp.
    records.sort(key=lambda r: r.get("timestamp_ns", 0))
    base_ts = records[0].get("timestamp_ns", 0)
    start_time = asyncio.get_event_loop().time()

    for record in records:
        ts = record.get("timestamp_ns", base_ts)
        delay = (ts - base_ts) / 1e9
        scheduled_time = start_time + delay
        now = asyncio.get_event_loop().time()
        if scheduled_time > now:
            await asyncio.sleep(scheduled_time - now)

        message = aio_pika.Message(
            body=json.dumps(record).encode(),
            content_type="application/json"
        )
        await trajectory_exchange.publish(message, routing_key="")
        print(f"[Trajectory] Published record with timestamp {ts}")


async def replay_ply(ply_exchange, ply_folder: Path):
    """
    Replay PLY files.
    If the filenames are numeric (indicating a timestamp), they will be replayed in real time.
    Otherwise, a fixed delay is used between files.
    """
    ply_files = sorted(ply_folder.glob("*.ply"), key=lambda p: p.stem)
    if not ply_files:
        return

    # Check if all filenames are numeric.
    if all(f.stem.isdigit() for f in ply_files):
        base_ts = int(ply_files[0].stem)
        start_time = asyncio.get_event_loop().time()
        for ply_file in ply_files:
            ts = int(ply_file.stem)
            delay = (ts - base_ts) / 1e9
            scheduled_time = start_time + delay
            now = asyncio.get_event_loop().time()
            if scheduled_time > now:
                await asyncio.sleep(scheduled_time - now)
            with open(ply_file, "rb") as f:
                ply_data = f.read()
            ply_data_b64 = base64.b64encode(ply_data).decode("utf-8")
            msg = {
                "ply_filename": ply_file.name,
                "ply_data_b64": ply_data_b64
            }
            message = aio_pika.Message(
                body=json.dumps(msg).encode(),
                content_type="application/json"
            )
            await ply_exchange.publish(message, routing_key="")
            print(f"[PLY] Published {ply_file.name} (timestamp {ts})")
    else:
        # Fallback: Use a fixed delay between PLY messages.
        for ply_file in ply_files:
            with open(ply_file, "rb") as f:
                ply_data = f.read()
            ply_data_b64 = base64.b64encode(ply_data).decode("utf-8")
            msg = {
                "ply_filename": ply_file.name,
                "ply_data_b64": ply_data_b64
            }
            message = aio_pika.Message(
                body=json.dumps(msg).encode(),
                content_type="application/json"
            )
            await ply_exchange.publish(message, routing_key="")
            print(f"[PLY] Published {ply_file.name} with fixed delay")
            await asyncio.sleep(1)  # Fixed delay of 1 second


async def publish_messages():
    connection = await aio_pika.connect_robust(RABBITMQ_URL, heartbeat=3600)
    channel = await connection.channel()

    # Declare the exchanges (fanout and durable).
    video_exchange = await channel.declare_exchange(
        VIDEO_FRAMES_EXCHANGE, aio_pika.ExchangeType.FANOUT, durable=True
    )
    imu_exchange = await channel.declare_exchange(
        IMU_DATA_EXCHANGE, aio_pika.ExchangeType.FANOUT, durable=True
    )
    trajectory_exchange = await channel.declare_exchange(
        TRAJECTORY_DATA_EXCHANGE, aio_pika.ExchangeType.FANOUT, durable=True
    )
    ply_exchange = await channel.declare_exchange(
        PLY_FANOUT_EXCHANGE, aio_pika.ExchangeType.FANOUT, durable=True
    )

    # Build a sorted list of recording folders.
    folders = sorted([f for f in DATA_ROOT.iterdir() if f.is_dir()])
    total_folders = len(folders)
    for i, recording_folder in enumerate(folders):
        mav0_path = recording_folder / "mav0"
        if not mav0_path.exists():
            continue

        print(f"\nProcessing recording: {recording_folder.name}")

        # Create a list of tasks for each data stream in the current recording folder.
        tasks = []

        # Images
        cam_data_path = mav0_path / "cam0" / "data"
        if cam_data_path.is_dir():
            tasks.append(asyncio.create_task(replay_images(video_exchange, cam_data_path)))

        # IMU
        imu_data_path = mav0_path / "imu0" / "data"
        if imu_data_path.is_dir():
            tasks.append(asyncio.create_task(replay_imu(imu_exchange, imu_data_path)))

        # Trajectory
        trajectory_file = mav0_path / "trajectory.txt"
        if trajectory_file.is_file():
            tasks.append(asyncio.create_task(replay_trajectory(trajectory_exchange, trajectory_file)))

        # PLY
        ply_folder = mav0_path / "ply"
        if ply_folder.is_dir():
            tasks.append(asyncio.create_task(replay_ply(ply_exchange, ply_folder)))

        if tasks:
            # Run all streams concurrently for this recording folder.
            await asyncio.gather(*tasks)

        # Insert a fixed 10-second gap before starting the next recording folder,
        # unless this is the last folder.
        if i < total_folders - 1:
            print("Waiting 10 seconds before next recording folder...")
            await asyncio.sleep(10)

    await connection.close()


if __name__ == "__main__":
    print("Waiting for 10 seconds for services to start...")
    time.sleep(10)
    print("Starting simulation...")
    asyncio.run(publish_messages())
