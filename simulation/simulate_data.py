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
import sys
from pathlib import Path
import numpy as np
import cv2
import ntplib
import socket
import threading

import aio_pika
from worldsystem_common import EXCHANGES, ROUTING_KEYS, declare_exchanges_sync, declare_exchanges

# Configuration: Use the same environment variables and exchange names as your live system.
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://rabbitmq")

# The root of your recorded data.
# Check if USE_FOLDER is set to use a specific folder, otherwise use /data or /simulation_data
USE_FOLDER = os.getenv("USE_FOLDER", "")
if USE_FOLDER and os.path.exists("/simulation_data"):
    DATA_ROOT = Path("/simulation_data")
    print(f"[INFO] Using simulation data from: {DATA_ROOT}")
else:
    DATA_ROOT = Path("/data")
    print(f"[INFO] Using default data path: {DATA_ROOT}")

# Add NTP client and time offset tracking
ntp_client = ntplib.NTPClient()
ntp_time_offset = 0.0  # Offset between system time and NTP time in seconds
last_ntp_sync = 0
NTP_SYNC_INTERVAL = 60  # Sync NTP every 60 seconds
NTP_SERVER = os.getenv("NTP_SERVER", "pool.ntp.org")

def sync_ntp_time():
    """Update the NTP time offset by querying an NTP server."""
    global ntp_time_offset, last_ntp_sync
    
    try:
        response = ntp_client.request(NTP_SERVER, timeout=5)
        # Calculate offset: NTP time - local time
        ntp_time_offset = response.offset
        last_ntp_sync = time.time()
        print(f"[NTP] Synchronized time, offset: {ntp_time_offset:.6f} seconds")
        sys.stdout.flush()
        return True
    except (ntplib.NTPException, socket.gaierror, socket.timeout) as e:
        print(f"[NTP] Synchronization failed: {str(e)}")
        sys.stdout.flush()
        return False

def get_ntp_time_ns():
    """Get current time in nanoseconds, synchronized with NTP."""
    current_time = time.time() + ntp_time_offset
    # Check if we need to resync
    if time.time() - last_ntp_sync > NTP_SYNC_INTERVAL:
        # Start a thread to sync NTP time without blocking
        threading.Thread(target=sync_ntp_time, daemon=True).start()
    return int(current_time * 1e9)  # Convert to nanoseconds

# Initialize NTP synchronization
print("[NTP] Initializing time synchronization...")
sys.stdout.flush()
sync_ntp_time()

# Debug output
print(f"[DEBUG] USE_FOLDER environment variable: {USE_FOLDER}")
print(f"[DEBUG] DATA_ROOT path: {DATA_ROOT}")
sys.stdout.flush()

async def replay_single_video(video_exchange, video_file: Path):
    """Replay a single video file by extracting frames and publishing them in real time."""
    # Define common resolutions (width, height)
    RESOLUTIONS = {
        "4K": (3840, 2160),    # 4K UHD
        "2K": (2560, 1440),    # 2K QHD
        "1080p": (1920, 1080), # Full HD
        "720p": (1280, 720),   # HD
        "480p": (854, 480),    # SD
        "360p": (640, 360),    # LD
        "240p": (426, 240)     # Very Low
    }
    
    current_resolution = "480p"  # Default resolution
    
    print(f"[SingleVideo] Processing {video_file.name}")
    sys.stdout.flush()
    
    # Open video file
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        print(f"[SingleVideo] Failed to open video file: {video_file.name}")
        sys.stdout.flush()
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[SingleVideo] Video info: {frame_count} frames at {fps} FPS")
    sys.stdout.flush()
    
    start_time = asyncio.get_event_loop().time()
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate timestamp for this frame
        frame_time_offset = frame_idx / fps
        frame_timestamp_ns = int(frame_time_offset * 1e9)
        
        # Calculate when to publish this frame
        scheduled_time = start_time + frame_time_offset
        now = asyncio.get_event_loop().time()
        if scheduled_time > now:
            await asyncio.sleep(scheduled_time - now)
        
        # Get original dimensions
        original_height, original_width = frame.shape[:2]
        
        # Get target resolution
        new_width, new_height = RESOLUTIONS[current_resolution]
        
        # Resize the frame if needed
        if (new_width, new_height) != (original_width, original_height):
            frame = cv2.resize(frame, (new_width, new_height), 
                             interpolation=cv2.INTER_AREA if new_width < original_width else cv2.INTER_LINEAR)
        
        # Encode frame as JPEG
        _, encoded_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        frame_bytes = encoded_frame.tobytes()
        
        # Get current time for server timestamp using NTP-synchronized time
        ntp_time_ns = str(get_ntp_time_ns())
        
        # Create message with metadata
        message = aio_pika.Message(
            body=frame_bytes,
            content_type="application/octet-stream",
            headers={
                "timestamp_ns": str(frame_timestamp_ns),
                "server_received": ntp_time_ns,
                "ntp_time": ntp_time_ns,
                "width": new_width,
                "height": new_height,
                "resolution": current_resolution,
                "original_width": original_width,
                "original_height": original_height,
                "source_format": "MP4_SINGLE",
                "output_format": "JPEG",
                "ntp_offset": ntp_time_offset,
                "video_name": video_file.name,  # Include video name
                "frame_index": frame_idx
            }
        )
        await exchanges['sensor_data'].publish(message, routing_key=ROUTING_KEYS['VIDEO_FRAMES'])
        
        frame_idx += 1
        
        if frame_idx % 30 == 0:  # Log every 30 frames
            print(f"[SingleVideo] Published frame {frame_idx}/{frame_count} from {video_file.name}")
            sys.stdout.flush()
    
    cap.release()
    print(f"[SingleVideo] Completed {video_file.name} - published {frame_idx} frames")
    sys.stdout.flush()


async def replay_video_segments(exchanges, video_path: Path):
    """Replay video segments by extracting frames and publishing them in real time."""
    # Define common resolutions (width, height)
    RESOLUTIONS = {
        "4K": (3840, 2160),    # 4K UHD
        "2K": (2560, 1440),    # 2K QHD
        "1080p": (1920, 1080), # Full HD
        "720p": (1280, 720),   # HD
        "480p": (854, 480),    # SD
        "360p": (640, 360),    # LD
        "240p": (426, 240)     # Very Low
    }
    
    current_resolution = "480p"  # Default resolution
    
    # Check if the directory exists
    if not video_path.exists() or not video_path.is_dir():
        print(f"[Videos] Directory does not exist or is not a directory: {video_path}")
        sys.stdout.flush()
        return
    
    # Get sorted list of video files
    video_files = sorted(video_path.glob("*_segment_*.mp4"))
    if not video_files:
        print(f"[Videos] No video segment files found in {video_path}")
        sys.stdout.flush()
        return
    
    print(f"[Videos] Found {len(video_files)} video segments in {video_path}")
    sys.stdout.flush()
    
    start_time = asyncio.get_event_loop().time()
    total_frames = 0
    
    for video_file in video_files:
        # Extract timestamp from filename (e.g., "0000_segment_000.mp4" -> 0 seconds)
        try:
            timestamp_seconds = int(video_file.name.split('_')[0])
        except (ValueError, IndexError):
            print(f"[Videos] Skipping {video_file.name}: invalid filename format")
            sys.stdout.flush()
            continue
        
        # Open video file
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"[Videos] Failed to open video file: {video_file.name}")
            sys.stdout.flush()
            continue
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[Videos] Processing {video_file.name} - {frame_count} frames at {fps} FPS")
        sys.stdout.flush()
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate timestamp for this frame
            frame_time_offset = frame_idx / fps
            frame_timestamp_ns = int((timestamp_seconds + frame_time_offset) * 1e9)
            
            # Calculate when to publish this frame
            scheduled_time = start_time + timestamp_seconds + frame_time_offset
            now = asyncio.get_event_loop().time()
            if scheduled_time > now:
                await asyncio.sleep(scheduled_time - now)
            
            # Get original dimensions
            original_height, original_width = frame.shape[:2]
            
            # Get target resolution
            new_width, new_height = RESOLUTIONS[current_resolution]
            
            # Resize the frame if needed
            if (new_width, new_height) != (original_width, original_height):
                frame = cv2.resize(frame, (new_width, new_height), 
                                 interpolation=cv2.INTER_AREA if new_width < original_width else cv2.INTER_LINEAR)
            
            # Encode frame as JPEG
            _, encoded_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            frame_bytes = encoded_frame.tobytes()
            
            # Get current time for server timestamp using NTP-synchronized time
            ntp_time_ns = str(get_ntp_time_ns())
            
            # Create message with metadata
            message = aio_pika.Message(
                body=frame_bytes,
                content_type="application/octet-stream",
                headers={
                    "timestamp_ns": str(frame_timestamp_ns),
                    "server_received": ntp_time_ns,
                    "ntp_time": ntp_time_ns,
                    "width": new_width,
                    "height": new_height,
                    "resolution": current_resolution,
                    "original_width": original_width,
                    "original_height": original_height,
                    "source_format": "MP4_FRAME",
                    "output_format": "JPEG",
                    "ntp_offset": ntp_time_offset,
                    "video_segment": video_file.name,
                    "frame_index": frame_idx
                }
            )
            await exchanges['sensor_data'].publish(message, routing_key=ROUTING_KEYS['VIDEO_FRAMES'])
            
            frame_idx += 1
            total_frames += 1
            
            if frame_idx % 30 == 0:  # Log every 30 frames
                print(f"[Videos] Published frame {frame_idx} from {video_file.name}")
                sys.stdout.flush()
        
        cap.release()
        print(f"[Videos] Completed {video_file.name} - published {frame_idx} frames")
        sys.stdout.flush()
    
    print(f"[Videos] Finished publishing {total_frames} total frames from video segments")
    sys.stdout.flush()


async def replay_images(exchanges, cam_data_path: Path):
    """Replay image messages in real time based on their filename timestamp (in nanoseconds)."""
    # Define common resolutions (width, height)
    RESOLUTIONS = {
        "4K": (3840, 2160),    # 4K UHD
        "2K": (2560, 1440),    # 2K QHD
        "1080p": (1920, 1080), # Full HD
        "720p": (1280, 720),   # HD
        "480p": (854, 480),    # SD
        "360p": (640, 360),    # LD
        "240p": (426, 240)     # Very Low
    }
    
    current_resolution = "480p"  # Start with HD
    
    # Check if the directory exists and is accessible
    if not cam_data_path.exists():
        print(f"[Images] Directory does not exist: {cam_data_path}")
        sys.stdout.flush()
        return
    
    if not cam_data_path.is_dir():
        print(f"[Images] Path is not a directory: {cam_data_path}")
        sys.stdout.flush()
        return
    
    # Get sorted list of image files - check for both JPG and PNG
    try:
        jpg_files = list(cam_data_path.glob("*.jpg"))
        png_files = list(cam_data_path.glob("*.png"))
    except Exception as e:
        print(f"[Images] Error accessing directory {cam_data_path}: {e}")
        sys.stdout.flush()
        return
    
    # Determine which format to use for this folder
    if jpg_files and png_files:
        # If both exist, prefer the format with more files
        if len(jpg_files) >= len(png_files):
            image_files = sorted(jpg_files, key=lambda p: int(p.stem))
            image_format = "JPEG"
        else:
            image_files = sorted(png_files, key=lambda p: int(p.stem))
            image_format = "PNG"
    elif jpg_files:
        image_files = sorted(jpg_files, key=lambda p: int(p.stem))
        image_format = "JPEG"
    elif png_files:
        image_files = sorted(png_files, key=lambda p: int(p.stem))
        image_format = "PNG"
    else:
        print(f"[Images] No image files found in {cam_data_path}")
        sys.stdout.flush()
        return
    
    print(f"[Images] Found {image_format} format for folder {cam_data_path.name} ({len(image_files)} files) - will convert to JPEG for RabbitMQ")
    sys.stdout.flush()
    
    # Debug: Check if all discovered files actually exist
    missing_files = [f for f in image_files if not f.exists()]
    if missing_files:
        print(f"[Images] Warning: {len(missing_files)} files from glob don't exist: {[f.name for f in missing_files[:5]]}")
        sys.stdout.flush()
        # Remove missing files from the list
        image_files = [f for f in image_files if f.exists()]
        print(f"[Images] After filtering, {len(image_files)} files remain")
        sys.stdout.flush()

    # Check if we still have files to process
    if not image_files:
        print(f"[Images] No valid image files found in {cam_data_path}")
        sys.stdout.flush()
        return
    
    # Use the timestamp of the first image as the base
    base_ts = int(image_files[0].stem)
    start_time = asyncio.get_event_loop().time()

    for img_file in image_files:
        try:
            ts = int(img_file.stem)
        except ValueError:
            print(f"[Images] Skipping {img_file.name}: invalid timestamp in filename")
            sys.stdout.flush()
            continue

        # Compute the delay
        delay = (ts - base_ts) / 1e9
        scheduled_time = start_time + delay
        now = asyncio.get_event_loop().time()
        if scheduled_time > now:
            await asyncio.sleep(scheduled_time - now)

        # Check if file still exists before trying to read it
        if not img_file.exists():
            print(f"[Images] Skipping {img_file.name}: file no longer exists")
            sys.stdout.flush()
            continue
            
        # Read and decode the image
        try:
            with open(img_file, "rb") as f:
                img_bytes = f.read()
        except FileNotFoundError:
            print(f"[Images] Skipping {img_file.name}: file not found during read")
            sys.stdout.flush()
            continue
        except Exception as e:
            print(f"[Images] Error reading {img_file.name}: {e}")
            sys.stdout.flush()
            continue
        
        # Convert to numpy array for processing
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            continue
            
        # Get original dimensions
        original_height, original_width = img.shape[:2]
        
        # Get target resolution
        new_width, new_height = RESOLUTIONS[current_resolution]
        
        # Resize the image
        if (new_width, new_height) != (original_width, original_height):
            img = cv2.resize(img, (new_width, new_height), 
                           interpolation=cv2.INTER_AREA if new_width < original_width else cv2.INTER_LINEAR)
            
        # Encode the processed image as JPEG (convert PNG to JPEG)
        _, encoded_img = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        processed_img_bytes = encoded_img.tobytes()
        
        # Get current time for server timestamp using NTP-synchronized time
        ntp_time_ns = str(get_ntp_time_ns())
        timestamp_ns = str(ts)

        # Create message with metadata
        message = aio_pika.Message(
            body=processed_img_bytes,
            content_type="application/octet-stream",
            headers={
                "timestamp_ns": timestamp_ns,
                "server_received": ntp_time_ns,
                "ntp_time": ntp_time_ns,  # NTP-synchronized time
                "width": new_width,
                "height": new_height,
                "resolution": current_resolution,
                "original_width": original_width,
                "original_height": original_height,
                "source_format": image_format,  # Original format (JPEG or PNG)
                "output_format": "JPEG",  # Always JPEG for RabbitMQ
                "ntp_offset": ntp_time_offset  # Include the NTP offset for debugging
            }
        )
        await exchanges['sensor_data'].publish(message, routing_key=ROUTING_KEYS['VIDEO_FRAMES'])
        conversion_note = f" (converted {image_format}→JPEG)" if image_format == "PNG" else ""
        print(f"[Images] Published {img_file.name} (timestamp {ts}, NTP time {ntp_time_ns}, resolution {current_resolution}{conversion_note})")
        sys.stdout.flush()


async def replay_imu(imu_exchange, imu_data_path: Path):
    """Replay IMU messages in real time based on their filename timestamp (in nanoseconds)."""
    # Check if the directory exists
    if not imu_data_path.exists():
        print(f"[IMU] Directory does not exist: {imu_data_path}")
        sys.stdout.flush()
        return
    
    if not imu_data_path.is_dir():
        print(f"[IMU] Path is not a directory: {imu_data_path}")
        sys.stdout.flush()
        return
    
    try:
        imu_files = sorted(imu_data_path.glob("*.txt"), key=lambda p: int(p.stem))
    except Exception as e:
        print(f"[IMU] Error accessing directory {imu_data_path}: {e}")
        sys.stdout.flush()
        return
        
    if not imu_files:
        print(f"[IMU] No IMU files found in {imu_data_path}")
        sys.stdout.flush()
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

        # Check if file still exists
        if not imu_file.exists():
            print(f"[IMU] Skipping {imu_file.name}: file no longer exists")
            sys.stdout.flush()
            continue
            
        try:
            with open(imu_file, "r") as f:
                imu_data = json.load(f)
        except FileNotFoundError:
            print(f"[IMU] Skipping {imu_file.name}: file not found during read")
            sys.stdout.flush()
            continue
        except Exception as e:
            print(f"[IMU] Failed to load {imu_file.name}: {e}")
            sys.stdout.flush()
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
        # IMU data no longer needed - skip publishing
        print(f"[IMU] Published {imu_file.name} (timestamp {ts})")
        sys.stdout.flush()


async def replay_trajectory(exchanges, trajectory_file: Path):
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
        await exchanges['assets'].publish(message, routing_key=ROUTING_KEYS['TRAJECTORY'])
        print(f"[Trajectory] Published record with timestamp {ts}")
        sys.stdout.flush()


async def replay_ply(exchanges, ply_folder: Path):
    """
    Replay PLY files.
    If the filenames are numeric (indicating a timestamp), they will be replayed in real time.
    Otherwise, a fixed delay is used between files.
    """
    # Check if the directory exists
    if not ply_folder.exists():
        print(f"[PLY] Directory does not exist: {ply_folder}")
        sys.stdout.flush()
        return
    
    if not ply_folder.is_dir():
        print(f"[PLY] Path is not a directory: {ply_folder}")
        sys.stdout.flush()
        return
    
    try:
        ply_files = sorted(ply_folder.glob("*.ply"), key=lambda p: p.stem)
    except Exception as e:
        print(f"[PLY] Error accessing directory {ply_folder}: {e}")
        sys.stdout.flush()
        return
        
    if not ply_files:
        print(f"[PLY] No PLY files found in {ply_folder}")
        sys.stdout.flush()
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
            
            # Check if file still exists
            if not ply_file.exists():
                print(f"[PLY] Skipping {ply_file.name}: file no longer exists")
                sys.stdout.flush()
                continue
                
            try:
                with open(ply_file, "rb") as f:
                    ply_data = f.read()
            except FileNotFoundError:
                print(f"[PLY] Skipping {ply_file.name}: file not found during read")
                sys.stdout.flush()
                continue
            except Exception as e:
                print(f"[PLY] Error reading {ply_file.name}: {e}")
                sys.stdout.flush()
                continue
            ply_data_b64 = base64.b64encode(ply_data).decode("utf-8")
            msg = {
                "ply_filename": ply_file.name,
                "ply_data_b64": ply_data_b64
            }
            message = aio_pika.Message(
                body=json.dumps(msg).encode(),
                content_type="application/json"
            )
            await exchanges['assets'].publish(message, routing_key=ROUTING_KEYS['PLY_FILE'])
            print(f"[PLY] Published {ply_file.name} (timestamp {ts})")
            sys.stdout.flush()
    else:
        # Fallback: Use a fixed delay between PLY messages.
        for ply_file in ply_files:
            # Check if file still exists
            if not ply_file.exists():
                print(f"[PLY] Skipping {ply_file.name}: file no longer exists")
                sys.stdout.flush()
                continue
                
            try:
                with open(ply_file, "rb") as f:
                    ply_data = f.read()
            except FileNotFoundError:
                print(f"[PLY] Skipping {ply_file.name}: file not found during read")
                sys.stdout.flush()
                continue
            except Exception as e:
                print(f"[PLY] Error reading {ply_file.name}: {e}")
                sys.stdout.flush()
                continue
            ply_data_b64 = base64.b64encode(ply_data).decode("utf-8")
            msg = {
                "ply_filename": ply_file.name,
                "ply_data_b64": ply_data_b64
            }
            message = aio_pika.Message(
                body=json.dumps(msg).encode(),
                content_type="application/json"
            )
            await exchanges['assets'].publish(message, routing_key=ROUTING_KEYS['PLY_FILE'])
            print(f"[PLY] Published {ply_file.name} with fixed delay")
            sys.stdout.flush()
            await asyncio.sleep(1)  # Fixed delay of 1 second


async def connect_to_rabbitmq_with_retry(max_retries=10, initial_delay=1):
    """Connect to RabbitMQ with exponential backoff retry mechanism."""
    for attempt in range(max_retries):
        try:
            print(f"[RabbitMQ] Connection attempt {attempt + 1}/{max_retries}...")
            connection = await aio_pika.connect_robust(RABBITMQ_URL, heartbeat=3600)
            print("[RabbitMQ] Successfully connected!")
            return connection
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[RabbitMQ] Failed to connect after {max_retries} attempts: {e}")
                raise
            
            delay = initial_delay * (2 ** attempt)  # Exponential backoff
            print(f"[RabbitMQ] Connection failed: {e}. Retrying in {delay} seconds...")
            await asyncio.sleep(delay)


async def publish_messages():
    connection = await connect_to_rabbitmq_with_retry()
    channel = await connection.channel()

    # Declare the exchanges (fanout and durable).
    # Declare new topic exchanges
    await declare_exchanges(channel)
    
    # Get exchange references
    exchanges = {}
    for exchange_name in EXCHANGES.keys():
        exchanges[exchange_name] = await channel.get_exchange(
            EXCHANGES[exchange_name]['name']
        )

    # Check if data root exists
    if not DATA_ROOT.exists():
        print(f"[ERROR] Data root directory does not exist: {DATA_ROOT}")
        sys.stdout.flush()
        return
    
    if not DATA_ROOT.is_dir():
        print(f"[ERROR] Data root is not a directory: {DATA_ROOT}")
        sys.stdout.flush()
        return
    
    # Check for different video file patterns
    # 1. Check for single video file (*.mp4)
    single_videos = list(DATA_ROOT.glob("*.mp4"))
    if single_videos and not any("_segment_" in v.name for v in single_videos):
        # Found single video file(s), not segments
        print(f"[INFO] Found {len(single_videos)} single video file(s)")
        sys.stdout.flush()
        for video_file in sorted(single_videos):
            print(f"[INFO] Processing single video: {video_file.name}")
            sys.stdout.flush()
            await replay_single_video(video_exchange, video_file)
        print(f"\n[INFO] Simulation completed! Processed {len(single_videos)} single video(s).")
        sys.stdout.flush()
        await connection.close()
        return
    
    # 2. Check if we're dealing with video segments directly in the root
    video_segments = list(DATA_ROOT.glob("*_segment_*.mp4"))
    if video_segments:
        print(f"[INFO] Found video segments directory with {len(video_segments)} files")
        sys.stdout.flush()
        # Handle video segments directly
        await replay_video_segments(video_exchange, DATA_ROOT)
        print(f"\n[INFO] Simulation completed! Processed video segments.")
        sys.stdout.flush()
        await connection.close()
        return
    
    # Build a sorted list of recording folders.
    try:
        folders = sorted([f for f in DATA_ROOT.iterdir() if f.is_dir()])
    except Exception as e:
        print(f"[ERROR] Cannot access data root directory {DATA_ROOT}: {e}")
        sys.stdout.flush()
        return
        
    total_folders = len(folders)
    print(f"[INFO] Found {total_folders} recording folders in {DATA_ROOT}")
    sys.stdout.flush()
    for i, recording_folder in enumerate(folders):
        mav0_path = recording_folder / "mav0"
        if not mav0_path.exists():
            continue

        print(f"\nProcessing recording: {recording_folder.name}")
        sys.stdout.flush()

        # Create a list of tasks for each data stream in the current recording folder.
        tasks = []

        # Check for video segments first
        video_segments_path = mav0_path / "video_segments"
        if video_segments_path.is_dir() and list(video_segments_path.glob("*_segment_*.mp4")):
            print(f"[INFO] Found video segments in {recording_folder.name}")
            tasks.append(asyncio.create_task(replay_video_segments(exchanges, video_segments_path)))
        else:
            # Images (fallback to original image replay)
            cam_data_path = mav0_path / "cam0" / "data"
            if cam_data_path.is_dir():
                tasks.append(asyncio.create_task(replay_images(exchanges, cam_data_path)))

        # IMU replay removed - no longer needed

        # Trajectory
        trajectory_file = mav0_path / "trajectory.txt"
        if trajectory_file.is_file():
            tasks.append(asyncio.create_task(replay_trajectory(exchanges, trajectory_file)))

        # PLY
        ply_folder = mav0_path / "ply"
        if ply_folder.is_dir():
            tasks.append(asyncio.create_task(replay_ply(exchanges, ply_folder)))

        if tasks:
            # Run all streams concurrently for this recording folder.
            await asyncio.gather(*tasks)

        # Insert a fixed 10-second gap before starting the next recording folder,
        # unless this is the last folder.
        if i < total_folders - 1:
            print("Waiting 10 seconds before next recording folder...")
            sys.stdout.flush()
            await asyncio.sleep(10)

    print(f"\n[INFO] Simulation completed! Processed {total_folders} recording folders.")
    sys.stdout.flush()
    await connection.close()


if __name__ == "__main__":
    print("Waiting for 15 seconds for services to start...")
    sys.stdout.flush()
    time.sleep(15)
    print("Starting simulation...")
    sys.stdout.flush()
    
    try:
        asyncio.run(publish_messages())
        print("Simulation finished successfully!")
        sys.stdout.flush()
    except Exception as e:
        print(f"[ERROR] Simulation failed with error: {e}")
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        sys.exit(1)
