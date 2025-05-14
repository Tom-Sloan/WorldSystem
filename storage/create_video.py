#!/usr/bin/env python3
"""
create_video.py

Generates video segments for each valid dataset found in the data directory.
A dataset is considered valid if it has a mav0 folder containing
either cam0/data with JPG images or cam1/data with PNG images.

Segments are created when a time gap larger than a specified threshold
is detected between consecutive images. Each segment is saved as a separate video file.

The script can optionally clean the data directory first.
Videos are saved in data/<timestamp>/mav0/video_segments/ as <timestamp>_segment_<n>.mp4.
Frame rate can be fixed or dynamically calculated per segment.
"""

import os
import argparse
from pathlib import Path
import cv2 # OpenCV for video processing
import re
import shutil # For directory operations

# Attempt to import the cleanup function
try:
    from .clean_up_data import run_cleanup
except ImportError:
    # Fallback if running as a standalone script and clean_up_data is in the same directory
    try:
        from clean_up_data import run_cleanup
    except ImportError:
        run_cleanup = None
        print("[!] Warning: clean_up_data.py not found. --clean_data option will not be available.")

DATA_PATH = Path("data")
IMAGE_EXTENSIONS_PRIORITY = {
    "cam0": (".jpg", "cam0/data"),
    "cam1": (".png", "cam1/data"),
}
VIDEO_SEGMENTS_SUBDIR = "video_segments"
DEFAULT_LAST_FRAME_DURATION_S = 0.1 # How long the last frame of a segment should be displayed

def get_image_timestamp_ns(image_filename: str) -> int | None:
    """Extracts timestamp in nanoseconds from image filename (e.g., '1747239245451970520.jpg')."""
    match = re.match(r"(\d+)\.(jpg|jpeg|png)", image_filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def _write_video_segment(segment_images: list, 
                         mav0_path: Path, 
                         dataset_name: str, 
                         segment_num: int, 
                         user_target_fps: int, 
                         fallback_fps: int,
                         video_width: int, 
                         video_height: int):
    """Writes a single video segment file using frame duplication for correct timing."""
    if not segment_images:
        print("[!] _write_video_segment called with no images. Skipping.")
        return

    segment_output_dir = mav0_path / VIDEO_SEGMENTS_SUBDIR
    segment_output_path = segment_output_dir / f"{dataset_name}_segment_{segment_num}.mp4"

    actual_fps_for_segment = 0.0
    if user_target_fps > 0:
        actual_fps_for_segment = float(user_target_fps)
        print(f"  [Seg {segment_num}] Using user-defined FPS: {actual_fps_for_segment:.2f}")
    else:
        if len(segment_images) > 1:
            # Duration up to the *beginning* of the last image frame
            duration_to_last_frame_start_ns = segment_images[-1]['ts'] - segment_images[0]['ts']
            if duration_to_last_frame_start_ns > 0:
                # Number of intervals is len(segment_images) -1 for calculation up to last frame start
                calculated_fps = (len(segment_images) - 1) / (duration_to_last_frame_start_ns / 1_000_000_000.0)
                actual_fps_for_segment = calculated_fps if calculated_fps > 0.1 else float(fallback_fps)
            else: # all images up to last have same timestamp or only one image effectively for this calculation
                actual_fps_for_segment = float(fallback_fps)
            print(f"  [Seg {segment_num}] Dynamically calculated FPS: {actual_fps_for_segment:.2f} (fallback: {fallback_fps})")
        elif len(segment_images) == 1: # Should ideally be caught by min_images_for_segment
            actual_fps_for_segment = 1.0
            print(f"  [Seg {segment_num}] Segment has 1 image. Using FPS: {actual_fps_for_segment:.2f}")
        else: # No images, though caught earlier
            print(f"  [Seg {segment_num}] No images for FPS calculation. This should not happen.")
            return

    print(f"  [Seg {segment_num}] Writing segment {segment_output_path} with FPS {actual_fps_for_segment:.2f}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(segment_output_path), fourcc, actual_fps_for_segment, (video_width, video_height))

    if not out.isOpened():
        print(f"[!] Error: VideoWriter failed to open for {segment_output_path}.")
        return

    for i, img_info_dict in enumerate(segment_images):
        img_path = img_info_dict["path"]
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[!] Warning: Could not read frame {img_path.name} for segment {segment_num}. Skipping.")
            continue

        if frame.shape[0] != video_height or frame.shape[1] != video_width:
            print(f"[!] Warning: Image {img_path.name} resolution ({frame.shape[1]}x{frame.shape[0]}) differs. Resizing.")
            frame = cv2.resize(frame, (video_width, video_height))

        duration_for_this_frame_s = 0.0
        if i < len(segment_images) - 1:
            duration_for_this_frame_s = (segment_images[i+1]['ts'] - img_info_dict['ts']) / 1_000_000_000.0
        else:
            duration_for_this_frame_s = DEFAULT_LAST_FRAME_DURATION_S # Last frame shows for default duration
        
        num_times_to_write_frame = max(1, int(round(duration_for_this_frame_s * actual_fps_for_segment)))
        
        for _ in range(num_times_to_write_frame):
            out.write(frame)
            
    out.release()
    print(f"    ✓ Segment {segment_num} successfully created: {segment_output_path}")

def create_video_for_dataset(timestamp_folder_path: Path, mav0_path: Path, 
                             gap_threshold_s: float, user_target_fps: int, 
                             min_images_for_segment: int, fallback_fps: int):
    """Creates video segments for a single dataset based on time gaps."""
    
    segments_output_dir = mav0_path / VIDEO_SEGMENTS_SUBDIR
    if segments_output_dir.exists():
        print(f"[i] Existing segments directory found: {segments_output_dir}. Clearing its contents.")
        try:
            shutil.rmtree(segments_output_dir)
        except Exception as e:
            print(f"[!] Error clearing segments directory {segments_output_dir}: {e}. Skipping dataset.")
            return
    try:
        segments_output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[!] Error creating segments directory {segments_output_dir}: {e}. Skipping dataset.")
        return

    selected_image_files = []
    image_dir_path = None

    for cam_key, (ext, cam_data_subdir) in IMAGE_EXTENSIONS_PRIORITY.items():
        current_image_dir = mav0_path / cam_data_subdir
        if current_image_dir.is_dir():
            found_files = sorted(
                [f for f in current_image_dir.iterdir() if f.is_file() and f.suffix.lower() == ext],
                key=lambda p: p.name
            )
            if found_files:
                selected_image_files = found_files
                image_dir_path = current_image_dir
                print(f"[i] Using {cam_key} ({ext}) images from: {image_dir_path}")
                break
    
    if not selected_image_files:
        print(f"[!] No suitable images found in mav0 subdirectories for {timestamp_folder_path.name}")
        return

    timed_images = []
    for img_path in selected_image_files:
        ts = get_image_timestamp_ns(img_path.name)
        if ts is not None:
            timed_images.append({"path": img_path, "ts": ts})
        else:
            print(f"[!] Could not extract timestamp from {img_path.name}, skipping image.")
    
    if not timed_images:
        print(f"[!] No images with valid timestamps found in {image_dir_path} for {timestamp_folder_path.name}.")
        return

    timed_images.sort(key=lambda x: x["ts"])

    if len(timed_images) < min_images_for_segment:
        print(f"[i] Not enough images ({len(timed_images)}) to meet minimum of {min_images_for_segment} for dataset {timestamp_folder_path.name}. Skipping.")
        return

    first_image_for_res_path = timed_images[0]["path"]
    first_image_for_res = cv2.imread(str(first_image_for_res_path))
    if first_image_for_res is None:
        print(f"[!] Error: Could not read first image {first_image_for_res_path} for resolution. Skipping dataset {timestamp_folder_path.name}.")
        return
    height, width, _ = first_image_for_res.shape

    current_segment_images = []
    segment_counter = 1
    
    print(f"[+] Processing {len(timed_images)} images for {timestamp_folder_path.name} into segments...")

    for i, img_info_dict in enumerate(timed_images):
        current_segment_images.append(img_info_dict)

        is_last_image_in_dataset = (i == len(timed_images) - 1)
        gap_detected = False
        if not is_last_image_in_dataset:
            time_diff_ns = timed_images[i+1]['ts'] - img_info_dict['ts']
            time_diff_s = time_diff_ns / 1_000_000_000.0
            if time_diff_s >= gap_threshold_s:
                gap_detected = True
        
        if is_last_image_in_dataset or gap_detected:
            if len(current_segment_images) >= min_images_for_segment:
                print(f"  Creating segment {segment_counter} for {timestamp_folder_path.name} with {len(current_segment_images)} images.")
                _write_video_segment(current_segment_images, mav0_path, timestamp_folder_path.name, 
                                     segment_counter, user_target_fps, fallback_fps, width, height)
                segment_counter += 1
            else:
                # Corrected f-string by accessing .name on the Path object directly
                image_name_for_log = current_segment_images[-1]["path"].name if current_segment_images else "N/A"
                print(f"  [i] Segment ending at image {image_name_for_log} has {len(current_segment_images)} images, less than min {min_images_for_segment}. Discarding.")
            current_segment_images = [] # Start new segment

    # Final check for any remaining images in current_segment_images (should be handled by is_last_image_in_dataset)
    if current_segment_images and len(current_segment_images) >= min_images_for_segment:
         print(f"  Creating final (safeguard) segment {segment_counter} for {timestamp_folder_path.name} with {len(current_segment_images)} images.")
         _write_video_segment(current_segment_images, mav0_path, timestamp_folder_path.name, 
                             segment_counter, user_target_fps, fallback_fps, width, height)

def main():
    parser = argparse.ArgumentParser(
        description="Generates video segments from image sequences.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--clean_data", action="store_true",
        help="Run data cleanup process before generating videos. Requires clean_up_data.py."
    )
    parser.add_argument(
        "--data_path", type=str, default=str(DATA_PATH),
        help=f"Path to the main data directory (default: {DATA_PATH})"
    )
    parser.add_argument(
        "--gap_threshold_seconds", type=float, default=2.0,
        help="Minimum time gap (seconds) between images to trigger a new video segment (default: 2.0)"
    )
    parser.add_argument(
        "--display_fps", type=int, default=0,
        help="Target FPS for video segments. If 0, FPS is calculated dynamically per segment (default: 0)"
    )
    parser.add_argument(
        "--min_images_for_segment", type=int, default=5,
        help="Minimum number of images required to form a video segment (default: 5)"
    )
    parser.add_argument(
        "--fallback_fps", type=int, default=10,
        help="Fallback FPS if dynamic calculation fails or results in very low FPS (default: 10)"
    )
    args = parser.parse_args()

    current_data_path = Path(args.data_path)

    if args.clean_data:
        if run_cleanup:
            print(f"[*] Running data cleanup on: {current_data_path}")
            # Assuming clean_up_data.py takes these args or has its own defaults
            # For now, calling it as it was, it might need its own arg parser updated if necessary
            run_cleanup(str(current_data_path)) 
            print("[*] Data cleanup finished.")
        else:
            print("[!] --clean_data was specified, but the cleanup function could not be imported. Skipping cleanup.")

    if not current_data_path.is_dir():
        print(f"[!] Error: Data path {current_data_path} does not exist or is not a directory.")
        return

    print(f"[*] Starting video generation process for directories in: {current_data_path}")
    timestamp_folders = [d for d in current_data_path.iterdir() if d.is_dir()]

    if not timestamp_folders:
        print(f"[i] No timestamped data folders found in {current_data_path}.")
        return

    for timestamp_folder in timestamp_folders:
        print(f"\nProcessing data folder: {timestamp_folder.name}")
        mav0_path = timestamp_folder / "mav0"
        if not mav0_path.is_dir():
            print(f"[!] 'mav0' directory not found in {timestamp_folder.name}. Skipping.")
            continue
        
        create_video_for_dataset(timestamp_folder, mav0_path, 
                                 args.gap_threshold_seconds, args.display_fps, 
                                 args.min_images_for_segment, args.fallback_fps)
    
    print("\n[*] Video generation process finished.")

if __name__ == "__main__":
    if cv2.__version__:
        print(f"[*] OpenCV version: {cv2.__version__}")
        main()
    else:
        print("[!] OpenCV (cv2) library is not available. Please install it.")
        print("    pip install opencv-python")
