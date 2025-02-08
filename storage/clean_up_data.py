#!/usr/bin/env python3
"""
clean_up_data.py

Go through each subfolder in /data and delete any that do not have:
1) A non-empty trajectory file,
2) At least one .jpg image, and
3) At least one .txt IMU file.
"""

import os
import shutil
from pathlib import Path

DATA_PATH = Path("data")

def has_valid_images(cam0_data_path: Path) -> bool:
    """
    Returns True if cam0/data exists and contains at least one .jpg file.
    """
    if not cam0_data_path.is_dir():
        return False
    image_files = list(cam0_data_path.glob("*.jpg"))
    return len(image_files) > 0

def has_valid_imu(imu0_data_path: Path) -> bool:
    """
    Returns True if imu0/data exists and contains at least one .txt file.
    """
    if not imu0_data_path.is_dir():
        return False
    imu_files = list(imu0_data_path.glob("*.txt"))
    return len(imu_files) > 0

def has_valid_trajectory(trajectory_file: Path) -> bool:
    """
    Returns True if trajectory.txt exists and is not empty.
    """
    if not trajectory_file.is_file():
        return False
    if os.path.getsize(trajectory_file) == 0:
        return False
    return True

def clean_up_data(data_folder: Path = DATA_PATH):
    """
    Iterate over each sub-directory in /data. If the directory does not contain
    valid images, IMU data, and a trajectory file, remove the entire directory.
    """
    if not data_folder.exists() or not data_folder.is_dir():
        print(f"[!] {data_folder} does not exist or is not a directory.")
        return

    # Print all folders before cleanup
    print("\nFolders before cleanup:")
    for subfolder in data_folder.iterdir():
        if subfolder.is_dir():
            print(f"  - {subfolder}")
    print()

    # Go through each subfolder, e.g. /data/20230101_120101/
    for subfolder in data_folder.iterdir():
        if not subfolder.is_dir():
            # Skip files or anything that is not a directory
            continue

        # We expect a structure like /data/<timestamp>/mav0
        mav0_path = subfolder / "mav0"
        if not mav0_path.is_dir():
            print(f"[-] {subfolder} has no 'mav0' folder. Attempting deletion...")
            try:
                shutil.rmtree(subfolder, ignore_errors=False)
                print(f"    ✓ Successfully deleted {subfolder}")
            except Exception as e:
                print(f"    ✗ Failed to delete {subfolder}: {str(e)}")
            continue

        cam0_data_path = mav0_path / "cam0" / "data"
        imu0_data_path = mav0_path / "imu0" / "data"
        trajectory_file = mav0_path / "trajectory.txt"

        images_ok = has_valid_images(cam0_data_path)
        imu_ok = has_valid_imu(imu0_data_path)
        trajectory_ok = has_valid_trajectory(trajectory_file)

        if not (images_ok and imu_ok and trajectory_ok):
            # If any check fails, remove this entire subfolder
            print(f"[-] Incomplete data in {subfolder}. Attempting deletion...")
            try:
                shutil.rmtree(subfolder, ignore_errors=False)
                print(f"    ✓ Successfully deleted {subfolder}")
            except Exception as e:
                print(f"    ✗ Failed to delete {subfolder}: {str(e)}")
        else:
            print(f"[+] {subfolder} has valid images, IMU data, and trajectory.")

if __name__ == "__main__":
    clean_up_data()
