#!/usr/bin/env python3
"""
clean_up_data.py

Go through each subfolder in /data and delete any that do not have:
1) At least one .jpg image in cam0/data OR one .png image in cam1/data.
IMU and Trajectory checks are SKIPPED BY DEFAULT.
Use --require_imu_check and --require_trajectory_check to enforce them.

NOW RUNS IN DRY RUN MODE BY DEFAULT. No folders will be deleted.
It will only print which folders WOULD be deleted.
"""

import os
import shutil
from pathlib import Path
import argparse

DATA_PATH = Path("data")

def has_valid_images(mav0_path: Path) -> bool:
    """
    Returns True if mav0/cam0/data exists and contains at least one .jpg file,
    OR if mav0/cam1/data exists and contains at least one .png file.
    """
    cam0_data_path = mav0_path / "cam0" / "data"
    cam1_data_path = mav0_path / "cam1" / "data"

    valid_cam0_jpgs = False
    if cam0_data_path.is_dir():
        image_files_jpg = list(cam0_data_path.glob("*.jpg"))
        if len(image_files_jpg) > 0:
            valid_cam0_jpgs = True

    valid_cam1_pngs = False
    if cam1_data_path.is_dir():
        image_files_png = list(cam1_data_path.glob("*.png"))
        if len(image_files_png) > 0:
            valid_cam1_pngs = True
            
    if valid_cam0_jpgs:
        print(f"    [Image Check] Found JPGs in {cam0_data_path}")
    elif valid_cam1_pngs:
        print(f"    [Image Check] Found PNGs in {cam1_data_path}")
    else:
        print(f"    [Image Check] No JPGs found in {cam0_data_path} AND no PNGs found in {cam1_data_path}")

    return valid_cam0_jpgs or valid_cam1_pngs

def has_valid_imu(mav0_path: Path) -> bool:
    """
    Returns True if imu0/data.csv or imu1/data.csv exists and is not empty.
    """
    imu0_csv = mav0_path / "imu0" / "data.csv"
    imu1_csv = mav0_path / "imu1" / "data.csv"

    valid_imu0 = False
    if imu0_csv.is_file() and os.path.getsize(imu0_csv) > 0:
        valid_imu0 = True

    valid_imu1 = False
    if imu1_csv.is_file() and os.path.getsize(imu1_csv) > 0:
        valid_imu1 = True
        
    if valid_imu0 or valid_imu1:
        print(f"    [IMU Check] Found valid IMU data (imu0: {valid_imu0}, imu1: {valid_imu1}).")
    else:
        print(f"    [IMU Check] No valid IMU data found in {imu0_csv} or {imu1_csv}")
        
    return valid_imu0 or valid_imu1

def has_valid_trajectory(trajectory_file: Path) -> bool:
    """
    Returns True if trajectory.txt exists and is not empty.
    """
    if not trajectory_file.is_file():
        print(f"    [Trajectory Check] File not found: {trajectory_file}")
        return False
    if os.path.getsize(trajectory_file) == 0:
        print(f"    [Trajectory Check] File is empty: {trajectory_file}")
        return False
    print(f"    [Trajectory Check] Found valid trajectory file: {trajectory_file}")
    return True

def run_cleanup(data_folder_str: str = "data", execute_deletion: bool = False, require_imu_check: bool = False, require_trajectory_check: bool = False):
    """
    Iterate over each sub-directory in the specified data folder.
    If the directory does not meet criteria, it will be marked for deletion.
    Actual deletion only occurs if execute_deletion is True.
    Image check is always performed. IMU and Trajectory checks are optional.
    """
    data_folder = Path(data_folder_str)
    if not data_folder.exists() or not data_folder.is_dir():
        print(f"[!] {data_folder} does not exist or is not a directory.")
        return

    print(f"\nStarting cleanup process for: {data_folder.resolve()}")
    if not execute_deletion:
        print("[!] RUNNING IN DRY RUN MODE. No files will be deleted.")
    else:
        print("[!] EXECUTE DELETION MODE ENABLED. Folders failing checks will be deleted.")
    
    print(f"[i] IMU Check: {'ENFORCED' if require_imu_check else 'SKIPPED (default)'}")
    print(f"[i] Trajectory Check: {'ENFORCED' if require_trajectory_check else 'SKIPPED (default)'}")

    folders_before = [f.name for f in data_folder.iterdir() if f.is_dir()]
    print(f"Folders scanned: {folders_before}")
    print()
    
    deleted_count = 0
    kept_count = 0

    for subfolder in data_folder.iterdir():
        if not subfolder.is_dir():
            continue

        print(f"Processing subfolder: {subfolder.name}")
        mav0_path = subfolder / "mav0"
        if not mav0_path.is_dir():
            print(f"  [-] {subfolder.name} has no 'mav0' folder.")
            if execute_deletion:
                print(f"    EXECUTING DELETION of {subfolder.name} (no mav0 folder)")
                try:
                    shutil.rmtree(subfolder, ignore_errors=False)
                    print(f"      ✓ Successfully deleted {subfolder.name}")
                    deleted_count +=1
                except Exception as e:
                    print(f"      ✗ Failed to delete {subfolder.name}: {str(e)}")
            else:
                print(f"    WOULD DELETE {subfolder.name} (no mav0 folder)")
                deleted_count+=1
            continue

        print(f"  Verifying data in: {mav0_path}")
        
        images_ok = has_valid_images(mav0_path)
        imu_ok = True # Assume OK if check is skipped
        trajectory_ok = True # Assume OK if check is skipped

        if require_imu_check:
            imu_ok = has_valid_imu(mav0_path)
        else:
            print("    [IMU Check] SKIPPED as per settings.")

        if require_trajectory_check:
            trajectory_file = mav0_path / "trajectory.txt"
            trajectory_ok = has_valid_trajectory(trajectory_file)
        else:
            print("    [Trajectory Check] SKIPPED as per settings.")
        
        print(f"  Checks for {subfolder.name}: Images OK: {images_ok}, IMU OK (actual/skipped): {imu_ok}, Trajectory OK (actual/skipped): {trajectory_ok}")

        if not images_ok or (require_imu_check and not imu_ok) or (require_trajectory_check and not trajectory_ok):
            print(f"  [-] Incomplete data in {subfolder.name}.")
            if execute_deletion:
                print(f"    EXECUTING DELETION of {subfolder.name} (failed checks)")
                try:
                    shutil.rmtree(subfolder, ignore_errors=False)
                    print(f"      ✓ Successfully deleted {subfolder.name}")
                    deleted_count +=1
                except Exception as e:
                    print(f"      ✗ Failed to delete {subfolder.name}: {str(e)}")
            else:
                print(f"    WOULD DELETE {subfolder.name} (failed checks)")
                deleted_count +=1
        else:
            print(f"  [+] {subfolder.name} has valid data based on current criteria. Kept.")
            kept_count += 1
    
    folders_after_query = data_folder.iterdir() # Re-query after potential deletions
    folders_after = [f.name for f in folders_after_query if f.is_dir()]
    print(f"\nSummary for {data_folder.resolve()}:")
    print(f"  Folders kept: {kept_count}")
    print(f"  Folders {'deleted' if execute_deletion else 'marked for deletion'}: {deleted_count}")
    if not execute_deletion:
         print(f"  Folders that would remain (dry run): {folders_after}") # Name might be confusing if execute_deletion is true, but structure is for dry run primarily
    else:
         print(f"  Folders remaining: {folders_after}")
    print(f"\nCleanup process for {data_folder.resolve()} finished.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Cleans up data folders based on image presence. "
            "IMU and Trajectory checks are SKIPPED BY DEFAULT.\n"
            "Runs in DRY RUN mode by default."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(DATA_PATH),
        help=f"Path to the main data directory (default: {DATA_PATH})"
    )
    parser.add_argument(
        "--execute_deletion",
        action="store_true",
        help="Actually delete folders that fail checks. Without this flag, script runs in DRY RUN mode."
    )
    parser.add_argument(
        "--require_imu_check",
        action="store_true",
        help="Enforce the check for valid IMU data. Skipped by default."
    )
    parser.add_argument(
        "--require_trajectory_check",
        action="store_true",
        help="Enforce the check for a valid trajectory.txt file. Skipped by default."
    )
    args = parser.parse_args()

    run_cleanup(data_folder_str=args.data_path, execute_deletion=args.execute_deletion, require_imu_check=args.require_imu_check, require_trajectory_check=args.require_trajectory_check)
