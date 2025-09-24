#!/usr/bin/env python3
"""
Setup script to symlink or copy data from GCS bucket to local data directory
"""

import os
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_data_from_gcs():
    """Copy or symlink data from GCS bucket to local data directory"""

    # Paths
    gcs_path = Path("/home/Hudini/gcs")
    local_raw_data = Path("data/raw")

    if not gcs_path.exists():
        logger.error(f"GCS bucket not found at {gcs_path}")
        return False

    # Create raw data directory
    local_raw_data.mkdir(parents=True, exist_ok=True)

    # Copy JSON files from GCS to local raw data
    json_files = list(gcs_path.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in GCS bucket")

    for json_file in json_files:
        target_file = local_raw_data / json_file.name

        # Skip if file already exists
        if target_file.exists():
            logger.info(f"File already exists: {target_file}")
            continue

        # Create symlink (more efficient than copying)
        try:
            target_file.symlink_to(json_file)
            logger.info(f"Created symlink: {target_file} -> {json_file}")
        except OSError as e:
            # Fallback to copying if symlink fails
            logger.warning(f"Symlink failed, copying instead: {e}")
            shutil.copy2(json_file, target_file)
            logger.info(f"Copied file: {json_file} -> {target_file}")

    logger.info("Data setup completed!")
    return True

def add_data_to_dvc():
    """Add raw data to DVC tracking"""
    import subprocess

    try:
        # Add the entire raw data directory to DVC
        result = subprocess.run(
            ["dvc", "add", "data/raw"],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("Successfully added data/raw to DVC tracking")
        logger.info(result.stdout)

        # Check if .gitignore was updated
        gitignore_path = Path(".gitignore")
        if gitignore_path.exists():
            with open(gitignore_path, "r") as f:
                content = f.read()
                if "/data/raw.dvc" not in content:
                    logger.info("Remember to commit data/raw.dvc to git!")

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to add data to DVC: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("DVC not found. Make sure DVC is installed.")
        return False

if __name__ == "__main__":
    logger.info("Setting up data from GCS bucket...")

    if setup_data_from_gcs():
        logger.info("Data setup successful!")

        # Optionally add to DVC tracking
        response = input("Add data to DVC tracking? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            add_data_to_dvc()
    else:
        logger.error("Data setup failed!")
        exit(1)