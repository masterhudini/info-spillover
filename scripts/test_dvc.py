#!/usr/bin/env python3
"""
Test DVC configuration and functionality
"""

import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dvc_config():
    """Test DVC configuration"""
    logger.info("Testing DVC configuration...")

    try:
        # Check DVC status
        result = subprocess.run(
            ["dvc", "status"],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"DVC status: {result.stdout}")

        # List remotes
        result = subprocess.run(
            ["dvc", "remote", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"DVC remotes:\n{result.stdout}")

        # Check remote connectivity
        result = subprocess.run(
            ["dvc", "config", "--list"],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"DVC config:\n{result.stdout}")

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"DVC command failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def test_gcs_access():
    """Test GCS bucket access"""
    logger.info("Testing GCS bucket access...")

    gcs_path = Path("/home/Hudini/gcs")
    if not gcs_path.exists():
        logger.error(f"GCS bucket not accessible at {gcs_path}")
        return False

    # Count files
    json_files = list(gcs_path.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in GCS bucket")

    # Test read access
    if json_files:
        test_file = json_files[0]
        try:
            size = test_file.stat().st_size
            logger.info(f"Successfully read file info: {test_file.name} ({size} bytes)")
            return True
        except Exception as e:
            logger.error(f"Cannot read file {test_file}: {e}")
            return False
    else:
        logger.warning("No JSON files found in GCS bucket")
        return False

if __name__ == "__main__":
    logger.info("Testing DVC and GCS setup...")

    success = True

    if not test_dvc_config():
        success = False

    if not test_gcs_access():
        success = False

    if success:
        logger.info("✅ All tests passed! DVC and GCS are properly configured.")
    else:
        logger.error("❌ Some tests failed. Check the configuration.")
        exit(1)