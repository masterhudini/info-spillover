#!/usr/bin/env python3
"""
Run complete ML experiment pipeline
"""

import subprocess
import sys
import logging
from pathlib import Path
import yaml
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run shell command and handle errors"""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")

    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Command failed: {command}")
        logger.error(f"Error: {result.stderr}")
        sys.exit(1)

    if result.stdout:
        logger.info(f"Output: {result.stdout}")

def main():
    parser = argparse.ArgumentParser(description="Run ML experiment pipeline")
    parser.add_argument("--config", default="experiments/configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--skip-data", action="store_true",
                       help="Skip data preparation step")
    parser.add_argument("--skip-features", action="store_true",
                       help="Skip feature engineering step")
    parser.add_argument("--only-train", action="store_true",
                       help="Only run training step")

    args = parser.parse_args()

    logger.info("Starting ML experiment pipeline...")

    # Validate config file exists
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)

    try:
        # Step 1: Data preparation
        if not args.skip_data and not args.only_train:
            run_command("python src/data/prepare_data.py", "Data preparation")

        # Step 2: Feature engineering
        if not args.skip_features and not args.only_train:
            run_command("python src/features/build_features.py", "Feature engineering")

        # Step 3: Model training
        run_command(f"python src/models/train_model.py --config {args.config}", "Model training")

        # Step 4: Model evaluation
        if not args.only_train:
            run_command("python src/models/evaluate_model.py", "Model evaluation")

        logger.info("Experiment pipeline completed successfully!")

    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()