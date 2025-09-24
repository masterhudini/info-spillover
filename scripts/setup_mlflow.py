#!/usr/bin/env python3
"""
Setup MLFlow experiments and configuration
"""

import mlflow
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_mlflow_directories():
    """Create necessary MLFlow directories"""
    directories = [
        "mlruns",
        "experiments/outputs",
        "experiments/logs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def setup_mlflow_tracking():
    """Initialize MLFlow tracking"""
    # Set tracking URI
    tracking_uri = "file:./mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLFlow tracking URI set to: {tracking_uri}")

    # Create default experiment
    try:
        experiment_name = "info_spillover_experiment"
        experiment_id = mlflow.create_experiment(
            experiment_name,
            tags={
                "project": "info_spillover",
                "version": "1.0",
                "environment": "development"
            }
        )
        logger.info(f"Created experiment '{experiment_name}' with ID: {experiment_id}")
    except mlflow.exceptions.MlflowException as e:
        if "already exists" in str(e):
            logger.info(f"Experiment '{experiment_name}' already exists")
            experiment = mlflow.get_experiment_by_name(experiment_name)
            logger.info(f"Using existing experiment ID: {experiment.experiment_id}")
        else:
            logger.error(f"Error creating experiment: {e}")

    # Set default experiment
    mlflow.set_experiment(experiment_name)

def create_sample_config():
    """Create sample MLFlow configuration"""
    config_path = Path("mlflow.env")

    if not config_path.exists():
        content = """# MLFlow Environment Configuration
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_DEFAULT_ARTIFACT_ROOT=./mlruns
MLFLOW_EXPERIMENT_NAME=info_spillover_experiment

# Server configuration
MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db
MLFLOW_HOST=0.0.0.0
MLFLOW_PORT=5000
"""
        with open(config_path, "w") as f:
            f.write(content)
        logger.info(f"Created MLFlow configuration: {config_path}")

def test_mlflow_setup():
    """Test MLFlow setup with a simple run"""
    try:
        with mlflow.start_run(run_name="setup_test"):
            # Log some test parameters and metrics
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 0.95)
            mlflow.set_tag("test_tag", "setup_validation")

            logger.info("✅ MLFlow test run completed successfully!")
            return True

    except Exception as e:
        logger.error(f"❌ MLFlow test run failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Setting up MLFlow...")

    # Setup directories
    setup_mlflow_directories()

    # Setup tracking
    setup_mlflow_tracking()

    # Create configuration
    create_sample_config()

    # Test setup
    if test_mlflow_setup():
        logger.info("🎉 MLFlow setup completed successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Start MLFlow UI: make mlflow-start")
        logger.info("2. Visit http://localhost:5000")
        logger.info("3. Run experiments with your data")
    else:
        logger.error("❌ MLFlow setup failed!")
        exit(1)