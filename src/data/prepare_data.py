import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.utils.mlflow_utils import MLFlowTracker
import mlflow
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data():
    """Prepare and split data for training"""
    logger.info("Starting data preparation...")

    # Initialize MLFlow tracking
    tracker = MLFlowTracker("info_spillover_experiment")

    with tracker.start_run(run_name="data_preparation"):
        # Load raw data
        raw_data_path = Path("data/raw")
        processed_data_path = Path("data/processed")
        processed_data_path.mkdir(parents=True, exist_ok=True)

        # Log parameters
        params = {
            "test_size": 0.2,
            "val_size": 0.25,  # 0.25 of remaining 80% = 20% of total
            "random_state": 42
        }
        tracker.log_params(params)

        # Split data (placeholder - adapt to your specific data)
        # This is a template - you'll need to modify based on your actual data
        logger.info("Splitting data into train/val/test sets...")

        # Placeholder for actual data loading
        # Replace this with your actual data loading logic
        # data = pd.read_csv(raw_data_path / "your_data.csv")

        # Example split logic
        # X_temp, X_test, y_temp, y_test = train_test_split(
        #     data.drop('target', axis=1), data['target'],
        #     test_size=params["test_size"], random_state=params["random_state"]
        # )
        #
        # X_train, X_val, y_train, y_val = train_test_split(
        #     X_temp, y_temp,
        #     test_size=params["val_size"], random_state=params["random_state"]
        # )

        # Save processed data
        # train_data = pd.concat([X_train, y_train], axis=1)
        # val_data = pd.concat([X_val, y_val], axis=1)
        # test_data = pd.concat([X_test, y_test], axis=1)

        # train_data.to_csv(processed_data_path / "train.csv", index=False)
        # val_data.to_csv(processed_data_path / "val.csv", index=False)
        # test_data.to_csv(processed_data_path / "test.csv", index=False)

        # Log metrics
        # metrics = {
        #     "train_samples": len(train_data),
        #     "val_samples": len(val_data),
        #     "test_samples": len(test_data),
        #     "total_features": len(X_train.columns)
        # }
        # tracker.log_metrics(metrics)

        logger.info("Data preparation completed successfully!")

if __name__ == "__main__":
    prepare_data()