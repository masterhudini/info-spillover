import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import yaml
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.mlflow_utils import MLFlowTracker
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(config_path):
    """Train model with given configuration"""
    logger.info("Starting model training...")

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize MLFlow tracking
    tracker = MLFlowTracker("info_spillover_experiment")

    with tracker.start_run(run_name="model_training"):
        # Log configuration
        tracker.log_params(config)
        tracker.log_config(config_path)

        # Load features
        with open("data/interim/features.pkl", "rb") as f:
            feature_pipeline = pickle.load(f)

        # Load data (placeholder)
        # processed_data_path = Path("data/processed")
        # train_data = pd.read_csv(processed_data_path / "train.csv")
        # val_data = pd.read_csv(processed_data_path / "val.csv")

        # Prepare data (placeholder - adapt to your data)
        # X_train = train_data.drop('target', axis=1)
        # y_train = train_data['target']
        # X_val = val_data.drop('target', axis=1)
        # y_val = val_data['target']

        # Apply feature transformations
        # X_train_transformed = feature_pipeline['selector'].transform(
        #     feature_pipeline['scaler'].transform(X_train)
        # )
        # X_val_transformed = feature_pipeline['selector'].transform(
        #     feature_pipeline['scaler'].transform(X_val)
        # )

        # Initialize model
        model_params = config.get('model', {})
        model = RandomForestClassifier(**model_params)

        # Train model (placeholder)
        logger.info("Training model...")
        # model.fit(X_train_transformed, y_train)

        # Evaluate on validation set
        # y_pred = model.predict(X_val_transformed)
        # y_pred_proba = model.predict_proba(X_val_transformed)

        # Calculate metrics (placeholder)
        # metrics = {
        #     'accuracy': accuracy_score(y_val, y_pred),
        #     'precision': precision_score(y_val, y_pred, average='weighted'),
        #     'recall': recall_score(y_val, y_pred, average='weighted'),
        #     'f1_score': f1_score(y_val, y_pred, average='weighted')
        # }

        # Log metrics
        # tracker.log_metrics(metrics)

        # Create plots directory
        plots_dir = Path("experiments/outputs/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        models_dir = Path("models/saved")
        models_dir.mkdir(parents=True, exist_ok=True)

        # with open(models_dir / "model.pkl", "wb") as f:
        #     pickle.dump(model, f)

        # Log model to MLFlow
        # tracker.log_model(model, "model", registered_model_name="info_spillover_model")

        # Save metrics to file for DVC
        metrics_output = Path("experiments/outputs")
        metrics_output.mkdir(parents=True, exist_ok=True)

        # tracker.save_metrics_to_file(metrics, str(metrics_output / "metrics.json"))

        logger.info("Model training completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()

    train_model(args.config)