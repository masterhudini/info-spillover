import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import yaml
import argparse
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
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

        # Load feature info
        with open("data/processed/feature_info.json", "r") as f:
            feature_info = json.load(f)

        feature_columns = feature_info['feature_columns']
        target_column = feature_info['target_column']

        # Load data
        processed_data_path = Path("data/processed")
        train_data = pd.read_csv(processed_data_path / "train.csv")
        val_data = pd.read_csv(processed_data_path / "val.csv")

        logger.info(f"Loaded training data: {train_data.shape}")
        logger.info(f"Loaded validation data: {val_data.shape}")

        # Prepare data
        X_train = train_data[feature_columns]
        y_train = train_data[target_column]
        X_val = val_data[feature_columns]
        y_val = val_data[target_column]

        logger.info(f"Target distribution (train): {y_train.value_counts().to_dict()}")
        logger.info(f"Target distribution (val): {y_val.value_counts().to_dict()}")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Initialize model
        model_config = config.get('model', {})
        model_type = model_config.get('type', 'random_forest')
        model_params = model_config.get('params', {})

        if model_type == 'random_forest':
            model = RandomForestClassifier(**model_params)
        elif model_type == 'logistic_regression':
            model = LogisticRegression(**model_params)
        elif model_type == 'svm':
            model = SVC(**model_params)
        else:
            logger.warning(f"Unknown model type {model_type}, using RandomForest")
            model = RandomForestClassifier(**model_params)

        logger.info(f"Training {model_type} model with params: {model_params}")

        # Train model
        logger.info("Training model...")
        model.fit(X_train_scaled, y_train)

        # Evaluate on validation set
        y_pred = model.predict(X_val_scaled)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted'),
            'recall': recall_score(y_val, y_pred, average='weighted'),
            'f1_score': f1_score(y_val, y_pred, average='weighted')
        }

        logger.info(f"Validation metrics: {metrics}")

        # Log metrics
        tracker.log_metrics(metrics)

        # Create plots directory
        plots_dir = Path("experiments/outputs/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Create confusion matrix plot
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_val, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Down', 'Neutral', 'Up'],
                   yticklabels=['Down', 'Neutral', 'Up'])
        plt.title('Confusion Matrix - Validation Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(plots_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Feature importance plot (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            importance_df = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)

            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title('Top 15 Feature Importances')
            plt.tight_layout()
            plt.savefig(plots_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()

        # Save model and scaler
        models_dir = Path("models/saved")
        models_dir.mkdir(parents=True, exist_ok=True)

        # Save pipeline components
        pipeline = {
            'model': model,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'model_type': model_type
        }

        with open(models_dir / "model.pkl", "wb") as f:
            pickle.dump(pipeline, f)

        # Log model to MLFlow
        tracker.log_model(model, "model", registered_model_name="info_spillover_model")

        # Save metrics to file for DVC
        metrics_output = Path("experiments/outputs")
        metrics_output.mkdir(parents=True, exist_ok=True)

        with open(metrics_output / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save classification report
        class_report = classification_report(y_val, y_pred, output_dict=True)
        with open(metrics_output / "classification_report.json", 'w') as f:
            json.dump(class_report, f, indent=2)

        logger.info("Model training completed successfully!")
        logger.info(f"Model saved to {models_dir / 'model.pkl'}")
        logger.info(f"Metrics saved to {metrics_output / 'metrics.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()

    train_model(args.config)