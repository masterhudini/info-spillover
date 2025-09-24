import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
from src.utils.mlflow_utils import MLFlowTracker
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_features():
    """Build features from processed data"""
    logger.info("Starting feature engineering...")

    # Initialize MLFlow tracking
    tracker = MLFlowTracker("info_spillover_experiment")

    with tracker.start_run(run_name="feature_engineering"):
        # Load processed data
        processed_data_path = Path("data/processed")
        interim_data_path = Path("data/interim")
        interim_data_path.mkdir(parents=True, exist_ok=True)

        # Parameters
        params = {
            "scaling_method": "standard",
            "feature_selection": "k_best",
            "k_features": 50,
            "random_state": 42
        }
        tracker.log_params(params)

        # Load data (placeholder - adapt to your data)
        # train_data = pd.read_csv(processed_data_path / "train.csv")
        # val_data = pd.read_csv(processed_data_path / "val.csv")
        # test_data = pd.read_csv(processed_data_path / "test.csv")

        # Feature engineering pipeline
        feature_pipeline = {
            'scaler': StandardScaler(),
            'selector': SelectKBest(f_classif, k=params["k_features"]),
            'encoders': {}
        }

        # Apply transformations
        logger.info("Applying feature transformations...")

        # Example feature engineering (adapt to your needs)
        # X_train = train_data.drop('target', axis=1)
        # y_train = train_data['target']

        # X_val = val_data.drop('target', axis=1)
        # y_val = val_data['target']

        # X_test = test_data.drop('target', axis=1)
        # y_test = test_data['target']

        # Fit and transform features
        # X_train_scaled = feature_pipeline['scaler'].fit_transform(X_train)
        # X_val_scaled = feature_pipeline['scaler'].transform(X_val)
        # X_test_scaled = feature_pipeline['scaler'].transform(X_test)

        # Feature selection
        # X_train_selected = feature_pipeline['selector'].fit_transform(X_train_scaled, y_train)
        # X_val_selected = feature_pipeline['selector'].transform(X_val_scaled)
        # X_test_selected = feature_pipeline['selector'].transform(X_test_scaled)

        # Save feature pipeline
        with open(interim_data_path / "features.pkl", "wb") as f:
            pickle.dump(feature_pipeline, f)

        # Log metrics
        # metrics = {
        #     "original_features": X_train.shape[1],
        #     "selected_features": X_train_selected.shape[1],
        #     "feature_reduction_ratio": X_train_selected.shape[1] / X_train.shape[1]
        # }
        # tracker.log_metrics(metrics)

        logger.info("Feature engineering completed successfully!")

if __name__ == "__main__":
    build_features()