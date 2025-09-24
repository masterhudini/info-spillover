import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.utils.mlflow_utils import MLFlowTracker
from src.data.bigquery_client import BigQueryClient
import mlflow
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data_to_bigquery():
    """Load raw JSON and CSV data to BigQuery"""
    logger.info("Loading data to BigQuery...")

    # Initialize BigQuery client
    bq_client = BigQueryClient()

    # Load JSON data (posts and comments)
    json_data_path = "/home/Hudini/gcs/raw/posts_n_comments"
    if os.path.exists(json_data_path):
        logger.info("Loading JSON data to BigQuery...")
        bq_client.load_json_data_to_bq(json_data_path)
    else:
        logger.warning(f"JSON data path {json_data_path} does not exist")

    # Load CSV data (prices)
    csv_data_path = "/home/Hudini/gcs/raw/prices"
    if os.path.exists(csv_data_path):
        logger.info("Loading CSV data to BigQuery...")
        bq_client.load_csv_data_to_bq(csv_data_path)
    else:
        logger.warning(f"CSV data path {csv_data_path} does not exist")

    logger.info("Data loading to BigQuery completed!")
    return bq_client


def prepare_data():
    """Prepare and split data for training"""
    logger.info("Starting data preparation...")

    # Initialize MLFlow tracking
    tracker = MLFlowTracker("info_spillover_experiment")

    with tracker.start_run(run_name="data_preparation"):
        processed_data_path = Path("data/processed")
        processed_data_path.mkdir(parents=True, exist_ok=True)

        # Log parameters
        params = {
            "test_size": 0.2,
            "val_size": 0.25,  # 0.25 of remaining 80% = 20% of total
            "random_state": 42,
            "start_date": "2021-01-01",
            "end_date": "2023-12-31"
        }
        tracker.log_params(params)

        # Load data to BigQuery if not already done
        logger.info("Ensuring data is loaded to BigQuery...")
        bq_client = load_data_to_bigquery()

        # Create combined dataset from BigQuery
        logger.info("Creating combined dataset from BigQuery...")
        data = bq_client.create_combined_dataset(
            start_date=params["start_date"],
            end_date=params["end_date"]
        )

        if data.empty:
            logger.warning("No data returned from BigQuery. Check data availability.")
            return

        # Feature engineering
        logger.info("Performing feature engineering...")

        # Create additional features
        data['sentiment_score'] = (
            data['positive_ratio'] * data['avg_positive_sentiment'] -
            data['negative_ratio'] * data['avg_negative_sentiment']
        ).fillna(0)

        data['activity_score'] = data['num_posts'] * data['num_comments']
        data['engagement_score'] = data['avg_post_score'] * data['avg_comment_score']

        # Handle missing values
        data = data.fillna(0)

        # Prepare features and target
        feature_columns = [
            'num_posts', 'num_comments', 'avg_positive_sentiment', 'avg_negative_sentiment',
            'avg_neutral_sentiment', 'positive_ratio', 'negative_ratio', 'neutral_ratio',
            'avg_post_score', 'avg_comment_score', 'price', 'market_cap', 'total_volume',
            'price_change_pct', 'sentiment_score', 'activity_score', 'engagement_score'
        ]

        # Filter valid rows (where we have next day price data)
        data_clean = data.dropna(subset=['price_direction'])

        if data_clean.empty:
            logger.warning("No valid data for training after filtering.")
            return

        X = data_clean[feature_columns]
        y = data_clean['price_direction']  # Target: -1, 0, 1 for price movement

        logger.info(f"Dataset shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")

        # Split data
        logger.info("Splitting data into train/val/test sets...")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=params["test_size"],
            random_state=params["random_state"], stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=params["val_size"],
            random_state=params["random_state"], stratify=y_temp
        )

        # Save processed data
        train_data = pd.concat([X_train, y_train], axis=1)
        val_data = pd.concat([X_val, y_val], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        train_data.to_csv(processed_data_path / "train.csv", index=False)
        val_data.to_csv(processed_data_path / "val.csv", index=False)
        test_data.to_csv(processed_data_path / "test.csv", index=False)

        # Save feature info for later use
        feature_info = {
            'feature_columns': feature_columns,
            'target_column': 'price_direction',
            'num_classes': len(y.unique())
        }

        import json
        with open(processed_data_path / "feature_info.json", 'w') as f:
            json.dump(feature_info, f, indent=2)

        # Log metrics
        metrics = {
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
            "total_features": len(feature_columns),
            "num_classes": len(y.unique()),
            "data_start_date": data_clean['date'].min().strftime('%Y-%m-%d'),
            "data_end_date": data_clean['date'].max().strftime('%Y-%m-%d')
        }
        tracker.log_metrics(metrics)

        logger.info("Data preparation completed successfully!")
        logger.info(f"Train: {len(train_data)} samples, Val: {len(val_data)} samples, Test: {len(test_data)} samples")

if __name__ == "__main__":
    prepare_data()