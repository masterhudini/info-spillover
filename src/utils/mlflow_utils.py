import mlflow
import mlflow.sklearn
import mlflow.pytorch
import os
from pathlib import Path
import yaml
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
import logging

class MLFlowTracker:
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        return mlflow.start_run(run_name=run_name, tags=tags)

    def log_params(self, params: Dict[str, Any]):
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_artifacts(self, artifacts_path: str):
        mlflow.log_artifacts(artifacts_path)

    def log_model(self, model, model_name: str, registered_model_name: Optional[str] = None):
        if hasattr(model, 'predict'):  # sklearn-like interface
            mlflow.sklearn.log_model(
                model,
                model_name,
                registered_model_name=registered_model_name
            )
        else:
            mlflow.log_artifact(model, model_name)

    def log_config(self, config_path: str):
        mlflow.log_artifact(config_path, "config")

    def save_metrics_to_file(self, metrics: Dict[str, float], output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

def get_or_create_experiment(experiment_name: str) -> str:
    """Get existing experiment or create new one"""
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        logging.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        logging.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
    return experiment_id

class ExperimentLogger:
    """Enhanced experiment logging with crypto data specifics"""

    def __init__(self, experiment_name: str = "info_spillover_experiment"):
        self.tracker = MLFlowTracker(experiment_name)
        self.logger = logging.getLogger(__name__)

    def log_data_info(self, data: pd.DataFrame, stage: str = "raw"):
        """Log dataset information"""
        info = {
            f"{stage}_samples": len(data),
            f"{stage}_features": len(data.columns),
            f"{stage}_memory_mb": data.memory_usage(deep=True).sum() / 1024**2,
        }

        # Add data quality metrics
        if 'subreddit' in data.columns:
            info[f"{stage}_subreddits"] = data['subreddit'].nunique()

        if 'created_utc' in data.columns:
            info[f"{stage}_date_range_days"] = (
                data['created_utc'].max() - data['created_utc'].min()
            ).days

        self.tracker.log_params(info)
        return info

    def log_model_performance(self, y_true, y_pred, stage: str = "validation"):
        """Log model performance metrics"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

        metrics = {
            f"{stage}_accuracy": accuracy_score(y_true, y_pred),
        }

        # Handle binary vs multiclass
        if len(np.unique(y_true)) == 2:
            metrics[f"{stage}_roc_auc"] = roc_auc_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

        metrics[f"{stage}_precision"] = precision
        metrics[f"{stage}_recall"] = recall
        metrics[f"{stage}_f1"] = f1

        self.tracker.log_metrics(metrics)
        return metrics

    def log_crypto_features(self, feature_names: list, importance_scores: Optional[np.ndarray] = None):
        """Log cryptocurrency-specific feature information"""
        self.tracker.log_params({"total_features": len(feature_names)})

        # Categorize features
        sentiment_features = [f for f in feature_names if 'sentiment' in f.lower()]
        temporal_features = [f for f in feature_names if any(t in f.lower() for t in ['time', 'day', 'hour'])]
        text_features = [f for f in feature_names if any(t in f.lower() for t in ['text', 'word', 'token'])]

        categories = {
            "sentiment_features": len(sentiment_features),
            "temporal_features": len(temporal_features),
            "text_features": len(text_features),
        }

        self.tracker.log_params(categories)

        # Log top important features if available
        if importance_scores is not None:
            top_indices = np.argsort(importance_scores)[-10:]
            top_features = [feature_names[i] for i in top_indices]

            for i, (feature, importance) in enumerate(zip(top_features, importance_scores[top_indices])):
                self.tracker.log_metrics({f"feature_importance_top_{i+1}": importance})
                self.tracker.log_params({f"top_feature_{i+1}": feature})

    def log_spillover_analysis(self, spillover_results: Dict[str, Any]):
        """Log information spillover analysis results"""
        if 'transfer_entropy' in spillover_results:
            self.tracker.log_metrics({"avg_transfer_entropy": spillover_results['transfer_entropy']})

        if 'correlation_matrix' in spillover_results:
            corr_matrix = spillover_results['correlation_matrix']
            self.tracker.log_metrics({
                "avg_correlation": np.mean(corr_matrix),
                "max_correlation": np.max(corr_matrix)
            })

        if 'granger_causality' in spillover_results:
            self.tracker.log_metrics({"granger_p_value": spillover_results['granger_causality']['p_value']})

    def create_run_context(self, run_name: str = None, tags: Dict[str, str] = None):
        """Create MLFlow run context manager"""
        return self.tracker.start_run(run_name=run_name, tags=tags)