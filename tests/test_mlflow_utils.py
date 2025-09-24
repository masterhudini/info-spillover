"""
Tests for MLFlow utilities
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.utils.mlflow_utils import MLFlowTracker, ExperimentLogger


class TestMLFlowTracker:
    """Test MLFlowTracker functionality"""

    def test_init_tracker(self, mlflow_tracking_uri):
        """Test tracker initialization"""
        tracker = MLFlowTracker("test_experiment")
        assert tracker.experiment_name == "test_experiment"

    def test_log_params(self, mlflow_tracking_uri):
        """Test parameter logging"""
        tracker = MLFlowTracker("test_experiment")

        with tracker.start_run():
            test_params = {
                "learning_rate": 0.01,
                "n_estimators": 100,
                "model_type": "random_forest"
            }
            tracker.log_params(test_params)

    def test_log_metrics(self, mlflow_tracking_uri):
        """Test metrics logging"""
        tracker = MLFlowTracker("test_experiment")

        with tracker.start_run():
            test_metrics = {
                "accuracy": 0.95,
                "precision": 0.89,
                "recall": 0.92
            }
            tracker.log_metrics(test_metrics)

    def test_log_model(self, mlflow_tracking_uri, sample_crypto_data, feature_columns):
        """Test model logging"""
        tracker = MLFlowTracker("test_experiment")

        # Train a simple model
        X = sample_crypto_data[feature_columns]
        y = sample_crypto_data['spillover_target']
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        with tracker.start_run():
            tracker.log_model(model, "test_model")


class TestExperimentLogger:
    """Test ExperimentLogger functionality"""

    def test_init_logger(self, mlflow_tracking_uri):
        """Test logger initialization"""
        logger = ExperimentLogger("test_experiment")
        assert logger.tracker.experiment_name == "test_experiment"

    def test_log_data_info(self, mlflow_tracking_uri, sample_crypto_data):
        """Test data information logging"""
        logger = ExperimentLogger("test_experiment")

        with logger.create_run_context():
            info = logger.log_data_info(sample_crypto_data, "test_data")

            assert "test_data_samples" in info
            assert info["test_data_samples"] == len(sample_crypto_data)
            assert "test_data_features" in info

    def test_log_model_performance(self, mlflow_tracking_uri, sample_crypto_data, feature_columns):
        """Test model performance logging"""
        logger = ExperimentLogger("test_experiment")

        # Generate predictions
        X = sample_crypto_data[feature_columns]
        y = sample_crypto_data['spillover_target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        with logger.create_run_context():
            metrics = logger.log_model_performance(y_test, y_pred, "test")

            assert "test_accuracy" in metrics
            assert "test_precision" in metrics
            assert "test_recall" in metrics
            assert "test_f1" in metrics
            assert 0 <= metrics["test_accuracy"] <= 1

    def test_log_crypto_features(self, mlflow_tracking_uri, feature_columns):
        """Test crypto feature logging"""
        logger = ExperimentLogger("test_experiment")

        # Create mock feature importance
        importance_scores = np.random.uniform(0, 1, len(feature_columns))

        with logger.create_run_context():
            logger.log_crypto_features(feature_columns, importance_scores)

    def test_log_spillover_analysis(self, mlflow_tracking_uri):
        """Test spillover analysis logging"""
        logger = ExperimentLogger("test_experiment")

        spillover_results = {
            'transfer_entropy': 0.75,
            'correlation_matrix': np.random.uniform(-0.5, 0.8, (5, 5)),
            'granger_causality': {'p_value': 0.05}
        }

        with logger.create_run_context():
            logger.log_spillover_analysis(spillover_results)

    def test_full_experiment_workflow(self, mlflow_tracking_uri, sample_crypto_data, feature_columns):
        """Test complete experiment logging workflow"""
        logger = ExperimentLogger("test_experiment")

        with logger.create_run_context(run_name="full_test_workflow"):
            # Log data info
            data_info = logger.log_data_info(sample_crypto_data, "raw")
            assert data_info["raw_samples"] > 0

            # Train model
            X = sample_crypto_data[feature_columns]
            y = sample_crypto_data['spillover_target']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            model = RandomForestClassifier(n_estimators=5, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Log performance
            metrics = logger.log_model_performance(y_test, y_pred, "test")
            assert "test_accuracy" in metrics

            # Log features
            logger.log_crypto_features(feature_columns, model.feature_importances_)

            # Log spillover analysis
            spillover_results = {
                'transfer_entropy': 0.65,
                'correlation_matrix': np.corrcoef(X.T),
                'granger_causality': {'p_value': 0.03}
            }
            logger.log_spillover_analysis(spillover_results)