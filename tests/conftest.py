"""
Pytest configuration and fixtures for info_spillover tests
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import os

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)

@pytest.fixture
def sample_crypto_data():
    """Generate sample cryptocurrency discussion data"""
    np.random.seed(42)
    n_samples = 100

    data = {
        'subreddit': np.random.choice(['Bitcoin', 'ethereum', 'CryptoCurrency'], n_samples),
        'post_score': np.random.poisson(10, n_samples),
        'num_comments': np.random.poisson(5, n_samples),
        'sentiment_score': np.random.normal(0, 1, n_samples),
        'text_length': np.random.lognormal(5, 1, n_samples),
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'author_karma': np.random.exponential(1000, n_samples),
        'is_weekend': np.random.randint(0, 2, n_samples),
        'btc_price_change': np.random.normal(0, 0.05, n_samples),
        'market_volatility': np.random.exponential(0.02, n_samples),
        'spillover_target': np.random.binomial(1, 0.3, n_samples)
    }

    return pd.DataFrame(data)

@pytest.fixture
def mlflow_tracking_uri(temp_dir):
    """Set up temporary MLFlow tracking for tests"""
    tracking_uri = f"sqlite:///{temp_dir}/test_mlflow.db"
    os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
    mlflow.set_tracking_uri(tracking_uri)

    # Create test experiment
    try:
        mlflow.create_experiment("test_experiment")
    except mlflow.exceptions.MlflowException:
        pass  # Experiment already exists

    mlflow.set_experiment("test_experiment")
    yield tracking_uri

    # Cleanup
    mlflow.end_run()

@pytest.fixture
def feature_columns():
    """Standard feature columns for testing"""
    return [
        'post_score', 'num_comments', 'sentiment_score', 'text_length',
        'hour_of_day', 'day_of_week', 'author_karma', 'is_weekend',
        'btc_price_change', 'market_volatility'
    ]

@pytest.fixture(scope="session")
def test_config():
    """Test configuration dictionary"""
    return {
        'experiment': {
            'name': 'test_experiment',
            'random_seed': 42
        },
        'model': {
            'name': 'random_forest',
            'n_estimators': 10,
            'max_depth': 3,
            'random_state': 42
        },
        'data': {
            'test_size': 0.2,
            'val_size': 0.25
        }
    }