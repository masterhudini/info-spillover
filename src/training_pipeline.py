#!/usr/bin/env python3
"""
Comprehensive Training Pipeline for Information Spillover Analysis

This pipeline integrates:
1. Data processing and feature engineering
2. Statistical validation with optimized parameters
3. Machine learning models with hyperparameter optimization
4. Hierarchical deep learning models (LSTM + GNN)
5. Spillover analysis with Diebold-Yilmaz methodology
6. MLflow experiment tracking
7. Model evaluation and economic validation

Based on academic methodology and Optuna-optimized hyperparameters.
"""

import os
import sys
import yaml
import json
import pickle
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import argparse

import pandas as pd
import numpy as np
import networkx as nx

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Deep Learning
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

# Project imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.bigquery_client import BigQueryClient
from src.data.hierarchical_data_processor import HierarchicalDataProcessor
from src.models.hierarchical_models import (
    HierarchicalModelBuilder, HierarchicalDataModule,
    SubredditTimeSeriesDataset
)
from src.analysis.diebold_yilmaz_spillover import DieboldYilmazSpillover
from src.utils.statistical_validation import StatisticalValidationFramework
from src.utils.enhanced_mlflow_tracker import EnhancedMLflowTracker
from src.utils.optuna_optimizer import OptunaMLflowOptimizer, OptimizationConfig
from src.evaluation.economic_evaluation import EconomicEvaluator

import mlflow
import mlflow.sklearn
import mlflow.pytorch

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveTrainingPipeline:
    """
    Comprehensive training pipeline integrating all components
    """

    def __init__(self, config_path: str):
        """Initialize pipeline with configuration"""

        self.config_path = config_path
        self.config = self._load_config()

        # Set random seeds for reproducibility
        self._set_random_seeds()

        # Initialize components
        self.bigquery_client = None
        self.data_processor = None
        self.statistical_validator = None
        self.mlflow_tracker = None
        self.optuna_optimizer = None

        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.features_df = None
        self.spillover_data = None
        self.network = None

        # Models storage
        self.ml_models = {}
        self.hierarchical_model = None
        self.spillover_results = {}

        # Results storage
        self.ml_results = {}
        self.statistical_results = {}
        self.economic_results = {}

        logger.info(f"🚀 Initialized Comprehensive Training Pipeline")
        logger.info(f"📋 Configuration: {config_path}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _set_random_seeds(self):
        """Set random seeds for reproducibility"""
        if self.config.get('reproducibility', {}).get('set_seeds', True):
            seed = self.config.get('experiment', {}).get('random_seed', 42)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            logger.info(f"🎯 Set random seeds to {seed}")

    def setup_mlflow_tracking(self):
        """Initialize MLflow tracking"""

        mlflow_config = self.config.get('mlflow', {})

        self.mlflow_tracker = EnhancedMLflowTracker(
            experiment_name=mlflow_config.get('experiment_name', 'comprehensive_training'),
            tracking_uri=mlflow_config.get('tracking_uri', 'sqlite:///comprehensive_training.db')
        )

        # Enable auto-logging
        if mlflow_config.get('autolog', {}).get('sklearn', True):
            mlflow.sklearn.autolog()

        if mlflow_config.get('autolog', {}).get('pytorch', True):
            mlflow.pytorch.autolog()

        logger.info("📊 MLflow tracking initialized")

    def load_data(self) -> pd.DataFrame:
        """Load and prepare data"""

        logger.info("📥 Loading data...")

        data_config = self.config.get('data', {})

        # Initialize BigQuery client if not testing
        try:
            self.bigquery_client = BigQueryClient()

            # Load data from BigQuery
            query = f"""
            SELECT *
            FROM `crypto_sentiment.processed_reddit_data`
            WHERE DATE(created_utc) BETWEEN '{data_config.get("start_date", "2021-01-01")}'
            AND '{data_config.get("end_date", "2023-12-31")}'
            ORDER BY created_utc
            """

            self.raw_data = self.bigquery_client.query_to_dataframe(query)
            logger.info(f"✅ Loaded {len(self.raw_data)} records from BigQuery")

        except Exception as e:
            logger.warning(f"⚠️ BigQuery loading failed: {e}")

            # Fallback: generate synthetic data for testing
            logger.info("🧪 Generating synthetic data for testing...")
            self.raw_data = self._generate_synthetic_data()

        return self.raw_data

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic data for testing"""

        np.random.seed(42)

        # Generate synthetic cryptocurrency sentiment data
        dates = pd.date_range('2023-01-01', periods=1000, freq='H')
        subreddits = ['Bitcoin', 'CryptoCurrency', 'ethereum', 'dogecoin', 'cardano']

        data = []
        for date in dates:
            for subreddit in subreddits:
                record = {
                    'created_utc': date,
                    'subreddit': subreddit,
                    'sentiment_positive': np.random.beta(2, 2),
                    'sentiment_negative': np.random.beta(2, 2),
                    'sentiment_neutral': np.random.beta(3, 2),
                    'sentiment_compound': np.random.normal(0, 0.3),
                    'emotion_joy': np.random.beta(2, 3),
                    'emotion_fear': np.random.beta(2, 3),
                    'emotion_anger': np.random.beta(1.5, 3),
                    'count_posts': np.random.poisson(50),
                    'count_comments': np.random.poisson(200),
                    'avg_score': np.random.normal(10, 5),
                    'price_btc': 30000 + np.random.normal(0, 2000),
                    'volume_btc': np.random.exponential(1000000),
                    'market_cap': np.random.exponential(500000000)
                }
                data.append(record)

        synthetic_df = pd.DataFrame(data)
        synthetic_df['created_utc'] = pd.to_datetime(synthetic_df['created_utc'])

        logger.info(f"🧪 Generated {len(synthetic_df)} synthetic records")
        return synthetic_df

    def process_data(self):
        """Process and engineer features"""

        logger.info("⚙️ Processing data and engineering features...")

        # Initialize hierarchical data processor
        processor_config = {
            'sequence_length': self.config['data']['sequence_length'],
            'prediction_horizon': self.config['data']['prediction_horizon'],
            'scaling_method': self.config['data']['scaling_method'],
            'subreddits': self.raw_data['subreddit'].unique().tolist()
        }

        self.data_processor = HierarchicalDataProcessor(processor_config)

        # Process the data
        processed_results = self.data_processor.process_hierarchical_data(self.raw_data)

        self.processed_data = processed_results['hierarchical_features']
        self.features_df = processed_results['ml_features']
        self.network = processed_results['network']

        logger.info(f"✅ Processed data shapes:")
        logger.info(f"   Hierarchical: {self.processed_data.shape}")
        logger.info(f"   ML Features: {self.features_df.shape}")
        logger.info(f"   Network nodes: {self.network.number_of_nodes()}")

    def validate_statistical_assumptions(self):
        """Validate statistical assumptions with optimized parameters"""

        logger.info("📊 Validating statistical assumptions...")

        # Get optimized statistical parameters
        stat_params = self.config.get('statistical_parameters', {})
        var_params = stat_params.get('var_model', {})
        bootstrap_params = stat_params.get('bootstrap', {})

        # Initialize statistical validator with optimized parameters
        self.statistical_validator = StatisticalValidationFramework(
            significance_level=var_params.get('significance_level', 0.012),
            bootstrap_iterations=bootstrap_params.get('iterations', 1800),
            max_lags=var_params.get('max_lags', 15)
        )

        # Prepare time series data for validation
        ts_data = self._prepare_time_series_data()

        with self.mlflow_tracker:
            # Validate VAR assumptions
            var_results = self.statistical_validator.validate_var_assumptions(
                ts_data, max_lags=var_params.get('max_lags', 15)
            )

            # Test spillover significance
            spillover_results = self.statistical_validator.test_spillover_significance(
                ts_data[['sentiment_compound', 'price_btc']].dropna()
            )

            # Validate ML models
            X = self.features_df.drop(['target'], axis=1, errors='ignore')
            y = self.features_df.get('target', self._create_target())

            ml_models_simple = {
                'RandomForest': RandomForestRegressor(**self.config['ml_models']['random_forest']),
                'Ridge': Ridge(**self.config['ml_models']['ridge'])
            }

            ml_results = self.statistical_validator.validate_ml_models(
                X, y, ml_models_simple, cv_method='time_series'
            )

            # Store results
            self.statistical_results = {
                'var_validation': var_results,
                'spillover_validation': spillover_results,
                'ml_validation': ml_results
            }

            # Generate comprehensive report
            validation_report = self.statistical_validator.generate_validation_report()

            logger.info(f"✅ Statistical validation completed:")
            logger.info(f"   Total tests: {validation_report['summary']['total_tests_performed']}")
            logger.info(f"   Pass rate: {validation_report['summary']['tests_passed'] / max(1, validation_report['summary']['total_tests_performed']):.2%}")

            return validation_report

    def _prepare_time_series_data(self) -> pd.DataFrame:
        """Prepare time series data for statistical validation"""

        # Group by time and aggregate across subreddits
        ts_data = self.raw_data.groupby('created_utc').agg({
            'sentiment_compound': 'mean',
            'sentiment_positive': 'mean',
            'sentiment_negative': 'mean',
            'price_btc': 'first',
            'volume_btc': 'first',
            'count_posts': 'sum',
            'count_comments': 'sum'
        }).fillna(method='ffill').reset_index()

        # Ensure numeric columns
        numeric_columns = ['sentiment_compound', 'sentiment_positive', 'sentiment_negative',
                          'price_btc', 'volume_btc', 'count_posts', 'count_comments']
        for col in numeric_columns:
            if col in ts_data.columns:
                ts_data[col] = pd.to_numeric(ts_data[col], errors='coerce')

        return ts_data.dropna()

    def _create_target(self) -> pd.Series:
        """Create target variable for ML models"""

        if 'price_btc' in self.features_df.columns:
            # Create price direction target
            price_returns = self.features_df['price_btc'].pct_change()
            target = (price_returns > 0).astype(int)  # 1 for price increase, 0 for decrease
        else:
            # Synthetic target for testing
            target = np.random.randint(0, 2, len(self.features_df))

        return pd.Series(target, index=self.features_df.index)

    def optimize_hyperparameters(self):
        """Optimize hyperparameters using Optuna"""

        logger.info("🎯 Optimizing hyperparameters with Optuna...")

        optimization_config = self.config.get('optimization', {})
        optuna_config = optimization_config.get('optuna', {})

        # Initialize Optuna optimizer
        config = OptimizationConfig(
            study_name="comprehensive_optimization",
            storage_url=None,  # Use in-memory storage
            n_trials=optuna_config.get('n_trials', 50),
            sampler_name=optuna_config.get('sampler', 'TPE'),
            pruner_name=optuna_config.get('pruner', 'Median'),
            direction=optuna_config.get('direction', 'maximize')
        )

        self.optuna_optimizer = OptunaMLflowOptimizer(
            config=config,
            mlflow_tracking_uri=self.mlflow_tracker.tracking_uri
        )

        # Prepare data for optimization
        X = self.features_df.drop(['target'], axis=1, errors='ignore')
        y = self.features_df.get('target', self._create_target())

        # Clean data
        finite_mask = np.isfinite(X.values).all(axis=1) & np.isfinite(y.values)
        X_clean = X[finite_mask]
        y_clean = y[finite_mask]

        if len(X_clean) > 50:  # Ensure sufficient data
            try:
                # Optimize ML models
                ml_optimization_result = self.optuna_optimizer.optimize_ml_models(
                    X_clean, y_clean,
                    model_types=['RandomForest', 'GradientBoosting']
                )

                logger.info(f"✅ ML optimization completed: best score {ml_optimization_result.best_score:.4f}")

                # Optimize statistical parameters
                ts_data = self._prepare_time_series_data()
                if len(ts_data) > 100:
                    statistical_optimization_result = self.optuna_optimizer.optimize_statistical_parameters(ts_data)
                    logger.info(f"✅ Statistical optimization completed: best score {statistical_optimization_result.best_score:.4f}")

                # Optimize spillover parameters
                spillover_data = ts_data[['sentiment_compound', 'price_btc']].dropna()
                if len(spillover_data) > 50:
                    spillover_optimization_result = self.optuna_optimizer.optimize_spillover_parameters(spillover_data)
                    logger.info(f"✅ Spillover optimization completed: best score {spillover_optimization_result.best_score:.4f}")

            except Exception as e:
                logger.warning(f"⚠️ Hyperparameter optimization failed: {e}")

        logger.info("🎯 Hyperparameter optimization phase completed")

    def train_ml_models(self):
        """Train machine learning models with optimized hyperparameters"""

        logger.info("🤖 Training machine learning models...")

        # Prepare data
        X = self.features_df.drop(['target'], axis=1, errors='ignore')
        y = self.features_df.get('target', self._create_target())

        # Clean data
        finite_mask = np.isfinite(X.values).all(axis=1) & np.isfinite(y.values)
        X_clean = X[finite_mask].fillna(0)
        y_clean = y[finite_mask]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        # Train models with optimized hyperparameters
        ml_config = self.config.get('ml_models', {})

        models_to_train = {
            'RandomForest': RandomForestRegressor(**ml_config.get('random_forest', {})),
            'GradientBoosting': GradientBoostingRegressor(**ml_config.get('gradient_boosting', {})),
            'SVR': SVR(**ml_config.get('svm', {})),
            'Ridge': Ridge(**ml_config.get('ridge', {}))
        }

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.get('cross_validation', {}).get('n_splits', 5))

        with self.mlflow_tracker:
            for name, model in models_to_train.items():
                logger.info(f"   Training {name}...")

                try:
                    # Train model
                    model.fit(X_scaled, y_clean)

                    # Cross-validation scores
                    cv_scores = cross_val_score(model, X_scaled, y_clean, cv=tscv, scoring='r2')

                    # Store model and results
                    self.ml_models[name] = {
                        'model': model,
                        'scaler': scaler,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'feature_columns': X_clean.columns.tolist()
                    }

                    # Log to MLflow
                    with mlflow.start_run(run_name=f"ml_model_{name}"):
                        mlflow.log_params(model.get_params())
                        mlflow.log_metric(f"{name}_cv_mean", cv_scores.mean())
                        mlflow.log_metric(f"{name}_cv_std", cv_scores.std())

                        if hasattr(model, 'feature_importances_'):
                            # Log feature importance
                            feature_importance = pd.DataFrame({
                                'feature': X_clean.columns,
                                'importance': model.feature_importances_
                            }).sort_values('importance', ascending=False)

                            mlflow.log_text(feature_importance.to_string(), f"{name}_feature_importance.txt")

                    logger.info(f"   ✅ {name}: CV Score = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

                except Exception as e:
                    logger.error(f"   ❌ {name} training failed: {e}")

        logger.info(f"✅ Trained {len(self.ml_models)} ML models")

    def train_hierarchical_models(self):
        """Train hierarchical deep learning models"""

        logger.info("🧠 Training hierarchical deep learning models...")

        try:
            # Configuration
            hierarchical_config = self.config.get('hierarchical_models', {})

            # Prepare datasets for each subreddit
            subreddits = self.processed_data['subreddit'].unique()
            train_datasets = {}

            # Feature and target columns
            feature_columns = [col for col in self.processed_data.columns
                              if any(pattern in col for pattern in
                              ['sentiment_', 'emotion_', 'count_', 'price_', 'volume_'])]

            target_columns = ['sentiment_compound']  # Simplified for this example

            for subreddit in subreddits:
                if len(self.processed_data[self.processed_data['subreddit'] == subreddit]) > 50:
                    dataset = SubredditTimeSeriesDataset(
                        df=self.processed_data,
                        subreddit=subreddit,
                        sequence_length=self.config['data']['sequence_length'],
                        prediction_horizon=self.config['data']['prediction_horizon'],
                        feature_columns=feature_columns,
                        target_columns=target_columns
                    )
                    train_datasets[subreddit] = dataset

            if len(train_datasets) > 0:
                # Build hierarchical model
                model_builder = HierarchicalModelBuilder(hierarchical_config)

                # Get feature dimension from first dataset
                first_dataset = list(train_datasets.values())[0]
                node_features = first_dataset.sequences.shape[-1]

                self.hierarchical_model = model_builder.build_hierarchical_model(
                    train_datasets, node_features
                )

                # Setup data module
                data_module = HierarchicalDataModule(
                    processed_data=self.processed_data,
                    network=self.network,
                    config=hierarchical_config
                )

                # Setup trainer
                mlflow_logger = MLFlowLogger(
                    experiment_name=self.mlflow_tracker.experiment_name,
                    tracking_uri=self.mlflow_tracker.tracking_uri
                )

                trainer = pl.Trainer(
                    max_epochs=hierarchical_config.get('training', {}).get('max_epochs', 50),
                    logger=mlflow_logger,
                    callbacks=[
                        EarlyStopping(
                            monitor='val_loss',
                            patience=hierarchical_config.get('training', {}).get('patience', 10)
                        ),
                        ModelCheckpoint(
                            monitor='val_loss',
                            mode='min',
                            save_top_k=1
                        )
                    ],
                    accelerator='auto'
                )

                # Train model
                trainer.fit(self.hierarchical_model, data_module)

                logger.info("✅ Hierarchical model training completed")

            else:
                logger.warning("⚠️ Insufficient data for hierarchical model training")

        except Exception as e:
            logger.error(f"❌ Hierarchical model training failed: {e}")

    def analyze_spillovers(self):
        """Perform spillover analysis using Diebold-Yilmaz methodology"""

        logger.info("🌊 Performing spillover analysis...")

        try:
            # Get optimized spillover parameters
            spillover_params = self.config.get('spillover_parameters', {})
            dy_params = spillover_params.get('diebold_yilmaz', {})

            # Prepare spillover data
            spillover_analyzer = DieboldYilmazSpillover(
                forecast_horizon=dy_params.get('forecast_horizon', 12),
                var_lags=dy_params.get('var_lags', 9),
                rolling_window=dy_params.get('rolling_window', 65)
            )

            # Prepare time series data
            ts_data = self._prepare_time_series_data()

            # Select variables for spillover analysis
            spillover_variables = ['sentiment_compound', 'sentiment_positive', 'sentiment_negative', 'price_btc']
            spillover_variables = [col for col in spillover_variables if col in ts_data.columns]

            if len(spillover_variables) >= 2 and len(ts_data) > 100:
                spillover_data = ts_data[spillover_variables].dropna()

                with self.mlflow_tracker:
                    # Compute spillover indices
                    spillover_results = spillover_analyzer.compute_spillover_indices(spillover_data)

                    # Compute rolling spillovers
                    rolling_spillovers = spillover_analyzer.compute_rolling_spillovers(spillover_data)

                    # Network analysis
                    spillover_network = spillover_analyzer.build_spillover_network(
                        spillover_results['spillover_table']
                    )

                    # Store results
                    self.spillover_results = {
                        'spillover_indices': spillover_results,
                        'rolling_spillovers': rolling_spillovers,
                        'network': spillover_network
                    }

                    # Log key metrics
                    total_spillover = spillover_results.get('total_spillover_index', 0)
                    mlflow.log_metric('total_spillover_index', total_spillover)

                    logger.info(f"✅ Spillover analysis completed: Total spillover = {total_spillover:.4f}")

            else:
                logger.warning("⚠️ Insufficient data for spillover analysis")

        except Exception as e:
            logger.error(f"❌ Spillover analysis failed: {e}")

    def evaluate_models(self):
        """Comprehensive model evaluation"""

        logger.info("📊 Evaluating models...")

        # Prepare evaluation data
        X = self.features_df.drop(['target'], axis=1, errors='ignore')
        y = self.features_df.get('target', self._create_target())

        # Clean data
        finite_mask = np.isfinite(X.values).all(axis=1) & np.isfinite(y.values)
        X_clean = X[finite_mask].fillna(0)
        y_clean = y[finite_mask]

        # Split data for evaluation
        split_idx = int(0.8 * len(X_clean))
        X_train, X_test = X_clean[:split_idx], X_clean[split_idx:]
        y_train, y_test = y_clean[:split_idx], y_clean[split_idx:]

        evaluation_results = {}

        with self.mlflow_tracker:
            # Evaluate ML models
            for name, model_info in self.ml_models.items():
                try:
                    model = model_info['model']
                    scaler = model_info['scaler']

                    # Make predictions
                    X_test_scaled = scaler.transform(X_test)
                    y_pred = model.predict(X_test_scaled)

                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    evaluation_results[name] = {
                        'mse': mse,
                        'mae': mae,
                        'r2': r2,
                        'cv_mean': model_info.get('cv_mean', 0),
                        'cv_std': model_info.get('cv_std', 0)
                    }

                    # Log to MLflow
                    with mlflow.start_run(run_name=f"evaluation_{name}"):
                        mlflow.log_metric(f"{name}_test_mse", mse)
                        mlflow.log_metric(f"{name}_test_mae", mae)
                        mlflow.log_metric(f"{name}_test_r2", r2)

                    logger.info(f"   {name}: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

                except Exception as e:
                    logger.error(f"   ❌ {name} evaluation failed: {e}")

            # Economic evaluation if available
            try:
                economic_evaluator = EconomicEvaluator()

                # Prepare price and prediction data
                if 'price_btc' in self.features_df.columns and len(self.ml_models) > 0:
                    prices = self.features_df['price_btc'].iloc[split_idx:].values

                    # Use best performing model
                    best_model_name = max(evaluation_results.keys(),
                                        key=lambda k: evaluation_results[k]['r2'])
                    best_model_info = self.ml_models[best_model_name]

                    # Generate predictions
                    model = best_model_info['model']
                    scaler = best_model_info['scaler']
                    X_test_scaled = scaler.transform(X_test)
                    predictions = model.predict(X_test_scaled)

                    # Economic evaluation
                    economic_results = economic_evaluator.evaluate_trading_strategy(
                        prices=prices,
                        predictions=predictions,
                        transaction_cost=0.001,
                        initial_capital=10000
                    )

                    self.economic_results = economic_results

                    # Log economic metrics
                    for metric, value in economic_results.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"economic_{metric}", value)

                    logger.info(f"✅ Economic evaluation completed: Sharpe ratio = {economic_results.get('sharpe_ratio', 0):.4f}")

            except Exception as e:
                logger.warning(f"⚠️ Economic evaluation failed: {e}")

        # Store evaluation results
        self.ml_results = evaluation_results

        logger.info(f"✅ Model evaluation completed for {len(evaluation_results)} models")

    def save_results(self):
        """Save all pipeline results"""

        logger.info("💾 Saving pipeline results...")

        # Create output directories
        output_config = self.config.get('outputs', {})

        models_dir = Path(output_config.get('models', 'models/comprehensive'))
        metrics_dir = Path(output_config.get('metrics', 'experiments/outputs/comprehensive'))
        artifacts_dir = Path(output_config.get('artifacts', 'experiments/artifacts/comprehensive'))

        for directory in [models_dir, metrics_dir, artifacts_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Save ML models
            for name, model_info in self.ml_models.items():
                model_path = models_dir / f"{name}_model_{timestamp}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model_info, f)
                logger.info(f"   ✅ Saved {name} model to {model_path}")

            # Save hierarchical model if available
            if self.hierarchical_model is not None:
                hierarchical_path = models_dir / f"hierarchical_model_{timestamp}.pt"
                torch.save(self.hierarchical_model.state_dict(), hierarchical_path)
                logger.info(f"   ✅ Saved hierarchical model to {hierarchical_path}")

            # Save evaluation results
            results_summary = {
                'timestamp': timestamp,
                'config': self.config,
                'ml_results': self.ml_results,
                'statistical_results': self.statistical_results,
                'spillover_results': self.spillover_results,
                'economic_results': self.economic_results
            }

            results_path = metrics_dir / f"pipeline_results_{timestamp}.json"
            with open(results_path, 'w') as f:
                json.dump(results_summary, f, indent=2, default=str)
            logger.info(f"   ✅ Saved results summary to {results_path}")

            # Save processed data
            if self.features_df is not None:
                features_path = artifacts_dir / f"features_{timestamp}.parquet"
                self.features_df.to_parquet(features_path)
                logger.info(f"   ✅ Saved features to {features_path}")

            # Save network if available
            if self.network is not None:
                network_path = artifacts_dir / f"network_{timestamp}.gml"
                nx.write_gml(self.network, network_path)
                logger.info(f"   ✅ Saved network to {network_path}")

        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")

    def run_comprehensive_pipeline(self):
        """Run the complete training pipeline"""

        logger.info("🚀 Starting Comprehensive Training Pipeline")
        logger.info("=" * 80)

        try:
            # Setup tracking
            self.setup_mlflow_tracking()

            # Data loading and processing
            self.load_data()
            self.process_data()

            # Statistical validation
            self.validate_statistical_assumptions()

            # Hyperparameter optimization
            self.optimize_hyperparameters()

            # Model training
            self.train_ml_models()
            self.train_hierarchical_models()

            # Analysis
            self.analyze_spillovers()

            # Evaluation
            self.evaluate_models()

            # Save results
            self.save_results()

            logger.info("=" * 80)
            logger.info("🎉 Comprehensive Training Pipeline Completed Successfully!")
            logger.info("=" * 80)

            # Print summary
            self._print_pipeline_summary()

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise

    def _print_pipeline_summary(self):
        """Print pipeline execution summary"""

        print("\n" + "=" * 60)
        print("🎯 PIPELINE EXECUTION SUMMARY")
        print("=" * 60)

        print(f"📊 Data Processing:")
        if self.raw_data is not None:
            print(f"   • Raw data records: {len(self.raw_data):,}")
        if self.processed_data is not None:
            print(f"   • Processed features: {self.processed_data.shape}")
        if self.features_df is not None:
            print(f"   • ML features: {self.features_df.shape}")

        print(f"\n🤖 Machine Learning Models:")
        for name, results in self.ml_results.items():
            r2_score = results.get('r2', 0)
            print(f"   • {name}: R² = {r2_score:.4f}")

        print(f"\n📈 Statistical Validation:")
        if self.statistical_results:
            print(f"   • VAR validation: {len(self.statistical_results.get('var_validation', {}))} tests")
            print(f"   • Spillover validation: {len(self.statistical_results.get('spillover_validation', {}))} tests")
            print(f"   • ML validation: {len(self.statistical_results.get('ml_validation', {}))} tests")

        print(f"\n🌊 Spillover Analysis:")
        if self.spillover_results:
            total_spillover = self.spillover_results.get('spillover_indices', {}).get('total_spillover_index', 0)
            print(f"   • Total spillover index: {total_spillover:.4f}")

        print(f"\n💰 Economic Results:")
        if self.economic_results:
            sharpe = self.economic_results.get('sharpe_ratio', 0)
            returns = self.economic_results.get('total_return', 0)
            print(f"   • Sharpe ratio: {sharpe:.4f}")
            print(f"   • Total return: {returns:.2%}")

        print("\n✅ Pipeline completed successfully!")
        print(f"🔗 MLflow UI: http://localhost:5000")
        print("=" * 60)


def main():
    """Main function to run the comprehensive training pipeline"""

    parser = argparse.ArgumentParser(description="Comprehensive Training Pipeline")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Initialize and run pipeline
    pipeline = ComprehensiveTrainingPipeline(args.config)
    pipeline.run_comprehensive_pipeline()


if __name__ == "__main__":
    main()