#!/usr/bin/env python3
"""
Test core training components without deep learning dependencies

Tests the essential training pipeline components:
1. Data processing
2. Statistical validation with optimized parameters
3. ML models with optimized hyperparameters
4. Optuna optimization
5. MLflow tracking
"""

import sys
import os
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Add project root to path
sys.path.append('/home/Hudini/projects/info_spillover')

from src.utils.statistical_validation import StatisticalValidationFramework
from src.utils.enhanced_mlflow_tracker import EnhancedMLflowTracker
from src.utils.optuna_optimizer import OptunaMLflowOptimizer, OptimizationConfig

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoreTrainingPipeline:
    """Core training pipeline with optimized hyperparameters"""

    def __init__(self, config_path: str):
        """Initialize core pipeline"""

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set random seed
        np.random.seed(self.config.get('experiment', {}).get('random_seed', 42))

        # Initialize components
        self.mlflow_tracker = None
        self.statistical_validator = None
        self.optuna_optimizer = None

        # Data and results
        self.data = None
        self.features_df = None
        self.models = {}
        self.results = {}

        logger.info("🚀 Core Training Pipeline initialized")

    def setup_tracking(self):
        """Setup MLflow tracking"""

        mlflow_config = self.config.get('mlflow', {})

        self.mlflow_tracker = EnhancedMLflowTracker(
            experiment_name=mlflow_config.get('experiment_name', 'core_training'),
            tracking_uri=mlflow_config.get('tracking_uri', 'sqlite:///core_training.db')
        )

        logger.info("📊 MLflow tracking setup complete")

    def generate_data(self):
        """Generate synthetic dataset for testing"""

        logger.info("🧪 Generating synthetic dataset...")

        # Generate time series data
        np.random.seed(42)
        n_samples = 1000
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='H')

        # Create features with realistic cryptocurrency patterns
        data = {
            'timestamp': dates,
            'sentiment_compound': np.random.normal(0, 0.3, n_samples),
            'sentiment_positive': np.random.beta(2, 2, n_samples),
            'sentiment_negative': np.random.beta(2, 2, n_samples),
            'sentiment_neutral': np.random.beta(3, 2, n_samples),
            'volume': np.random.exponential(1000, n_samples),
            'price': 30000 + np.cumsum(np.random.normal(0, 100, n_samples)),
            'volatility': np.abs(np.random.normal(0.02, 0.01, n_samples)),
            'posts_count': np.random.poisson(20, n_samples),
            'comments_count': np.random.poisson(50, n_samples)
        }

        self.data = pd.DataFrame(data)

        # Engineer features
        self.data['price_returns'] = self.data['price'].pct_change()
        self.data['price_sma_5'] = self.data['price'].rolling(5).mean()
        self.data['volume_sma_5'] = self.data['volume'].rolling(5).mean()
        self.data['sentiment_ma_3'] = self.data['sentiment_compound'].rolling(3).mean()

        # Create target (price direction)
        self.data['target'] = (self.data['price_returns'] > 0).astype(int)

        # Prepare features DataFrame
        feature_columns = [
            'sentiment_compound', 'sentiment_positive', 'sentiment_negative',
            'volume', 'volatility', 'posts_count', 'comments_count',
            'price_sma_5', 'volume_sma_5', 'sentiment_ma_3'
        ]

        self.features_df = self.data[feature_columns + ['target']].dropna()

        logger.info(f"✅ Generated dataset: {self.features_df.shape}")

    def validate_statistical_assumptions(self):
        """Test statistical validation with optimized parameters"""

        logger.info("📊 Testing statistical validation...")

        # Get optimized parameters
        stat_params = self.config.get('statistical_parameters', {})
        var_params = stat_params.get('var_model', {})
        bootstrap_params = stat_params.get('bootstrap', {})

        # Initialize with optimized parameters
        self.statistical_validator = StatisticalValidationFramework(
            significance_level=var_params.get('significance_level', 0.012),
            bootstrap_iterations=bootstrap_params.get('iterations', 1800)
        )

        # Prepare time series data
        ts_data = self.data[['sentiment_compound', 'price', 'volume']].dropna()

        with self.mlflow_tracker:
            # VAR validation
            var_results = self.statistical_validator.validate_var_assumptions(
                ts_data, max_lags=var_params.get('max_lags', 15)
            )

            # Spillover testing
            spillover_results = self.statistical_validator.test_spillover_significance(
                ts_data[['sentiment_compound', 'price']]
            )

            # ML validation
            X = self.features_df.drop('target', axis=1)
            y = self.features_df['target']

            ml_models = {
                'RandomForest': RandomForestRegressor(**self.config['ml_models']['random_forest']),
                'Ridge': Ridge(**self.config['ml_models']['ridge'])
            }

            ml_results = self.statistical_validator.validate_ml_models(
                X, y, ml_models, cv_method='time_series'
            )

            # Generate report
            validation_report = self.statistical_validator.generate_validation_report()

            total_tests = validation_report['summary']['total_tests_performed']
            pass_rate = validation_report['summary']['tests_passed'] / max(1, total_tests)

            logger.info(f"✅ Statistical validation: {total_tests} tests, {pass_rate:.2%} pass rate")

            return validation_report

    def optimize_hyperparameters(self):
        """Test hyperparameter optimization"""

        logger.info("🎯 Testing Optuna optimization...")

        # Setup optimizer
        config = OptimizationConfig(
            study_name="core_optimization_test",
            storage_url=None,  # In-memory
            n_trials=20,  # Reduced for testing
            sampler_name="TPE",
            pruner_name="Median",
            direction="maximize"
        )

        self.optuna_optimizer = OptunaMLflowOptimizer(
            config=config,
            mlflow_tracking_uri=self.mlflow_tracker.tracking_uri
        )

        # Prepare data
        X = self.features_df.drop('target', axis=1)
        y = self.features_df['target']

        # Clean data
        finite_mask = np.isfinite(X.values).all(axis=1) & np.isfinite(y.values)
        X_clean = X[finite_mask]
        y_clean = y[finite_mask]

        try:
            # Test ML optimization
            ml_result = self.optuna_optimizer.optimize_ml_models(
                X_clean, y_clean, model_types=['RandomForest']
            )

            logger.info(f"✅ ML optimization: best score {ml_result.best_score:.4f}")

            # Test statistical optimization
            ts_data = self.data[['sentiment_compound', 'price', 'volume']].dropna()

            if len(ts_data) > 100:
                stat_result = self.optuna_optimizer.optimize_statistical_parameters(ts_data)
                logger.info(f"✅ Statistical optimization: best score {stat_result.best_score:.4f}")

        except Exception as e:
            logger.warning(f"⚠️ Optimization test failed: {e}")

    def train_optimized_models(self):
        """Train models with optimized hyperparameters"""

        logger.info("🤖 Training optimized ML models...")

        # Prepare data
        X = self.features_df.drop('target', axis=1)
        y = self.features_df['target']

        # Clean and scale
        finite_mask = np.isfinite(X.values).all(axis=1) & np.isfinite(y.values)
        X_clean = X[finite_mask].fillna(0)
        y_clean = y[finite_mask]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        # Get optimized hyperparameters
        ml_config = self.config.get('ml_models', {})

        models_config = {
            'RandomForest': ml_config.get('random_forest', {}),
            'GradientBoosting': ml_config.get('gradient_boosting', {}),
            'SVR': ml_config.get('svm', {}),
            'Ridge': ml_config.get('ridge', {})
        }

        # Cross-validation setup
        tscv = TimeSeriesSplit(n_splits=5)

        with self.mlflow_tracker:
            for name, params in models_config.items():
                logger.info(f"   Training {name}...")

                try:
                    # Initialize model
                    if name == 'RandomForest':
                        model = RandomForestRegressor(**params)
                    elif name == 'GradientBoosting':
                        model = GradientBoostingRegressor(**params)
                    elif name == 'SVR':
                        model = SVR(**params)
                    elif name == 'Ridge':
                        model = Ridge(**params)

                    # Train and evaluate
                    model.fit(X_scaled, y_clean)

                    # Cross-validation
                    cv_scores = cross_val_score(model, X_scaled, y_clean, cv=tscv, scoring='r2')

                    # Store results
                    self.models[name] = {
                        'model': model,
                        'scaler': scaler,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'params': params
                    }

                    logger.info(f"   ✅ {name}: CV Score = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

                except Exception as e:
                    logger.error(f"   ❌ {name} failed: {e}")

    def evaluate_models(self):
        """Evaluate trained models"""

        logger.info("📊 Evaluating models...")

        # Prepare test data
        X = self.features_df.drop('target', axis=1)
        y = self.features_df['target']

        finite_mask = np.isfinite(X.values).all(axis=1) & np.isfinite(y.values)
        X_clean = X[finite_mask].fillna(0)
        y_clean = y[finite_mask]

        # Split for evaluation
        split_idx = int(0.8 * len(X_clean))
        X_test = X_clean[split_idx:]
        y_test = y_clean[split_idx:]

        evaluation_results = {}

        for name, model_info in self.models.items():
            try:
                model = model_info['model']
                scaler = model_info['scaler']

                X_test_scaled = scaler.transform(X_test)
                y_pred = model.predict(X_test_scaled)

                # Metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                evaluation_results[name] = {
                    'mse': mse,
                    'r2': r2,
                    'cv_mean': model_info['cv_mean'],
                    'cv_std': model_info['cv_std']
                }

                logger.info(f"   {name}: R² = {r2:.4f}, MSE = {mse:.4f}")

            except Exception as e:
                logger.error(f"   ❌ {name} evaluation failed: {e}")

        self.results = evaluation_results
        return evaluation_results

    def save_results(self):
        """Save models and results"""

        logger.info("💾 Saving results...")

        # Create output directory
        output_dir = Path("experiments/outputs/core_training")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save models
        for name, model_info in self.models.items():
            model_path = output_dir / f"{name}_optimized.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model_info, f)
            logger.info(f"   ✅ Saved {name} to {model_path}")

        # Save results summary
        results_summary = {
            'config': self.config,
            'evaluation_results': self.results,
            'dataset_shape': self.features_df.shape,
            'feature_columns': self.features_df.drop('target', axis=1).columns.tolist()
        }

        results_path = output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)

        logger.info(f"   ✅ Saved results to {results_path}")

    def run_core_pipeline(self):
        """Run the core training pipeline"""

        logger.info("🚀 STARTING CORE TRAINING PIPELINE")
        logger.info("="*60)

        try:
            # Setup
            self.setup_tracking()

            # Data preparation
            self.generate_data()

            # Statistical validation
            self.validate_statistical_assumptions()

            # Hyperparameter optimization
            self.optimize_hyperparameters()

            # Model training
            self.train_optimized_models()

            # Evaluation
            self.evaluate_models()

            # Save results
            self.save_results()

            logger.info("="*60)
            logger.info("🎉 CORE TRAINING PIPELINE COMPLETED!")
            logger.info("="*60)

            # Print summary
            self.print_summary()

        except Exception as e:
            logger.error(f"❌ Core pipeline failed: {e}")
            raise

    def print_summary(self):
        """Print execution summary"""

        print("\n" + "="*50)
        print("🎯 CORE TRAINING PIPELINE SUMMARY")
        print("="*50)

        print(f"📊 Dataset: {self.features_df.shape}")
        print(f"🧪 Features: {len(self.features_df.columns) - 1}")

        print(f"\n🤖 Trained Models:")
        best_model = ""
        best_score = -1

        for name, results in self.results.items():
            r2 = results['r2']
            cv_score = results['cv_mean']
            print(f"   • {name}: R² = {r2:.4f}, CV = {cv_score:.4f}")

            if r2 > best_score:
                best_score = r2
                best_model = name

        if best_model:
            print(f"\n🏆 Best Model: {best_model} (R² = {best_score:.4f})")

        print(f"\n✅ Training completed with optimized hyperparameters!")
        print("="*50)


def main():
    """Run core training pipeline test"""

    print("🧪 TESTING CORE TRAINING PIPELINE")
    print("="*60)

    config_path = "/home/Hudini/projects/info_spillover/experiments/configs/optimized_hyperparameters.yaml"

    try:
        pipeline = CoreTrainingPipeline(config_path)
        pipeline.run_core_pipeline()

        print("\n🎉 Core training pipeline test completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()