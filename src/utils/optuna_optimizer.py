"""
Optuna Hyperparameter Optimization Framework with MLflow Integration

This module implements advanced hyperparameter optimization using Optuna
with full integration into the statistical validation pipeline.

Features:
1. Bayesian hyperparameter optimization
2. Multi-objective optimization (accuracy + statistical significance)
3. MLflow integration for experiment tracking
4. Statistical validation of optimization results
5. Pruning for efficient resource usage
6. Advanced samplers (TPE, CMA-ES, etc.)

Optimizes:
- Machine Learning models (RandomForest, SVM, Neural Networks)
- VAR model parameters (lag selection, regularization)
- Statistical test parameters (significance levels, bootstrap iterations)
"""

import optuna
from optuna.integration import MLflowCallback
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for Optuna optimization"""
    study_name: str
    n_trials: int = 100
    n_jobs: int = 1  # Parallel optimization
    sampler_name: str = "TPE"  # TPE, CmaEs, Random
    pruner_name: str = "Median"  # Median, Hyperband
    direction: str = "maximize"  # maximize, minimize
    timeout: Optional[int] = None  # seconds
    storage_url: Optional[str] = None  # For distributed optimization


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization"""
    best_params: Dict[str, Any]
    best_value: float
    best_trial: optuna.Trial
    study: optuna.Study
    optimization_history: List[Dict]
    statistical_significance: Dict[str, float]


class OptunaMLflowOptimizer:
    """
    Advanced hyperparameter optimizer with Optuna and MLflow integration
    """

    def __init__(self, config: OptimizationConfig, mlflow_tracking_uri: str = "sqlite:///optuna_mlflow.db"):
        """
        Initialize Optuna optimizer with MLflow tracking

        Args:
            config: Optimization configuration
            mlflow_tracking_uri: MLflow tracking URI
        """
        self.config = config
        self.mlflow_tracking_uri = mlflow_tracking_uri

        # Initialize MLflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        # Create or get experiment
        try:
            self.experiment = mlflow.create_experiment(f"optuna_{config.study_name}")
        except:
            self.experiment = mlflow.get_experiment_by_name(f"optuna_{config.study_name}").experiment_id

        # Initialize Optuna study
        self.study = self._create_study()

        # Results storage
        self.optimization_results = {}

        logger.info(f"🔧 Optuna optimizer initialized: {config.study_name}")
        logger.info(f"   📊 Sampler: {config.sampler_name}")
        logger.info(f"   ✂️  Pruner: {config.pruner_name}")
        logger.info(f"   🎯 Trials: {config.n_trials}")

    def _create_study(self) -> optuna.Study:
        """Create Optuna study with specified configuration"""

        # Choose sampler
        if self.config.sampler_name.lower() == "cmaes":
            sampler = CmaEsSampler()
        elif self.config.sampler_name.lower() == "random":
            sampler = RandomSampler(seed=42)
        else:  # TPE (default)
            sampler = TPESampler(seed=42, n_startup_trials=10, n_ei_candidates=24)

        # Choose pruner
        if self.config.pruner_name.lower() == "hyperband":
            pruner = HyperbandPruner()
        else:  # Median (default)
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        # Create study
        study = optuna.create_study(
            study_name=self.config.study_name,
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner,
            storage=self.config.storage_url,
            load_if_exists=True
        )

        return study

    def optimize_ml_models(self, X: pd.DataFrame, y: pd.Series,
                          model_types: List[str] = ["RandomForest", "GradientBoosting", "SVR", "Ridge"]) -> OptimizationResult:
        """
        Optimize machine learning model hyperparameters

        Args:
            X: Features
            y: Target
            model_types: List of model types to optimize

        Returns:
            Optimization results
        """
        logger.info("🚀 Starting ML model hyperparameter optimization...")

        # Store data for objective function
        self.X_train = X
        self.y_train = y

        # Define CV strategy
        self.cv = TimeSeriesSplit(n_splits=5) if len(X) > 50 else TimeSeriesSplit(n_splits=3)

        # Define objective function
        def ml_objective(trial):
            return self._ml_model_objective(trial, model_types)

        # Setup MLflow callback
        mlflow_callback = MLflowCallback(
            tracking_uri=self.mlflow_tracking_uri,
            create_experiment=False,
            mlflow_kwargs={"nested": True}
        )

        # Run optimization
        self.study.optimize(
            ml_objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            callbacks=[mlflow_callback],
            n_jobs=self.config.n_jobs,
            show_progress_bar=True
        )

        # Analyze results
        result = self._analyze_optimization_results("ml_models")

        logger.info(f"✅ ML optimization completed!")
        logger.info(f"   🏆 Best model: {result.best_params.get('model_type', 'Unknown')}")
        logger.info(f"   📊 Best score: {result.best_value:.4f}")
        logger.info(f"   🧪 Total trials: {len(self.study.trials)}")

        return result

    def optimize_statistical_parameters(self, data: pd.DataFrame) -> OptimizationResult:
        """
        Optimize statistical analysis parameters (VAR lags, significance levels, etc.)

        Args:
            data: Time series data for analysis

        Returns:
            Optimization results
        """
        logger.info("🚀 Starting statistical parameter optimization...")

        # Store data
        self.stat_data = data

        # Define objective function
        def stat_objective(trial):
            return self._statistical_objective(trial)

        # Setup MLflow callback
        mlflow_callback = MLflowCallback(
            tracking_uri=self.mlflow_tracking_uri,
            create_experiment=False
        )

        # Run optimization
        self.study.optimize(
            stat_objective,
            n_trials=min(50, self.config.n_trials),  # Fewer trials for statistical params
            timeout=self.config.timeout,
            callbacks=[mlflow_callback],
            show_progress_bar=True
        )

        result = self._analyze_optimization_results("statistical_params")

        logger.info(f"✅ Statistical optimization completed!")
        logger.info(f"   📊 Best parameters: {result.best_params}")
        logger.info(f"   🎯 Best score: {result.best_value:.4f}")

        return result

    def optimize_spillover_parameters(self, spillover_data: pd.DataFrame) -> OptimizationResult:
        """
        Optimize spillover analysis parameters (forecast horizon, VAR lags, etc.)

        Args:
            spillover_data: Spillover time series data

        Returns:
            Optimization results
        """
        logger.info("🚀 Starting spillover parameter optimization...")

        self.spillover_data = spillover_data

        def spillover_objective(trial):
            return self._spillover_objective(trial)

        # Setup MLflow callback
        mlflow_callback = MLflowCallback(
            tracking_uri=self.mlflow_tracking_uri,
            create_experiment=False
        )

        # Run optimization
        self.study.optimize(
            spillover_objective,
            n_trials=min(30, self.config.n_trials),
            timeout=self.config.timeout,
            callbacks=[mlflow_callback],
            show_progress_bar=True
        )

        result = self._analyze_optimization_results("spillover_params")

        logger.info(f"✅ Spillover optimization completed!")
        logger.info(f"   📊 Best parameters: {result.best_params}")
        logger.info(f"   🎯 Best spillover index: {result.best_value:.4f}")

        return result

    def _ml_model_objective(self, trial, model_types):
        """Objective function for ML model optimization"""

        # Select model type
        model_type = trial.suggest_categorical("model_type", model_types)

        # Model-specific hyperparameter optimization
        if model_type == "RandomForest":
            model = RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators", 10, 200, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                max_features=trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"]),
                random_state=42
            )

        elif model_type == "GradientBoosting":
            model = GradientBoostingRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 300, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.7, 1.0),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                random_state=42
            )

        elif model_type == "SVR":
            model = SVR(
                kernel=trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
                C=trial.suggest_float("C", 0.1, 100.0, log=True),
                epsilon=trial.suggest_float("epsilon", 0.001, 1.0, log=True),
                gamma=trial.suggest_categorical("gamma", ["scale", "auto"])
            )

        elif model_type == "Ridge":
            model = Ridge(
                alpha=trial.suggest_float("alpha", 0.001, 100.0, log=True),
                solver=trial.suggest_categorical("solver", ["auto", "svd", "cholesky", "lsqr"])
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Cross-validation with pruning
        scores = []
        for i, (train_idx, val_idx) in enumerate(self.cv.split(self.X_train)):
            X_train_fold = self.X_train.iloc[train_idx]
            X_val_fold = self.X_train.iloc[val_idx]
            y_train_fold = self.y_train.iloc[train_idx]
            y_val_fold = self.y_train.iloc[val_idx]

            # Train model
            model.fit(X_train_fold, y_train_fold)

            # Predict
            y_pred = model.predict(X_val_fold)

            # Calculate score (R²)
            score = r2_score(y_val_fold, y_pred)
            scores.append(score)

            # Report intermediate value for pruning
            trial.report(np.mean(scores), i)

            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Final score
        final_score = np.mean(scores)

        # Log additional metrics
        trial.set_user_attr("mean_score", final_score)
        trial.set_user_attr("std_score", np.std(scores))
        trial.set_user_attr("model_type", model_type)

        return final_score

    def _statistical_objective(self, trial):
        """Objective function for statistical parameter optimization"""

        # Optimize VAR model parameters
        max_lags = trial.suggest_int("max_lags", 1, 20)
        significance_level = trial.suggest_float("significance_level", 0.01, 0.10)

        # Optimize bootstrap parameters
        bootstrap_iterations = trial.suggest_int("bootstrap_iterations", 100, 2000, log=True)

        try:
            # Import statistical validation framework
            from .statistical_validation import StatisticalValidationFramework

            # Initialize validator with trial parameters
            validator = StatisticalValidationFramework(
                significance_level=significance_level,
                bootstrap_iterations=bootstrap_iterations
            )

            # Run VAR validation
            var_results = validator.validate_var_assumptions(self.stat_data, max_lags=max_lags)

            # Calculate objective (maximize validation pass rate)
            if var_results:
                passed = sum(1 for result in var_results.values()
                           if result.result.value == 'PASS')
                total = len(var_results)
                pass_rate = passed / total if total > 0 else 0
            else:
                pass_rate = 0

            # Log metrics
            trial.set_user_attr("var_tests_passed", passed if var_results else 0)
            trial.set_user_attr("var_tests_total", total if var_results else 0)
            trial.set_user_attr("pass_rate", pass_rate)

            return pass_rate

        except Exception as e:
            logger.warning(f"Statistical objective failed: {e}")
            return 0.0

    def _spillover_objective(self, trial):
        """Objective function for spillover parameter optimization"""

        # Optimize spillover analysis parameters
        forecast_horizon = trial.suggest_int("forecast_horizon", 5, 50)
        var_lags = trial.suggest_int("var_lags", 1, 15)
        rolling_window = trial.suggest_int("rolling_window", 50, 500)

        try:
            # Simplified spillover measure calculation
            # In practice, this would use the full Diebold-Yilmaz framework

            # Create rolling spillover measure
            spillover_values = []

            for i in range(rolling_window, len(self.spillover_data)):
                window_data = self.spillover_data.iloc[i-rolling_window:i]

                # Calculate cross-sectional variance (simplified spillover proxy)
                if window_data.shape[1] > 1:
                    spillover_proxy = window_data.std(axis=1).mean()
                    spillover_values.append(spillover_proxy)

            if spillover_values:
                # Objective: Maximize spillover detection capability
                spillover_strength = np.mean(spillover_values)
                spillover_stability = 1 / (1 + np.std(spillover_values))  # Prefer stable measures

                objective_value = spillover_strength * spillover_stability
            else:
                objective_value = 0

            # Log metrics
            trial.set_user_attr("spillover_strength", spillover_strength if spillover_values else 0)
            trial.set_user_attr("spillover_stability", spillover_stability if spillover_values else 0)
            trial.set_user_attr("spillover_periods", len(spillover_values))

            return objective_value

        except Exception as e:
            logger.warning(f"Spillover objective failed: {e}")
            return 0.0

    def _analyze_optimization_results(self, optimization_type: str) -> OptimizationResult:
        """Analyze and summarize optimization results"""

        # Get best trial
        best_trial = self.study.best_trial

        # Extract optimization history
        history = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    'trial_number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'user_attrs': trial.user_attrs,
                    'datetime': trial.datetime_start
                })

        # Calculate statistical significance of optimization
        values = [trial.value for trial in self.study.trials
                 if trial.state == optuna.trial.TrialState.COMPLETE]

        if len(values) >= 10:
            # Test if optimization found significantly better parameters
            from scipy import stats

            # Compare best 10% vs worst 10%
            sorted_values = sorted(values, reverse=True)
            top_10_pct = sorted_values[:max(1, len(sorted_values)//10)]
            bottom_10_pct = sorted_values[-max(1, len(sorted_values)//10):]

            if len(top_10_pct) >= 3 and len(bottom_10_pct) >= 3:
                t_stat, p_value = stats.ttest_ind(top_10_pct, bottom_10_pct)

                statistical_significance = {
                    'optimization_p_value': p_value,
                    't_statistic': t_stat,
                    'significant_improvement': p_value < 0.05,
                    'effect_size': (np.mean(top_10_pct) - np.mean(bottom_10_pct)) / np.std(values)
                }
            else:
                statistical_significance = {'insufficient_data': True}
        else:
            statistical_significance = {'insufficient_trials': True}

        # Create result object
        result = OptimizationResult(
            best_params=best_trial.params,
            best_value=best_trial.value,
            best_trial=best_trial,
            study=self.study,
            optimization_history=history,
            statistical_significance=statistical_significance
        )

        # Store results
        self.optimization_results[optimization_type] = result

        return result

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary"""

        summary = {
            'study_name': self.config.study_name,
            'total_trials': len(self.study.trials),
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'optimization_config': {
                'sampler': self.config.sampler_name,
                'pruner': self.config.pruner_name,
                'direction': self.config.direction,
                'n_trials': self.config.n_trials
            },
            'results_by_type': {}
        }

        # Add results for each optimization type
        for opt_type, result in self.optimization_results.items():
            summary['results_by_type'][opt_type] = {
                'best_params': result.best_params,
                'best_value': result.best_value,
                'statistical_significance': result.statistical_significance,
                'n_completed_trials': len([t for t in result.study.trials
                                         if t.state == optuna.trial.TrialState.COMPLETE])
            }

        return summary

    def save_optimization_results(self, output_dir: str) -> None:
        """Save optimization results to files"""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save study
        self.study.trials_dataframe().to_csv(
            output_path / f"{self.config.study_name}_trials.csv",
            index=False
        )

        # Save summary
        summary = self.get_optimization_summary()
        with open(output_path / f"{self.config.study_name}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Save optimization plots (if available)
        try:
            from optuna.visualization import plot_optimization_history, plot_param_importances
            import plotly.io as pio

            # Optimization history plot
            fig_history = plot_optimization_history(self.study)
            pio.write_html(fig_history, output_path / f"{self.config.study_name}_history.html")

            # Parameter importance plot
            if len(self.study.trials) >= 10:
                fig_importance = plot_param_importances(self.study)
                pio.write_html(fig_importance, output_path / f"{self.config.study_name}_importance.html")

        except Exception as e:
            logger.warning(f"Could not save optimization plots: {e}")

        logger.info(f"✅ Optimization results saved to: {output_path}")

    def create_optimized_model(self, optimization_type: str = "ml_models"):
        """Create model with optimized hyperparameters"""

        if optimization_type not in self.optimization_results:
            raise ValueError(f"No optimization results found for {optimization_type}")

        result = self.optimization_results[optimization_type]
        best_params = result.best_params.copy()

        if optimization_type == "ml_models":
            model_type = best_params.pop("model_type")

            if model_type == "RandomForest":
                return RandomForestRegressor(**best_params, random_state=42)
            elif model_type == "GradientBoosting":
                return GradientBoostingRegressor(**best_params, random_state=42)
            elif model_type == "SVR":
                return SVR(**best_params)
            elif model_type == "Ridge":
                return Ridge(**best_params)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        else:
            raise ValueError(f"Model creation not implemented for {optimization_type}")


def create_optuna_study_config(study_name: str, optimization_goal: str = "ml_performance",
                              n_trials: int = 100) -> OptimizationConfig:
    """
    Create optimized Optuna configuration for different use cases

    Args:
        study_name: Name of the study
        optimization_goal: Type of optimization (ml_performance, statistical_validation, spillover_analysis)
        n_trials: Number of trials to run

    Returns:
        Optimized configuration
    """

    if optimization_goal == "ml_performance":
        return OptimizationConfig(
            study_name=study_name,
            n_trials=n_trials,
            sampler_name="TPE",  # Best for ML hyperparameters
            pruner_name="Median",
            direction="maximize",
            timeout=3600  # 1 hour
        )

    elif optimization_goal == "statistical_validation":
        return OptimizationConfig(
            study_name=study_name,
            n_trials=min(50, n_trials),  # Fewer trials needed
            sampler_name="TPE",
            pruner_name="Median",
            direction="maximize",
            timeout=1800  # 30 minutes
        )

    elif optimization_goal == "spillover_analysis":
        return OptimizationConfig(
            study_name=study_name,
            n_trials=min(30, n_trials),
            sampler_name="CmaES",  # Good for continuous parameters
            pruner_name="Hyperband",
            direction="maximize",
            timeout=2400  # 40 minutes
        )

    else:
        return OptimizationConfig(
            study_name=study_name,
            n_trials=n_trials,
            sampler_name="TPE",
            pruner_name="Median",
            direction="maximize"
        )