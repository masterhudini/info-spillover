#!/usr/bin/env python3
"""
Sample MLFlow experiment demonstrating cryptocurrency information spillover analysis
"""

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.mlflow_utils import ExperimentLogger, MLFlowTracker

def generate_sample_crypto_data(n_samples=1000):
    """Generate synthetic cryptocurrency discussion data"""
    np.random.seed(42)

    # Simulate different subreddits
    subreddits = ['Bitcoin', 'ethereum', 'CryptoCurrency', 'btc', 'Ripple']
    subreddit_data = np.random.choice(subreddits, n_samples)

    # Generate synthetic features
    data = {
        'subreddit': subreddit_data,
        'post_score': np.random.poisson(10, n_samples),
        'num_comments': np.random.poisson(5, n_samples),
        'sentiment_score': np.random.normal(0, 1, n_samples),
        'text_length': np.random.lognormal(5, 1, n_samples),
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'author_karma': np.random.exponential(1000, n_samples),
        'is_weekend': np.random.randint(0, 2, n_samples),
        'btc_price_change': np.random.normal(0, 0.05, n_samples),
        'market_volatility': np.random.exponential(0.02, n_samples)
    }

    df = pd.DataFrame(data)

    # Create target variable: information spillover indicator
    # Higher scores, more comments, and certain subreddits increase spillover probability
    spillover_prob = (
        0.1 +
        0.3 * (df['post_score'] > df['post_score'].quantile(0.8)) +
        0.2 * (df['num_comments'] > df['num_comments'].quantile(0.7)) +
        0.2 * (df['sentiment_score'].abs() > 1.5) +
        0.1 * (df['subreddit'].isin(['Bitcoin', 'CryptoCurrency'])) +
        0.1 * (df['btc_price_change'].abs() > 0.1)
    )

    df['spillover_target'] = np.random.binomial(1, spillover_prob)
    return df

def run_sample_experiment():
    """Run a sample information spillover experiment with MLFlow tracking"""

    # Initialize experiment logger
    logger = ExperimentLogger("info_spillover_experiment")

    with logger.create_run_context(
        run_name="sample_spillover_analysis",
        tags={
            "experiment_type": "sample",
            "data_type": "synthetic",
            "model_type": "random_forest",
            "analysis": "information_spillover"
        }
    ):
        print("🚀 Starting sample information spillover experiment...")

        # Generate sample data
        print("📊 Generating synthetic cryptocurrency data...")
        df = generate_sample_crypto_data(n_samples=2000)

        # Log data information
        data_info = logger.log_data_info(df, "synthetic_raw")
        print(f"Generated {data_info['synthetic_raw_samples']} samples with {data_info['synthetic_raw_features']} features")

        # Prepare features
        print("🔧 Preparing features...")
        feature_cols = [
            'post_score', 'num_comments', 'sentiment_score', 'text_length',
            'hour_of_day', 'day_of_week', 'author_karma', 'is_weekend',
            'btc_price_change', 'market_volatility'
        ]

        X = df[feature_cols]
        y = df['spillover_target']

        # Log feature information
        logger.log_crypto_features(feature_cols)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Log split information
        logger.tracker.log_params({
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "positive_rate_train": y_train.mean(),
            "positive_rate_test": y_test.mean()
        })

        # Train model
        print("🤖 Training Random Forest model...")
        model_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42
        }

        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)

        # Log model parameters
        logger.tracker.log_params(model_params)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Log performance
        train_metrics = logger.log_model_performance(y_train, y_pred_train, "training")
        test_metrics = logger.log_model_performance(y_test, y_pred_test, "test")

        print(f"📈 Test Accuracy: {test_metrics['test_accuracy']:.3f}")
        print(f"📈 Test F1-Score: {test_metrics['test_f1']:.3f}")

        # Log feature importance
        logger.log_crypto_features(feature_cols, model.feature_importances_)

        # Create and save visualizations
        print("📊 Creating visualizations...")
        plots_dir = Path("experiments/outputs/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Feature importance plot
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)

        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Feature Importance for Information Spillover Prediction')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(plots_dir / "feature_importance_sample.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Log the plot as artifact
        logger.tracker.log_artifacts(str(plots_dir))

        # Simulate spillover analysis results
        print("🔄 Performing spillover analysis...")
        spillover_results = {
            'transfer_entropy': np.random.uniform(0.1, 0.8),
            'correlation_matrix': np.random.uniform(-0.5, 0.8, (5, 5)),
            'granger_causality': {'p_value': np.random.uniform(0.001, 0.1)}
        }

        logger.log_spillover_analysis(spillover_results)

        # Log model
        logger.tracker.log_model(model, "spillover_classifier")

        print("✅ Sample experiment completed successfully!")
        print(f"📋 Results logged to MLFlow experiment: info_spillover_experiment")
        print(f"🎯 Transfer Entropy: {spillover_results['transfer_entropy']:.3f}")
        print(f"🎯 Granger Causality p-value: {spillover_results['granger_causality']['p_value']:.4f}")

        # Log final summary
        logger.tracker.log_params({"experiment_status": "completed"})
        mlflow.set_tag("data_quality", "high")

        return {
            "test_accuracy": test_metrics['test_accuracy'],
            "spillover_entropy": spillover_results['transfer_entropy'],
            "model": model
        }

if __name__ == "__main__":
    try:
        results = run_sample_experiment()
        print("\n🎉 Experiment completed successfully!")
        print("\nNext steps:")
        print("1. Start MLFlow UI: make mlflow-start")
        print("2. Visit http://localhost:5000 to view results")
        print("3. Check the 'info_spillover_experiment' experiment")

    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)