import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.mlflow_utils import MLFlowTracker
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model():
    """Evaluate trained model on test set"""
    logger.info("Starting model evaluation...")

    # Initialize MLFlow tracking
    tracker = MLFlowTracker("info_spillover_experiment")

    with tracker.start_run(run_name="model_evaluation"):
        # Load model
        with open("models/saved/model.pkl", "rb") as f:
            model = pickle.load(f)

        # Load feature pipeline
        with open("data/interim/features.pkl", "rb") as f:
            feature_pipeline = pickle.load(f)

        # Load test data (placeholder)
        # test_data = pd.read_csv("data/processed/test.csv")
        # X_test = test_data.drop('target', axis=1)
        # y_test = test_data['target']

        # Apply feature transformations
        # X_test_transformed = feature_pipeline['selector'].transform(
        #     feature_pipeline['scaler'].transform(X_test)
        # )

        # Make predictions
        # y_pred = model.predict(X_test_transformed)
        # y_pred_proba = model.predict_proba(X_test_transformed)

        # Calculate detailed metrics (placeholder)
        # test_accuracy = accuracy_score(y_test, y_pred)
        # test_precision = precision_score(y_test, y_pred, average='weighted')
        # test_recall = recall_score(y_test, y_pred, average='weighted')
        # test_f1 = f1_score(y_test, y_pred, average='weighted')

        # evaluation_metrics = {
        #     'test_accuracy': test_accuracy,
        #     'test_precision': test_precision,
        #     'test_recall': test_recall,
        #     'test_f1_score': test_f1,
        #     'test_samples': len(y_test)
        # }

        # Generate classification report
        # class_report = classification_report(y_test, y_pred, output_dict=True)

        # Create confusion matrix plot
        # plt.figure(figsize=(10, 8))
        # cm = confusion_matrix(y_test, y_pred)
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        # plt.title('Confusion Matrix')
        # plt.ylabel('True Label')
        # plt.xlabel('Predicted Label')
        # plt.tight_layout()

        # Save plots
        plots_dir = Path("experiments/outputs/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        # plt.savefig(plots_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        # plt.close()

        # Feature importance plot (for tree-based models)
        # if hasattr(model, 'feature_importances_'):
        #     plt.figure(figsize=(12, 8))
        #     feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
        #     importance_df = pd.DataFrame({
        #         'feature': feature_names,
        #         'importance': model.feature_importances_
        #     }).sort_values('importance', ascending=False).head(20)

        #     sns.barplot(data=importance_df, x='importance', y='feature')
        #     plt.title('Top 20 Feature Importances')
        #     plt.tight_layout()
        #     plt.savefig(plots_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
        #     plt.close()

        # Log metrics to MLFlow
        # tracker.log_metrics(evaluation_metrics)

        # Save evaluation results
        output_dir = Path("experiments/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        # evaluation_results = {
        #     'metrics': evaluation_metrics,
        #     'classification_report': class_report
        # }

        # with open(output_dir / "evaluation_results.json", 'w') as f:
        #     json.dump(evaluation_results, f, indent=2)

        logger.info("Model evaluation completed successfully!")

if __name__ == "__main__":
    evaluate_model()