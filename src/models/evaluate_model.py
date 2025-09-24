import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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
        # Load model pipeline
        with open("models/saved/model.pkl", "rb") as f:
            pipeline = pickle.load(f)

        model = pipeline['model']
        scaler = pipeline['scaler']
        feature_columns = pipeline['feature_columns']
        model_type = pipeline['model_type']

        logger.info(f"Loaded {model_type} model")

        # Load feature info
        with open("data/processed/feature_info.json", "r") as f:
            feature_info = json.load(f)

        target_column = feature_info['target_column']

        # Load test data
        test_data = pd.read_csv("data/processed/test.csv")
        logger.info(f"Loaded test data: {test_data.shape}")

        X_test = test_data[feature_columns]
        y_test = test_data[target_column]

        logger.info(f"Test target distribution: {y_test.value_counts().to_dict()}")

        # Apply feature transformations
        X_test_scaled = scaler.transform(X_test)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test_scaled)
            logger.info("Prediction probabilities computed")

        # Calculate detailed metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, average='weighted')
        test_recall = recall_score(y_test, y_pred, average='weighted')
        test_f1 = f1_score(y_test, y_pred, average='weighted')

        evaluation_metrics = {
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1_score': test_f1,
            'test_samples': len(y_test)
        }

        logger.info(f"Test metrics: {evaluation_metrics}")

        # Generate classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)

        # Create plots directory
        plots_dir = Path("experiments/outputs/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Create confusion matrix plot
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Down', 'Neutral', 'Up'],
                   yticklabels=['Down', 'Neutral', 'Up'])
        plt.title('Confusion Matrix - Test Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(plots_dir / "test_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Feature importance plot (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            importance_df = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)

            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title('Top 20 Feature Importances - Test Set')
            plt.tight_layout()
            plt.savefig(plots_dir / "test_feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()

        # Create prediction distribution plot
        plt.figure(figsize=(10, 6))
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        true_counts = y_test.value_counts().sort_index()

        x_labels = ['Down', 'Neutral', 'Up']
        x_pos = np.arange(len(x_labels))

        width = 0.35
        plt.bar(x_pos - width/2, true_counts.values, width, label='True', alpha=0.7)
        plt.bar(x_pos + width/2, pred_counts.values, width, label='Predicted', alpha=0.7)

        plt.xlabel('Price Direction')
        plt.ylabel('Count')
        plt.title('True vs Predicted Distribution - Test Set')
        plt.xticks(x_pos, x_labels)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "prediction_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Log metrics to MLFlow
        tracker.log_metrics(evaluation_metrics)

        # Save evaluation results
        output_dir = Path("experiments/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        evaluation_results = {
            'metrics': evaluation_metrics,
            'classification_report': class_report
        }

        with open(output_dir / "evaluation_results.json", 'w') as f:
            json.dump(evaluation_results, f, indent=2)

        # Save detailed results
        results_df = pd.DataFrame({
            'y_true': y_test,
            'y_pred': y_pred
        })

        if hasattr(model, 'predict_proba'):
            proba_df = pd.DataFrame(y_pred_proba, columns=[f'prob_class_{i}' for i in range(y_pred_proba.shape[1])])
            results_df = pd.concat([results_df, proba_df], axis=1)

        results_df.to_csv(output_dir / "test_predictions.csv", index=False)

        logger.info("Model evaluation completed successfully!")
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        logger.info(f"Test F1 score: {test_f1:.4f}")
        logger.info(f"Results saved to {output_dir}")

        return evaluation_metrics

if __name__ == "__main__":
    evaluate_model()