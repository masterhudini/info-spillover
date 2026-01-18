import os
import sys
import logging
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

from src.data.bigquery_client import BigQueryClient
from src.data.hierarchical_data_processor import HierarchicalDataProcessor
from src.modules.modeling_pipeline import run_modeling
from src.modules.evaluation.backtesting import backtest_from_predictions, backtest_curve_from_predictions


def setup_logging(project_root: Path):
    """Setup logging to both file and console"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = project_root / f"training_{timestamp}.log"

    # Create formatters
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info(f"Logging to file: {log_file}")
    return log_file


def main():
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=str(project_root / ".env"))

    # Setup logging
    setup_logging(project_root)

    hyperparams_path = os.getenv("MODEL_HYPERPARAMS_PATH")
    if not hyperparams_path:
        raise ValueError("Missing required env var: MODEL_HYPERPARAMS_PATH")
    hyperparams_path = str(project_root / hyperparams_path)

    start_date = os.getenv("START_DATE")
    end_date = os.getenv("END_DATE")
    if not start_date or not end_date:
        raise ValueError("Missing START_DATE/END_DATE")

    bq_client = BigQueryClient(
        project_id=os.getenv("BIGQUERY_PROJECT_ID"),
        dataset_id=os.getenv("BIGQUERY_DATASET_ID"),
        sentiment_table=os.getenv("BIGQUERY_SENTIMENT_TABLE"),
    )

    processor = HierarchicalDataProcessor(bq_client)
    batch_days_env = os.getenv("PIPELINE_BATCH_DAYS")
    if batch_days_env:
        feature_env = os.getenv("MODEL_FEATURE_COLUMNS")
        if not feature_env:
            raise ValueError("Missing required env var: MODEL_FEATURE_COLUMNS")
        feature_columns = [s.strip() for s in feature_env.split(",") if s.strip()]
        keep_columns = (
            feature_columns
            + ["subreddit", "symbol", "post_created_utc", "price_log_return", "return_direction", "next_return"]
        )
        df, network, _ = processor.process_full_pipeline_chunked(
            start_date=start_date,
            end_date=end_date,
            include_price_targets=True,
            batch_days=int(batch_days_env),
            keep_columns=keep_columns,
        )
    else:
        df, network, _ = processor.process_full_pipeline(
            start_date=start_date,
            end_date=end_date,
            include_price_targets=True,
        )

    modeling_results = run_modeling(df, network, hyperparams_path)
    backtest_metrics = backtest_from_predictions(modeling_results)

    output_path = os.getenv("BACKTEST_OUTPUT_PATH")
    if not output_path:
        raise ValueError("Missing required env var: BACKTEST_OUTPUT_PATH")
    output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(str(backtest_metrics))

    predictions_path = os.getenv("MODEL_PREDICTIONS_PATH", str(project_root / "results" / "predictions.csv"))
    metrics_path = os.getenv("MODEL_METRICS_PATH", str(project_root / "results" / "training_metrics.json"))
    curve_path = os.getenv("BACKTEST_CURVE_PATH", str(project_root / "results" / "backtest_curve.csv"))

    if predictions_path:
        rows = []
        for symbol, payload in modeling_results.items():
            preds = payload.get("test_predictions", [])
            probs = payload.get("test_probabilities", [])
            trues = payload.get("test_true", [])
            returns = payload.get("test_returns", [])
            timestamps = payload.get("test_timestamps", [])
            for ts, pred, true, ret, prob in zip(timestamps, preds, trues, returns, probs):
                rows.append(
                    {
                        "timestamp": ts,
                        "symbol": symbol,
                        "pred_class": int(pred),
                        "true_class": int(true),
                        "prob_down": float(prob[0]),
                        "prob_neutral": float(prob[1]),
                        "prob_up": float(prob[2]),
                        "return": float(ret),
                    }
                )
        pd.DataFrame(rows).to_csv(predictions_path, index=False)
        logging.info("Predictions saved to: %s", predictions_path)

    if metrics_path:
        metrics_payload = {
            "training_metrics": {
                symbol: {
                    "train": payload.get("train"),
                    "val": payload.get("val"),
                    "test": payload.get("test"),
                }
                for symbol, payload in modeling_results.items()
            },
            "backtest_metrics": backtest_metrics,
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_payload, f, indent=2, sort_keys=True, default=str)
        logging.info("Training metrics saved to: %s", metrics_path)

    if curve_path:
        curve = backtest_curve_from_predictions(modeling_results)
        curve.to_csv(curve_path, index=False)
        logging.info("Backtest curve saved to: %s", curve_path)

    logging.info("Backtest metrics:")
    logging.info(backtest_metrics)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"Training failed with error: {e}")
        raise
