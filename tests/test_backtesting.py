import os
import numpy as np

from src.modules.evaluation.backtesting import backtest_from_predictions


def test_backtest_basic():
    os.environ["BACKTEST_TRANSACTION_COST"] = "0.0"
    os.environ["BACKTEST_MAX_POSITION"] = "1.0"
    os.environ["BACKTEST_PERIODS_PER_YEAR"] = "10"

    results = {
        "BTC": {
            "test_predictions": np.array([2, 2, 0, 1]),
            "test_returns": np.array([0.01, 0.02, -0.01, 0.0]),
            "test_timestamps": [
                "2024-01-01 00:00:00",
                "2024-01-01 02:00:00",
                "2024-01-01 04:00:00",
                "2024-01-01 06:00:00",
            ],
        }
    }

    metrics = backtest_from_predictions(results)
    assert "cumulative_return" in metrics
    assert "annualized_sharpe" in metrics
    assert metrics["num_periods"] == 4
