import os
import numpy as np
import pandas as pd

from src.modules.modeling_pipeline import run_modeling
from src.modules.evaluation.backtesting import backtest_from_predictions


def _make_processed_data():
    ts = pd.date_range("2024-01-01", periods=6, freq="2h")
    rows = []
    for t in ts:
        rows.append(
            {
                "subreddit": "A",
                "symbol": "BTC",
                "post_created_utc": t,
                "compound_sentiment": 0.1,
                "sentiment_score": 0.1,
                "post_count": 1,
                "comment_count": 1,
                "avg_positive_sentiment": 0.2,
                "avg_negative_sentiment": 0.1,
                "avg_neutral_sentiment": 0.7,
                "net_spillover": 0.0,
                "pagerank": 0.5,
                "in_degree": 1.0,
                "out_degree": 1.0,
                "clustering": 0.0,
                "return_direction": 1,
                "price_log_return": 0.01,
            }
        )
        rows.append(
            {
                "subreddit": "B",
                "symbol": "ETH",
                "post_created_utc": t,
                "compound_sentiment": -0.1,
                "sentiment_score": -0.1,
                "post_count": 1,
                "comment_count": 1,
                "avg_positive_sentiment": 0.1,
                "avg_negative_sentiment": 0.2,
                "avg_neutral_sentiment": 0.7,
                "net_spillover": 0.0,
                "pagerank": 0.5,
                "in_degree": 1.0,
                "out_degree": 1.0,
                "clustering": 0.0,
                "return_direction": -1,
                "price_log_return": -0.01,
            }
        )
    return pd.DataFrame(rows)


def _make_network():
    import networkx as nx

    g = nx.DiGraph()
    g.add_node("A")
    g.add_node("B")
    g.add_edge("A", "B", weight=0.2)
    g.add_edge("B", "A", weight=0.1)
    return g


def test_integration_modeling_backtest(tmp_path):
    os.environ["SUBREDDITS_SELECTED"] = "A,B"
    os.environ["SYMBOLS_SELECTED"] = "BTC,ETH"
    os.environ["MODEL_FEATURE_COLUMNS"] = (
        "compound_sentiment,sentiment_score,post_count,comment_count,"
        "avg_positive_sentiment,avg_negative_sentiment,avg_neutral_sentiment,"
        "net_spillover,pagerank,in_degree,out_degree,clustering"
    )
    os.environ["DY_RESAMPLE_FREQ"] = "2h"
    os.environ["DY_FFILL_LIMIT"] = "0"
    os.environ["MODEL_VAL_SPLIT"] = "0.2"
    os.environ["MODEL_TEST_SPLIT"] = "0.2"
    os.environ["BACKTEST_TRANSACTION_COST"] = "0.0"
    os.environ["BACKTEST_MAX_POSITION"] = "1.0"
    os.environ["BACKTEST_PERIODS_PER_YEAR"] = "10"

    hyperparams_path = tmp_path / "hyper.yaml"
    hyperparams_path.write_text(
        "model:\\n"
        "  hidden_dim: 4\\n"
        "  gnn_dim: 4\\n"
        "  epochs: 1\\n"
        "  lr: 0.01\\n"
        "  batch_size: 2\\n"
    )

    results = run_modeling(_make_processed_data(), _make_network(), str(hyperparams_path))
    metrics = backtest_from_predictions(results)

    assert metrics["num_periods"] > 0
