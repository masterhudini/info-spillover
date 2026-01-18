import numpy as np
import pandas as pd

from src.modules.data.modeling_data_module import ModelingDataModule


def _make_processed_data():
    ts = pd.date_range("2024-01-01", periods=4, freq="2h")
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


def test_data_module_builds_features_and_targets():
    df = _make_processed_data()
    module = ModelingDataModule(
        processed_data=df,
        network=_make_network(),
        feature_columns=[
            "compound_sentiment",
            "sentiment_score",
            "post_count",
            "comment_count",
            "avg_positive_sentiment",
            "avg_negative_sentiment",
            "avg_neutral_sentiment",
            "net_spillover",
            "pagerank",
            "in_degree",
            "out_degree",
            "clustering",
        ],
        dy_resample_freq="2h",
        ffill_limit=0,
        subreddits=["A", "B"],
        symbols=["BTC", "ETH"],
    )

    features, timestamps = module.build_feature_tensor()
    targets = module.build_targets(timestamps)
    returns = module.build_return_series(timestamps)

    assert features.shape[1] == 2
    assert features.shape[2] == 12
    assert len(timestamps) == features.shape[0]
    assert set(targets.keys()) == {"BTC", "ETH"}
    assert set(returns.keys()) == {"BTC", "ETH"}


def test_data_module_rejects_missing_network_edges():
    import networkx as nx

    df = _make_processed_data()
    g = nx.DiGraph()
    g.add_node("A")
    g.add_node("B")
    try:
        ModelingDataModule(
            processed_data=df,
            network=g,
            feature_columns=["compound_sentiment"],
            dy_resample_freq="2h",
            ffill_limit=0,
            subreddits=["A", "B"],
            symbols=["BTC", "ETH"],
        )
    except ValueError as exc:
        assert "zero edges" in str(exc)
    else:
        raise AssertionError("Expected error for empty network edges")
