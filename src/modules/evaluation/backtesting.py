import os
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def _load_config() -> Tuple[float, float, int, float]:
    tx_cost = os.getenv("BACKTEST_TRANSACTION_COST")
    if not tx_cost:
        raise ValueError("Missing required env var: BACKTEST_TRANSACTION_COST")
    max_pos = os.getenv("BACKTEST_MAX_POSITION")
    if not max_pos:
        raise ValueError("Missing required env var: BACKTEST_MAX_POSITION")
    periods = os.getenv("BACKTEST_PERIODS_PER_YEAR")
    if not periods:
        raise ValueError("Missing required env var: BACKTEST_PERIODS_PER_YEAR")
    threshold = os.getenv("RETURN_DIRECTION_THRESHOLD")
    if not threshold:
        raise ValueError("Missing required env var: RETURN_DIRECTION_THRESHOLD")
    return float(tx_cost), float(max_pos), int(periods), float(threshold)


def _direction_to_weight(direction: int) -> float:
    if direction == 2:  # mapped from +1
        return 1.0
    if direction == 0:  # mapped from -1
        return -1.0
    return 0.0


def _returns_to_class(returns: np.ndarray, threshold: float) -> np.ndarray:
    classes = np.full_like(returns, fill_value=1, dtype=np.int64)
    classes[returns > threshold] = 2
    classes[returns < -threshold] = 0
    return classes


def _build_prediction_frame(results: Dict[str, Dict]) -> pd.DataFrame:
    records = []
    for symbol, payload in results.items():
        preds = payload.get("test_predictions")
        returns = payload.get("test_returns")
        timestamps = payload.get("test_timestamps")
        if preds is None or returns is None or timestamps is None:
            raise ValueError(f"Missing predictions/returns/timestamps for symbol: {symbol}")
        if not (len(preds) == len(returns) == len(timestamps)):
            raise ValueError(f"Length mismatch for symbol: {symbol}")
        for ts, pred, ret in zip(timestamps, preds, returns):
            records.append({"timestamp": ts, "symbol": symbol, "pred": int(pred), "ret": float(ret)})

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No prediction records for backtest")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _compute_portfolio_series(
    df: pd.DataFrame,
    tx_cost: float,
    max_pos: float,
) -> Tuple[np.ndarray, List[float]]:
    df = df.sort_values(["timestamp", "symbol"])
    all_symbols = sorted(df["symbol"].unique())
    if not all_symbols:
        raise ValueError("No symbols available for backtest")

    weights_prev = {s: 0.0 for s in all_symbols}
    portfolio_log_returns = []
    portfolio_turnover = []

    for _, group in df.groupby("timestamp"):
        weights = {s: 0.0 for s in all_symbols}
        for _, row in group.iterrows():
            weights[row["symbol"]] = _direction_to_weight(row["pred"])

        total_abs = sum(abs(w) for w in weights.values())
        if total_abs > 0:
            scale = min(1.0, max_pos * len(weights) / total_abs)
        else:
            scale = 0.0
        for s in weights:
            weights[s] *= scale

        turnover = sum(abs(weights[s] - weights_prev[s]) for s in all_symbols)
        portfolio_turnover.append(turnover)
        cost = turnover * tx_cost

        ret = 0.0
        for _, row in group.iterrows():
            ret += weights[row["symbol"]] * row["ret"]
        ret -= cost

        portfolio_log_returns.append(ret)
        weights_prev = weights

    return np.array(portfolio_log_returns, dtype=np.float32), portfolio_turnover


def backtest_curve_from_predictions(results: Dict[str, Dict]) -> pd.DataFrame:
    tx_cost, max_pos, _, _ = _load_config()
    df = _build_prediction_frame(results)
    df = df.sort_values(["timestamp", "symbol"])
    portfolio_log_returns, portfolio_turnover = _compute_portfolio_series(df, tx_cost, max_pos)

    by_ts = df.groupby("timestamp").size().index
    curve = pd.DataFrame(
        {
            "timestamp": by_ts,
            "portfolio_log_return": portfolio_log_returns,
            "turnover": portfolio_turnover,
        }
    )
    curve["cumulative_return"] = np.exp(curve["portfolio_log_return"].cumsum()) - 1.0
    return curve


def backtest_from_predictions(results: Dict[str, Dict]) -> Dict[str, float]:
    tx_cost, max_pos, periods_per_year, threshold = _load_config()

    df = _build_prediction_frame(results)
    df = df.sort_values(["timestamp", "symbol"])

    all_symbols = sorted(df["symbol"].unique())

    if df["ret"].isna().any():
        bad = df[df["ret"].isna()].head(5)
        raise ValueError(f"NaN returns in backtest data: {bad.to_dict(orient='records')}")
    if not np.isfinite(df["ret"].to_numpy()).all():
        bad = df[~np.isfinite(df["ret"])].head(5)
        raise ValueError(f"Non-finite returns in backtest data: {bad.to_dict(orient='records')}")

    if df["pred"].isna().any():
        bad = df[df["pred"].isna()].head(5)
        raise ValueError(f"NaN predictions in backtest data: {bad.to_dict(orient='records')}")
    invalid_preds = df[~df["pred"].isin([0, 1, 2])]
    if not invalid_preds.empty:
        raise ValueError(f"Invalid prediction classes: {invalid_preds['pred'].unique().tolist()}")

    symbol_counts = df.groupby("timestamp")["symbol"].nunique()
    missing_ts = symbol_counts[symbol_counts != len(all_symbols)]
    if not missing_ts.empty:
        sample = missing_ts.head(5).to_dict()
        raise ValueError(f"Missing symbols for some timestamps in backtest: {sample}")
    portfolio_log_returns, portfolio_turnover = _compute_portfolio_series(df, tx_cost, max_pos)
    all_preds = []
    all_realized = []

    for _, group in df.groupby("timestamp"):
        realized = _returns_to_class(group["ret"].to_numpy(), threshold)
        all_realized.append(realized)
        all_preds.append(group["pred"].to_numpy())

    if portfolio_log_returns.size == 0:
        raise ValueError("No portfolio returns computed")

    cumulative = float(np.exp(portfolio_log_returns.sum()) - 1.0)
    mean = float(portfolio_log_returns.mean())
    std = float(portfolio_log_returns.std())
    sharpe = (mean / std * np.sqrt(periods_per_year)) if std > 0 else 0.0

    cumulative_curve = np.exp(np.cumsum(portfolio_log_returns))
    peak = np.maximum.accumulate(cumulative_curve)
    drawdown = (cumulative_curve - peak) / peak
    max_drawdown = float(drawdown.min()) if drawdown.size else 0.0

    all_preds = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    all_realized = np.concatenate(all_realized) if all_realized else np.array([], dtype=np.int64)
    if all_preds.size == 0 or all_realized.size == 0:
        raise ValueError("No realized classes available for signal quality metrics")

    accuracy = float((all_preds == all_realized).mean())
    def _precision_recall(cls: int) -> Tuple[float, float]:
        tp = ((all_preds == cls) & (all_realized == cls)).sum()
        fp = ((all_preds == cls) & (all_realized != cls)).sum()
        fn = ((all_preds != cls) & (all_realized == cls)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return float(precision), float(recall)

    long_precision, long_recall = _precision_recall(2)
    short_precision, short_recall = _precision_recall(0)

    return {
        "cumulative_return": cumulative,
        "annualized_sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "avg_turnover": float(np.mean(portfolio_turnover)) if portfolio_turnover else 0.0,
        "num_periods": int(portfolio_log_returns.size),
        "signal_accuracy": accuracy,
        "long_precision": long_precision,
        "long_recall": long_recall,
        "short_precision": short_precision,
        "short_recall": short_recall,
    }
