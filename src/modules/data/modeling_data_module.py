import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class ModelingDataModule:
    """Prepare time-aligned node features and targets for GNN modeling."""

    def __init__(
        self,
        processed_data: pd.DataFrame,
        network,
        feature_columns: List[str],
        dy_resample_freq: str,
        ffill_limit: int,
        subreddits: List[str],
        symbols: List[str],
        target_col: str = "return_direction",
        timestamp_col: str = "post_created_utc",
    ):
        self.processed_data = processed_data
        self.network = network
        self.feature_columns = feature_columns
        self.dy_resample_freq = dy_resample_freq
        self.ffill_limit = ffill_limit
        self.subreddits = subreddits
        self.symbols = symbols
        self.target_col = target_col
        self.timestamp_col = timestamp_col

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if self.timestamp_col not in self.processed_data.columns:
            raise ValueError(f"Missing timestamp column: {self.timestamp_col}")
        if self.target_col not in self.processed_data.columns:
            raise ValueError(f"Missing target column: {self.target_col}")
        if "subreddit" not in self.processed_data.columns:
            raise ValueError("Missing subreddit column")
        if "symbol" not in self.processed_data.columns:
            raise ValueError("Missing symbol column (price targets must be merged)")
        if "price_log_return" not in self.processed_data.columns:
            raise ValueError("Missing price_log_return column (required for backtesting)")

        missing = [c for c in self.feature_columns if c not in self.processed_data.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        data_subs = set(self.processed_data["subreddit"].unique())
        expected_subs = set(self.subreddits)
        if data_subs != expected_subs:
            raise ValueError(f"Subreddit mismatch. Data={sorted(data_subs)}, expected={sorted(expected_subs)}")

        if self.network is None or not hasattr(self.network, "nodes"):
            raise ValueError("Missing DY network")
        network_nodes = set(self.network.nodes())
        if network_nodes != expected_subs:
            raise ValueError(f"Network nodes mismatch. Network={sorted(network_nodes)}, expected={sorted(expected_subs)}")
        if self.network.number_of_edges() == 0:
            raise ValueError("DY network has zero edges (no fallback allowed)")

        if not self.symbols:
            raise ValueError("Symbols list is empty")

    def build_feature_tensor(self) -> Tuple[np.ndarray, List[pd.Timestamp]]:
        df = self.processed_data.copy()
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        df["ts"] = df[self.timestamp_col].dt.floor(self.dy_resample_freq)

        panel = None
        for feature in self.feature_columns:
            pivot = df.pivot_table(
                index="ts",
                columns="subreddit",
                values=feature,
                aggfunc="mean",
            )
            if self.ffill_limit > 0:
                pivot = pivot.ffill(limit=self.ffill_limit)

            stacked = pivot.stack(dropna=False).rename(feature)
            if panel is None:
                panel = stacked.to_frame()
            else:
                panel[feature] = stacked

        panel = panel.dropna()
        if panel.empty:
            raise ValueError("No complete feature rows after resample/ffill")

        counts = panel.groupby(level=0).size()
        valid_ts = counts[counts == len(self.subreddits)].index
        panel = panel.loc[panel.index.get_level_values(0).isin(valid_ts)]
        if panel.empty:
            raise ValueError("No timestamps with full subreddit coverage")

        timestamps = sorted(panel.index.get_level_values(0).unique())
        num_nodes = len(self.subreddits)
        num_features = len(self.feature_columns)
        features = np.zeros((len(timestamps), num_nodes, num_features), dtype=np.float32)

        for idx, ts in enumerate(timestamps):
            rows = panel.loc[ts].reindex(self.subreddits)
            if rows.isna().any().any():
                raise ValueError(f"Missing features after alignment at {ts}")
            features[idx] = rows[self.feature_columns].values.astype(np.float32)

        return features, timestamps

    def build_targets(self, timestamps: List[pd.Timestamp]) -> Dict[str, np.ndarray]:
        df = self.processed_data.copy()
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        df["ts"] = df[self.timestamp_col].dt.floor(self.dy_resample_freq)

        targets = {}
        for symbol in self.symbols:
            sym = df[df["symbol"] == symbol].copy()
            series = sym.groupby("ts")[self.target_col].first()
            aligned = series.reindex(timestamps)
            if aligned.isna().all():
                raise ValueError(f"No targets available for symbol: {symbol}")
            # Keep as float to preserve NaN, which will be filtered by mask in run_modeling
            targets[symbol] = aligned.values.astype(np.float64)

        return targets

    def build_return_series(self, timestamps: List[pd.Timestamp]) -> Dict[str, np.ndarray]:
        df = self.processed_data.copy()
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        df["ts"] = df[self.timestamp_col].dt.floor(self.dy_resample_freq)

        returns = {}
        for symbol in self.symbols:
            sym = df[df["symbol"] == symbol].copy()
            series = sym.groupby("ts")["price_log_return"].first()
            aligned = series.reindex(timestamps)
            if aligned.isna().all():
                raise ValueError(f"No price_log_return available for symbol: {symbol}")
            returns[symbol] = aligned.values.astype(np.float32)
        return returns

    def time_split_indices(self, n: int, val_frac: float, test_frac: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if val_frac + test_frac >= 1.0:
            raise ValueError("val_frac + test_frac must be < 1.0")
        train_end = int(n * (1 - val_frac - test_frac))
        val_end = int(n * (1 - test_frac))
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, n)
        return train_idx, val_idx, test_idx
