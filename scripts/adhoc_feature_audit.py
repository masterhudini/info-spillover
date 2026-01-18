import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from statsmodels.tsa.stattools import adfuller, kpss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bigquery_client import BigQueryClient
from src.data.hierarchical_data_processor import HierarchicalDataProcessor


def require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ValueError(f"Missing required env var: {name}")
    return value


def parse_int_list(csv_value: str) -> list[int]:
    return [int(x.strip()) for x in csv_value.split(",") if x.strip()]


def build_features(df: pd.DataFrame, window_bins: int, lag_cols: list[str], lags: list[int]) -> pd.DataFrame:
    df = df.copy()
    denom = df["comment_count"].replace(0, np.nan)
    df["pos_ratio"] = df["positive_comments"] / denom
    df["neg_ratio"] = df["negative_comments"] / denom
    df["neu_ratio"] = df["neutral_comments"] / denom
    df["activity_log"] = np.log1p(df["post_count"] + df["comment_count"])

    df["compound_change"] = df.groupby("subreddit")["compound_sentiment"].diff()
    df["activity_change"] = df.groupby("subreddit")["activity_log"].diff()

    rolling_std = (
        df.groupby("subreddit")["compound_sentiment"]
        .rolling(window_bins, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )
    df["compound_shock"] = df["compound_change"] / rolling_std.replace(0, np.nan)

    for lag in lags:
        for col in lag_cols:
            df[f"{col}_lag{lag}"] = df.groupby("subreddit")[col].shift(lag)

    return df


def stationarity_report(
    df: pd.DataFrame,
    feature_cols: list[str],
    adf_pvalue_max: float,
    kpss_pvalue_min: float,
    min_points: int = 20
) -> pd.DataFrame:
    records = []
    for col in feature_cols:
        failures = []
        total = 0
        for subreddit, sdf in df.groupby("subreddit"):
            series = sdf[col].dropna()
            if len(series) < min_points:
                continue
            if series.nunique() <= 1:
                failures.append(
                    {
                        "subreddit": subreddit,
                        "adf_p": None,
                        "kpss_p": None,
                        "reason": "constant_series",
                    }
                )
                total += 1
                continue
            total += 1
            adf_stat, adf_p, _, _, _, _ = adfuller(series, autolag="AIC")
            kpss_stat, kpss_p, _, _ = kpss(series, regression="c", nlags="auto")
            if not (adf_p < adf_pvalue_max and kpss_p > kpss_pvalue_min):
                failures.append(
                    {
                        "subreddit": subreddit,
                        "adf_p": adf_p,
                        "kpss_p": kpss_p,
                    }
                )
        pass_rate = 1.0 if total == 0 else (total - len(failures)) / total
        records.append(
            {
                "feature": col,
                "series_tested": total,
                "failures": len(failures),
                "pass_rate": pass_rate,
                "failure_details": failures,
            }
        )
    return pd.DataFrame(records).sort_values(["pass_rate", "series_tested"], ascending=[True, False])


def correlation_pairs(df: pd.DataFrame, feature_cols: list[str], threshold: float) -> pd.DataFrame:
    numeric_df = df[feature_cols].astype(float)
    corr = numeric_df.corr()
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if np.isfinite(value) and abs(value) >= threshold:
                pairs.append(
                    {"feature_a": cols[i], "feature_b": cols[j], "corr": value}
                )
    if not pairs:
        return pd.DataFrame(columns=["feature_a", "feature_b", "corr"])
    return pd.DataFrame(pairs).sort_values("corr", key=lambda s: s.abs(), ascending=False)


def target_correlations(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> pd.DataFrame:
    records = []
    for col in feature_cols:
        series = df[[col, target_col]].dropna()
        if series.empty:
            continue
        corr = series[col].corr(series[target_col], method="spearman")
        records.append({"feature": col, "spearman_corr": corr})
    return pd.DataFrame(records).sort_values("spearman_corr", key=lambda s: s.abs(), ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adhoc feature audit for DY modeling.")
    parser.add_argument("--lags", required=True, help="Comma-separated lags, e.g. 1,2,3")
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument("--pass-rate", type=float, default=0.8)
    parser.add_argument("--include-target", action="store_true")
    parser.add_argument("--output", help="Output directory for reports")
    args = parser.parse_args()

    load_dotenv(dotenv_path=str(PROJECT_ROOT / ".env"))

    start_date = require_env("START_DATE")
    end_date = require_env("END_DATE")
    adf_pvalue_max = float(require_env("DY_ADF_PVALUE_MAX"))
    kpss_pvalue_min = float(require_env("DY_KPSS_PVALUE_MIN"))
    window_minutes = parse_int_list(require_env("TEMPORAL_WINDOWS_MINUTES"))
    interval_seconds = int(require_env("SENTIMENT_AGG_INTERVAL_SECONDS"))

    lags = parse_int_list(args.lags)
    if not lags:
        raise ValueError("At least one lag is required")

    window_bins = max(1, int(round((min(window_minutes) * 60) / interval_seconds)))

    bq_client = BigQueryClient(
        project_id=require_env("BIGQUERY_PROJECT_ID"),
        dataset_id=require_env("BIGQUERY_DATASET_ID"),
        sentiment_table=require_env("BIGQUERY_SENTIMENT_TABLE"),
    )
    processor = HierarchicalDataProcessor(bigquery_client=bq_client)

    df = processor.load_and_validate_data(start_date, end_date)
    df = processor.process_hierarchical_features(df)
    df, network = processor.build_network_features(df)

    total_spillover = processor.processing_log.get("network_building", {}).get("total_spillover")
    if total_spillover is not None:
        df["total_spillover"] = float(total_spillover)

    for subreddit in df["subreddit"].unique():
        if subreddit in network.nodes:
            mask = df["subreddit"] == subreddit
            df.loc[mask, "to_spillover"] = network.nodes[subreddit].get("to_spillover")
            df.loc[mask, "from_spillover"] = network.nodes[subreddit].get("from_spillover")

    lag_cols = [
        "compound_sentiment",
        "sentiment_score",
        "net_spillover",
        "to_spillover",
        "from_spillover",
        "activity_log",
        "compound_change",
        "compound_shock",
    ]
    df = build_features(df, window_bins, lag_cols, lags)

    candidate_cols = []
    for col in df.columns:
        if col in ["subreddit", "post_created_utc", "score", "snapped_at", "symbol"]:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            candidate_cols.append(col)

    stationarity_df = stationarity_report(
        df,
        candidate_cols,
        adf_pvalue_max=adf_pvalue_max,
        kpss_pvalue_min=kpss_pvalue_min,
    )

    pass_cols = stationarity_df[stationarity_df["pass_rate"] >= args.pass_rate]["feature"].tolist()
    corr_pairs = correlation_pairs(df, pass_cols, args.corr_threshold)

    target_corr_df = None
    if args.include_target:
        symbols = [s.strip() for s in require_env("SYMBOLS_SELECTED").split(",") if s.strip()]
        price_data = bq_client.get_price_data(symbols=symbols, start_date=start_date, end_date=end_date)
        df_targets = processor.create_targets(df, price_data)
        target_corr_df = target_correlations(df_targets, pass_cols, "return_direction")

    summary = {
        "rows": len(df),
        "subreddits": df["subreddit"].nunique(),
        "window_bins": window_bins,
        "lag_list": lags,
        "features_total": len(candidate_cols),
        "features_pass_rate": len(pass_cols),
        "pass_rate_threshold": args.pass_rate,
        "corr_threshold": args.corr_threshold,
    }

    print("=== Feature audit summary ===")
    print(json.dumps(summary, indent=2))
    print("\nTop stationarity failures:")
    print(stationarity_df.head(15)[["feature", "series_tested", "failures", "pass_rate"]])
    print("\nHigh correlation pairs (filtered):")
    print(corr_pairs.head(15))
    if target_corr_df is not None:
        print("\nTop target correlations (return_direction, spearman):")
        print(target_corr_df.head(15))

    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        stationarity_df.to_csv(output_dir / "stationarity_report.csv", index=False)
        corr_pairs.to_csv(output_dir / "correlation_pairs.csv", index=False)
        if target_corr_df is not None:
            target_corr_df.to_csv(output_dir / "target_correlations.csv", index=False)
        with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
