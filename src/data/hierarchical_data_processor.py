"""
Hierarchical Data Processor for Cryptocurrency Sentiment Analysis
Based on scientific methodology for information spillover analysis
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import logging
from datetime import datetime, timedelta
import warnings

from src.data.bigquery_client import BigQueryClient
from src.data.transform_validator import TransformValidator, TransformValidationError

# Statistical and econometric libraries
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import InterpolationWarning
import networkx as nx
from scipy import stats
from scipy.stats import jarque_bera

warnings.filterwarnings("ignore", category=InterpolationWarning)

logger = logging.getLogger(__name__)


class DataQualityValidator:
    """
    Data quality validation following academic standards for financial time series
    """

    def __init__(self):
        self.quality_report = {}

    def validate_timestamps(self, df: pd.DataFrame, timestamp_col: str = 'post_created_utc') -> Dict:
        """Validate timestamp consistency and gaps"""
        logger.info("Validating timestamp quality...")

        # Convert to datetime if needed
        if df[timestamp_col].dtype == 'object':
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Check for duplicates
        duplicates = df[timestamp_col].duplicated().sum()

        # Check for chronological order
        is_sorted = df[timestamp_col].is_monotonic_increasing

        # Identify gaps > 24 hours
        time_diffs = df[timestamp_col].diff()
        large_gaps = (time_diffs > timedelta(hours=24)).sum()

        # Distribution of time differences
        median_gap = time_diffs.median()
        mean_gap = time_diffs.mean()

        validation_report = {
            'total_records': len(df),
            'duplicate_timestamps': duplicates,
            'chronologically_ordered': is_sorted,
            'large_gaps_24h': large_gaps,
            'median_gap_minutes': median_gap.total_seconds() / 60,
            'mean_gap_minutes': mean_gap.total_seconds() / 60,
            'time_range': {
                'start': df[timestamp_col].min(),
                'end': df[timestamp_col].max(),
                'span_days': (df[timestamp_col].max() - df[timestamp_col].min()).days
            }
        }

        self.quality_report['timestamps'] = validation_report
        return validation_report

    def validate_sentiment_scores(self, df: pd.DataFrame,
                                score_col: str = 'sentiment_score',
                                label_col: Optional[str] = None) -> Dict:
        """Validate sentiment scores according to VADER methodology"""
        logger.info("Validating sentiment scores...")

        # Check score range (VADER compound is -1 to +1)
        score_min = df[score_col].min()
        score_max = df[score_col].max()
        valid_range = (score_min >= -1) and (score_max <= 1)

        # Check for missing values
        missing_scores = df[score_col].isna().sum()
        missing_labels = df[label_col].isna().sum() if label_col else None

        # Distribution analysis
        score_distribution = {
            'mean': df[score_col].mean(),
            'std': df[score_col].std(),
            'median': df[score_col].median(),
            'skewness': stats.skew(df[score_col].dropna()),
            'kurtosis': stats.kurtosis(df[score_col].dropna())
        }

        # Label distribution
        label_counts = df[label_col].value_counts().to_dict() if label_col else {}

        # Check for anomalies (neutral should be near 0, not extreme values)
        neutral_extreme = (
            ((df[label_col] == 'neutral') &
            ((df[score_col] < -0.3) | (df[score_col] > 0.3))).sum()
            if label_col else None
        )

        validation_report = {
            'valid_score_range': valid_range,
            'score_range': {'min': score_min, 'max': score_max},
            'missing_scores': missing_scores,
            'missing_labels': missing_labels,
            'score_distribution': score_distribution,
            'label_distribution': label_counts,
            'neutral_extreme_scores': neutral_extreme
        }

        self.quality_report['sentiment'] = validation_report
        return validation_report

    def validate_subreddit_coverage(self, df: pd.DataFrame,
                                   subreddit_col: str = 'subreddit') -> Dict:
        """Validate subreddit data coverage and balance"""
        logger.info("Validating subreddit coverage...")

        # Count posts per subreddit
        subreddit_counts = df[subreddit_col].value_counts()

        # Temporal coverage per subreddit
        temporal_coverage = df.groupby(subreddit_col)['post_created_utc'].agg([
            'min', 'max', 'count'
        ])
        temporal_coverage['span_days'] = (temporal_coverage['max'] -
                                        temporal_coverage['min']).dt.days

        # Balance analysis (Gini coefficient)
        counts = subreddit_counts.values
        n = len(counts)
        index = np.arange(1, n + 1)
        gini = ((2 * index - n - 1) * np.sort(counts)).sum() / (n * counts.sum())

        validation_report = {
            'total_subreddits': len(subreddit_counts),
            'post_counts': subreddit_counts.to_dict(),
            'balance_gini': gini,
            'temporal_coverage': temporal_coverage.to_dict(),
            'min_posts_subreddit': subreddit_counts.min(),
            'max_posts_subreddit': subreddit_counts.max(),
            'median_posts_subreddit': subreddit_counts.median()
        }

        self.quality_report['subreddit_coverage'] = validation_report
        return validation_report


class HierarchicalFeatureEngineer:
    """
    Hierarchical feature engineering following Diebold-Yilmaz methodology
    """

    def __init__(self, window_minutes: List[int], interval_seconds: int):
        self.window_minutes = window_minutes
        self.interval_seconds = interval_seconds
        self.scalers = {}
        self.validator = TransformValidator(strict=True)

    def compute_compound_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cast compound sentiment to float and validate range.
        VADER compound scores should be in [-1, 1].
        """
        logger.info("Computing compound sentiment scores...")

        if 'compound_sentiment' not in df.columns:
            raise ValueError("Missing required column: compound_sentiment")

        before = df['compound_sentiment'].copy()
        df['compound_sentiment'] = df['compound_sentiment'].astype(float)

        # Validate conversion didn't introduce NaN
        self.validator.validate_astype(before, df['compound_sentiment'], 'compound_sentiment', 'float')

        # Validate VADER range [-1, 1]
        self.validator.validate_sentiment_score(df['compound_sentiment'], 'compound_sentiment')

        return df

    def compute_temporal_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute temporal aggregates following Bollen et al. (2011) methodology
        """
        logger.info("Computing temporal aggregates...")

        # Ensure proper datetime index
        df = df.set_index('post_created_utc').sort_index()

        # Initialize results container
        aggregated_features = []
        window_bins_short = max(
            1, int(round((min(self.window_minutes) * 60) / self.interval_seconds))
        )

        for subreddit in df['subreddit'].unique():
            logger.info(f"Processing temporal aggregates for {subreddit}")

            subreddit_data = df[df['subreddit'] == subreddit].copy()

            # Log transform of activity (log1p handles zeros)
            activity_sum = subreddit_data['post_count'] + subreddit_data['comment_count']
            subreddit_data['activity_log'] = np.log1p(activity_sum)

            # Validate log transform
            self.validator.validate_log_transform(
                subreddit_data['activity_log'],
                f'activity_log_{subreddit}',
                original_min=activity_sum.min()
            )

            # Diff transforms (first element will be NaN - expected)
            subreddit_data['activity_change'] = subreddit_data['activity_log'].diff()
            subreddit_data['compound_change'] = subreddit_data['compound_sentiment'].diff()

            # Rolling std for shock normalization
            rolling_std = subreddit_data['compound_sentiment'].rolling(
                window_bins_short, min_periods=1
            ).std()

            # Compound shock = change / rolling_std (replace 0 with NaN to avoid inf)
            subreddit_data['compound_shock'] = subreddit_data['compound_change'] / rolling_std.replace(0, np.nan)

            # Validate no inf values in shock
            if np.isinf(subreddit_data['compound_shock']).any():
                inf_count = np.isinf(subreddit_data['compound_shock']).sum()
                logger.warning(f"compound_shock for {subreddit} has {inf_count} inf values, replacing with NaN")
                subreddit_data['compound_shock'] = subreddit_data['compound_shock'].replace([np.inf, -np.inf], np.nan)

            # For each time window in minutes, translated into bins based on aggregation interval
            for window_minutes in self.window_minutes:
                window_bins = max(
                    1, int(round((window_minutes * 60) / self.interval_seconds))
                )

                # Rolling statistics over aggregated data
                rolling = subreddit_data['compound_sentiment'].rolling(window_bins, min_periods=1)

                subreddit_data[f'sentiment_mean_{window_bins}bins'] = rolling.mean()
                subreddit_data[f'sentiment_std_{window_bins}bins'] = rolling.std()
                subreddit_data[f'sentiment_skew_{window_bins}bins'] = rolling.skew()
                subreddit_data[f'sentiment_kurt_{window_bins}bins'] = rolling.apply(
                    lambda x: stats.kurtosis(x) if len(x) > 3 else np.nan
                )

                # Volume-weighted sentiment: sum(sentiment * weight) / sum(weight)
                if 'score' in subreddit_data.columns:
                    # Ensure positive weights (scores can be negative on Reddit)
                    weights = subreddit_data['score'].clip(lower=0) + 1  # +1 to avoid zero weights

                    weighted_sum = (subreddit_data['compound_sentiment'] * weights).rolling(
                        window_bins, min_periods=1
                    ).sum()
                    weight_sum = weights.rolling(window_bins, min_periods=1).sum()

                    subreddit_data[f'vwsentiment_{window_bins}bins'] = (weighted_sum / weight_sum).fillna(0)

                # Activity count in window (count of non-null sentiment values)
                subreddit_data[f'obs_count_{window_bins}bins'] = (
                    subreddit_data['compound_sentiment']
                    .rolling(window_bins, min_periods=1)
                    .count()
                )

            aggregated_features.append(subreddit_data)

        # Combine all subreddits
        result_df = pd.concat(aggregated_features, ignore_index=False)

        # Validate concat preserved total rows
        expected_rows = sum(len(part) for part in aggregated_features)
        if len(result_df) != expected_rows:
            raise TransformValidationError(
                f"concat lost rows: expected {expected_rows}, got {len(result_df)}"
            )

        return result_df.reset_index()


class DieboldYilmazSpilloverBuilder:
    """
    Build Diebold-Yilmaz spillover network using VAR and FEVD
    """

    def __init__(
        self,
        max_lags: int,
        forecast_horizon: int,
        adf_pvalue_max: float,
        kpss_pvalue_min: float,
        ljungbox_pvalue_min: float,
        jarque_bera_pvalue_min: float,
        ffill_limit: int,
        diff_subreddits: List[str]
    ):
        self.max_lags = max_lags
        self.forecast_horizon = forecast_horizon
        self.adf_pvalue_max = adf_pvalue_max
        self.kpss_pvalue_min = kpss_pvalue_min
        self.ljungbox_pvalue_min = ljungbox_pvalue_min
        self.jarque_bera_pvalue_min = jarque_bera_pvalue_min
        self.ffill_limit = ffill_limit
        self.diff_subreddits = diff_subreddits

    def _validate_stationarity(self, ts_matrix: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Validate stationarity with ADF/KPSS (p<0.05 for ADF, p>0.05 for KPSS)."""
        results = {}
        for col in ts_matrix.columns:
            series = ts_matrix[col].dropna()
            adf_stat, adf_p, _, _, _, _ = adfuller(series, autolag='AIC')
            kpss_stat, kpss_p, _, _ = kpss(series, regression='c', nlags='auto')
            results[col] = {
                'adf_p': float(adf_p),
                'kpss_p': float(kpss_p)
            }
        failures = [
            k for k, v in results.items()
            if not (v['adf_p'] < self.adf_pvalue_max and v['kpss_p'] > self.kpss_pvalue_min)
        ]
        if failures:
            logger.error("Non-stationary series details (ADF/KPSS thresholds):")
            for name in failures:
                logger.error(
                    "%s: ADF p=%.6f (max %.2f), KPSS p=%.6f (min %.2f)",
                    name,
                    results[name]['adf_p'],
                    self.adf_pvalue_max,
                    results[name]['kpss_p'],
                    self.kpss_pvalue_min
                )
            raise ValueError(f"Non-stationary series detected (ADF/KPSS): {failures}")
        return results

    def _validate_var_residuals(self, results) -> Dict[str, Dict[str, float]]:
        """Validate VAR residuals with Ljung-Box and Jarque-Bera tests."""
        resid = pd.DataFrame(results.resid, columns=results.names)
        n_obs = resid.shape[0]
        max_lags = max(1, min(10, n_obs // 5))
        report = {}
        for col in resid.columns:
            series = resid[col].dropna()
            lb_p = acorr_ljungbox(series, lags=[max_lags], return_df=True)['lb_pvalue'].iloc[0]
            jb_stat, jb_p = jarque_bera(series)
            report[col] = {
                'ljungbox_p': float(lb_p),
                'jarque_bera_p': float(jb_p)
            }
        lb_fail = [k for k, v in report.items() if v['ljungbox_p'] < self.ljungbox_pvalue_min]
        jb_fail = [k for k, v in report.items() if v['jarque_bera_p'] < self.jarque_bera_pvalue_min]
        if lb_fail:
            raise ValueError(f"VAR residual autocorrelation detected (Ljung-Box): {lb_fail}")
        if jb_fail:
            raise ValueError(f"VAR residual non-normality detected (Jarque-Bera): {jb_fail}")
        return report

    def prepare_time_series(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Prepare a wide time series matrix for VAR/FEVD"""
        logger.info("Preparing time series for Diebold-Yilmaz spillover analysis...")

        if 'post_created_utc' not in df.columns:
            raise ValueError("Missing required timestamp column: post_created_utc")

        df = df.copy()
        df['post_created_utc'] = pd.to_datetime(df['post_created_utc'])

        pivot = (
            df.pivot_table(
                index='post_created_utc',
                columns='subreddit',
                values='compound_sentiment',
                aggfunc='mean'
            )
            .sort_index()
        )

        pivot = pivot.resample(freq).mean()
        if self.ffill_limit > 0:
            pivot = pivot.ffill(limit=self.ffill_limit)

        if self.diff_subreddits:
            missing = [s for s in self.diff_subreddits if s not in pivot.columns]
            if missing:
                raise ValueError(f"Requested diff subreddits not in data: {missing}")
            pivot[self.diff_subreddits] = pivot[self.diff_subreddits].diff()

        pivot = pivot.dropna()

        if pivot.shape[0] <= self.max_lags:
            raise ValueError(
                f"Insufficient observations for VAR: {pivot.shape[0]} rows with max_lags={self.max_lags}"
            )

        return pivot

    def compute_spillover_matrix(self, ts_matrix: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Compute DY spillover matrix from VAR FEVD"""
        stationarity_report = self._validate_stationarity(ts_matrix)
        model = VAR(ts_matrix)
        results = model.fit(self.max_lags)

        if not results.is_stable():
            raise ValueError("VAR model is unstable (roots outside unit circle)")

        residuals_report = self._validate_var_residuals(results)

        fevd = results.fevd(self.forecast_horizon)
        decomposition = fevd.decomp[-1]

        if decomposition.shape[0] != decomposition.shape[1]:
            raise ValueError("FEVD decomposition matrix must be square")

        row_sums = decomposition.sum(axis=1, keepdims=True)
        if np.any(row_sums == 0):
            raise ValueError("FEVD row sums contain zeros, cannot normalize")

        normalized = decomposition / row_sums

        spillover = pd.DataFrame(
            normalized,
            index=ts_matrix.columns,
            columns=ts_matrix.columns
        )

        off_diag = spillover.values.copy()
        np.fill_diagonal(off_diag, 0.0)
        total_spillover = off_diag.sum() / spillover.shape[0]

        metrics = {
            'total_spillover': float(total_spillover),
            'stationarity': stationarity_report,
            'residuals': residuals_report
        }

        return spillover, metrics

    def build_spillover_network(self, spillover_matrix: pd.DataFrame) -> nx.DiGraph:
        """Build directed network from DY spillover matrix"""
        G = nx.DiGraph()
        subreddits = list(spillover_matrix.index)

        for subreddit in subreddits:
            G.add_node(subreddit)

        for source in subreddits:
            for target in subreddits:
                if source == target:
                    continue
                weight = float(spillover_matrix.loc[target, source])
                G.add_edge(source, target, weight=weight)

        self._compute_network_metrics(G, spillover_matrix)
        self._log_network_summary(G)

        return G

    def _compute_network_metrics(self, G: nx.DiGraph, spillover_matrix: pd.DataFrame):
        """Compute network centrality metrics"""

        # In-degree and out-degree
        in_degrees = dict(G.in_degree(weight='weight'))
        out_degrees = dict(G.out_degree(weight='weight'))

        # PageRank centrality
        pagerank = nx.pagerank(G, weight='weight')

        # Clustering coefficient
        clustering = nx.clustering(G.to_undirected(), weight='weight')

        diag = pd.Series(np.diag(spillover_matrix.values), index=spillover_matrix.index)
        to_spillover = spillover_matrix.sum(axis=0) - diag
        from_spillover = spillover_matrix.sum(axis=1) - diag
        net_spillover = to_spillover - from_spillover

        # Add metrics as node attributes
        for node in G.nodes():
            G.nodes[node]['in_degree'] = in_degrees.get(node, 0)
            G.nodes[node]['out_degree'] = out_degrees.get(node, 0)
            G.nodes[node]['net_spillover'] = float(net_spillover.loc[node])
            G.nodes[node]['pagerank'] = pagerank.get(node, 0)
            G.nodes[node]['clustering'] = clustering.get(node, 0)
            G.nodes[node]['to_spillover'] = float(to_spillover.loc[node])
            G.nodes[node]['from_spillover'] = float(from_spillover.loc[node])

    def _log_network_summary(self, G: nx.DiGraph, top_k: int = 5) -> None:
        """Log a compact summary of the spillover network for sanity checks."""
        if G.number_of_nodes() == 0:
            logger.warning("Spillover network is empty")
            return

        edges = [
            (u, v, float(data.get("weight", 0.0)))
            for u, v, data in G.edges(data=True)
        ]
        edges_sorted = sorted(
            edges,
            key=lambda item: item[2],
            reverse=True,
        )
        top_edges = edges_sorted[:top_k]
        node_spillover = sorted(
            ((node, float(G.nodes[node].get("net_spillover", 0.0))) for node in G.nodes()),
            key=lambda item: item[1],
            reverse=True,
        )
        top_net = node_spillover[:top_k]

        logger.info("Spillover network summary: nodes=%d edges=%d", G.number_of_nodes(), G.number_of_edges())
        if edges:
            weights = np.array([w for _, _, w in edges], dtype=float)
            logger.info(
                "Spillover edge weights: min=%.6f mean=%.6f max=%.6f",
                weights.min(),
                weights.mean(),
                weights.max(),
            )
        logger.info("Top spillover edges: %s", top_edges)
        logger.info("Top net_spillover nodes: %s", top_net)

    def get_spillover_matrix(self, G: nx.DiGraph) -> pd.DataFrame:
        """Extract spillover matrix from network"""
        nodes = list(G.nodes())
        n = len(nodes)

        spillover_matrix = np.zeros((n, n))
        for i, source in enumerate(nodes):
            for j, target in enumerate(nodes):
                if G.has_edge(source, target):
                    spillover_matrix[i, j] = G[source][target]['weight']

        return pd.DataFrame(spillover_matrix, index=nodes, columns=nodes)


class HierarchicalDataProcessor:
    """
    Main processor combining all hierarchical data processing components
    """

    def __init__(self, bigquery_client: BigQueryClient = None, temporal_windows: List[int] = None):
        self.bq_client = bigquery_client or BigQueryClient()
        self.validator = DataQualityValidator()
        if temporal_windows is None:
            temporal_windows = os.getenv("TEMPORAL_WINDOWS_MINUTES")
            if not temporal_windows:
                raise ValueError("Missing required env var: TEMPORAL_WINDOWS_MINUTES")
            window_minutes = [int(x.strip()) for x in temporal_windows.split(",") if x.strip()]
        else:
            window_minutes = temporal_windows
        if not window_minutes:
            raise ValueError("TEMPORAL_WINDOWS_MINUTES must contain at least one integer")

        interval_seconds = os.getenv("SENTIMENT_AGG_INTERVAL_SECONDS")
        if not interval_seconds:
            raise ValueError("Missing required env var: SENTIMENT_AGG_INTERVAL_SECONDS")
        interval_seconds = int(interval_seconds)

        dy_resample_freq = os.getenv("DY_RESAMPLE_FREQ")
        if not dy_resample_freq:
            raise ValueError("Missing required env var: DY_RESAMPLE_FREQ")
        self.dy_resample_freq = dy_resample_freq

        dy_max_lags = os.getenv("DY_MAX_LAGS")
        if not dy_max_lags:
            raise ValueError("Missing required env var: DY_MAX_LAGS")
        dy_max_lags = int(dy_max_lags)

        dy_forecast_horizon = os.getenv("DY_FORECAST_HORIZON")
        if not dy_forecast_horizon:
            raise ValueError("Missing required env var: DY_FORECAST_HORIZON")
        dy_forecast_horizon = int(dy_forecast_horizon)

        dy_adf_pvalue_max = os.getenv("DY_ADF_PVALUE_MAX")
        if not dy_adf_pvalue_max:
            raise ValueError("Missing required env var: DY_ADF_PVALUE_MAX")
        dy_adf_pvalue_max = float(dy_adf_pvalue_max)

        dy_kpss_pvalue_min = os.getenv("DY_KPSS_PVALUE_MIN")
        if not dy_kpss_pvalue_min:
            raise ValueError("Missing required env var: DY_KPSS_PVALUE_MIN")
        dy_kpss_pvalue_min = float(dy_kpss_pvalue_min)

        dy_ljungbox_pvalue_min = os.getenv("DY_LJUNGBOX_PVALUE_MIN")
        if not dy_ljungbox_pvalue_min:
            raise ValueError("Missing required env var: DY_LJUNGBOX_PVALUE_MIN")
        dy_ljungbox_pvalue_min = float(dy_ljungbox_pvalue_min)

        dy_jarque_bera_pvalue_min = os.getenv("DY_JARQUE_BERA_PVALUE_MIN")
        if not dy_jarque_bera_pvalue_min:
            raise ValueError("Missing required env var: DY_JARQUE_BERA_PVALUE_MIN")
        dy_jarque_bera_pvalue_min = float(dy_jarque_bera_pvalue_min)

        dy_ffill_limit = os.getenv("DY_FFILL_LIMIT")
        if not dy_ffill_limit:
            raise ValueError("Missing required env var: DY_FFILL_LIMIT")
        dy_ffill_limit = int(dy_ffill_limit)

        diff_subs_env = os.getenv("DY_DIFF_SUBREDDITS", "")
        diff_subreddits = [s.strip() for s in diff_subs_env.split(",") if s.strip()]

        feature_lags = os.getenv("FEATURE_LAGS")
        if not feature_lags:
            raise ValueError("Missing required env var: FEATURE_LAGS")
        self.feature_lags = [int(x.strip()) for x in feature_lags.split(",") if x.strip()]
        if not self.feature_lags:
            raise ValueError("FEATURE_LAGS must contain at least one integer")

        price_adf_pvalue_max = os.getenv("PRICE_ADF_PVALUE_MAX")
        if not price_adf_pvalue_max:
            raise ValueError("Missing required env var: PRICE_ADF_PVALUE_MAX")
        self.price_adf_pvalue_max = float(price_adf_pvalue_max)

        price_kpss_pvalue_min = os.getenv("PRICE_KPSS_PVALUE_MIN")
        if not price_kpss_pvalue_min:
            raise ValueError("Missing required env var: PRICE_KPSS_PVALUE_MIN")
        self.price_kpss_pvalue_min = float(price_kpss_pvalue_min)

        return_direction_threshold = os.getenv("RETURN_DIRECTION_THRESHOLD")
        if not return_direction_threshold:
            raise ValueError("Missing required env var: RETURN_DIRECTION_THRESHOLD")
        self.return_direction_threshold = float(return_direction_threshold)

        self.price_resample_freq = self.dy_resample_freq

        price_merge_tolerance = os.getenv("PRICE_MERGE_TOLERANCE")
        if not price_merge_tolerance:
            raise ValueError("Missing required env var: PRICE_MERGE_TOLERANCE")
        self.price_merge_tolerance = price_merge_tolerance

        subreddits_env = os.getenv("SUBREDDITS_SELECTED")
        if not subreddits_env:
            raise ValueError("Missing required env var: SUBREDDITS_SELECTED")
        self.selected_subreddits = [s.strip() for s in subreddits_env.split(",") if s.strip()]
        if not self.selected_subreddits:
            raise ValueError("SUBREDDITS_SELECTED must contain at least one subreddit")

        # Load subreddit -> symbol mapping for price merge
        symbol_map_env = os.getenv("SUBREDDIT_SYMBOL_MAP")
        if not symbol_map_env:
            raise ValueError("Missing required env var: SUBREDDIT_SYMBOL_MAP")
        # Parse format: "subreddit1:SYMBOL1,subreddit2:SYMBOL2,..."
        self.subreddit_symbol_map = {}
        for pair in symbol_map_env.split(","):
            if ":" not in pair:
                raise ValueError(f"Invalid SUBREDDIT_SYMBOL_MAP format, expected 'subreddit:SYMBOL': {pair}")
            subreddit, symbol = pair.strip().split(":", 1)
            self.subreddit_symbol_map[subreddit.strip()] = symbol.strip()

        # Validate all selected subreddits have symbol mapping
        unmapped = [s for s in self.selected_subreddits if s not in self.subreddit_symbol_map]
        if unmapped:
            raise ValueError(f"Subreddits missing symbol mapping: {unmapped}")

        self.transform_validator = TransformValidator(strict=True)

        self.feature_engineer = HierarchicalFeatureEngineer(
            window_minutes=window_minutes,
            interval_seconds=interval_seconds
        )
        self.network_builder = DieboldYilmazSpilloverBuilder(
            max_lags=dy_max_lags,
            forecast_horizon=dy_forecast_horizon,
            adf_pvalue_max=dy_adf_pvalue_max,
            kpss_pvalue_min=dy_kpss_pvalue_min,
            ljungbox_pvalue_min=dy_ljungbox_pvalue_min,
            jarque_bera_pvalue_min=dy_jarque_bera_pvalue_min,
            ffill_limit=dy_ffill_limit,
            diff_subreddits=diff_subreddits
        )

        self.processing_log = {}

    def load_and_validate_data(self, start_date: str = None,
                             end_date: str = None) -> pd.DataFrame:
        """Load data from BigQuery and perform validation with strict quality checks"""
        logger.info("Loading and validating data...")

        # Load raw data
        if not hasattr(self.bq_client, 'get_post_sentiment_aggregation'):
            raise ValueError("BigQuery client does not implement get_post_sentiment_aggregation")

        df = self.bq_client.get_post_sentiment_aggregation(
            start_date, end_date, subreddits=self.selected_subreddits
        )
        # Rename 'date' to 'post_created_utc' for consistency
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'post_created_utc'})

        if 'avg_compound_sentiment' not in df.columns:
            raise ValueError("Missing required column from BigQuery: avg_compound_sentiment")
        df['compound_sentiment'] = df['avg_compound_sentiment'].astype(float)

        # Map other columns for consistency
        column_mapping = {
            'avg_post_score': 'score',
            'num_posts': 'post_count',
            'num_comments': 'comment_count'
        }
        df = df.rename(columns=column_mapping)

        # For aggregated data, create sentiment_score from compound_sentiment
        if 'sentiment_score' not in df.columns:
            df['sentiment_score'] = df['compound_sentiment']

        if df.empty:
            raise ValueError("No data returned from BigQuery")

        # Track raw data count for validation report
        if 'raw_reddit_rows' not in self.processing_log:
            self.processing_log['raw_reddit_rows'] = 0
        self.processing_log['raw_reddit_rows'] += len(df)

        # Data validation
        logger.info("Performing data quality validation...")

        # Remove duplicates
        initial_size = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_size - len(df)

        # Validate data quality
        timestamp_report = self.validator.validate_timestamps(df)
        sentiment_report = self.validator.validate_sentiment_scores(
            df, 'sentiment_score', None
        )
        coverage_report = self.validator.validate_subreddit_coverage(df)

        # Log validation results
        self.processing_log['validation'] = {
            'duplicates_removed': duplicates_removed,
            'timestamp_validation': timestamp_report,
            'sentiment_validation': sentiment_report,
            'coverage_validation': coverage_report
        }

        logger.info(f"Data validation complete. Records: {len(df)}")
        logger.info(f"Time range: {timestamp_report['time_range']['start']} to {timestamp_report['time_range']['end']}")
        logger.info(f"Subreddits: {coverage_report['total_subreddits']}")

        return df

    def _enforce_minimums(self, df: pd.DataFrame, timestamp_report: Dict, coverage_report: Dict) -> None:
        # STRICT DATA QUALITY CHECKS - Reject insufficient data
        min_records = 100  # Minimum records for meaningful analysis
        min_subreddits = 3  # Minimum subreddits for spillover analysis
        min_days = 5       # Minimum days for temporal analysis

        if len(df) < min_records:
            raise ValueError(
                f"❌ INSUFFICIENT DATA: Only {len(df)} records found, need at least {min_records} for meaningful analysis"
            )

        if coverage_report['total_subreddits'] < min_subreddits:
            raise ValueError(
                f"❌ INSUFFICIENT SUBREDDITS: Only {coverage_report['total_subreddits']} subreddits found, need at least {min_subreddits} for spillover analysis"
            )

        # Check temporal coverage
        date_range = pd.to_datetime(timestamp_report['time_range']['end']) - pd.to_datetime(timestamp_report['time_range']['start'])
        if date_range.days < min_days:
            raise ValueError(
                f"❌ INSUFFICIENT TIME COVERAGE: Only {date_range.days} days of data, need at least {min_days} days for temporal analysis"
            )

        # Check for data density per subreddit
        subreddit_counts = df.groupby('subreddit').size()
        min_per_subreddit = 10
        sparse_subreddits = subreddit_counts[subreddit_counts < min_per_subreddit]
        if len(sparse_subreddits) == len(subreddit_counts):
            raise ValueError(
                f"❌ SPARSE DATA: All subreddits have <{min_per_subreddit} records. Cannot build reliable spillover model."
            )

        logger.info("✅ Data quality checks PASSED - sufficient data for analysis")

    def process_hierarchical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all hierarchical features"""
        logger.info("Processing hierarchical features...")

        # Level 1: Post-level features
        df = self.feature_engineer.compute_compound_sentiment(df)

        # Level 2: Temporal aggregates
        df = self.feature_engineer.compute_temporal_aggregates(df)

        # Log feature engineering results
        self.processing_log['feature_engineering'] = {
            'features_created': [col for col in df.columns if any(
                pattern in col for pattern in ['sentiment_', 'emotion_', 'ratio_', 'count_']
            )],
            'feature_count': len([col for col in df.columns if any(
                pattern in col for pattern in ['sentiment_', 'emotion_', 'ratio_', 'count_']
            )])
        }

        return df

    def _apply_network_features(self, df: pd.DataFrame, G: nx.DiGraph) -> pd.DataFrame:
        """Attach DY network metrics and lagged network features to data."""
        network_features = []
        for subreddit in df['subreddit'].unique():
            if subreddit in G.nodes():
                subreddit_data = df[df['subreddit'] == subreddit].copy()

                # Add network metrics as features
                subreddit_data['net_spillover'] = G.nodes[subreddit]['net_spillover']
                subreddit_data['pagerank'] = G.nodes[subreddit]['pagerank']
                subreddit_data['in_degree'] = G.nodes[subreddit]['in_degree']
                subreddit_data['out_degree'] = G.nodes[subreddit]['out_degree']
                subreddit_data['clustering'] = G.nodes[subreddit]['clustering']
                subreddit_data['to_spillover'] = G.nodes[subreddit]['to_spillover']
                subreddit_data['from_spillover'] = G.nodes[subreddit]['from_spillover']

                network_features.append(subreddit_data)

        # Combine results
        if not network_features:
            raise ValueError("No network features computed: spillover network contains no matching nodes")
        df_with_network = pd.concat(network_features, ignore_index=True)
        lag_sources = ['compound_change', 'compound_shock']
        for lag in self.feature_lags:
            for col in lag_sources:
                df_with_network[f'{col}_lag{lag}'] = df_with_network.groupby('subreddit')[col].shift(lag)

        return df_with_network

    def build_network_features(self, df: pd.DataFrame,
                             network_freq: Optional[str] = None) -> Tuple[pd.DataFrame, nx.DiGraph]:
        """Build network structure features"""
        logger.info("Building Diebold-Yilmaz spillover network features...")

        if network_freq is None:
            network_freq = self.dy_resample_freq

        ts_matrix = self.network_builder.prepare_time_series(df, network_freq)
        spillover_matrix, spillover_metrics = self.network_builder.compute_spillover_matrix(ts_matrix)
        G = self.network_builder.build_spillover_network(spillover_matrix)

        # Add network features to dataframe
        df_with_network = self._apply_network_features(df, G)

        # Log network building results
        self.processing_log['network_building'] = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'connected_components': len(list(nx.weakly_connected_components(G))),
            'total_spillover': spillover_metrics.get('total_spillover')
        }

        return df_with_network, G

    def _compute_batch_overlap_days(self) -> Tuple[int, int]:
        max_window = max(self.feature_engineer.window_minutes)
        max_lag = max(self.feature_lags) if self.feature_lags else 0
        lag_minutes = (max_lag * self.feature_engineer.interval_seconds) / 60.0
        start_overlap_minutes = max_window + lag_minutes
        start_overlap_days = int(np.ceil(start_overlap_minutes / (60.0 * 24.0)))

        try:
            end_overlap_minutes = pd.Timedelta(self.dy_resample_freq).total_seconds() / 60.0
        except ValueError:
            end_overlap_minutes = pd.tseries.frequencies.to_offset(self.dy_resample_freq).delta.total_seconds() / 60.0
        end_overlap_days = int(np.ceil(end_overlap_minutes / (60.0 * 24.0)))

        return max(0, start_overlap_days), max(0, end_overlap_days)

    def process_full_pipeline_chunked(
        self,
        start_date: str,
        end_date: str,
        include_price_targets: bool = True,
        batch_days: int = 60,
        keep_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, nx.DiGraph, Dict]:
        """Run the pipeline in time batches with a global DY network."""
        logger.info("Starting chunked hierarchical data processing pipeline...")
        if batch_days <= 0:
            raise ValueError("batch_days must be a positive integer")

        start_time = datetime.now()
        start_dt = pd.to_datetime(start_date).date()
        end_dt = pd.to_datetime(end_date).date()
        if end_dt < start_dt:
            raise ValueError("end_date must be >= start_date")

        start_overlap_days, end_overlap_days = self._compute_batch_overlap_days()

        # Build global spillover network once
        df_network = self.load_and_validate_data(start_date, end_date)
        timestamp_report = self.validator.validate_timestamps(df_network, timestamp_col='post_created_utc')
        coverage_report = self.validator.validate_subreddit_coverage(df_network)
        self._enforce_minimums(df_network, timestamp_report, coverage_report)
        ts_matrix = self.network_builder.prepare_time_series(df_network, self.dy_resample_freq)
        spillover_matrix, spillover_metrics = self.network_builder.compute_spillover_matrix(ts_matrix)
        G = self.network_builder.build_spillover_network(spillover_matrix)

        self.processing_log['network_building'] = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'connected_components': len(list(nx.weakly_connected_components(G))),
            'total_spillover': spillover_metrics.get('total_spillover')
        }
        del df_network
        if include_price_targets:
            price_data_full = self.bq_client.get_price_data(start_date=start_date, end_date=end_date)
            self.processing_log['raw_price_rows'] = len(price_data_full)
            self._validate_price_log_return_stationarity(price_data_full)
            del price_data_full

        chunks = []
        current = start_dt
        while current <= end_dt:
            chunk_start = current
            chunk_end = min(end_dt, current + timedelta(days=batch_days - 1))
            query_start = chunk_start - timedelta(days=start_overlap_days)
            query_end = chunk_end + timedelta(days=end_overlap_days)

            logger.info("Processing chunk %s to %s (query %s to %s)", chunk_start, chunk_end, query_start, query_end)
            df = self.load_and_validate_data(query_start.isoformat(), query_end.isoformat())
            df = self.process_hierarchical_features(df)

            if include_price_targets:
                price_data = self.bq_client.get_price_data(
                    start_date=query_start.isoformat(),
                    end_date=query_end.isoformat()
                )
                df = self.create_targets(df, price_data)
            else:
                df = self.create_targets(df)

            df = self._apply_network_features(df, G)

            df = df.dropna(subset=['compound_sentiment', 'post_created_utc'])
            df['post_created_utc'] = pd.to_datetime(df['post_created_utc'])
            mask = df['post_created_utc'].dt.date.between(chunk_start, chunk_end)
            df = df.loc[mask].copy()

            if keep_columns:
                missing = [c for c in keep_columns if c not in df.columns]
                if missing:
                    raise ValueError(f"Missing required columns for keep_columns: {missing}")
                df = df[keep_columns]

            chunks.append(df)
            current = chunk_end + timedelta(days=1)

        if not chunks:
            raise ValueError("No chunks processed; check date range")

        df_final = pd.concat(chunks, ignore_index=True)
        timestamp_report = self.validator.validate_timestamps(df_final, timestamp_col='post_created_utc')
        coverage_report = self.validator.validate_subreddit_coverage(df_final)
        self._enforce_minimums(df_final, timestamp_report, coverage_report)
        if include_price_targets:
            self._validate_next_return_stationarity(df_final)
        processing_time = datetime.now() - start_time

        self.processing_log.update({
            'processing_time_seconds': processing_time.total_seconds(),
            'final_dataset_shape': df_final.shape,
            'final_features': list(df_final.columns),
            'status': 'success'
        })

        logger.info("Chunked pipeline completed successfully in %s", processing_time)
        logger.info("Final dataset shape: %s", df_final.shape)

        # Generate and save validation report + processed data
        raw_reddit_count = self.processing_log.get('raw_reddit_rows', 0)
        raw_price_count = self.processing_log.get('raw_price_rows', 0)
        report = self.generate_validation_report(raw_reddit_count, raw_price_count, df_final)
        logger.info("\n" + report)

        # Save processed data to CSV
        from datetime import datetime as dt
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'processed')
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f'processed_data_{dt.now().strftime("%Y%m%d_%H%M%S")}.csv')
        df_final.to_csv(csv_path, index=False)
        logger.info(f"Processed data saved to: {csv_path}")

        return df_final, G, self.processing_log

    def create_targets(self, df: pd.DataFrame, price_data: pd.DataFrame = None,
                       network_graph: nx.DiGraph = None) -> pd.DataFrame:
        """
        Create target variables for prediction.

        NEW APPROACH: Price series is the main axis (regular 2h grid).
        Reddit data is aggregated to same 2h buckets and merged.
        - Sentiment columns: forward-filled where no posts in bucket
        - Activity columns: filled with 0 where no posts in bucket
        - Changes computed AFTER filling
        """
        logger.info("Creating target variables...")

        if price_data is None:
            raise ValueError("price_data is required for create_targets")

        # Standardize datetime types
        if 'snapped_at' in price_data.columns:
            price_data['snapped_at'] = pd.to_datetime(price_data['snapped_at']).dt.tz_localize(None)

        # --- Step 1: Create regular price grid (main axis) per symbol ---
        logger.info("Step 1: Creating price grid as main axis...")
        price_data = price_data.sort_values(['symbol', 'snapped_at']).copy()

        price_resampled_list = []
        for symbol, sdf in price_data.groupby('symbol'):
            sdf = sdf.set_index('snapped_at').sort_index()
            sdf = sdf.resample(self.price_resample_freq).last()

            before_dropna = len(sdf)
            sdf = sdf.dropna(subset=['price'])
            after_dropna = len(sdf)

            if before_dropna > 0 and before_dropna != after_dropna:
                drop_pct = (before_dropna - after_dropna) / before_dropna
                if drop_pct > 0.1:
                    logger.warning(f"Symbol {symbol}: dropped {drop_pct:.1%} periods without price")

            # Compute log returns: log(price_t+1 / price_t)
            sdf['price_log_return'] = np.log(sdf['price']).diff().shift(-1)
            sdf['symbol'] = symbol
            sdf = sdf.reset_index().rename(columns={'snapped_at': 'timestamp'})
            price_resampled_list.append(sdf)

        if not price_resampled_list:
            raise ValueError("No price data after resampling")

        price_grid = pd.concat(price_resampled_list, ignore_index=True)
        logger.info(f"Price grid: {len(price_grid)} rows, symbols: {price_grid['symbol'].unique().tolist()}")

        # --- Step 2: Aggregate Reddit data to same 2h buckets per subreddit ---
        logger.info("Step 2: Aggregating Reddit data to 2h buckets...")

        if 'post_created_utc' not in df.columns:
            raise ValueError("Missing required column: post_created_utc")
        df['post_created_utc'] = pd.to_datetime(df['post_created_utc']).dt.tz_localize(None)

        # Add symbol based on subreddit mapping
        df['symbol'] = df['subreddit'].map(self.subreddit_symbol_map)
        unmapped = df['symbol'].isna().sum()
        if unmapped > 0:
            unmapped_subs = df[df['symbol'].isna()]['subreddit'].unique().tolist()
            raise ValueError(f"Unmapped subreddits: {unmapped_subs}")

        # Create timestamp bucket (floor to resample freq)
        df['timestamp'] = df['post_created_utc'].dt.floor(self.price_resample_freq)

        # Aggregate per (subreddit, timestamp bucket)
        # Sentiment: mean in bucket, Activity: sum in bucket
        agg_dict = {
            'compound_sentiment': 'mean',
            'post_count': 'sum',
            'comment_count': 'sum',
        }
        # Add score if available
        if 'score' in df.columns:
            agg_dict['score'] = 'sum'

        reddit_agg = df.groupby(['subreddit', 'symbol', 'timestamp']).agg(agg_dict).reset_index()
        reddit_agg['activity'] = reddit_agg['post_count'] + reddit_agg['comment_count']

        logger.info(f"Reddit aggregated: {len(reddit_agg)} bucket-subreddit pairs")

        # --- Step 3: Merge price grid with Reddit data per symbol ---
        logger.info("Step 3: Merging price grid with Reddit data...")

        # For each symbol, get all subreddits that map to it
        symbol_to_subreddits = {}
        for sub, sym in self.subreddit_symbol_map.items():
            if sym not in symbol_to_subreddits:
                symbol_to_subreddits[sym] = []
            symbol_to_subreddits[sym].append(sub)

        merged_parts = []
        for symbol in price_grid['symbol'].unique():
            price_sym = price_grid[price_grid['symbol'] == symbol].copy()
            subreddits_for_symbol = symbol_to_subreddits.get(symbol, [])

            if not subreddits_for_symbol:
                logger.warning(f"No subreddits mapped to symbol {symbol}")
                continue

            for subreddit in subreddits_for_symbol:
                reddit_sub = reddit_agg[
                    (reddit_agg['subreddit'] == subreddit) &
                    (reddit_agg['symbol'] == symbol)
                ].copy()

                # Build columns to merge
                merge_cols = ['timestamp', 'compound_sentiment', 'activity']
                if 'score' in reddit_sub.columns:
                    merge_cols.append('score')

                # Left merge: keep all price timestamps, join Reddit where available
                merged = price_sym.merge(
                    reddit_sub[merge_cols],
                    on='timestamp',
                    how='left'
                )
                merged['subreddit'] = subreddit

                # Fill missing values:
                # - activity: 0 (no posts = zero activity)
                # - compound_sentiment: forward fill (last known sentiment)
                merged['activity'] = merged['activity'].fillna(0)
                merged['compound_sentiment'] = merged['compound_sentiment'].ffill()

                if 'score' in merged.columns:
                    merged['score'] = merged['score'].fillna(0)

                merged_parts.append(merged)

        if not merged_parts:
            raise ValueError("No data after merge - check symbol mapping")

        result = pd.concat(merged_parts, ignore_index=True)
        logger.info(f"Merged dataset: {len(result)} rows")

        # --- Step 4: Compute derived features AFTER filling ---
        logger.info("Step 4: Computing derived features on filled data...")

        result = result.sort_values(['subreddit', 'timestamp'])

        # Activity change (log scale)
        result['activity_log'] = np.log1p(result['activity'])
        result['activity_change'] = result.groupby('subreddit')['activity_log'].diff()

        # Sentiment change
        result['compound_change'] = result.groupby('subreddit')['compound_sentiment'].diff()

        # Compound shock (change normalized by rolling std)
        window_bins = max(1, int(round((min(self.feature_engineer.window_minutes) * 60) / self.feature_engineer.interval_seconds)))
        result['compound_shock'] = result.groupby('subreddit').apply(
            lambda g: g['compound_change'] / g['compound_sentiment'].rolling(window_bins, min_periods=1).std().replace(0, np.nan),
            include_groups=False
        ).reset_index(level=0, drop=True)
        result['compound_shock'] = result['compound_shock'].replace([np.inf, -np.inf], np.nan)

        # --- Step 4c: Add feature lags ---
        logger.info("Adding feature lags...")
        lag_columns = ['compound_change', 'compound_shock']
        for col in lag_columns:
            if col in result.columns:
                for lag in self.feature_lags:
                    result[f'{col}_lag{lag}'] = result.groupby('subreddit')[col].shift(lag)

        # --- Step 5: Create target variables ---
        logger.info("Step 5: Creating target variables...")

        result['next_return'] = result['price_log_return']
        result['return_direction'] = np.where(
            result['next_return'] > self.return_direction_threshold, 1,
            np.where(result['next_return'] < -self.return_direction_threshold, -1, 0)
        )

        # Next sentiment (for sentiment prediction task)
        result['next_sentiment'] = result.groupby('subreddit')['compound_sentiment'].shift(-1)
        result['sentiment_change_target'] = result['next_sentiment'] - result['compound_sentiment']

        # Add post_created_utc alias for backward compatibility with pipeline
        result['post_created_utc'] = result['timestamp']

        # Validate
        valid_returns = result['return_direction'].dropna()
        if not valid_returns.empty:
            self.transform_validator.validate_range(
                valid_returns, 'return_direction', min_val=-1, max_val=1
            )

        # Log coverage stats
        total = len(result)
        with_return = result['next_return'].notna().sum()
        with_sentiment = result['compound_sentiment'].notna().sum()
        logger.info(f"Coverage - next_return: {with_return}/{total} ({100*with_return/total:.1f}%), "
                   f"sentiment: {with_sentiment}/{total} ({100*with_sentiment/total:.1f}%)")

        # Store validation stats for final report
        self._target_validation_stats = {
            'total_rows': total,
            'next_return_coverage': with_return,
            'next_return_coverage_pct': 100 * with_return / total if total > 0 else 0,
            'sentiment_coverage': with_sentiment,
            'sentiment_coverage_pct': 100 * with_sentiment / total if total > 0 else 0,
            'symbols': result['symbol'].unique().tolist(),
            'subreddits': result['subreddit'].unique().tolist(),
            'time_range': (result['timestamp'].min(), result['timestamp'].max()),
            'return_direction_dist': result['return_direction'].value_counts().to_dict(),
        }

        return result

    def generate_validation_report(self, raw_reddit_count: int, raw_price_count: int,
                                   processed_df: pd.DataFrame, output_dir: str = None) -> str:
        """
        Generate final validation report comparing raw data to processed output.
        Saves to project folder and returns report string.
        """
        from datetime import datetime

        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'reports')
        os.makedirs(output_dir, exist_ok=True)

        if 'timestamp' in processed_df.columns:
            time_col = 'timestamp'
        elif 'post_created_utc' in processed_df.columns:
            time_col = 'post_created_utc'
        else:
            raise ValueError("Missing timestamp column for validation report")

        report_lines = [
            "=" * 70,
            "PIPELINE VALIDATION REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
            "1. RAW DATA INPUT",
            "-" * 40,
            f"   Reddit posts/comments: {raw_reddit_count:,}",
            f"   Price records: {raw_price_count:,}",
            "",
            "2. PROCESSED OUTPUT",
            "-" * 40,
            f"   Final rows: {len(processed_df):,}",
            f"   Columns: {len(processed_df.columns)}",
            f"   Symbols: {processed_df['symbol'].nunique()} - {processed_df['symbol'].unique().tolist()}",
            f"   Subreddits: {processed_df['subreddit'].nunique()} - {processed_df['subreddit'].unique().tolist()}",
            "",
            "3. TIME RANGE",
            "-" * 40,
            f"   Start: {processed_df[time_col].min()}",
            f"   End: {processed_df[time_col].max()}",
            f"   Duration: {processed_df[time_col].max() - processed_df[time_col].min()}",
            "",
            "4. COVERAGE STATS",
            "-" * 40,
        ]

        feature_env = os.getenv("MODEL_FEATURE_COLUMNS", "")
        feature_columns = [c.strip() for c in feature_env.split(",") if c.strip()]
        report_features = [c for c in feature_columns if c in processed_df.columns]

        # Per-subreddit stats
        for sub in sorted(processed_df['subreddit'].unique()):
            sub_df = processed_df[processed_df['subreddit'] == sub]
            total = len(sub_df)
            report_lines.append(f"   {sub}:")
            report_lines.append(f"      Rows: {total:,}")
            if 'activity' in sub_df.columns:
                with_activity = (sub_df['activity'] > 0).sum()
                report_lines.append(f"      With activity: {with_activity:,} ({100*with_activity/total:.1f}%)")
            elif 'post_count' in sub_df.columns and 'comment_count' in sub_df.columns:
                with_activity = ((sub_df['post_count'] + sub_df['comment_count']) > 0).sum()
                report_lines.append(f"      With activity: {with_activity:,} ({100*with_activity/total:.1f}%)")
            else:
                report_lines.append("      With activity: N/A (missing column)")

        report_lines.extend([
            "",
            "5. TARGET DISTRIBUTION",
            "-" * 40,
            f"   return_direction:",
        ])

        rd_counts = processed_df['return_direction'].value_counts().sort_index()
        for val, cnt in rd_counts.items():
            label = {-1: 'DOWN', 0: 'NEUTRAL', 1: 'UP'}.get(val, str(val))
            report_lines.append(f"      {label}: {cnt:,} ({100*cnt/len(processed_df):.1f}%)")

        report_lines.extend([
            "",
            "6. FEATURE STATS",
            "-" * 40,
        ])

        if not report_features:
            report_features = [
                c for c in
                ['compound_sentiment', 'compound_change', 'compound_shock',
                 'activity', 'activity_change', 'price_log_return']
                if c in processed_df.columns
            ]
        for feat in report_features:
            if feat in processed_df.columns:
                col = processed_df[feat]
                na_pct = 100 * col.isna().sum() / len(col)
                report_lines.append(f"   {feat}:")
                report_lines.append(f"      NaN: {col.isna().sum():,} ({na_pct:.1f}%)")
                if not col.dropna().empty:
                    report_lines.append(f"      Mean: {col.mean():.4f}, Std: {col.std():.4f}")
                    report_lines.append(f"      Min: {col.min():.4f}, Max: {col.max():.4f}")

        report_lines.extend([
            "",
            "7. DATA QUALITY CHECKS",
            "-" * 40,
        ])

        # Check for issues
        issues = []
        checks_run = False
        if 'compound_sentiment' in processed_df.columns:
            checks_run = True
            if processed_df['compound_sentiment'].isna().any():
                issues.append("WARNING: NaN values in compound_sentiment after ffill")
        if 'return_direction' in processed_df.columns:
            checks_run = True
            if (processed_df['return_direction'] == 0).sum() / len(processed_df) > 0.9:
                issues.append("WARNING: >90% of return_direction is NEUTRAL")
        if 'activity' in processed_df.columns:
            checks_run = True
            if processed_df['activity'].sum() == 0:
                issues.append("ERROR: All activity is 0")

        if issues:
            for issue in issues:
                report_lines.append(f"   {issue}")
        elif checks_run:
            report_lines.append("   All checks PASSED")
        else:
            report_lines.append("   No checks run (required columns missing)")

        report_lines.extend([
            "",
            "=" * 70,
            "END OF REPORT",
            "=" * 70,
        ])

        report = "\n".join(report_lines)

        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(output_dir, f'validation_report_{timestamp}.txt')
        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"Validation report saved to: {report_path}")

        return report

    def _validate_price_log_return_stationarity(self, price_data: pd.DataFrame) -> None:
        """Validate price log returns are stationary using ADF and KPSS tests."""
        if price_data.empty:
            raise ValueError("No price data available for stationarity checks")
        if 'snapped_at' in price_data.columns:
            price_data['snapped_at'] = pd.to_datetime(price_data['snapped_at']).dt.tz_localize(None).astype('datetime64[ns]')
        price_data = price_data.sort_values(['symbol', 'snapped_at']).copy()
        resampled = []
        for symbol, sdf in price_data.groupby('symbol'):
            before_resample = len(sdf)
            sdf = sdf.set_index('snapped_at').sort_index()
            sdf = sdf.resample(self.price_resample_freq).last()

            # Validate and log dropna
            before_dropna = len(sdf)
            sdf = sdf.dropna(subset=['price'])
            after_dropna = len(sdf)

            if before_dropna > 0 and before_dropna != after_dropna:
                drop_pct = (before_dropna - after_dropna) / before_dropna
                if drop_pct > 0.1:
                    raise TransformValidationError(
                        f"Symbol {symbol}: {drop_pct:.1%} of resampled periods have no price data. "
                        f"This indicates data quality issues."
                    )
                logger.info(f"Stationarity check - {symbol}: {after_dropna} periods after dropna")

            sdf['symbol'] = symbol
            resampled.append(sdf.reset_index())

        if not resampled:
            raise ValueError("No price data available after resampling")
        price_data = pd.concat(resampled, ignore_index=True)
        price_data['price_log_return'] = price_data.groupby('symbol')['price'].transform(
            lambda s: np.log(s).diff().shift(-1)
        )

        # Validate stationarity per symbol
        for symbol, series in price_data.groupby('symbol')['price_log_return']:
            clean = series.dropna()
            if clean.empty:
                raise ValueError(f"No price_log_return data for symbol: {symbol}")

            adf_stat, adf_p, _, _, _, _ = adfuller(clean, autolag='AIC')
            kpss_stat, kpss_p, _, _ = kpss(clean, regression='c', nlags='auto')

            logger.info(
                "Stationarity test %s: ADF stat=%.6f p=%.6f, KPSS stat=%.6f p=%.6f",
                symbol,
                adf_stat,
                adf_p,
                kpss_stat,
                kpss_p,
            )

            if not (adf_p < self.price_adf_pvalue_max and kpss_p > self.price_kpss_pvalue_min):
                raise ValueError(
                    f"Non-stationary price_log_return on full range for {symbol} "
                    f"(ADF p={adf_p:.6f}, KPSS p={kpss_p:.6f}). "
                    f"Required: ADF p<{self.price_adf_pvalue_max}, KPSS p>{self.price_kpss_pvalue_min}"
                )

    def _validate_next_return_stationarity(self, df: pd.DataFrame) -> None:
        """Validate next_return (merged price returns) are stationary per symbol."""
        if 'timestamp' in df.columns:
            time_col = 'timestamp'
        elif 'post_created_utc' in df.columns:
            time_col = 'post_created_utc'
        else:
            raise ValueError("Missing timestamp column for next_return stationarity checks")

        # Deduplicate per (symbol, timestamp) to avoid repeated price returns across subreddits.
        subset = df[['symbol', time_col, 'next_return']].copy()
        subset[time_col] = pd.to_datetime(subset[time_col])

        conflicts = (
            subset.groupby(['symbol', time_col])['next_return']
            .nunique(dropna=True)
        )
        if (conflicts > 1).any():
            bad = conflicts[conflicts > 1].head(5)
            raise ValueError(
                f"Conflicting next_return values for identical (symbol, {time_col}) buckets: "
                f"{bad.to_dict()}"
            )

        deduped = subset.drop_duplicates(subset=['symbol', time_col], keep='first')

        for symbol, series in deduped.groupby('symbol')['next_return']:
            clean = series.dropna()
            total = len(series)
            non_null = len(clean)

            if clean.empty:
                raise ValueError(f"No merged next_return data for symbol: {symbol}")

            # Log coverage
            coverage_pct = non_null / total if total > 0 else 0
            logger.info(f"next_return coverage for {symbol}: {non_null}/{total} ({coverage_pct:.1%})")

            if coverage_pct < 0.5:
                logger.warning(
                    f"Low coverage for {symbol}: only {coverage_pct:.1%} of posts have next_return. "
                    f"Check PRICE_MERGE_TOLERANCE or data alignment."
                )

            adf_stat, adf_p, _, _, _, _ = adfuller(clean, autolag='AIC')
            kpss_stat, kpss_p, _, _ = kpss(clean, regression='c', nlags='auto')

            logger.info(
                "Stationarity test merged %s: ADF stat=%.6f p=%.6f, KPSS stat=%.6f p=%.6f",
                symbol,
                adf_stat,
                adf_p,
                kpss_stat,
                kpss_p,
            )

            if not (adf_p < self.price_adf_pvalue_max and kpss_p > self.price_kpss_pvalue_min):
                raise ValueError(
                    f"Non-stationary next_return after merge for {symbol} "
                    f"(ADF p={adf_p:.6f}, KPSS p={kpss_p:.6f}). "
                    f"Required: ADF p<{self.price_adf_pvalue_max}, KPSS p>{self.price_kpss_pvalue_min}"
                )

    def process_full_pipeline(self, start_date: str = None, end_date: str = None,
                            include_price_targets: bool = True) -> Tuple[pd.DataFrame, nx.DiGraph, Dict]:
        """Run the complete hierarchical data processing pipeline"""
        logger.info("Starting full hierarchical data processing pipeline...")

        start_time = datetime.now()

        try:
            # Step 1: Load and validate
            df = self.load_and_validate_data(start_date, end_date)
            timestamp_report = self.validator.validate_timestamps(df, timestamp_col='post_created_utc')
            coverage_report = self.validator.validate_subreddit_coverage(df)
            self._enforce_minimums(df, timestamp_report, coverage_report)

            # Step 2: Feature engineering
            df = self.process_hierarchical_features(df)

            # Step 3: Network features
            _, network = self.build_network_features(df)

            # Step 4: Create targets
            if include_price_targets:
                price_data = self.bq_client.get_price_data(start_date=start_date, end_date=end_date)
                self._validate_price_log_return_stationarity(price_data)
                df = self.create_targets(df, price_data)
                self._validate_next_return_stationarity(df)
            else:
                df = self.create_targets(df)

            df = self._apply_network_features(df, network)

            # Final cleanup
            df = df.dropna(subset=['compound_sentiment', 'post_created_utc'])

            processing_time = datetime.now() - start_time

            # Complete processing log
            self.processing_log.update({
                'processing_time_seconds': processing_time.total_seconds(),
                'final_dataset_shape': df.shape,
                'final_features': list(df.columns),
                'status': 'success'
            })

            logger.info(f"Pipeline completed successfully in {processing_time}")
            logger.info(f"Final dataset shape: {df.shape}")

            # Generate and save validation report + processed data
            raw_reddit_count = self.processing_log.get('raw_reddit_rows', 0)
            raw_price_count = self.processing_log.get('raw_price_rows', 0)
            report = self.generate_validation_report(raw_reddit_count, raw_price_count, df)
            logger.info("\n" + report)

            # Save processed data to CSV
            from datetime import datetime as dt
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'data',
                'processed',
            )
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, f'processed_data_{dt.now().strftime("%Y%m%d_%H%M%S")}.csv')
            df.to_csv(csv_path, index=False)
            logger.info(f"Processed data saved to: {csv_path}")

            return df, network, self.processing_log

        except Exception as e:
            self.processing_log.update({
                'status': 'failed',
                'error': str(e),
                'processing_time_seconds': (datetime.now() - start_time).total_seconds()
            })
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """Example usage of the hierarchical data processor"""

    # Initialize processor
    processor = HierarchicalDataProcessor()

    # Run pipeline
    try:
        df, network, log = processor.process_full_pipeline(
            start_date='2021-01-01',
            end_date='2023-12-31'
        )

        print("Pipeline completed successfully!")
        print(f"Dataset shape: {df.shape}")
        print(f"Network nodes: {network.number_of_nodes()}")
        print(f"Network edges: {network.number_of_edges()}")

        # Save results
        output_dir = Path("data/processed/hierarchical")
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_parquet(output_dir / "hierarchical_features.parquet", index=False)
        nx.write_gml(network, output_dir / "causality_network.gml")

        with open(output_dir / "processing_log.json", 'w') as f:
            json.dump(log, f, indent=2, default=str)

        print("Results saved successfully!")

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")


if __name__ == "__main__":
    main()
