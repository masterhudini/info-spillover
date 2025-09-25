"""
Hierarchical Data Processor for Cryptocurrency Sentiment Analysis
Based on scientific methodology for information spillover analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json
from pathlib import Path
import logging
from datetime import datetime, timedelta
import warnings

from google.cloud import bigquery
from src.data.bigquery_client import BigQueryClient

# Statistical and econometric libraries
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
import networkx as nx
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats
from scipy.stats import jarque_bera, normaltest

logging.basicConfig(level=logging.INFO)
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
                                label_col: str = 'sentiment_label') -> Dict:
        """Validate sentiment scores according to VADER methodology"""
        logger.info("Validating sentiment scores...")

        # Check score range
        score_min = df[score_col].min()
        score_max = df[score_col].max()
        valid_range = (score_min >= 0) and (score_max <= 1)

        # Check for missing values
        missing_scores = df[score_col].isna().sum()
        missing_labels = df[label_col].isna().sum()

        # Distribution analysis
        score_distribution = {
            'mean': df[score_col].mean(),
            'std': df[score_col].std(),
            'median': df[score_col].median(),
            'skewness': stats.skew(df[score_col].dropna()),
            'kurtosis': stats.kurtosis(df[score_col].dropna())
        }

        # Label distribution
        label_counts = df[label_col].value_counts().to_dict()

        # Check for anomalies (scores near 0 or 1 should be rare for neutral)
        neutral_low_confidence = ((df[label_col] == 'neutral') &
                                 (df[score_col] < 0.6)).sum()

        validation_report = {
            'valid_score_range': valid_range,
            'score_range': {'min': score_min, 'max': score_max},
            'missing_scores': missing_scores,
            'missing_labels': missing_labels,
            'score_distribution': score_distribution,
            'label_distribution': label_counts,
            'neutral_low_confidence': neutral_low_confidence
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

    def __init__(self, window_hours: List[int] = [1, 6, 24]):
        self.window_hours = window_hours
        self.scalers = {}

    def compute_compound_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute compound sentiment score following VADER methodology
        Hutto & Gilbert (2014)
        """
        logger.info("Computing compound sentiment scores...")

        # Map labels to numerical values
        label_map = {'positive': 1, 'neutral': 0, 'negative': -1}
        df['sentiment_numeric'] = df['sentiment_label'].map(label_map)

        # Compound score: (numeric_sentiment) * confidence
        df['compound_sentiment'] = df['sentiment_numeric'] * df['sentiment_score']

        # Normalize to [-1, 1] range
        df['compound_sentiment'] = np.clip(df['compound_sentiment'], -1, 1)

        return df

    def extract_emotion_features(self, df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
        """
        Extract emotion category flags based on Mohammad & Turney (2013)
        """
        logger.info("Extracting emotion features...")

        # Fear indicators (keyword-based approach)
        fear_keywords = ['crash', 'dump', 'bear', 'fear', 'panic', 'sell', 'drop', 'fall']
        greed_keywords = ['moon', 'pump', 'bull', 'buy', 'hodl', 'diamond', 'rocket', 'lambo']
        uncertainty_keywords = ['maybe', 'perhaps', 'uncertain', 'confused', 'doubt']

        # Create binary features
        if text_col in df.columns:
            df['has_fear_keywords'] = df[text_col].str.lower().str.contains(
                '|'.join(fear_keywords), na=False
            )
            df['has_greed_keywords'] = df[text_col].str.lower().str.contains(
                '|'.join(greed_keywords), na=False
            )
            df['has_uncertainty_keywords'] = df[text_col].str.lower().str.contains(
                '|'.join(uncertainty_keywords), na=False
            )

            # Emotion intensity scores
            df['fear_score'] = (df['sentiment_numeric'] == -1) & df['has_fear_keywords']
            df['greed_score'] = (df['sentiment_numeric'] == 1) & df['has_greed_keywords']
            df['fomo_score'] = (df['sentiment_numeric'] == 1) & df['has_uncertainty_keywords']
        else:
            # Fallback to sentiment-based emotion detection
            df['fear_score'] = (df['sentiment_numeric'] == -1) & (df['sentiment_score'] > 0.8)
            df['greed_score'] = (df['sentiment_numeric'] == 1) & (df['sentiment_score'] > 0.8)
            df['fomo_score'] = (df['sentiment_numeric'] == 1) & (df['sentiment_score'] < 0.7)

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

        for subreddit in df['subreddit'].unique():
            logger.info(f"Processing temporal aggregates for {subreddit}")

            subreddit_data = df[df['subreddit'] == subreddit].copy()

            # For each time window (convert hours to periods - assuming data is roughly hourly)
            for window_h in self.window_hours:
                window_periods = max(1, window_h)  # Use hours as periods directly

                # Rolling statistics
                rolling = subreddit_data['compound_sentiment'].rolling(window_periods, min_periods=1)

                subreddit_data[f'sentiment_mean_{window_h}h'] = rolling.mean()
                subreddit_data[f'sentiment_std_{window_h}h'] = rolling.std()
                subreddit_data[f'sentiment_skew_{window_h}h'] = rolling.skew()
                subreddit_data[f'sentiment_kurt_{window_h}h'] = rolling.apply(
                    lambda x: stats.kurtosis(x) if len(x) > 3 else np.nan
                )

                # Volume-weighted sentiment
                if 'score' in subreddit_data.columns:  # post score as volume proxy
                    weights = subreddit_data['score'].rolling(window_periods, min_periods=1)
                    sentiment_vol = subreddit_data['compound_sentiment'].rolling(window_periods, min_periods=1)

                    subreddit_data[f'vwsentiment_{window_h}h'] = (
                        (sentiment_vol.mean() * weights.mean()) / weights.mean()
                    ).fillna(0)

                # Activity metrics - since created_utc is already the index, don't specify 'on'
                try:
                    subreddit_data[f'post_count_{window_h}h'] = subreddit_data.rolling(
                        window_periods, min_periods=1
                    ).size()
                except Exception:
                    # Simple fallback - use expanding window count
                    subreddit_data[f'post_count_{window_h}h'] = range(1, len(subreddit_data) + 1)

                # Emotion ratios
                for emotion in ['fear', 'greed', 'fomo']:
                    emotion_rolling = subreddit_data[f'{emotion}_score'].rolling(window_periods, min_periods=1)
                    subreddit_data[f'{emotion}_ratio_{window_h}h'] = emotion_rolling.mean()

            aggregated_features.append(subreddit_data)

        # Combine all subreddits
        result_df = pd.concat(aggregated_features, ignore_index=False)

        return result_df.reset_index()


class GrangerCausalityNetworkBuilder:
    """
    Build Granger causality networks following Billio et al. (2012)
    """

    def __init__(self, max_lags: int = 5, significance_level: float = 0.05):
        self.max_lags = max_lags
        self.significance_level = significance_level
        self.causality_networks = {}

    def prepare_time_series(self, df: pd.DataFrame,
                          freq: str = '1H') -> Dict[str, pd.Series]:
        """Prepare time series for Granger causality testing"""
        logger.info("Preparing time series for Granger causality analysis...")

        # Group by subreddit and resample to regular frequency
        time_series = {}

        for subreddit in df['subreddit'].unique():
            subreddit_data = df[df['subreddit'] == subreddit].copy()
            # Use the correct timestamp column name
            timestamp_col = 'created_utc' if 'created_utc' in subreddit_data.columns else 'post_created_utc'

            # Ensure timestamp column is datetime
            if timestamp_col in subreddit_data.columns:
                subreddit_data[timestamp_col] = pd.to_datetime(subreddit_data[timestamp_col])
                subreddit_data = subreddit_data.set_index(timestamp_col).sort_index()
            else:
                logger.warning(f"No valid timestamp column found for {subreddit}, skipping network analysis")
                continue

            # Resample to regular frequency, forward-fill missing values
            ts = subreddit_data['compound_sentiment'].resample(freq).mean()
            ts = ts.fillna(method='ffill', limit=3)  # Max 3 consecutive fills

            # Remove remaining NaN values
            ts = ts.dropna()

            if len(ts) > 100:  # Minimum observations for reliable testing
                time_series[subreddit] = ts
            else:
                logger.warning(f"Insufficient data for {subreddit}: {len(ts)} observations")

        return time_series

    def test_granger_causality(self, ts1: pd.Series, ts2: pd.Series,
                             max_lags: int = None) -> Dict:
        """Test Granger causality between two time series"""
        if max_lags is None:
            max_lags = self.max_lags

        try:
            # Align time series
            aligned = pd.concat([ts1, ts2], axis=1, join='inner')
            aligned.columns = ['ts1', 'ts2']
            aligned = aligned.dropna()

            if len(aligned) < max_lags * 3:  # Minimum sample size
                return {'significant': False, 'p_value': 1.0, 'f_stat': 0.0, 'lags_used': 0}

            # Test ts1 -> ts2 (ts1 Granger causes ts2)
            result = grangercausalitytests(aligned[['ts2', 'ts1']], max_lags, verbose=False)

            # Get the result for optimal lag (use first available lag)
            if not result:
                return {'significant': False, 'p_value': 1.0, 'f_stat': 0.0, 'lags_used': 0}

            # Use first available lag result
            optimal_lag = list(result.keys())[0]

            # Extract statistics for optimal lag - simplified approach
            try:
                test_result = result[optimal_lag][0]['ssr_ftest']
                f_stat = test_result[0]  # F-statistic
                p_value = test_result[1]  # p-value
            except (KeyError, IndexError, TypeError):
                # Fallback - use any available test result
                f_stat = 0.0
                p_value = 1.0
                for test_name, test_data in result[optimal_lag][0].items():
                    if isinstance(test_data, (list, tuple)) and len(test_data) >= 2:
                        f_stat = test_data[0]
                        p_value = test_data[1]
                        break

            causality_result = {
                'significant': p_value < self.significance_level,
                'p_value': p_value,
                'f_stat': f_stat,
                'lags_used': optimal_lag,
                'sample_size': len(aligned)
            }

            return causality_result

        except Exception as e:
            logger.warning(f"Granger causality test failed: {str(e)}")
            return {'significant': False, 'p_value': 1.0, 'f_stat': 0.0, 'lags_used': 0}

    def build_causality_network(self, time_series: Dict[str, pd.Series],
                               time_window: str = None) -> nx.DiGraph:
        """Build directed network based on Granger causality"""
        logger.info("Building Granger causality network...")

        subreddits = list(time_series.keys())
        G = nx.DiGraph()

        # Add nodes
        for subreddit in subreddits:
            G.add_node(subreddit, size=len(time_series[subreddit]))

        # Test all pairs for causality
        causality_matrix = np.zeros((len(subreddits), len(subreddits)))

        for i, sr1 in enumerate(subreddits):
            for j, sr2 in enumerate(subreddits):
                if i != j:
                    # Test sr1 -> sr2
                    result = self.test_granger_causality(
                        time_series[sr1], time_series[sr2]
                    )

                    if result['significant']:
                        weight = result['f_stat'] / (1 + result['p_value'])  # Normalized weight
                        G.add_edge(sr1, sr2,
                                 weight=weight,
                                 p_value=result['p_value'],
                                 f_stat=result['f_stat'],
                                 lags=result['lags_used'])

                        causality_matrix[i, j] = weight

        # Compute network metrics
        self._compute_network_metrics(G, subreddits)

        if time_window:
            self.causality_networks[time_window] = G

        return G

    def _compute_network_metrics(self, G: nx.DiGraph, subreddits: List[str]):
        """Compute network centrality metrics"""

        # In-degree and out-degree
        in_degrees = dict(G.in_degree(weight='weight'))
        out_degrees = dict(G.out_degree(weight='weight'))

        # PageRank centrality
        try:
            pagerank = nx.pagerank(G, weight='weight')
        except:
            pagerank = {node: 0 for node in G.nodes()}

        # Clustering coefficient
        try:
            clustering = nx.clustering(G.to_undirected(), weight='weight')
        except:
            clustering = {node: 0 for node in G.nodes()}

        # Add metrics as node attributes
        for node in G.nodes():
            G.nodes[node]['in_degree'] = in_degrees.get(node, 0)
            G.nodes[node]['out_degree'] = out_degrees.get(node, 0)
            G.nodes[node]['net_spillover'] = out_degrees.get(node, 0) - in_degrees.get(node, 0)
            G.nodes[node]['pagerank'] = pagerank.get(node, 0)
            G.nodes[node]['clustering'] = clustering.get(node, 0)

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
        # Use provided temporal windows or default
        window_hours = temporal_windows or [1, 6, 24]
        self.feature_engineer = HierarchicalFeatureEngineer(window_hours=window_hours)
        self.network_builder = GrangerCausalityNetworkBuilder()

        self.processing_log = {}

    def load_and_validate_data(self, start_date: str = None,
                             end_date: str = None) -> pd.DataFrame:
        """Load data from BigQuery and perform validation"""
        logger.info("Loading and validating data...")

        # Load raw data
        if hasattr(self.bq_client, 'get_post_sentiment_aggregation'):
            df = self.bq_client.get_post_sentiment_aggregation(start_date, end_date)
            # Rename 'date' to 'post_created_utc' for consistency
            if 'date' in df.columns:
                df = df.rename(columns={'date': 'post_created_utc'})

            # Create compound sentiment from aggregated sentiments
            if 'avg_positive_sentiment' in df.columns:
                df['compound_sentiment'] = (
                    df['avg_positive_sentiment'].fillna(0) -
                    df['avg_negative_sentiment'].fillna(0)
                )

            # Map other columns for consistency
            column_mapping = {
                'avg_post_score': 'score',
                'num_posts': 'post_count',
                'num_comments': 'comment_count'
            }
            df = df.rename(columns=column_mapping)

            # For aggregated data, create sentiment_score from compound_sentiment
            if 'compound_sentiment' in df.columns and 'sentiment_score' not in df.columns:
                df['sentiment_score'] = df['compound_sentiment']

            # Create sentiment_label from compound_sentiment
            if 'compound_sentiment' in df.columns and 'sentiment_label' not in df.columns:
                df['sentiment_label'] = df['compound_sentiment'].apply(
                    lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral')
                )
        else:
            # Fallback query
            query = f"""
            SELECT
                subreddit,
                post_created_utc,
                comment_sentiment_label as sentiment_label,
                comment_sentiment_score as sentiment_score,
                comment_body as text,
                post_score as score
            FROM `{self.bq_client.project_id}.{self.bq_client.dataset_id}.posts_comments`
            WHERE comment_id IS NOT NULL
            """
            if start_date:
                query += f" AND DATE(post_created_utc) >= '{start_date}'"
            if end_date:
                query += f" AND DATE(post_created_utc) <= '{end_date}'"

            df = self.bq_client.query_data(query)

        if df.empty:
            raise ValueError("No data returned from BigQuery")

        # Data validation
        logger.info("Performing data quality validation...")

        # Remove duplicates
        initial_size = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_size - len(df)

        # Validate data quality
        timestamp_report = self.validator.validate_timestamps(df)
        sentiment_report = self.validator.validate_sentiment_scores(
            df, 'sentiment_score', 'sentiment_label'
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

    def process_hierarchical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all hierarchical features"""
        logger.info("Processing hierarchical features...")

        # Level 1: Post-level features
        df = self.feature_engineer.compute_compound_sentiment(df)
        df = self.feature_engineer.extract_emotion_features(df)

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

    def build_network_features(self, df: pd.DataFrame,
                             network_freq: str = '1H') -> Tuple[pd.DataFrame, nx.DiGraph]:
        """Build network structure features"""
        logger.info("Building network features...")

        # Prepare time series
        time_series = self.network_builder.prepare_time_series(df, network_freq)

        # Build causality network
        G = self.network_builder.build_causality_network(time_series)

        # Add network features to dataframe
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

                network_features.append(subreddit_data)

        # Combine results
        if network_features:
            df_with_network = pd.concat(network_features, ignore_index=True)
        else:
            # Fallback: add zero network features
            df_with_network = df.copy()
            for feature in ['net_spillover', 'pagerank', 'in_degree', 'out_degree', 'clustering']:
                df_with_network[feature] = 0.0

        # Log network building results
        self.processing_log['network_building'] = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'connected_components': len(list(nx.weakly_connected_components(G)))
        }

        return df_with_network, G

    def create_targets(self, df: pd.DataFrame, price_data: pd.DataFrame = None) -> pd.DataFrame:
        """Create target variables for prediction"""
        logger.info("Creating target variables...")

        # If price data is provided, create price-based targets
        if price_data is not None:
            # Ensure compatible datetime types for merging (standardize to ns precision)
            if 'post_created_utc' in df.columns:
                df['post_created_utc'] = pd.to_datetime(df['post_created_utc']).dt.tz_localize(None).astype('datetime64[ns]')
            if 'created_utc' in df.columns:
                df['created_utc'] = pd.to_datetime(df['created_utc']).dt.tz_localize(None).astype('datetime64[ns]')

            if 'snapped_at' in price_data.columns:
                price_data['snapped_at'] = pd.to_datetime(price_data['snapped_at']).dt.tz_localize(None).astype('datetime64[ns]')

            # Use the correct timestamp column
            timestamp_col = 'created_utc' if 'created_utc' in df.columns else 'post_created_utc'

            # Merge with price data (assuming it has timestamp alignment)
            df = pd.merge_asof(
                df.sort_values(timestamp_col),
                price_data.sort_values('snapped_at'),
                left_on=timestamp_col,
                right_on='snapped_at',
                direction='nearest',
                tolerance=pd.Timedelta('1H')
            )

            # Create price return targets
            df = df.sort_values(['symbol', timestamp_col])
            df['next_return'] = df.groupby('symbol')['price'].pct_change().shift(-1)
            df['return_direction'] = np.where(df['next_return'] > 0.001, 1,
                                    np.where(df['next_return'] < -0.001, -1, 0))

        # Create sentiment-based targets - use correct timestamp column
        timestamp_col = 'created_utc' if 'created_utc' in df.columns else 'post_created_utc'
        df = df.sort_values(['subreddit', timestamp_col])
        df['next_sentiment'] = df.groupby('subreddit')['compound_sentiment'].shift(-1)
        df['sentiment_change'] = df['next_sentiment'] - df['compound_sentiment']

        return df

    def process_full_pipeline(self, start_date: str = None, end_date: str = None,
                            include_price_targets: bool = True) -> Tuple[pd.DataFrame, nx.DiGraph, Dict]:
        """Run the complete hierarchical data processing pipeline"""
        logger.info("Starting full hierarchical data processing pipeline...")

        start_time = datetime.now()

        try:
            # Step 1: Load and validate
            df = self.load_and_validate_data(start_date, end_date)

            # Step 2: Feature engineering
            df = self.process_hierarchical_features(df)

            # Step 3: Network features
            df, network = self.build_network_features(df)

            # Step 4: Create targets
            if include_price_targets:
                price_data = self.bq_client.get_price_data(start_date=start_date, end_date=end_date)
                df = self.create_targets(df, price_data)
            else:
                df = self.create_targets(df)

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