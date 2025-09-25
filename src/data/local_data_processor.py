"""
Local data processor for JSON files from GCS mount
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
from scipy import stats
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class LocalDataProcessor:
    """Process local JSON files instead of BigQuery"""

    def __init__(self, data_path: str = "~/gcs/raw"):
        """Initialize with local data path"""
        self.data_path = Path(data_path).expanduser()
        self.posts_path = self.data_path / "posts_n_comments"
        self.prices_path = self.data_path / "prices"

        # Subreddit mapping
        self.subreddit_mapping = {
            'Altcoin': 'altcoin',
            'Bitcoin': 'bitcoin',
            'BitcoinBeginners': 'bitcoinbeginners',
            'BitcoinMarkets': 'bitcoinmarkets',
            'CryptoCurrency': 'cryptocurrency',
            'CryptoMarkets': 'cryptomarkets',
            'CryptoMoonShots': 'cryptomoonshots',
            'CryptoTechnology': 'cryptotechnology',
            'DeFi': 'defi',
            'Ripple': 'ripple',
            'SatoshiStreetBets': 'satoshistreetbets',
            'Tronix': 'tronix',
            'XRP': 'xrp',
            'binance': 'binance',
            'btc': 'btc',
            'cardano': 'cardano',
            'dogecoin': 'dogecoin',
            'ethereum': 'ethereum',
            'ethtrader': 'ethtrader',
            'solana': 'solana'
        }

        logger.info(f"Initialized LocalDataProcessor with data path: {self.data_path}")

    def load_json_files(self) -> pd.DataFrame:
        """Load all JSON files and combine into DataFrame"""

        logger.info("Loading JSON files from local storage...")
        all_data = []

        # Get all JSON files
        json_files = list(self.posts_path.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files")

        def load_single_file(file_path):
            """Load single JSON file"""
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract subreddit from filename
                filename = file_path.stem
                subreddit = filename.split('_batch_')[0]

                # Convert to records
                records = []
                for item in data:
                    record = {
                        'subreddit': self.subreddit_mapping.get(subreddit, subreddit.lower()),
                        'id': item.get('id', ''),
                        'created_utc': item.get('created_utc', 0),
                        'title': item.get('title', ''),
                        'selftext': item.get('selftext', ''),
                        'score': item.get('score', 0),
                        'num_comments': item.get('num_comments', 0),
                        'author': item.get('author', ''),
                        'upvote_ratio': item.get('upvote_ratio', 0.5),
                        'is_self': item.get('is_self', True),
                        'is_video': item.get('is_video', False),
                        'is_original_content': item.get('is_original_content', False),
                        'stickied': item.get('stickied', False),
                        'over_18': item.get('over_18', False),
                        'spoiler': item.get('spoiler', False),
                        'locked': item.get('locked', False),
                        'gilded': item.get('gilded', 0),
                        'all_awardings_count': item.get('all_awardings_count', 0),
                        'total_awards_received': item.get('total_awards_received', 0)
                    }
                    records.append(record)

                logger.info(f"Loaded {len(records)} records from {file_path.name}")
                return records

            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                return []

        # Load files in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(load_single_file, json_files))

        # Combine all data
        for records in results:
            all_data.extend(records)

        logger.info(f"Total records loaded: {len(all_data)}")

        # Convert to DataFrame
        df = pd.DataFrame(all_data)

        # Convert timestamps
        df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s', errors='coerce')
        df = df.dropna(subset=['created_utc'])

        # Sort by timestamp
        df = df.sort_values('created_utc').reset_index(drop=True)

        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Date range: {df['created_utc'].min()} to {df['created_utc'].max()}")
        logger.info(f"Subreddits: {df['subreddit'].nunique()}")

        return df

    def load_price_data(self) -> pd.DataFrame:
        """Load price data (stub - implement if price files exist)"""

        price_files = list(self.prices_path.glob("*.json")) if self.prices_path.exists() else []

        if not price_files:
            logger.info("No price files found, generating synthetic price data")
            return self._generate_synthetic_prices()

        logger.info(f"Found {len(price_files)} price files")
        # TODO: Implement actual price loading if files exist
        return self._generate_synthetic_prices()

    def _generate_synthetic_prices(self) -> pd.DataFrame:
        """Generate synthetic price data for testing"""

        # Generate 2 years of daily prices
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2023, 12, 31)
        dates = pd.date_range(start_date, end_date, freq='D')

        # Cryptocurrencies to generate prices for
        cryptocurrencies = [
            'bitcoin', 'ethereum', 'cardano', 'solana', 'dogecoin',
            'ripple', 'binancecoin', 'tron', 'defi'
        ]

        price_data = []
        np.random.seed(42)  # For reproducible results

        for crypto in cryptocurrencies:
            # Initial price (different for each crypto)
            initial_prices = {
                'bitcoin': 46000,
                'ethereum': 3700,
                'cardano': 1.30,
                'solana': 170,
                'dogecoin': 0.17,
                'ripple': 0.83,
                'binancecoin': 520,
                'tron': 0.062,
                'defi': 2800
            }

            initial_price = initial_prices.get(crypto, 100)

            # Generate price series with random walk + trend
            prices = [initial_price]

            for _ in range(len(dates) - 1):
                # Daily returns: random walk with slight positive trend
                daily_return = np.random.normal(0.001, 0.05)  # 0.1% average daily return, 5% volatility
                new_price = prices[-1] * (1 + daily_return)
                prices.append(max(new_price, 0.01))  # Prevent negative prices

            # Create records
            for date, price in zip(dates, prices):
                price_data.append({
                    'date': date,
                    'crypto': crypto,
                    'price_usd': price,
                    'volume_24h': np.random.lognormal(15, 1.5),  # Random volume
                    'market_cap': price * np.random.lognormal(18, 0.5)  # Random market cap
                })

        df_prices = pd.DataFrame(price_data)
        logger.info(f"Generated synthetic price data: {df_prices.shape}")

        return df_prices

    def calculate_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate sentiment and engagement features"""

        logger.info("Calculating sentiment and engagement features...")

        # Simple sentiment proxy using score and text length
        df['text_length'] = (df['title'].fillna('').str.len() +
                            df['selftext'].fillna('').str.len())

        # Engagement features
        df['engagement_score'] = (df['score'] + df['num_comments']).fillna(0)
        df['comment_rate'] = np.where(df['score'] > 0, df['num_comments'] / df['score'], 0)
        df['text_engagement'] = np.where(df['text_length'] > 0,
                                        df['engagement_score'] / np.log1p(df['text_length']),
                                        0)

        # Sentiment proxies
        df['sentiment_score'] = np.where(df['score'] >= 0,
                                        np.log1p(df['score']) / 10,
                                        -np.log1p(-df['score']) / 10)

        # Bullish/bearish keywords (simple approach)
        bullish_words = ['moon', 'bull', 'buy', 'hodl', 'pump', 'up', 'rise', 'gain']
        bearish_words = ['dump', 'crash', 'bear', 'sell', 'down', 'fall', 'drop', 'loss']

        text = (df['title'].fillna('') + ' ' + df['selftext'].fillna('')).str.lower()

        df['bullish_mentions'] = sum(text.str.count(word) for word in bullish_words)
        df['bearish_mentions'] = sum(text.str.count(word) for word in bearish_words)
        df['sentiment_polarity'] = df['bullish_mentions'] - df['bearish_mentions']

        logger.info("Sentiment features calculated")
        return df

    def aggregate_hourly_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data by hour for time series analysis"""

        logger.info("Aggregating data by hour...")

        # Create hourly time index
        df['hour'] = df['created_utc'].dt.floor('H')

        # Aggregation functions
        agg_funcs = {
            'score': ['sum', 'mean', 'std', 'count'],
            'num_comments': ['sum', 'mean'],
            'engagement_score': ['sum', 'mean'],
            'sentiment_score': ['mean', 'sum', 'std'],
            'sentiment_polarity': ['sum', 'mean'],
            'bullish_mentions': ['sum'],
            'bearish_mentions': ['sum'],
            'text_length': ['sum', 'mean'],
            'upvote_ratio': ['mean'],
            'gilded': ['sum'],
            'total_awards_received': ['sum']
        }

        # Group by subreddit and hour
        hourly = df.groupby(['subreddit', 'hour']).agg(agg_funcs).reset_index()

        # Flatten column names
        hourly.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in hourly.columns]

        # Rename for clarity
        column_mapping = {
            'score_count': 'post_count',
            'score_sum': 'total_score',
            'score_mean': 'avg_score',
            'score_std': 'score_volatility'
        }
        hourly = hourly.rename(columns=column_mapping)

        # Fill missing volatility with 0
        hourly['score_volatility'] = hourly['score_volatility'].fillna(0)

        logger.info(f"Hourly aggregated data shape: {hourly.shape}")
        logger.info(f"Time range: {hourly['hour'].min()} to {hourly['hour'].max()}")

        return hourly

    def create_granger_causality_network(self, hourly_df: pd.DataFrame) -> nx.DiGraph:
        """Create network based on Granger causality between subreddits"""

        logger.info("Creating Granger causality network...")

        # Pivot to get subreddit time series
        pivot_data = hourly_df.pivot_table(
            index='hour',
            columns='subreddit',
            values='total_score',
            fill_value=0
        )

        # Ensure we have enough data points
        min_obs = 48  # 48 hours minimum
        if len(pivot_data) < min_obs:
            logger.warning(f"Insufficient data for Granger causality: {len(pivot_data)} < {min_obs}")
            # Return simple correlation-based network
            return self._create_correlation_network(pivot_data)

        network = nx.DiGraph()
        subreddits = pivot_data.columns.tolist()

        # Add all nodes
        for subreddit in subreddits:
            network.add_node(subreddit,
                           total_posts=int(pivot_data[subreddit].sum()),
                           avg_activity=float(pivot_data[subreddit].mean()))

        # Test Granger causality between each pair
        for i, sub1 in enumerate(subreddits):
            for j, sub2 in enumerate(subreddits):
                if i != j:
                    try:
                        # Simple causality test using correlation with lag
                        series1 = pivot_data[sub1].values
                        series2 = pivot_data[sub2].values

                        # Test if sub1 Granger-causes sub2
                        causality_score = self._simple_causality_test(series1, series2)

                        if causality_score > 0.1:  # Threshold for significance
                            network.add_edge(sub1, sub2,
                                           weight=causality_score,
                                           causality_score=causality_score)

                    except Exception as e:
                        logger.warning(f"Error testing causality {sub1} -> {sub2}: {e}")

        logger.info(f"Created network with {network.number_of_nodes()} nodes and {network.number_of_edges()} edges")

        return network

    def _simple_causality_test(self, x: np.ndarray, y: np.ndarray, max_lags: int = 5) -> float:
        """Simple causality test based on lagged correlations"""

        if len(x) < 10 or len(y) < 10:
            return 0.0

        max_correlation = 0.0

        for lag in range(1, min(max_lags + 1, len(x) - 1)):
            if lag >= len(x):
                break

            try:
                # Correlation between lagged x and current y
                x_lagged = x[:-lag]
                y_current = y[lag:]

                if len(x_lagged) > 0 and len(y_current) > 0:
                    corr, p_value = stats.pearsonr(x_lagged, y_current)

                    if not np.isnan(corr) and p_value < 0.1:  # Significant at 10% level
                        max_correlation = max(max_correlation, abs(corr))

            except Exception:
                continue

        return max_correlation

    def _create_correlation_network(self, pivot_data: pd.DataFrame) -> nx.DiGraph:
        """Create network based on correlations (fallback)"""

        logger.info("Creating correlation-based network (fallback)")

        network = nx.DiGraph()
        subreddits = pivot_data.columns.tolist()

        # Add nodes
        for subreddit in subreddits:
            network.add_node(subreddit,
                           total_posts=int(pivot_data[subreddit].sum()),
                           avg_activity=float(pivot_data[subreddit].mean()))

        # Add edges based on correlation
        correlation_matrix = pivot_data.corr()

        for i, sub1 in enumerate(subreddits):
            for j, sub2 in enumerate(subreddits):
                if i != j:
                    corr = correlation_matrix.loc[sub1, sub2]

                    if not np.isnan(corr) and abs(corr) > 0.3:  # Threshold
                        network.add_edge(sub1, sub2,
                                       weight=abs(corr),
                                       correlation=corr)

        return network

    def process_full_pipeline(self, start_date: str = None, end_date: str = None) -> Tuple[pd.DataFrame, nx.DiGraph, Dict]:
        """Process complete pipeline from local JSON files"""

        logger.info("Starting local data processing pipeline...")
        start_time = datetime.now()

        # Step 1: Load raw data
        raw_df = self.load_json_files()

        # Step 2: Filter by date if specified
        if start_date and end_date:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            raw_df = raw_df[
                (raw_df['created_utc'] >= start_dt) &
                (raw_df['created_utc'] <= end_dt)
            ]
            logger.info(f"Filtered data to date range: {raw_df.shape}")

        # Step 3: Calculate features
        featured_df = self.calculate_sentiment_features(raw_df)

        # Step 4: Aggregate hourly
        hourly_df = self.aggregate_hourly_data(featured_df)

        # Step 5: Load price data
        price_df = self.load_price_data()

        # Step 6: Create network
        network = self.create_granger_causality_network(hourly_df)

        # Step 7: Combine with prices (simple merge by date)
        # Create daily aggregates from hourly data
        daily_df = hourly_df.copy()
        daily_df['date'] = daily_df['hour'].dt.date
        daily_agg = daily_df.groupby(['subreddit', 'date']).agg({
            'total_score': 'sum',
            'post_count': 'sum',
            'sentiment_score_mean': 'mean',
            'sentiment_polarity_sum': 'sum',
            'bullish_mentions_sum': 'sum',
            'bearish_mentions_sum': 'sum'
        }).reset_index()

        # Merge with price data (many-to-many join)
        price_df['date'] = price_df['date'].dt.date

        combined_data = []
        for _, row in daily_agg.iterrows():
            subreddit = row['subreddit']
            date = row['date']

            # Find relevant crypto for this subreddit
            crypto_mapping = {
                'bitcoin': 'bitcoin',
                'ethereum': 'ethereum',
                'cardano': 'cardano',
                'solana': 'solana',
                'dogecoin': 'dogecoin',
                'ripple': 'ripple',
                'xrp': 'ripple',
                'binance': 'binancecoin',
                'tronix': 'tron',
                'defi': 'defi'
            }

            crypto = crypto_mapping.get(subreddit, 'bitcoin')  # Default to bitcoin

            # Get price for this date
            price_row = price_df[
                (price_df['crypto'] == crypto) &
                (price_df['date'] == date)
            ]

            if not price_row.empty:
                combined_row = row.to_dict()
                combined_row.update({
                    'crypto': crypto,
                    'price_usd': price_row.iloc[0]['price_usd'],
                    'volume_24h': price_row.iloc[0]['volume_24h'],
                    'market_cap': price_row.iloc[0]['market_cap']
                })
                combined_data.append(combined_row)

        final_df = pd.DataFrame(combined_data)

        # Processing log
        end_time = datetime.now()
        processing_log = {
            'processing_time_seconds': (end_time - start_time).total_seconds(),
            'raw_records': len(raw_df),
            'processed_records': len(final_df),
            'subreddits': final_df['subreddit'].nunique() if len(final_df) > 0 else 0,
            'date_range': {
                'start': str(final_df['date'].min()) if len(final_df) > 0 else None,
                'end': str(final_df['date'].max()) if len(final_df) > 0 else None
            },
            'network_stats': {
                'nodes': network.number_of_nodes(),
                'edges': network.number_of_edges()
            }
        }

        logger.info(f"Local processing pipeline completed in {processing_log['processing_time_seconds']:.2f}s")
        logger.info(f"Final dataset shape: {final_df.shape}")

        return final_df, network, processing_log