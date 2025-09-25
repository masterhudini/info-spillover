#!/usr/bin/env python3
"""
Upload JSON data from GCS mount to BigQuery tables
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import logging
from google.cloud import bigquery
from concurrent.futures import ThreadPoolExecutor
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_json_files_to_dataframe(json_path: str) -> pd.DataFrame:
    """Load all JSON files and convert to DataFrame"""

    json_dir = Path(json_path).expanduser()
    json_files = list(json_dir.glob("*.json"))

    print(f"📂 Found {len(json_files)} JSON files")

    # Subreddit mapping for consistent naming
    subreddit_mapping = {
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

    def process_file(file_path):
        """Process single JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract subreddit from filename
            filename = file_path.stem
            subreddit_key = filename.split('_batch_')[0]
            subreddit = subreddit_mapping.get(subreddit_key, subreddit_key.lower())

            records = []
            for item in data:
                # Convert timestamp
                created_utc = item.get('created_utc', 0)
                if created_utc:
                    post_created_utc = datetime.fromtimestamp(created_utc, tz=timezone.utc)
                else:
                    continue  # Skip records without timestamp

                # Basic sentiment analysis (simple heuristic)
                title = item.get('title', '')
                selftext = item.get('selftext', '')
                text = f"{title} {selftext}".lower()

                # Simple sentiment scoring
                positive_words = ['good', 'great', 'excellent', 'amazing', 'awesome', 'love', 'buy', 'bullish', 'moon', 'pump', 'up', 'gain', 'profit', 'hodl']
                negative_words = ['bad', 'terrible', 'awful', 'hate', 'sell', 'bearish', 'dump', 'crash', 'down', 'loss', 'drop', 'fear']

                pos_count = sum(1 for word in positive_words if word in text)
                neg_count = sum(1 for word in negative_words if word in text)

                if pos_count > neg_count:
                    sentiment = 'positive'
                    sentiment_score = min(1.0, pos_count / 10)
                elif neg_count > pos_count:
                    sentiment = 'negative'
                    sentiment_score = max(-1.0, -neg_count / 10)
                else:
                    sentiment = 'neutral'
                    sentiment_score = 0.0

                record = {
                    'post_id': str(item.get('id', '')),
                    'subreddit': subreddit,
                    'post_created_utc': post_created_utc,
                    'post_title': title,
                    'post_selftext': selftext,
                    'post_score': int(item.get('score', 0)),
                    'post_num_comments': int(item.get('num_comments', 0)),
                    'post_upvote_ratio': float(item.get('upvote_ratio', 0.5)),
                    'post_author': str(item.get('author', '')),
                    'post_gilded': int(item.get('gilded', 0)),
                    'post_sentiment_label': sentiment,
                    'post_sentiment_score': sentiment_score,

                    # Comment fields (for now, use post data as proxy)
                    'comment_id': f"{item.get('id', '')}_comment",
                    'comment_body': selftext,  # Use selftext as comment proxy
                    'comment_score': max(1, int(item.get('num_comments', 0) // 10)),  # Proxy comment score
                    'comment_created_utc': post_created_utc,
                    'comment_sentiment_label': sentiment,
                    'comment_sentiment_score': sentiment_score
                }

                records.append(record)

            logger.info(f"✅ Processed {file_path.name}: {len(records)} records")
            return records

        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            return []

    # Process files in parallel
    print("🔄 Processing JSON files...")
    all_records = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_file, json_files))

    for records in results:
        all_records.extend(records)

    # Create DataFrame
    df = pd.DataFrame(all_records)

    print(f"📊 DATASET SUMMARY:")
    print(f"   Total records: {len(df):,}")
    print(f"   Date range: {df['post_created_utc'].min()} to {df['post_created_utc'].max()}")
    print(f"   Subreddits: {df['subreddit'].nunique()}")
    print(f"   Subreddit list: {', '.join(df['subreddit'].unique())}")

    return df

def load_crypto_prices_from_csv(prices_path: str) -> pd.DataFrame:
    """Load crypto price data from CSV files"""

    print("💰 Loading crypto price data from CSV files...")

    prices_dir = Path(prices_path).expanduser()
    csv_files = list(prices_dir.glob("*.csv"))

    print(f"📂 Found {len(csv_files)} CSV price files")

    # Symbol mapping from filename
    symbol_mapping = {
        'btc-usd-max.csv': 'BTC',
        'eth-usd-max.csv': 'ETH',
        'ada-usd-max.csv': 'ADA',
        'sol-usd-max.csv': 'SOL',
        'xrp-usd-max.csv': 'XRP',
        'bnb-usd-max.csv': 'BNB',
        'ltc-usd-max.csv': 'LTC'
    }

    # Crypto name mapping
    name_mapping = {
        'BTC': 'Bitcoin',
        'ETH': 'Ethereum',
        'ADA': 'Cardano',
        'SOL': 'Solana',
        'XRP': 'XRP',
        'BNB': 'Binance Coin',
        'LTC': 'Litecoin'
    }

    all_price_data = []

    for csv_file in csv_files:
        filename = csv_file.name
        symbol = symbol_mapping.get(filename)

        if not symbol:
            logger.warning(f"Unknown price file: {filename}")
            continue

        try:
            print(f"📈 Loading {symbol} prices from {filename}")

            # Load CSV
            df = pd.read_csv(csv_file)

            # Parse timestamp
            df['snapped_at'] = pd.to_datetime(df['snapped_at'], utc=True)

            # Create standardized records
            for _, row in df.iterrows():
                record = {
                    'crypto_id': symbol,
                    'symbol': symbol,
                    'name': name_mapping.get(symbol, symbol),
                    'snapped_at': row['snapped_at'],
                    'price_usd': float(row['price']) if pd.notna(row['price']) else 0.0,
                    'market_cap_usd': float(row['market_cap']) if pd.notna(row['market_cap']) else 0.0,
                    'volume_24h': float(row['total_volume']) if pd.notna(row['total_volume']) else 0.0,
                    'price_change_24h': 0.0,  # Calculate later if needed
                    'volume_change_24h': 0.0   # Calculate later if needed
                }
                all_price_data.append(record)

            logger.info(f"✅ Loaded {len(df)} price records for {symbol}")

        except Exception as e:
            logger.error(f"❌ Error loading {csv_file}: {e}")
            continue

    # Create final DataFrame
    df_prices = pd.DataFrame(all_price_data)

    # Filter to reasonable date range (2021-2024 to match posts data)
    start_filter = pd.to_datetime('2021-01-01', utc=True)
    end_filter = pd.to_datetime('2024-12-31', utc=True)
    df_prices = df_prices[
        (df_prices['snapped_at'] >= start_filter) &
        (df_prices['snapped_at'] <= end_filter)
    ].copy()

    # Sort by symbol and date
    df_prices = df_prices.sort_values(['symbol', 'snapped_at']).reset_index(drop=True)

    print(f"💹 PRICE DATA SUMMARY:")
    print(f"   Total records: {len(df_prices):,}")
    print(f"   Date range: {df_prices['snapped_at'].min()} to {df_prices['snapped_at'].max()}")
    print(f"   Cryptos: {df_prices['symbol'].nunique()}")
    print(f"   Symbols: {', '.join(df_prices['symbol'].unique())}")

    return df_prices

def upload_to_bigquery(df_posts: pd.DataFrame, df_prices: pd.DataFrame,
                      project_id: str = 'informationspillover',
                      dataset_id: str = 'spillover_statistical_test'):
    """Upload DataFrames to BigQuery"""

    print(f"⬆️  UPLOADING TO BIGQUERY: {project_id}.{dataset_id}")

    client = bigquery.Client(project=project_id)

    # Upload posts_comments table
    posts_table_id = f"{project_id}.{dataset_id}.posts_comments"

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE"  # Replace existing data
    )

    print("📤 Uploading posts_comments...")
    job = client.load_table_from_dataframe(df_posts, posts_table_id, job_config=job_config)
    job.result()  # Wait for completion

    table = client.get_table(posts_table_id)
    print(f"✅ Uploaded {table.num_rows:,} rows to posts_comments")

    # Upload crypto_prices table
    prices_table_id = f"{project_id}.{dataset_id}.crypto_prices"

    print("📤 Uploading crypto_prices...")
    job = client.load_table_from_dataframe(df_prices, prices_table_id, job_config=job_config)
    job.result()  # Wait for completion

    table = client.get_table(prices_table_id)
    print(f"✅ Uploaded {table.num_rows:,} rows to crypto_prices")

    print(f"\n🎉 DATA UPLOAD COMPLETED!")
    print(f"   Dataset: {dataset_id}")
    print(f"   Posts: {df_posts.shape[0]:,} records")
    print(f"   Prices: {df_prices.shape[0]:,} records")

def main():
    """Main upload process"""

    print("🚀 UPLOADING JSON DATA TO BIGQUERY")
    print("=" * 50)

    # Load JSON data
    json_path = "~/gcs/raw/posts_n_comments"
    df_posts = load_json_files_to_dataframe(json_path)

    if df_posts.empty:
        print("❌ No data loaded from JSON files")
        return False

    # Load price data from CSV files
    prices_path = "~/gcs/raw/prices"
    df_prices = load_crypto_prices_from_csv(prices_path)

    # Upload to BigQuery
    upload_to_bigquery(df_posts, df_prices)

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)