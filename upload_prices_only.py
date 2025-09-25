#!/usr/bin/env python3
"""
Upload only crypto prices to BigQuery
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_crypto_prices_from_csv(prices_path: str) -> pd.DataFrame:
    """Load crypto price data from CSV files"""

    print("💰 Loading crypto price data from CSV files...")

    prices_dir = Path(prices_path).expanduser()
    csv_files = list(prices_dir.glob("*.csv"))

    print(f"📂 Found {len(csv_files)} CSV price files")

    symbol_mapping = {
        'btc-usd-max.csv': 'BTC',
        'eth-usd-max.csv': 'ETH',
        'ada-usd-max.csv': 'ADA',
        'sol-usd-max.csv': 'SOL',
        'xrp-usd-max.csv': 'XRP',
        'bnb-usd-max.csv': 'BNB',
        'ltc-usd-max.csv': 'LTC'
    }

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
            continue

        try:
            print(f"📈 Loading {symbol} prices from {filename}")
            df = pd.read_csv(csv_file)
            df['snapped_at'] = pd.to_datetime(df['snapped_at'], utc=True)

            # Filter to 2021-2024 only
            start_filter = pd.to_datetime('2021-01-01', utc=True)
            end_filter = pd.to_datetime('2024-12-31', utc=True)
            df = df[
                (df['snapped_at'] >= start_filter) &
                (df['snapped_at'] <= end_filter)
            ]

            for _, row in df.iterrows():
                record = {
                    'crypto_id': symbol,
                    'symbol': symbol,
                    'name': name_mapping.get(symbol, symbol),
                    'snapped_at': row['snapped_at'],
                    'price_usd': float(row['price']) if pd.notna(row['price']) else 0.0,
                    'market_cap_usd': float(row['market_cap']) if pd.notna(row['market_cap']) else 0.0,
                    'volume_24h': float(row['total_volume']) if pd.notna(row['total_volume']) else 0.0,
                    'price_change_24h': 0.0,
                    'volume_change_24h': 0.0
                }
                all_price_data.append(record)

            logger.info(f"✅ Loaded {len(df)} price records for {symbol}")

        except Exception as e:
            logger.error(f"❌ Error loading {csv_file}: {e}")
            continue

    df_prices = pd.DataFrame(all_price_data)
    df_prices = df_prices.sort_values(['symbol', 'snapped_at']).reset_index(drop=True)

    print(f"💹 FILTERED PRICE DATA:")
    print(f"   Total records: {len(df_prices):,}")
    print(f"   Date range: {df_prices['snapped_at'].min()} to {df_prices['snapped_at'].max()}")
    print(f"   Symbols: {', '.join(df_prices['symbol'].unique())}")

    return df_prices

def upload_prices_to_bigquery(df_prices: pd.DataFrame):
    """Upload just prices"""

    client = bigquery.Client(project='informationspillover')
    prices_table_id = "informationspillover.spillover_statistical_test.crypto_prices"

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE"
    )

    print("📤 Uploading crypto_prices...")
    job = client.load_table_from_dataframe(df_prices, prices_table_id, job_config=job_config)
    job.result()

    table = client.get_table(prices_table_id)
    print(f"✅ Uploaded {table.num_rows:,} rows to crypto_prices")

def main():
    prices_path = "~/gcs/raw/prices"
    df_prices = load_crypto_prices_from_csv(prices_path)

    if df_prices.empty:
        print("❌ No price data loaded")
        return False

    upload_prices_to_bigquery(df_prices)
    print("🎉 PRICES UPLOADED SUCCESSFULLY!")
    return True

if __name__ == "__main__":
    main()