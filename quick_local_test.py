#!/usr/bin/env python3
"""
Quick test with local JSON data from GCS mount
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_local_test():
    """Test with actual local JSON data"""

    print("🧪 QUICK LOCAL DATA TEST")
    print("=" * 50)

    # Load a few JSON files to test data structure
    json_path = Path("~/gcs/raw/posts_n_comments").expanduser()
    json_files = list(json_path.glob("*.json"))[:5]  # Just first 5 files

    print(f"📂 Found {len(json_files)} files to test")

    all_data = []

    for file_path in json_files:
        print(f"📄 Loading: {file_path.name}")

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Extract subreddit from filename
            subreddit = file_path.stem.split('_batch_')[0].lower()

            for item in data[:100]:  # Just first 100 records per file
                record = {
                    'subreddit': subreddit,
                    'created_utc': pd.to_datetime(item.get('created_utc', 0), unit='s'),
                    'title': item.get('title', ''),
                    'score': item.get('score', 0),
                    'num_comments': item.get('num_comments', 0)
                }
                all_data.append(record)

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            continue

    # Create DataFrame
    df = pd.DataFrame(all_data)

    print(f"\n📊 TEST DATA SUMMARY:")
    print(f"   Total records: {len(df)}")
    print(f"   Date range: {df['created_utc'].min()} to {df['created_utc'].max()}")
    print(f"   Subreddits: {df['subreddit'].nunique()}")
    print(f"   Unique subreddits: {', '.join(df['subreddit'].unique()[:10])}")

    # Basic analysis
    print(f"\n📈 BASIC STATS:")
    print(f"   Avg score: {df['score'].mean():.2f}")
    print(f"   Avg comments: {df['num_comments'].mean():.2f}")
    print(f"   Posts per subreddit: {df['subreddit'].value_counts().head()}")

    # Check if we can create a simple time series
    daily_counts = df.groupby(['subreddit', df['created_utc'].dt.date]).size()

    print(f"\n⏰ TIME SERIES TEST:")
    print(f"   Daily data points: {len(daily_counts)}")
    print(f"   Date range suitable for analysis: {'✅' if len(daily_counts) > 30 else '❌'}")

    # Create sample config for this data
    config = {
        'data': {
            'start_date': str(df['created_utc'].min().date()),
            'end_date': str(df['created_utc'].max().date()),
            'subreddits_found': df['subreddit'].unique().tolist()
        },
        'recommended_settings': {
            'max_epochs': 3,
            'hidden_dim': 32,
            'batch_size': 16
        }
    }

    print(f"\n🔧 RECOMMENDED CONFIG:")
    print(f"   Date range: {config['data']['start_date']} to {config['data']['end_date']}")
    print(f"   Subreddits: {len(config['data']['subreddits_found'])}")

    return True

if __name__ == "__main__":
    quick_local_test()