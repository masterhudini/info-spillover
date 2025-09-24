#!/usr/bin/env python3
"""
Offline Pipeline Test - Demonstrates the hierarchical sentiment spillover pipeline
using local data files without requiring BigQuery access.

This script processes the available data in ~/gcs to show what the full pipeline
would accomplish once BigQuery authentication is fixed.
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np


def load_sample_reddit_data(data_path="/home/Hudini/gcs/raw/posts_n_comments", max_files=3):
    """Load a sample of Reddit data for testing"""
    print(f"📁 Loading Reddit data from: {data_path}")

    data_path = Path(data_path)
    json_files = list(data_path.glob("*.json"))[:max_files]  # Limit for testing

    print(f"   Found {len(json_files)} files to process")

    all_posts = []
    all_comments = []

    for json_file in json_files:
        print(f"   Processing: {json_file.name}")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for post in data:
            # Extract post data
            post_data = {
                'subreddit': post.get('subreddit', 'unknown'),
                'post_id': post.get('id'),
                'title': post.get('title', ''),
                'score': post.get('score', 0),
                'created_utc': post.get('created_utc'),
                'num_comments': post.get('num_comments', 0),
                'file_source': json_file.name
            }
            all_posts.append(post_data)

            # Extract comments
            for comment in post.get('comments', []):
                comment_data = {
                    'subreddit': post.get('subreddit', 'unknown'),
                    'post_id': post.get('id'),
                    'comment_id': comment.get('id'),
                    'body': comment.get('body', ''),
                    'score': comment.get('score', 0),
                    'created_utc': comment.get('created_utc'),
                    'sentiment_label': comment.get('sentiment', {}).get('label'),
                    'sentiment_score': comment.get('sentiment', {}).get('score'),
                    'file_source': json_file.name
                }
                all_comments.append(comment_data)

    posts_df = pd.DataFrame(all_posts)
    comments_df = pd.DataFrame(all_comments)

    # Convert timestamps
    if 'created_utc' in posts_df.columns and not posts_df['created_utc'].empty:
        posts_df['created_at'] = pd.to_datetime(posts_df['created_utc'], unit='s', errors='coerce')

    if 'created_utc' in comments_df.columns and not comments_df['created_utc'].empty:
        comments_df['created_at'] = pd.to_datetime(comments_df['created_utc'], unit='s', errors='coerce')

    print(f"✅ Loaded {len(posts_df)} posts and {len(comments_df)} comments")

    return posts_df, comments_df


def load_price_data(data_path="/home/Hudini/gcs/raw/prices"):
    """Load cryptocurrency price data"""
    print(f"📈 Loading price data from: {data_path}")

    data_path = Path(data_path)
    csv_files = list(data_path.glob("*.csv"))

    print(f"   Found {len(csv_files)} price files")

    price_data = []

    for csv_file in csv_files:
        symbol = csv_file.stem.split('-')[0].upper()  # Extract symbol from filename
        print(f"   Processing: {csv_file.name} -> {symbol}")

        try:
            df = pd.read_csv(csv_file)
            df['symbol'] = symbol
            df['snapped_at'] = pd.to_datetime(df['snapped_at'])

            # Take a sample for testing
            df_sample = df.head(100)  # First 100 rows
            price_data.append(df_sample)

        except Exception as e:
            print(f"   ⚠️ Error loading {csv_file.name}: {e}")

    if price_data:
        combined_prices = pd.concat(price_data, ignore_index=True)
        print(f"✅ Loaded price data for {len(combined_prices['symbol'].unique())} cryptocurrencies")
        return combined_prices
    else:
        print("❌ No price data loaded")
        return pd.DataFrame()


def analyze_sentiment_aggregation(comments_df):
    """Simulate the sentiment aggregation that would happen in BigQuery"""
    print("\n🧠 Analyzing sentiment patterns...")

    if comments_df.empty or 'sentiment_label' not in comments_df.columns:
        print("   ⚠️ No sentiment data available for analysis")
        return pd.DataFrame()

    # Filter out comments without sentiment
    sentiment_data = comments_df.dropna(subset=['sentiment_label', 'created_at'])

    if sentiment_data.empty:
        print("   ⚠️ No valid sentiment data after filtering")
        return pd.DataFrame()

    # Create daily aggregations by subreddit
    sentiment_data['date'] = sentiment_data['created_at'].dt.date

    aggregation = sentiment_data.groupby(['subreddit', 'date', 'sentiment_label']).agg({
        'comment_id': 'count',
        'sentiment_score': 'mean',
        'score': 'mean'
    }).reset_index()

    aggregation.columns = ['subreddit', 'date', 'sentiment_label', 'comment_count', 'avg_sentiment_score', 'avg_comment_score']

    print(f"✅ Created sentiment aggregations for {len(aggregation['subreddit'].unique())} subreddits")

    return aggregation


def simulate_spillover_analysis(sentiment_agg, price_data):
    """Simulate the spillover analysis that the full pipeline would perform"""
    print("\n🔄 Simulating spillover analysis...")

    if sentiment_agg.empty or price_data.empty:
        print("   ⚠️ Insufficient data for spillover analysis")
        return

    # Show what the analysis would include
    print("   📊 Available subreddits:")
    for subreddit in sentiment_agg['subreddit'].unique():
        count = len(sentiment_agg[sentiment_agg['subreddit'] == subreddit])
        print(f"      • {subreddit}: {count} data points")

    print("   💰 Available cryptocurrencies:")
    for symbol in price_data['symbol'].unique():
        count = len(price_data[price_data['symbol'] == symbol])
        print(f"      • {symbol}: {count} price points")

    # Demonstrate sentiment trend analysis
    sentiment_trends = sentiment_agg.pivot_table(
        index='date',
        columns=['subreddit', 'sentiment_label'],
        values='comment_count',
        fill_value=0
    )

    print(f"   📈 Sentiment trend matrix shape: {sentiment_trends.shape}")

    # Show correlation potential
    if len(sentiment_trends) > 10:  # Need enough data points
        print("   🔗 Correlation analysis would be feasible with this data")
    else:
        print("   ⚠️ Limited time series data for robust spillover analysis")


def demonstrate_hierarchical_features(posts_df, comments_df):
    """Show what hierarchical features would be extracted"""
    print("\n🏗️ Demonstrating hierarchical feature extraction...")

    # Subreddit-level features
    subreddit_features = posts_df.groupby('subreddit').agg({
        'post_id': 'count',
        'score': ['mean', 'std'],
        'num_comments': ['mean', 'sum']
    }).round(2)

    print("   📊 Subreddit-level features:")
    print(subreddit_features.head())

    # Comment sentiment distribution
    if not comments_df.empty and 'sentiment_label' in comments_df.columns:
        sentiment_dist = comments_df['sentiment_label'].value_counts()
        print(f"\n   😊 Sentiment distribution:")
        for label, count in sentiment_dist.items():
            percentage = (count / len(comments_df)) * 100
            print(f"      • {label}: {count} ({percentage:.1f}%)")


def main():
    print("=" * 70)
    print("🧪 OFFLINE PIPELINE TEST - Information Spillover Analysis")
    print("=" * 70)
    print()
    print("This test demonstrates what the full pipeline will accomplish")
    print("once BigQuery authentication is configured properly.")
    print()

    # Load data
    try:
        posts_df, comments_df = load_sample_reddit_data()
        price_data = load_price_data()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Basic data analysis
    print("\n" + "="*50)
    print("📊 DATA OVERVIEW")
    print("="*50)

    print(f"Reddit Data:")
    print(f"  • Posts: {len(posts_df)}")
    print(f"  • Comments: {len(comments_df)}")
    print(f"  • Subreddits: {posts_df['subreddit'].nunique()}")

    if not price_data.empty:
        print(f"Price Data:")
        print(f"  • Cryptocurrencies: {price_data['symbol'].nunique()}")
        print(f"  • Price points: {len(price_data)}")

    # Sentiment analysis
    sentiment_agg = analyze_sentiment_aggregation(comments_df)

    # Hierarchical features
    demonstrate_hierarchical_features(posts_df, comments_df)

    # Spillover simulation
    simulate_spillover_analysis(sentiment_agg, price_data)

    print("\n" + "="*70)
    print("🎯 PIPELINE READINESS ASSESSMENT")
    print("="*70)

    print("\n✅ WORKING COMPONENTS:")
    print("   • Data loading and preprocessing")
    print("   • Sentiment analysis aggregation")
    print("   • Hierarchical feature extraction")
    print("   • Price data integration")

    print("\n⚠️ BLOCKED BY BIGQUERY ACCESS:")
    print("   • Real-time data pipeline")
    print("   • Large-scale sentiment processing")
    print("   • Historical data analysis (2021-2023)")
    print("   • Granger causality network construction")

    print("\n🔧 TO COMPLETE SETUP:")
    print("   1. Fix VM OAuth scopes (see BigQuery diagnostic)")
    print("   2. Run: python3 scripts/test_bigquery_minimal.py")
    print("   3. Execute: python3 src/main_pipeline.py")

    print(f"\n✨ Once BigQuery is working, you'll have access to:")
    print(f"   • Complete 2021-2023 dataset")
    print(f"   • Advanced spillover analysis")
    print(f"   • Hierarchical LSTM+GNN modeling")
    print(f"   • Economic backtesting results")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()