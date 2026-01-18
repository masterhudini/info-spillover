"""BigQuery client for data processing and analysis"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Union
import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import logging

logger = logging.getLogger(__name__)


class BigQueryClient:
    """Client for BigQuery operations"""

    def __init__(self, project_id: str = None, dataset_id: str = "info_spillover",
                 sentiment_table: str = None, aggregation_interval_seconds: int = None):
        """Initialize BigQuery client

        Args:
            project_id: Google Cloud project ID. If None, uses default from environment
            dataset_id: BigQuery dataset name
            sentiment_table: Name of sentiment table to use
        """
        # Initialize BigQuery client with default credentials
        try:
            self.client = bigquery.Client(project=project_id)
            self.project_id = project_id or self.client.project
            self.dataset_id = dataset_id
            if sentiment_table is None:
                sentiment_table = os.getenv("BIGQUERY_SENTIMENT_TABLE")
                if not sentiment_table:
                    raise ValueError("Missing required env var: BIGQUERY_SENTIMENT_TABLE")
            self.sentiment_table = sentiment_table
            if aggregation_interval_seconds is None:
                aggregation_interval_seconds = os.getenv("SENTIMENT_AGG_INTERVAL_SECONDS")
                if not aggregation_interval_seconds:
                    raise ValueError("Missing required env var: SENTIMENT_AGG_INTERVAL_SECONDS")
            self.aggregation_interval_seconds = int(aggregation_interval_seconds)
            self.dataset_ref = self.client.dataset(dataset_id)

            logger.info(f"BigQuery client initialized for project: {self.project_id}")

            # Validate connection and permissions
            self._validate_connection()

            # Create dataset if it doesn't exist
            self._ensure_dataset_exists()

        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client: {str(e)}")
            logger.error("Please ensure Google Cloud credentials are properly configured:")
            logger.error("1. Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
            logger.error("2. Or run 'gcloud auth application-default login'")
            logger.error("3. Ensure the service account has BigQuery permissions")
            raise

    def _validate_connection(self):
        """Validate BigQuery connection and permissions"""
        try:
            # Simple query to test connection
            test_query = "SELECT 1 as test"
            query_job = self.client.query(test_query)
            results = list(query_job.result())

            logger.info("✅ BigQuery connection validated successfully")

            # Check if we can list datasets (basic permission check)
            datasets = list(self.client.list_datasets())
            logger.info(f"✅ BigQuery permissions validated - can access {len(datasets)} datasets")

        except Exception as e:
            logger.error(f"❌ BigQuery validation failed: {str(e)}")
            raise ConnectionError(f"Cannot connect to BigQuery: {str(e)}")

    def _ensure_dataset_exists(self):
        """Create dataset if it doesn't exist"""
        try:
            self.client.get_dataset(self.dataset_ref)
            logger.info(f"Dataset {self.dataset_id} already exists")
        except NotFound:
            dataset = bigquery.Dataset(self.dataset_ref)
            dataset.location = "US"
            dataset.description = "Information spillover analysis data"

            dataset = self.client.create_dataset(dataset, timeout=30)
            logger.info(f"Created dataset {self.dataset_id}")

    def create_posts_table(self):
        """Create table for Reddit posts and comments"""
        table_id = f"{self.project_id}.{self.dataset_id}.posts_comments"

        schema = [
            bigquery.SchemaField("subreddit", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("post_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("post_title", "STRING"),
            bigquery.SchemaField("post_url", "STRING"),
            bigquery.SchemaField("post_score", "INTEGER"),
            bigquery.SchemaField("post_text", "STRING"),
            bigquery.SchemaField("post_created_utc", "TIMESTAMP"),
            bigquery.SchemaField("post_created_at", "DATETIME"),
            bigquery.SchemaField("num_comments", "INTEGER"),
            bigquery.SchemaField("comment_id", "STRING"),
            bigquery.SchemaField("comment_author", "STRING"),
            bigquery.SchemaField("comment_body", "STRING"),
            bigquery.SchemaField("comment_score", "INTEGER"),
            bigquery.SchemaField("comment_created_utc", "TIMESTAMP"),
            bigquery.SchemaField("comment_created_at", "DATETIME"),
            bigquery.SchemaField("comment_sentiment_label", "STRING"),
            bigquery.SchemaField("comment_sentiment_score", "FLOAT"),
            bigquery.SchemaField("batch_file", "STRING"),
        ]

        table = bigquery.Table(table_id, schema=schema)

        # Set partitioning by date
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="post_created_utc"
        )

        try:
            table = self.client.create_table(table)
            logger.info(f"Created table {table_id}")
            return table
        except Exception as e:
            if "Already Exists" in str(e):
                logger.info(f"Table {table_id} already exists")
                return self.client.get_table(table_id)
            else:
                raise e

    def create_prices_table(self):
        """Create table for cryptocurrency prices"""
        table_id = f"{self.project_id}.{self.dataset_id}.crypto_prices"

        schema = [
            bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("snapped_at", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("price", "FLOAT"),
            bigquery.SchemaField("market_cap", "FLOAT"),
            bigquery.SchemaField("total_volume", "FLOAT"),
        ]

        table = bigquery.Table(table_id, schema=schema)

        # Set partitioning by date
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="snapped_at"
        )

        try:
            table = self.client.create_table(table)
            logger.info(f"Created table {table_id}")
            return table
        except Exception as e:
            if "Already Exists" in str(e):
                logger.info(f"Table {table_id} already exists")
                return self.client.get_table(table_id)
            else:
                raise e

    def load_json_data_to_bq(self, json_files_path: str):
        """Load JSON data from files to BigQuery

        Args:
            json_files_path: Path to directory containing JSON files
        """
        self.create_posts_table()
        table_id = f"{self.project_id}.{self.dataset_id}.posts_comments"

        json_path = Path(json_files_path)
        json_files = list(json_path.glob("*.json"))

        logger.info(f"Found {len(json_files)} JSON files to process")

        for json_file in json_files:
            logger.info(f"Processing {json_file.name}")

            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Flatten the data structure
            rows_to_insert = []
            for post in data:
                post_data = {
                    'subreddit': post.get('subreddit'),
                    'post_id': post.get('id'),
                    'post_title': post.get('title'),
                    'post_url': post.get('url'),
                    'post_score': post.get('score'),
                    'post_text': post.get('text'),
                    'post_created_utc': pd.to_datetime(post.get('created_utc'), unit='s') if post.get('created_utc') else None,
                    'post_created_at': pd.to_datetime(post.get('created_at')) if post.get('created_at') else None,
                    'num_comments': post.get('num_comments'),
                    'batch_file': json_file.name,
                }

                # If there are no comments, still insert the post
                if not post.get('comments'):
                    rows_to_insert.append({**post_data, **{
                        'comment_id': None,
                        'comment_author': None,
                        'comment_body': None,
                        'comment_score': None,
                        'comment_created_utc': None,
                        'comment_created_at': None,
                        'comment_sentiment_label': None,
                        'comment_sentiment_score': None,
                    }})
                else:
                    # Insert one row per comment
                    for comment in post.get('comments', []):
                        comment_data = {
                            'comment_id': comment.get('id'),
                            'comment_author': comment.get('author'),
                            'comment_body': comment.get('body'),
                            'comment_score': comment.get('score'),
                            'comment_created_utc': pd.to_datetime(comment.get('created_utc'), unit='s') if comment.get('created_utc') else None,
                            'comment_created_at': pd.to_datetime(comment.get('created_at')) if comment.get('created_at') else None,
                            'comment_sentiment_label': comment.get('sentiment', {}).get('label'),
                            'comment_sentiment_score': comment.get('sentiment', {}).get('score'),
                        }
                        rows_to_insert.append({**post_data, **comment_data})

            # Insert data in batches
            if rows_to_insert:
                errors = self.client.insert_rows_json(
                    self.client.get_table(table_id),
                    rows_to_insert
                )
                if errors:
                    logger.error(f"Errors inserting data from {json_file.name}: {errors}")
                else:
                    logger.info(f"Successfully inserted {len(rows_to_insert)} rows from {json_file.name}")

    def load_csv_data_to_bq(self, csv_files_path: str):
        """Load CSV price data to BigQuery

        Args:
            csv_files_path: Path to directory containing CSV files
        """
        self.create_prices_table()
        table_id = f"{self.project_id}.{self.dataset_id}.crypto_prices"

        csv_path = Path(csv_files_path)
        csv_files = list(csv_path.glob("*.csv"))

        logger.info(f"Found {len(csv_files)} CSV files to process")

        for csv_file in csv_files:
            logger.info(f"Processing {csv_file.name}")

            # Extract symbol from filename (e.g., btc-usd-max.csv -> BTC)
            symbol = csv_file.stem.split('-')[0].upper()

            # Read CSV
            df = pd.read_csv(csv_file)
            df['symbol'] = symbol
            df['snapped_at'] = pd.to_datetime(df['snapped_at'])

            # Reorder columns to match schema
            df = df[['symbol', 'snapped_at', 'price', 'market_cap', 'total_volume']]

            # Load to BigQuery
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_APPEND",
                schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
            )

            job = self.client.load_table_from_dataframe(
                df, table_id, job_config=job_config
            )
            job.result()  # Wait for job to complete

            logger.info(f"Successfully loaded {len(df)} rows from {csv_file.name}")

    def query_data(self, query: str) -> pd.DataFrame:
        """Execute a query and return results as DataFrame

        Args:
            query: SQL query string

        Returns:
            DataFrame with query results
        """
        return self.client.query(query).to_dataframe()

    def get_post_sentiment_aggregation(self, start_date: str = None, end_date: str = None,
                                       subreddits: Optional[List[str]] = None) -> pd.DataFrame:
        """Get aggregated sentiment data by subreddit at 5-minute intervals

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            subreddits: Optional list of subreddits to include

        Returns:
            DataFrame with sentiment aggregations at 5-minute intervals
        """
        conditions = []
        if start_date and end_date:
            conditions.append(f"DATE(post_created_utc) BETWEEN '{start_date}' AND '{end_date}'")
        elif start_date:
            conditions.append(f"DATE(post_created_utc) >= '{start_date}'")
        elif end_date:
            conditions.append(f"DATE(post_created_utc) <= '{end_date}'")

        if subreddits is None:
            subreddits_env = os.getenv("SUBREDDITS_SELECTED")
            if subreddits_env:
                subreddits = [s.strip() for s in subreddits_env.split(",") if s.strip()]

        if subreddits:
            subs = "', '".join(subreddits)
            conditions.append(f"subreddit IN ('{subs}')")

        date_filter = ""
        if conditions:
            date_filter = "WHERE " + " AND ".join(conditions)

        interval_seconds = self.aggregation_interval_seconds
        query = f"""
        SELECT
            subreddit,
            -- Create 5-minute bins: truncate to minute, then round down to nearest 5-minute interval
            TIMESTAMP_SECONDS(
                CAST(FLOOR(UNIX_SECONDS(post_created_utc) / {interval_seconds}) * {interval_seconds} AS INT64)
            ) as post_created_utc,
            COUNT(DISTINCT post_id) as num_posts,
            COUNT(DISTINCT comment_id) as num_comments,
            -- Use new_sentiment_* columns for averages
            AVG(new_sentiment_positive) as avg_positive_sentiment,
            AVG(new_sentiment_negative) as avg_negative_sentiment,
            AVG(new_sentiment_neutral) as avg_neutral_sentiment,
            AVG(new_sentiment_compound) as avg_compound_sentiment,
            -- Calculate counts based on new_sentiment_label
            COUNTIF(new_sentiment_label = 'positive') as positive_comments,
            COUNTIF(new_sentiment_label = 'negative') as negative_comments,
            COUNTIF(new_sentiment_label = 'neutral') as neutral_comments,
            -- Note: post_score and comment_score not available in enhanced_sentiment table
            0.0 as avg_post_score,
            0.0 as avg_comment_score
        FROM `{self.project_id}.{self.dataset_id}.{self.sentiment_table}`
        {date_filter}
        GROUP BY subreddit, TIMESTAMP_SECONDS(CAST(FLOOR(UNIX_SECONDS(post_created_utc) / {interval_seconds}) * {interval_seconds} AS INT64))
        ORDER BY subreddit, post_created_utc
        """

        return self.query_data(query)

    def get_price_data(self, symbols: List[str] = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get cryptocurrency price data

        Args:
            symbols: List of crypto symbols (e.g., ['BTC', 'ETH'])
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with price data
        """
        where_clauses = []

        if symbols:
            symbols_str = "', '".join(symbols)
            where_clauses.append(f"symbol IN ('{symbols_str}')")

        if start_date:
            where_clauses.append(f"DATE(snapped_at) >= '{start_date}'")

        if end_date:
            where_clauses.append(f"DATE(snapped_at) <= '{end_date}'")

        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
        SELECT
            symbol,
            DATE(snapped_at) as date,
            snapped_at,
            price_usd as price,
            market_cap_usd as market_cap,
            volume_24h as total_volume
        FROM `{self.project_id}.{self.dataset_id}.crypto_prices`
        {where_clause}
        ORDER BY symbol, snapped_at
        """

        return self.query_data(query)

    def create_combined_dataset(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Create combined dataset with sentiment and 5-minute price data for analysis

        Uses {self.sentiment_table} table with new_sentiment_* columns.
        Joins daily sentiment data with 5-minute crypto prices.
        Each 5-minute price observation gets matched with sentiment from that day.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with combined data for ML training (5-minute granularity)
        """
        date_filter = ""
        if start_date and end_date:
            date_filter = f"AND DATE(p.snapped_at) BETWEEN '{start_date}' AND '{end_date}'"
        elif start_date:
            date_filter = f"AND DATE(p.snapped_at) >= '{start_date}'"
        elif end_date:
            date_filter = f"AND DATE(p.snapped_at) <= '{end_date}'"

        query = f"""
        WITH sentiment_data AS (
            SELECT
                subreddit,
                DATE(post_created_utc) as date,
                COUNT(DISTINCT post_id) as num_posts,
                COUNT(DISTINCT comment_id) as num_comments,
                -- Use new_sentiment_positive/negative/neutral for averages
                AVG(new_sentiment_positive) as avg_positive_sentiment,
                AVG(new_sentiment_negative) as avg_negative_sentiment,
                AVG(new_sentiment_neutral) as avg_neutral_sentiment,
                AVG(new_sentiment_compound) as avg_compound_sentiment,
                -- Calculate ratios based on new_sentiment_label
                COUNTIF(new_sentiment_label = 'positive') / NULLIF(COUNT(comment_id), 0) as positive_ratio,
                COUNTIF(new_sentiment_label = 'negative') / NULLIF(COUNT(comment_id), 0) as negative_ratio,
                COUNTIF(new_sentiment_label = 'neutral') / NULLIF(COUNT(comment_id), 0) as neutral_ratio,
                -- Note: {self.sentiment_table} doesn't have post_score/comment_score
                -- Using placeholders for now (can be joined from posts_comments if needed)
                0.0 as avg_post_score,
                0.0 as avg_comment_score
            FROM `{self.project_id}.{self.dataset_id}.{self.sentiment_table}`
            WHERE comment_id IS NOT NULL
            GROUP BY subreddit, DATE(post_created_utc)
        ),
        price_data_5min AS (
            SELECT
                symbol,
                snapped_at,
                DATE(snapped_at) as date,
                price_usd as price,
                market_cap_usd as market_cap,
                volume_24h as total_volume,
                LAG(price_usd, 1) OVER (PARTITION BY symbol ORDER BY snapped_at) as prev_price,
                LEAD(price_usd, 1) OVER (PARTITION BY symbol ORDER BY snapped_at) as next_price
            FROM `{self.project_id}.{self.dataset_id}.crypto_prices`
        )
        SELECT
            p.symbol,
            p.snapped_at,
            p.date,
            p.price,
            p.market_cap,
            p.total_volume,
            p.prev_price,
            p.next_price,
            CASE
                WHEN p.next_price > p.price THEN 1
                WHEN p.next_price < p.price THEN -1
                ELSE 0
            END as price_direction,
            CASE WHEN p.prev_price > 0 THEN (p.price - p.prev_price) / p.prev_price ELSE 0 END as price_change_pct,
            s.subreddit,
            s.num_posts,
            s.num_comments,
            s.avg_positive_sentiment,
            s.avg_negative_sentiment,
            s.avg_neutral_sentiment,
            s.avg_compound_sentiment,
            s.positive_ratio,
            s.negative_ratio,
            s.neutral_ratio,
            s.avg_post_score,
            s.avg_comment_score
        FROM price_data_5min p
        LEFT JOIN sentiment_data s ON p.date = s.date
        WHERE 1=1 {date_filter}
        ORDER BY p.snapped_at, p.symbol, s.subreddit
        """

        return self.query_data(query)
