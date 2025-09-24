"""BigQuery client for data processing and analysis"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Union
import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BigQueryClient:
    """Client for BigQuery operations"""

    def __init__(self, project_id: str = None, dataset_id: str = "info_spillover"):
        """Initialize BigQuery client

        Args:
            project_id: Google Cloud project ID. If None, uses default from environment
            dataset_id: BigQuery dataset name
        """
        # Initialize BigQuery client with default credentials
        try:
            self.client = bigquery.Client(project=project_id)
            self.project_id = project_id or self.client.project
            self.dataset_id = dataset_id
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

    def get_post_sentiment_aggregation(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get aggregated sentiment data by subreddit and date

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with sentiment aggregations
        """
        date_filter = ""
        if start_date and end_date:
            date_filter = f"WHERE DATE(post_created_utc) BETWEEN '{start_date}' AND '{end_date}'"
        elif start_date:
            date_filter = f"WHERE DATE(post_created_utc) >= '{start_date}'"
        elif end_date:
            date_filter = f"WHERE DATE(post_created_utc) <= '{end_date}'"

        query = f"""
        SELECT
            subreddit,
            DATE(post_created_utc) as date,
            COUNT(DISTINCT post_id) as num_posts,
            COUNT(comment_id) as num_comments,
            AVG(CASE WHEN comment_sentiment_label = 'positive' THEN comment_sentiment_score END) as avg_positive_sentiment,
            AVG(CASE WHEN comment_sentiment_label = 'negative' THEN comment_sentiment_score END) as avg_negative_sentiment,
            AVG(CASE WHEN comment_sentiment_label = 'neutral' THEN comment_sentiment_score END) as avg_neutral_sentiment,
            SUM(CASE WHEN comment_sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_comments,
            SUM(CASE WHEN comment_sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_comments,
            SUM(CASE WHEN comment_sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_comments,
            AVG(post_score) as avg_post_score,
            AVG(comment_score) as avg_comment_score
        FROM `{self.project_id}.{self.dataset_id}.posts_comments`
        {date_filter}
        GROUP BY subreddit, DATE(post_created_utc)
        ORDER BY subreddit, date
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
            price,
            market_cap,
            total_volume
        FROM `{self.project_id}.{self.dataset_id}.crypto_prices`
        {where_clause}
        ORDER BY symbol, snapped_at
        """

        return self.query_data(query)

    def create_combined_dataset(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Create combined dataset with sentiment and price data for analysis

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with combined data for ML training
        """
        date_filter = ""
        if start_date and end_date:
            date_filter = f"AND DATE(s.date) BETWEEN '{start_date}' AND '{end_date}'"
        elif start_date:
            date_filter = f"AND DATE(s.date) >= '{start_date}'"
        elif end_date:
            date_filter = f"AND DATE(s.date) <= '{end_date}'"

        query = f"""
        WITH sentiment_data AS (
            SELECT
                subreddit,
                DATE(post_created_utc) as date,
                COUNT(DISTINCT post_id) as num_posts,
                COUNT(comment_id) as num_comments,
                AVG(CASE WHEN comment_sentiment_label = 'positive' THEN comment_sentiment_score END) as avg_positive_sentiment,
                AVG(CASE WHEN comment_sentiment_label = 'negative' THEN comment_sentiment_score END) as avg_negative_sentiment,
                AVG(CASE WHEN comment_sentiment_label = 'neutral' THEN comment_sentiment_score END) as avg_neutral_sentiment,
                SUM(CASE WHEN comment_sentiment_label = 'positive' THEN 1 ELSE 0 END) / COUNT(comment_id) as positive_ratio,
                SUM(CASE WHEN comment_sentiment_label = 'negative' THEN 1 ELSE 0 END) / COUNT(comment_id) as negative_ratio,
                SUM(CASE WHEN comment_sentiment_label = 'neutral' THEN 1 ELSE 0 END) / COUNT(comment_id) as neutral_ratio,
                AVG(post_score) as avg_post_score,
                AVG(comment_score) as avg_comment_score
            FROM `{self.project_id}.{self.dataset_id}.posts_comments`
            WHERE comment_id IS NOT NULL
            GROUP BY subreddit, DATE(post_created_utc)
        ),
        price_data AS (
            SELECT
                symbol,
                DATE(snapped_at) as date,
                price,
                market_cap,
                total_volume,
                LAG(price, 1) OVER (PARTITION BY symbol ORDER BY DATE(snapped_at)) as prev_price,
                LEAD(price, 1) OVER (PARTITION BY symbol ORDER BY DATE(snapped_at)) as next_price
            FROM `{self.project_id}.{self.dataset_id}.crypto_prices`
        )
        SELECT
            s.*,
            p.symbol,
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
            CASE WHEN p.prev_price > 0 THEN (p.price - p.prev_price) / p.prev_price ELSE 0 END as price_change_pct
        FROM sentiment_data s
        JOIN price_data p ON s.date = p.date
        WHERE 1=1 {date_filter}
        ORDER BY s.date, s.subreddit, p.symbol
        """

        return self.query_data(query)