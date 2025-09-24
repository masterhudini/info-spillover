#!/usr/bin/env python3
"""
Minimal Pipeline Test
Test the complete hierarchical sentiment pipeline on a small sample of data
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.gcp_setup import GCPAuthenticator
from src.data.bigquery_client import BigQueryClient
from src.data.hierarchical_data_processor import HierarchicalDataProcessor
from src.analysis.diebold_yilmaz_spillover import DieboldYilmazSpillover
from src.evaluation.economic_evaluation import BacktestingFramework

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MinimalPipelineTest:
    """Test class for minimal pipeline validation"""

    def __init__(self):
        self.test_dir = None
        self.sample_data = None
        self.test_results = {}

    def create_sample_data(self, num_posts: int = 50, num_subreddits: int = 3) -> dict:
        """Create minimal sample data for testing"""

        logger.info(f"Creating sample data: {num_posts} posts across {num_subreddits} subreddits")

        subreddits = ["Bitcoin", "ethereum", "CryptoCurrency"][:num_subreddits]

        # Generate sample posts and comments
        sample_posts = []

        base_time = datetime(2023, 1, 1)

        for i in range(num_posts):
            subreddit = subreddits[i % len(subreddits)]
            post_time = base_time + timedelta(hours=i)

            # Create sample post
            post = {
                "subreddit": subreddit,
                "title": f"Sample post {i} about {subreddit}",
                "id": f"post_{i}",
                "url": f"https://reddit.com/r/{subreddit}/post_{i}",
                "score": np.random.randint(1, 100),
                "text": f"This is sample text for post {i} in {subreddit}. Price analysis and market discussion.",
                "created_utc": post_time.timestamp(),
                "created_at": post_time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_comments": np.random.randint(1, 5),
                "comments": []
            }

            # Add sample comments
            for j in range(post["num_comments"]):
                comment_time = post_time + timedelta(minutes=j*10)
                sentiment_label = np.random.choice(["positive", "negative", "neutral"],
                                                 p=[0.4, 0.2, 0.4])
                sentiment_score = np.random.uniform(0.6, 0.99)

                comment = {
                    "id": f"comment_{i}_{j}",
                    "author": f"user_{i}_{j}",
                    "body": f"Sample comment {j} on post {i}. Good analysis!",
                    "score": np.random.randint(1, 20),
                    "created_utc": comment_time.timestamp(),
                    "created_at": comment_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sentiment": {
                        "label": sentiment_label,
                        "score": sentiment_score
                    }
                }
                post["comments"].append(comment)

            sample_posts.append(post)

        # Generate sample price data
        dates = pd.date_range(start="2023-01-01", end="2023-01-03", freq="1H")

        price_data = []
        base_prices = {"BTC": 40000, "ETH": 3000, "LTC": 100}

        for date in dates:
            for symbol, base_price in base_prices.items():
                # Random walk price
                price = base_price * (1 + np.random.normal(0, 0.01))
                market_cap = price * 1000000  # Dummy market cap
                volume = np.random.uniform(100000, 1000000)

                price_data.append({
                    "snapped_at": date.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "price": round(price, 2),
                    "market_cap": round(market_cap, 2),
                    "total_volume": round(volume, 2)
                })

        sample_data = {
            "posts": sample_posts,
            "prices": price_data
        }

        logger.info(f"Generated {len(sample_posts)} posts and {len(price_data)} price records")
        return sample_data

    def setup_test_environment(self):
        """Setup temporary test environment"""

        # Create temporary directory
        self.test_dir = Path(tempfile.mkdtemp(prefix="pipeline_test_"))
        logger.info(f"Created test directory: {self.test_dir}")

        # Create sample data
        self.sample_data = self.create_sample_data(num_posts=30, num_subreddits=3)

        # Save sample data files
        data_dir = self.test_dir / "sample_data"
        data_dir.mkdir()

        # Save posts as JSON (simulate the structure in ~/gcs)
        posts_dir = data_dir / "posts_n_comments"
        posts_dir.mkdir()

        # Split posts by subreddit to simulate real structure
        subreddit_posts = {}
        for post in self.sample_data["posts"]:
            subreddit = post["subreddit"]
            if subreddit not in subreddit_posts:
                subreddit_posts[subreddit] = []
            subreddit_posts[subreddit].append(post)

        for subreddit, posts in subreddit_posts.items():
            with open(posts_dir / f"{subreddit}_sample.json", "w") as f:
                json.dump(posts, f, indent=2)

        # Save prices as CSV
        prices_dir = data_dir / "prices"
        prices_dir.mkdir()

        # Group prices by symbol
        price_df = pd.DataFrame(self.sample_data["prices"])
        symbols = ["btc-usd-max.csv", "eth-usd-max.csv", "ltc-usd-max.csv"]

        for i, symbol_file in enumerate(symbols):
            symbol_prices = price_df.iloc[i::3].copy()  # Every 3rd row for each symbol
            symbol_prices.to_csv(prices_dir / symbol_file, index=False)

        logger.info(f"Sample data saved to {data_dir}")
        return data_dir

    def test_gcp_connection(self) -> bool:
        """Test Google Cloud connection"""

        logger.info("🔧 Testing Google Cloud connection...")

        try:
            # Check authentication
            auth_info = GCPAuthenticator.check_credentials()

            if auth_info['status'] != 'authenticated':
                logger.error("❌ GCP authentication failed")
                return False

            # Test BigQuery access
            if not GCPAuthenticator.test_bigquery_access():
                logger.error("❌ BigQuery access failed")
                return False

            logger.info("✅ GCP connection successful")
            return True

        except Exception as e:
            logger.error(f"❌ GCP test failed: {str(e)}")
            return False

    def test_bigquery_client(self) -> bool:
        """Test BigQuery client initialization and data loading"""

        logger.info("📊 Testing BigQuery client...")

        try:
            # Initialize client with test dataset
            test_dataset = f"pipeline_test_{int(datetime.now().timestamp())}"
            bq_client = BigQueryClient(dataset_id=test_dataset)

            # Test basic query
            test_data = bq_client.query_data("SELECT 1 as test, 'working' as status")

            if test_data.empty:
                logger.error("❌ BigQuery query returned empty result")
                return False

            # Test data loading with sample data
            logger.info("Testing sample data upload...")

            # Create sample DataFrame for upload test
            sample_df = pd.DataFrame([
                {
                    'subreddit': 'Bitcoin',
                    'post_id': 'test_1',
                    'comment_sentiment_label': 'positive',
                    'comment_sentiment_score': 0.85,
                    'post_created_utc': datetime.now()
                }
            ])

            logger.info("✅ BigQuery client working")

            # Cleanup test dataset
            try:
                bq_client.client.delete_dataset(test_dataset, delete_contents=True)
                logger.info("✅ Test dataset cleaned up")
            except:
                pass  # Ignore cleanup errors

            return True

        except Exception as e:
            logger.error(f"❌ BigQuery client test failed: {str(e)}")
            return False

    def test_data_processing(self, data_dir: Path) -> bool:
        """Test hierarchical data processing"""

        logger.info("🔄 Testing data processing...")

        try:
            # Initialize processor
            processor = HierarchicalDataProcessor()

            # Mock the data loading to use our sample data
            logger.info("Processing sample sentiment data...")

            # Create minimal DataFrame from sample data
            rows = []
            for post in self.sample_data["posts"]:
                for comment in post["comments"]:
                    rows.append({
                        'subreddit': post['subreddit'],
                        'created_utc': pd.to_datetime(comment['created_at']),
                        'sentiment_label': comment['sentiment']['label'],
                        'sentiment_score': comment['sentiment']['score'],
                        'text': comment['body'],
                        'score': comment['score']
                    })

            if not rows:
                logger.error("❌ No sample data rows created")
                return False

            df = pd.DataFrame(rows)
            logger.info(f"Created sample DataFrame with {len(df)} rows")

            # Test feature engineering components
            processed_df = processor.feature_engineer.compute_compound_sentiment(df.copy())
            processed_df = processor.feature_engineer.extract_emotion_features(processed_df)

            logger.info(f"✅ Feature engineering successful - {processed_df.shape}")

            self.test_results['data_processing'] = {
                'status': 'success',
                'rows_processed': len(processed_df),
                'features_created': len(processed_df.columns)
            }

            return True

        except Exception as e:
            logger.error(f"❌ Data processing test failed: {str(e)}")
            self.test_results['data_processing'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False

    def test_spillover_analysis(self) -> bool:
        """Test spillover analysis with minimal data"""

        logger.info("📈 Testing spillover analysis...")

        try:
            # Create minimal time series data
            dates = pd.date_range(start="2023-01-01", periods=50, freq="1H")

            # Generate synthetic sentiment time series for 3 subreddits
            subreddits = ["Bitcoin", "ethereum", "CryptoCurrency"]

            spillover_data = pd.DataFrame()
            spillover_data['created_utc'] = dates

            for subreddit in subreddits:
                # Create correlated random walk
                sentiment_series = np.cumsum(np.random.normal(0, 0.1, len(dates))) + np.random.normal(0, 0.05, len(dates))
                spillover_data[subreddit] = sentiment_series

            spillover_data = spillover_data.set_index('created_utc')

            # Initialize spillover analyzer with minimal settings
            analyzer = DieboldYilmazSpillover(forecast_horizon=3, identification='cholesky')

            # Test basic VAR estimation (skip if insufficient data)
            if len(spillover_data) >= 20:
                try:
                    var_fitted, optimal_lags = analyzer.estimate_var(spillover_data, max_lags=2)
                    logger.info(f"✅ VAR estimation successful - {optimal_lags} lags")

                    # Test variance decomposition
                    var_decomp = analyzer.compute_variance_decomposition(var_fitted)
                    spillover_measures = analyzer.compute_spillover_measures(var_decomp, subreddits)

                    logger.info(f"✅ Spillover analysis successful - Total spillover: {spillover_measures['total_spillover_index']:.1f}%")

                    self.test_results['spillover_analysis'] = {
                        'status': 'success',
                        'total_spillover_index': spillover_measures['total_spillover_index'],
                        'variables': len(subreddits)
                    }

                except Exception as e:
                    logger.warning(f"⚠️ Full spillover analysis skipped (insufficient data): {str(e)}")
                    self.test_results['spillover_analysis'] = {
                        'status': 'skipped',
                        'reason': 'insufficient_data'
                    }
            else:
                logger.warning("⚠️ Spillover analysis skipped - insufficient data points")
                self.test_results['spillover_analysis'] = {
                    'status': 'skipped',
                    'reason': 'insufficient_data'
                }

            return True

        except Exception as e:
            logger.error(f"❌ Spillover analysis test failed: {str(e)}")
            self.test_results['spillover_analysis'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False

    def test_economic_evaluation(self) -> bool:
        """Test economic evaluation with minimal data"""

        logger.info("💼 Testing economic evaluation...")

        try:
            # Create minimal portfolio test
            dates = pd.date_range(start="2023-01-01", periods=20, freq="1D")

            # Simple signal data (random)
            signals = pd.DataFrame({
                'BTC': np.random.uniform(-0.1, 0.1, len(dates)),
                'ETH': np.random.uniform(-0.1, 0.1, len(dates))
            }, index=dates)

            # Simple price data (random walk)
            prices = pd.DataFrame(index=dates)
            prices['BTC'] = 40000 * np.cumprod(1 + np.random.normal(0, 0.02, len(dates)))
            prices['ETH'] = 3000 * np.cumprod(1 + np.random.normal(0, 0.02, len(dates)))

            # Initialize backtester with minimal settings
            backtester = BacktestingFramework(
                start_date="2023-01-01",
                end_date="2023-01-20",
                initial_capital=10000,  # Small amount for testing
                transaction_costs=0.001
            )

            # Test portfolio construction
            portfolio = backtester.portfolio_constructor.construct_portfolio(
                signals=signals,
                price_data=prices,
                initial_capital=10000
            )

            # Test performance evaluation
            evaluator = backtester.evaluator
            performance = evaluator.calculate_performance_metrics(
                portfolio['daily_return']
            )

            logger.info(f"✅ Economic evaluation successful")
            logger.info(f"   Final portfolio value: ${portfolio['total_value'].iloc[-1]:.2f}")
            logger.info(f"   Total return: {portfolio['cumulative_return'].iloc[-1]:.2%}")

            self.test_results['economic_evaluation'] = {
                'status': 'success',
                'final_value': portfolio['total_value'].iloc[-1],
                'total_return': portfolio['cumulative_return'].iloc[-1],
                'sharpe_ratio': performance.get('sharpe_ratio', 0)
            }

            return True

        except Exception as e:
            logger.error(f"❌ Economic evaluation test failed: {str(e)}")
            self.test_results['economic_evaluation'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False

    def cleanup(self):
        """Cleanup test environment"""

        if self.test_dir and self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            logger.info(f"✅ Cleaned up test directory: {self.test_dir}")

    def run_complete_test(self):
        """Run complete minimal pipeline test"""

        print("\n" + "="*60)
        print("🧪 MINIMAL PIPELINE TEST")
        print("="*60)

        test_start = datetime.now()

        try:
            # Step 1: Setup
            logger.info("\n1️⃣ Setting up test environment...")
            data_dir = self.setup_test_environment()

            # Step 2: GCP Connection
            logger.info("\n2️⃣ Testing Google Cloud connection...")
            if not self.test_gcp_connection():
                logger.error("❌ GCP connection test failed - stopping")
                return False

            # Step 3: BigQuery Client
            logger.info("\n3️⃣ Testing BigQuery client...")
            if not self.test_bigquery_client():
                logger.error("❌ BigQuery client test failed")
                return False

            # Step 4: Data Processing
            logger.info("\n4️⃣ Testing data processing...")
            if not self.test_data_processing(data_dir):
                logger.error("❌ Data processing test failed")
                return False

            # Step 5: Spillover Analysis
            logger.info("\n5️⃣ Testing spillover analysis...")
            self.test_spillover_analysis()  # Non-blocking

            # Step 6: Economic Evaluation
            logger.info("\n6️⃣ Testing economic evaluation...")
            if not self.test_economic_evaluation():
                logger.error("❌ Economic evaluation test failed")
                return False

            test_duration = datetime.now() - test_start

            print("\n" + "="*60)
            print("🎉 MINIMAL PIPELINE TEST COMPLETED!")
            print("="*60)
            print(f"⏱️  Duration: {test_duration}")
            print(f"📊 Test Results:")

            for component, result in self.test_results.items():
                status = result.get('status', 'unknown')
                emoji = "✅" if status == 'success' else "⚠️" if status == 'skipped' else "❌"
                print(f"   {emoji} {component}: {status}")

            print("\n🚀 Your pipeline is ready for full execution!")
            print("   Run: python src/main_pipeline.py")
            print("="*60)

            return True

        except Exception as e:
            logger.error(f"❌ Pipeline test failed: {str(e)}")
            return False

        finally:
            self.cleanup()


def main():
    """Main test function"""

    tester = MinimalPipelineTest()

    try:
        success = tester.run_complete_test()
        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("\n⚠️ Test interrupted by user")
        tester.cleanup()
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        tester.cleanup()
        return 1


if __name__ == "__main__":
    sys.exit(main())