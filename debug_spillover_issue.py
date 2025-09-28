#!/usr/bin/env python3
"""
Debug script to trace the exact spillover to GNN issue
"""

import pandas as pd
import numpy as np
import networkx as nx
import logging
from datetime import datetime, timedelta

# Add project root to path
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.analysis.diebold_yilmaz_spillover import DieboldYilmazSpillover
from src.models.hierarchical_models import HierarchicalCollator, SubredditTimeSeriesDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_minimal_spillover_test():
    """Create minimal test data to reproduce the spillover->GNN issue"""

    logger.info("=== DEBUGGING SPILLOVER -> GNN ISSUE ===")

    # 1. Create minimal time series data for VAR analysis
    logger.info("1. Creating minimal VAR data...")
    dates = pd.date_range('2024-01-01', periods=50, freq='H')
    subreddits = ['bitcoin', 'ethereum', 'crypto']

    # Create VAR-compatible time series data
    var_data = pd.DataFrame(index=dates)
    for subreddit in subreddits:
        # Create correlated sentiment series
        var_data[subreddit] = np.cumsum(np.random.normal(0, 0.1, len(dates)))

    logger.info(f"VAR data shape: {var_data.shape}")
    logger.info(f"Subreddits: {var_data.columns.tolist()}")

    # 2. Run Diebold-Yilmaz spillover analysis
    logger.info("2. Running spillover analysis...")
    analyzer = DieboldYilmazSpillover(forecast_horizon=5, identification='cholesky')

    try:
        spillover_results = analyzer.analyze_spillover_dynamics(
            var_data,
            save_results=False
        )
        logger.info("✅ Spillover analysis completed")
    except Exception as e:
        logger.error(f"❌ Spillover analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Examine the spillover network structure
    logger.info("3. Examining spillover network...")
    network = spillover_results.get('network')

    if network:
        logger.info(f"Network nodes: {list(network.nodes())}")
        logger.info(f"Network edges: {list(network.edges())}")
        logger.info(f"Node attributes: {dict(network.nodes(data=True))}")
        logger.info(f"Edge attributes: {[(u, v, data) for u, v, data in network.edges(data=True)]}")
    else:
        logger.error("❌ No network created")
        return

    # 4. Create hierarchical datasets like the real pipeline
    logger.info("4. Creating hierarchical datasets...")

    # Create expanded dataset for hierarchical models
    expanded_data = []
    for date in dates:
        for subreddit in subreddits:
            expanded_data.append({
                'created_utc': date,
                'subreddit': subreddit,
                'compound_sentiment': var_data.loc[date, subreddit],
                'sentiment_positive': np.random.uniform(0, 1),
                'sentiment_negative': np.random.uniform(0, 1),
                'emotion_joy': np.random.uniform(0, 1),
                'ratio_upvote': np.random.uniform(0.5, 1),
                'count_comments': np.random.randint(1, 10)
            })

    expanded_df = pd.DataFrame(expanded_data)
    logger.info(f"Expanded data shape: {expanded_df.shape}")

    # Create datasets like hierarchical models would
    datasets = {}
    for subreddit in subreddits:
        try:
            dataset = SubredditTimeSeriesDataset(
                df=expanded_df,
                subreddit=subreddit,
                sequence_length=5,
                prediction_horizon=1
            )
            datasets[subreddit] = dataset
            logger.info(f"Created dataset for {subreddit}: {len(dataset)} sequences")
        except Exception as e:
            logger.error(f"Failed to create dataset for {subreddit}: {e}")
            return

    # 5. Test the hierarchical collator with the spillover network
    logger.info("5. Testing hierarchical collator...")

    collator = HierarchicalCollator(datasets, network)

    # Simulate a batch with different scenarios
    test_scenarios = [
        ("All subreddits", list(subreddits)),
        ("Missing one", subreddits[:-1]),
        ("Single subreddit", [subreddits[0]]),
    ]

    for scenario_name, available_subreddits in test_scenarios:
        logger.info(f"\n--- Testing: {scenario_name} ---")
        logger.info(f"Available: {available_subreddits}")

        try:
            graph_data = collator._create_graph_data(available_subreddits)

            if graph_data:
                logger.info(f"✅ Graph created: {graph_data.num_nodes} nodes, {graph_data.edge_index.shape[1]} edges")

                # Check for indexing issues
                if graph_data.edge_index.numel() > 0:
                    max_idx = graph_data.edge_index.max().item()
                    min_idx = graph_data.edge_index.min().item()

                    if max_idx >= graph_data.num_nodes:
                        logger.error(f"❌ EDGE INDEX OUT OF BOUNDS: {max_idx} >= {graph_data.num_nodes}")
                    elif min_idx < 0:
                        logger.error(f"❌ NEGATIVE EDGE INDEX: {min_idx}")
                    else:
                        logger.info(f"✅ Edge indices valid: [{min_idx}, {max_idx}]")
                else:
                    logger.info("No edges in graph")
            else:
                logger.warning("No graph data created")

        except Exception as e:
            logger.error(f"❌ Graph creation failed: {e}")
            import traceback
            traceback.print_exc()

    # 6. Identify the core issue
    logger.info("\n=== DIAGNOSIS ===")

    # Check spillover measures structure
    static_results = spillover_results.get('static', {})

    logger.info("Spillover results structure:")
    logger.info(f"- net_spillovers keys: {list(static_results.get('net_spillovers', {}).keys())}")
    logger.info(f"- pairwise_spillovers type: {type(static_results.get('pairwise_spillovers'))}")

    pairwise = static_results.get('pairwise_spillovers')
    if isinstance(pairwise, pd.DataFrame):
        logger.info(f"- pairwise DataFrame shape: {pairwise.shape}")
        logger.info(f"- pairwise DataFrame index: {pairwise.index.tolist()}")
        logger.info(f"- pairwise DataFrame columns: {pairwise.columns.tolist()}")
    elif isinstance(pairwise, dict):
        logger.info(f"- pairwise dict keys: {list(pairwise.keys())[:5]}...")

    # Check network nodes vs dataset subreddits
    network_nodes = set(network.nodes()) if network else set()
    dataset_subreddits = set(datasets.keys())

    logger.info(f"Network nodes: {network_nodes}")
    logger.info(f"Dataset subreddits: {dataset_subreddits}")
    logger.info(f"Intersection: {network_nodes & dataset_subreddits}")
    logger.info(f"Network only: {network_nodes - dataset_subreddits}")
    logger.info(f"Datasets only: {dataset_subreddits - network_nodes}")


if __name__ == "__main__":
    create_minimal_spillover_test()