#!/usr/bin/env python3
"""
Test pipeline on small dataset to verify functionality
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Set up minimal logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_small_pipeline():
    """Test pipeline with reduced dataset and fast settings"""

    print("🧪 TESTING PIPELINE ON SMALL DATASET")
    print("=" * 60)

    try:
        # Import after path setup
        from src.main_pipeline import HierarchicalSentimentPipeline

        # Create test config with small dataset
        test_config = {
            'experiment': {
                'name': 'test_small_pipeline',
                'description': 'Small test run',
                'version': '1.0'
            },
            'data': {
                'start_date': '2023-01-01',  # Just 3 months
                'end_date': '2023-03-31',
                'bigquery': {
                    'project_id': 'informationspillover',
                    'dataset_id': 'spillover_statistical_test'
                },
                'preprocessing': {
                    'min_observations_per_subreddit': 10,  # Much lower threshold
                    'max_missing_ratio': 0.5,
                    'stationarity_transformation': 'first_difference',
                    'outlier_removal': True,
                    'outlier_threshold': 3.0
                }
            },
            'feature_engineering': {
                'temporal_windows': [1, 6],  # Reduced windows
                'sentiment_analysis': {
                    'enabled': True,
                    'batch_size': 100  # Smaller batches
                },
                'network_construction': {
                    'min_connections': 2,  # Lower threshold
                    'causality_lags': 3    # Fewer lags
                }
            },
            'spillover': {
                'forecast_horizon': 5,     # Reduced horizon
                'var_lags': 2,             # Fewer lags
                'rolling_window': 30,      # Smaller window
                'dynamic_analysis': True,
                'window_size': 30
            },
            'hierarchical_model': {
                'subreddit_model_type': 'lstm',
                'hidden_dim': 32,          # Much smaller
                'num_layers': 1,           # Single layer
                'dropout': 0.2,
                'learning_rate': 0.01,     # Higher for faster convergence
                'batch_size': 16,          # Smaller batches
                'max_epochs': 5,           # Very few epochs
                'early_stopping_patience': 3,
                'gnn_hidden_dim': 16,      # Smaller GNN
                'gnn_num_layers': 1,
                'gnn_type': 'GCN',
                'attention_heads': 2
            },
            'backtesting': {
                'initial_capital': 10000,
                'transaction_cost': 0.001,
                'max_position_size': 0.1,
                'rebalancing_frequency': 'weekly',  # Less frequent
                'lookback_period': 10,              # Shorter lookback
                'min_observations': 5
            },
            'mlflow': {
                'tracking_uri': 'sqlite:///test_mlflow.db',
                'experiment_name': 'test_small_pipeline',
                'run_name': f'test_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            },
            'output_dir': 'test_results',
            'save_intermediate_results': True,
            'generate_plots': True,
            'plot_config': {
                'figure_size': [8, 6],
                'dpi': 100
            },
            'computing': {
                'n_jobs': 1,  # Single core
                'chunk_size': 100
            },
            'logging': {
                'level': 'INFO'
            },
            'random_state': 42,
            'torch_seed': 42,
            'numpy_seed': 42,
            'validation': {
                'cross_validation_folds': 2,  # Minimal CV
                'test_split_ratio': 0.2
            },
            'advanced': {
                'enable_feature_selection': False,  # Skip feature selection
                'enable_hyperparameter_tuning': False,
                'use_gpu': False
            },
            'monitoring': {
                'enable_progress_tracking': True,
                'log_frequency': 10
            }
        }

        # Save test config
        import yaml
        test_config_path = 'test_config.yaml'
        with open(test_config_path, 'w') as f:
            yaml.dump(test_config, f, indent=2)

        print(f"✅ Created test config: {test_config_path}")
        print(f"📅 Date range: {test_config['data']['start_date']} to {test_config['data']['end_date']}")
        print(f"🔬 Small model: {test_config['hierarchical_model']['hidden_dim']} hidden dims, {test_config['hierarchical_model']['max_epochs']} epochs")

        # Initialize pipeline with test config
        print("\n🚀 Initializing test pipeline...")
        pipeline = HierarchicalSentimentPipeline(test_config_path)

        # Run pipeline
        print("🏃 Running test pipeline...")
        start_time = datetime.now()

        results = pipeline.run_complete_pipeline()

        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 60)
        print("🎉 TEST PIPELINE COMPLETED!")
        print("=" * 60)
        print(f"⏱️  Total time: {duration}")
        print(f"📊 Data shape: {results['processed_data'].shape}")

        if 'spillover_results' in results:
            spillover_idx = results['spillover_results'].get('static', {}).get('total_spillover_index', 0)
            print(f"🔄 Spillover index: {spillover_idx:.2f}%")

        if 'backtest_results' in results:
            metrics = results['backtest_results'].get('performance_metrics', {})
            if metrics:
                print(f"📈 Sharpe ratio: {metrics.get('sharpe_ratio', 0):.3f}")

        print(f"📁 Results saved to: {results['output_directory']}")
        print(f"📋 Reports generated in results folder")

        # Cleanup
        if os.path.exists(test_config_path):
            os.remove(test_config_path)

        return True

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_small_pipeline()
    sys.exit(0 if success else 1)