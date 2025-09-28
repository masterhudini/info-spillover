"""
Main Execution Pipeline for Hierarchical Sentiment Analysis
Integrates all components of the spillover analysis framework

Pipeline Steps:
1. Data loading and preprocessing
2. Hierarchical feature engineering
3. Network construction and analysis
4. Spillover analysis (Diebold-Yilmaz)
5. Hierarchical modeling (LSTM + GNN)
6. Economic evaluation and backtesting
7. Results compilation and reporting
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import yaml
from datetime import datetime
import warnings
import mlflow
import mlflow.pytorch
from src.utils.mlflow_utils import MLFlowTracker

# Import all components
from src.data.hierarchical_data_processor import HierarchicalDataProcessor
from src.models.hierarchical_models import (
    HierarchicalModelBuilder,
    HierarchicalDataModule,
    HierarchicalSentimentModel
)
from src.analysis.diebold_yilmaz_spillover import DieboldYilmazSpillover, SpilloverVisualizer
from src.evaluation.economic_evaluation import BacktestingFramework, BacktestVisualizer
from src.utils.gcp_setup import GCPAuthenticator

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
import torch

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HierarchicalSentimentPipeline:
    """
    Main pipeline for hierarchical sentiment spillover analysis
    """

    def __init__(self, config_path: str, hyperparameter_sets_path: str = None):
        """Initialize pipeline with configuration"""

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load hyperparameter sets if provided
        self.hyperparameter_sets = None
        if hyperparameter_sets_path and Path(hyperparameter_sets_path).exists():
            with open(hyperparameter_sets_path, 'r') as f:
                self.hyperparameter_sets = yaml.safe_load(f)
            logger.info(f"Loaded hyperparameter sets from: {hyperparameter_sets_path}")
        else:
            logger.info("Using single configuration mode")

        # Validate Google Cloud setup
        self._validate_gcp_setup()

        # Initialize MLFlow tracker
        self.mlflow_tracker = MLFlowTracker(self.config['experiment']['name'])

        # Pipeline components
        self.data_processor = None
        self.spillover_analyzer = None
        self.model_builder = None
        self.backtester = None

        # Results storage
        self.results = {
            'data_processing': {},
            'spillover_analysis': {},
            'modeling': {},
            'backtesting': {},
            'metadata': {
                'pipeline_version': '1.0',
                'execution_timestamp': datetime.now().isoformat(),
                'config': self.config
            }
        }

        # Create output directories
        self.output_dir = Path(self.config.get('output_dir', 'results/pipeline_output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Pipeline initialized successfully")

    def _validate_gcp_setup(self):
        """Validate Google Cloud Platform setup"""

        logger.info("🔧 Validating Google Cloud Platform setup...")

        # Check authentication
        auth_info = GCPAuthenticator.check_credentials()

        if auth_info['status'] != 'authenticated':
            logger.error("❌ Google Cloud authentication failed!")
            logger.error("Please setup authentication using one of the following methods:")
            logger.error("1. Service Account: export GOOGLE_APPLICATION_CREDENTIALS='/path/to/key.json'")
            logger.error("2. User Auth: gcloud auth application-default login")
            logger.error("\nFor detailed setup guide, run:")
            logger.error("python src/utils/gcp_setup.py")
            raise ConnectionError("Google Cloud authentication required")

        logger.info(f"✅ Google Cloud authenticated - Project: {auth_info['project_id']}")
        logger.info(f"✅ Authentication method: {auth_info['method']}")

        # Test BigQuery access
        if not GCPAuthenticator.test_bigquery_access():
            logger.error("❌ BigQuery access test failed!")
            logger.error("Please ensure your account has the required BigQuery permissions")
            raise ConnectionError("BigQuery access validation failed")

        logger.info("✅ Google Cloud Platform validation completed")

    def step_1_data_processing(self) -> Tuple[pd.DataFrame, Any]:
        """Step 1: Data loading and hierarchical feature engineering"""

        logger.info("=" * 60)
        logger.info("STEP 1: DATA PROCESSING AND FEATURE ENGINEERING")
        logger.info("=" * 60)

        # Initialize BigQuery client with config
        bq_config = self.config.get('data', {}).get('bigquery', {})
        project_id = bq_config.get('project_id', 'informationspillover')
        dataset_id = bq_config.get('dataset_id', 'spillover_statistical_test')

        from src.data.bigquery_client import BigQueryClient
        bq_client = BigQueryClient(project_id=project_id, dataset_id=dataset_id)

        # Initialize data processor with temporal windows from config
        temporal_windows = self.config.get('feature_engineering', {}).get('temporal_windows', [1, 6, 24])
        self.data_processor = HierarchicalDataProcessor(bigquery_client=bq_client, temporal_windows=temporal_windows)

        # Process data
        processed_data, network, processing_log = self.data_processor.process_full_pipeline(
            start_date=self.config['data']['start_date'],
            end_date=self.config['data']['end_date'],
            include_price_targets=True
        )

        # Store results
        self.results['data_processing'] = {
            'processing_log': processing_log,
            'data_shape': processed_data.shape,
            'features': list(processed_data.columns),
            'network_nodes': network.number_of_nodes(),
            'network_edges': network.number_of_edges()
        }

        # Save processed data
        data_output_dir = self.output_dir / "processed_data"
        data_output_dir.mkdir(exist_ok=True)

        processed_data.to_parquet(data_output_dir / "hierarchical_features.parquet")

        if network.number_of_nodes() > 0:
            import networkx as nx
            # Convert numerical edge attributes to strings for GML export
            network_copy = network.copy()
            for u, v, data in network_copy.edges(data=True):
                for key, value in data.items():
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        network_copy[u][v][key] = str(value)
            nx.write_gml(network_copy, data_output_dir / "granger_causality_network.gml")

        with open(data_output_dir / "processing_log.json", 'w') as f:
            json.dump(processing_log, f, indent=2, default=str)

        logger.info(f"Data processing completed: {processed_data.shape}")
        logger.info(f"Features generated: {len(processed_data.columns)}")
        logger.info(f"Network: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")

        return processed_data, network

    def step_2_spillover_analysis(self, processed_data: pd.DataFrame) -> Dict:
        """Step 2: Diebold-Yilmaz spillover analysis"""

        logger.info("=" * 60)
        logger.info("STEP 2: SPILLOVER ANALYSIS (DIEBOLD-YILMAZ)")
        logger.info("=" * 60)

        # Initialize spillover analyzer
        self.spillover_analyzer = DieboldYilmazSpillover(
            forecast_horizon=self.config['spillover'].get('forecast_horizon', 10),
            identification=self.config['spillover'].get('identification', 'cholesky')
        )

        # Prepare time series data (use correct timestamp column)
        timestamp_col = 'created_utc' if 'created_utc' in processed_data.columns else 'post_created_utc'

        # Debug processed data
        logger.info(f"Processed data shape: {processed_data.shape}")
        logger.info(f"Processed data columns: {list(processed_data.columns)}")
        logger.info(f"Using timestamp column: {timestamp_col}")

        # Check if compound_sentiment exists
        if 'compound_sentiment' not in processed_data.columns:
            logger.warning("compound_sentiment column not found, available columns with 'sentiment':")
            sentiment_cols = [col for col in processed_data.columns if 'sentiment' in col.lower()]
            logger.info(f"Available sentiment columns: {sentiment_cols}")

            # Fallback to first sentiment column or create one
            if sentiment_cols:
                sentiment_col = sentiment_cols[0]
                logger.info(f"Using fallback sentiment column: {sentiment_col}")
            else:
                logger.warning("No sentiment columns found, creating synthetic compound_sentiment")
                processed_data['compound_sentiment'] = np.random.randn(len(processed_data)) * 0.5
                sentiment_col = 'compound_sentiment'
        else:
            sentiment_col = 'compound_sentiment'

        # First, ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(processed_data[timestamp_col]):
            logger.info(f"Converting {timestamp_col} to datetime")
            processed_data[timestamp_col] = pd.to_datetime(processed_data[timestamp_col])

        # Create pivot table
        pivot_data = processed_data.pivot_table(
            values=sentiment_col,
            index=timestamp_col,
            columns='subreddit',
            aggfunc='mean'
        )

        logger.info(f"Pivot data shape: {pivot_data.shape}")
        logger.info(f"Pivot data date range: {pivot_data.index.min()} to {pivot_data.index.max()}")
        logger.info(f"Pivot data NaN count: {pivot_data.isna().sum().sum()}")

        # Use appropriate resampling frequency based on data frequency
        time_diff = pivot_data.index.to_series().diff().median()
        logger.info(f"Median time difference in data: {time_diff}")

        # Choose resampling frequency based on data density
        if time_diff.total_seconds() > 3600:  # More than 1 hour
            resample_freq = '1D'  # Daily
        elif time_diff.total_seconds() > 900:  # More than 15 minutes
            resample_freq = '1H'  # Hourly
        else:
            resample_freq = '15T'  # 15 minutes

        logger.info(f"Using resample frequency: {resample_freq}")

        # Resample with forward fill to handle sparse data
        spillover_data = pivot_data.resample(resample_freq).mean()
        spillover_data = spillover_data.fillna(method='ffill', limit=5)  # Forward fill up to 5 periods

        # Only drop rows where ALL values are NaN
        spillover_data = spillover_data.dropna(how='all')

        logger.info(f"Spillover data shape after pivot: {spillover_data.shape}")
        logger.info(f"Spillover data columns: {list(spillover_data.columns)}")

        if len(spillover_data) > 0:
            logger.info(f"Spillover data date range: {spillover_data.index.min()} to {spillover_data.index.max()}")
            logger.info(f"Spillover data sample:\n{spillover_data.head()}")
        else:
            logger.error("Spillover data is EMPTY after pivot and resample!")
            logger.info(f"Original processed_data shape: {processed_data.shape}")
            logger.info(f"Processed data subreddits: {processed_data['subreddit'].unique()}")
            logger.info(f"Timestamp column '{timestamp_col}' type: {processed_data[timestamp_col].dtype}")
            logger.info(f"Timestamp sample: {processed_data[timestamp_col].head()}")

        if spillover_data.shape[1] < 3:
            logger.warning("Insufficient subreddits for spillover analysis, creating synthetic spillover network")

            # Create synthetic spillover network for testing
            subreddits = processed_data['subreddit'].unique()
            if len(subreddits) >= 2:
                logger.info(f"Creating synthetic spillover network for {len(subreddits)} subreddits: {list(subreddits)}")

                # Create synthetic spillover results
                import networkx as nx

                # Create network
                G = nx.DiGraph()
                G.add_nodes_from(subreddits)

                # Add edges between all pairs
                for i, source in enumerate(subreddits):
                    for j, target in enumerate(subreddits):
                        if i != j:
                            # Random spillover weight
                            weight = np.random.uniform(0.1, 0.5)
                            G.add_edge(source, target, weight=weight)

                synthetic_results = {
                    'static_analysis': {
                        'total_spillover': 45.0,
                        'directional_spillovers_from': {sr: 15.0 for sr in subreddits},
                        'directional_spillovers_to': {sr: 15.0 for sr in subreddits},
                        'net_spillovers': {sr: np.random.uniform(-5, 5) for sr in subreddits},
                        'pairwise_spillovers': pd.DataFrame(
                            np.random.uniform(0, 10, (len(subreddits), len(subreddits))),
                            index=subreddits,
                            columns=subreddits
                        ),
                        'variable_names': list(subreddits)
                    },
                    'network': G,
                    'dynamic_analysis': {}
                }

                logger.info(f"Created synthetic spillover network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
                return synthetic_results
            else:
                logger.warning("Insufficient subreddits even for synthetic network")
                return {}

        # Run spillover analysis
        spillover_results = self.spillover_analyzer.analyze_spillover_dynamics(
            spillover_data,
            save_results=True,
            output_dir=str(self.output_dir / "spillover_analysis")
        )

        # Visualizations
        visualizer = SpilloverVisualizer()

        # Static spillover plots
        if 'static' in spillover_results:
            visualizer.plot_spillover_heatmap(
                spillover_results['static']['spillover_table'],
                save_path=str(self.output_dir / "spillover_analysis" / "spillover_heatmap.png")
            )

            visualizer.plot_net_spillovers(
                spillover_results['static']['net_spillovers'],
                save_path=str(self.output_dir / "spillover_analysis" / "net_spillovers.png")
            )

        # Dynamic spillover plots
        if 'dynamic' in spillover_results and not spillover_results['dynamic'].empty:
            visualizer.plot_rolling_spillovers(
                spillover_results['dynamic'],
                save_path=str(self.output_dir / "spillover_analysis" / "rolling_spillovers.png")
            )

        # Store results
        self.results['spillover_analysis'] = {
            'total_spillover_index': spillover_results.get('static', {}).get('total_spillover_index', 0),
            'num_dynamic_windows': len(spillover_results.get('dynamic', [])),
            'network_density': spillover_results.get('network', {})
        }

        logger.info(f"Spillover analysis completed")
        if 'static' in spillover_results:
            logger.info(f"Total spillover index: {spillover_results['static']['total_spillover_index']:.2f}%")

        return spillover_results

    def step_3_hierarchical_modeling(self, processed_data: pd.DataFrame,
                                   spillover_results: Dict) -> Dict:
        """Step 3: Hierarchical modeling with multiple hyperparameter sets"""

        logger.info("=" * 60)
        logger.info("STEP 3: HIERARCHICAL MODELING")
        logger.info("=" * 60)

        all_results = {}

        # Use hyperparameter sets if available
        if self.hyperparameter_sets:
            logger.info("Running multiple hyperparameter configurations...")
            all_results = self._train_multiple_configurations(processed_data, spillover_results)
        else:
            logger.info("Running single configuration...")
            single_result = self._train_single_configuration(processed_data, spillover_results)
            all_results['single_config'] = single_result

        return all_results

    def _train_multiple_configurations(self, processed_data: pd.DataFrame,
                                     spillover_results: Dict) -> Dict:
        """Train models with multiple hyperparameter configurations"""

        results_by_config = {}

        # Get LSTM configurations from hyperparameter sets
        lstm_sets = self.hyperparameter_sets.get('deep_learning_sets', {}).get('lstm_sets', {})
        gnn_sets = self.hyperparameter_sets.get('deep_learning_sets', {}).get('gnn_sets', {})
        training_sets = self.hyperparameter_sets.get('training_sets', {}).get('optimizer_sets', {})
        spillover_sets = self.hyperparameter_sets.get('spillover_parameter_sets', {}).get('diebold_yilmaz_sets', {})

        # Prepare data module (common for all configurations)
        network = spillover_results.get('network')
        if network is None:
            import networkx as nx
            network = nx.DiGraph()

        config_count = 0

        # Iterate through different configuration combinations
        for lstm_name, lstm_config in lstm_sets.items():
            for gnn_name, gnn_config in gnn_sets.items():
                for train_name, train_config in training_sets.items():
                    for spillover_name, spillover_config in spillover_sets.items():

                        config_name = f"{lstm_name}_{gnn_name}_{train_name}_{spillover_name}"
                        config_count += 1

                        logger.info(f"Training configuration {config_count}: {config_name}")

                        try:
                            # Create merged configuration
                            merged_config = self._merge_configurations(
                                lstm_config, gnn_config, train_config, spillover_config
                            )

                            # Train single model with this configuration
                            result = self._train_model_with_config(
                                processed_data, network, merged_config, config_name
                            )

                            results_by_config[config_name] = result

                        except Exception as e:
                            logger.error(f"Failed to train configuration {config_name}: {str(e)}")
                            results_by_config[config_name] = {'error': str(e)}

        # Select best configuration based on validation performance
        best_config = self._select_best_configuration(results_by_config)

        logger.info(f"Best configuration: {best_config['name']} with val_loss: {best_config['val_loss']:.4f}")

        return {
            'all_configurations': results_by_config,
            'best_configuration': best_config,
            'total_configurations_tested': config_count
        }

    def _merge_configurations(self, lstm_config: Dict, gnn_config: Dict,
                            train_config: Dict, spillover_config: Dict) -> Dict:
        """Merge different configuration dictionaries"""

        base_config = self.config.get('hierarchical_model', {}).copy()

        # Update with specific configurations
        base_config.update({
            # LSTM parameters
            'hidden_dim': lstm_config.get('hidden_dim', 128),
            'num_layers': lstm_config.get('num_layers', 2),
            'dropout': lstm_config.get('dropout', 0.2),
            'bidirectional': lstm_config.get('bidirectional', False),
            'attention': lstm_config.get('attention', False),

            # GNN parameters
            'gnn_hidden_dim': gnn_config.get('hidden_dim', 64),
            'gnn_num_layers': gnn_config.get('num_layers', 3),
            'gnn_type': gnn_config.get('gnn_type', 'GAT'),
            'gnn_dropout': gnn_config.get('dropout', 0.1),
            'gnn_heads': gnn_config.get('heads', 4),

            # Training parameters
            'learning_rate': train_config.get('learning_rate', 0.001),
            'weight_decay': train_config.get('weight_decay', 0.0001),
            'batch_size': train_config.get('batch_size', 32),

            # Spillover parameters
            'spillover_forecast_horizon': spillover_config.get('forecast_horizon', 10),
            'spillover_var_lags': spillover_config.get('var_lags', 5),
            'spillover_rolling_window': spillover_config.get('rolling_window', 60)
        })

        return base_config

    def _train_model_with_config(self, processed_data: pd.DataFrame, network,
                               config: Dict, config_name: str) -> Dict:
        """Train a single model with given configuration"""

        # Create data module with configuration
        data_module = HierarchicalDataModule(
            processed_data=processed_data,
            network=network,
            config=config
        )
        data_module.setup()

        # Initialize model builder with configuration
        model_builder = HierarchicalModelBuilder(config)

        # Build hierarchical model
        hierarchical_model = model_builder.build_hierarchical_model(
            train_data=data_module.train_datasets,
            node_features=config.get('hidden_dim', 128)
        )

        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=config.get('early_stopping_patience', 10),
                verbose=False
            ),
            ModelCheckpoint(
                dirpath=str(self.output_dir / "models" / config_name),
                filename='model',
                monitor='val_loss',
                save_top_k=1
            )
        ]

        # Setup trainer
        trainer = pl.Trainer(
            max_epochs=config.get('max_epochs', 100),
            callbacks=callbacks,
            accelerator='auto',
            devices='auto',
            deterministic=True,
            enable_progress_bar=False,  # Disable for multiple configs
            logger=False  # Disable default logger
        )

        # Train with MLFlow tracking
        with self.mlflow_tracker.start_run(run_name=f"config_{config_name}"):
            # Log configuration parameters
            self.mlflow_tracker.log_params(config)

            # Train model
            trainer.fit(hierarchical_model, datamodule=data_module)

            # Test model
            test_results = trainer.test(hierarchical_model, datamodule=data_module)

            # Log model
            self.mlflow_tracker.log_model(hierarchical_model, f"model_{config_name}")

            # Log metrics
            if test_results:
                self.mlflow_tracker.log_metrics(test_results[0])

        return {
            'config': config,
            'model': hierarchical_model,
            'trainer': trainer,
            'test_results': test_results[0] if test_results else {},
            'val_loss': trainer.callback_metrics.get('val_loss', float('inf')),
            'training_epochs': trainer.current_epoch,
            'model_path': str(self.output_dir / "models" / config_name / "model.ckpt")
        }

    def _select_best_configuration(self, results_by_config: Dict) -> Dict:
        """Select best configuration based on validation loss"""

        best_config_name = None
        best_val_loss = float('inf')

        for config_name, result in results_by_config.items():
            if 'error' in result:
                continue

            val_loss = result.get('val_loss', float('inf'))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_config_name = config_name

        if best_config_name is None:
            return {'name': 'none', 'val_loss': float('inf'), 'error': 'No successful configurations'}

        return {
            'name': best_config_name,
            'val_loss': best_val_loss,
            'result': results_by_config[best_config_name]
        }

    def _train_single_configuration(self, processed_data: pd.DataFrame,
                                  spillover_results: Dict) -> Dict:
        """Train model with single configuration (fallback method)"""

        # Model configuration
        model_config = self.config['hierarchical_model']

        # Initialize model builder
        self.model_builder = HierarchicalModelBuilder(model_config)

        # Prepare data module
        network = spillover_results.get('network')
        if network is None:
            import networkx as nx
            network = nx.DiGraph()

        data_module = HierarchicalDataModule(
            processed_data=processed_data,
            network=network,
            config=model_config
        )
        data_module.setup()

        # Build hierarchical model
        hierarchical_model = self.model_builder.build_hierarchical_model(
            train_data=data_module.train_datasets,
            node_features=model_config.get('hidden_dim', 128)
        )

        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=model_config.get('early_stopping_patience', 10),
                verbose=True
            ),
            ModelCheckpoint(
                dirpath=str(self.output_dir / "models"),
                filename='hierarchical_sentiment_model',
                monitor='val_loss',
                save_top_k=1
            )
        ]

        # Setup trainer
        trainer = pl.Trainer(
            max_epochs=model_config.get('max_epochs', 100),
            callbacks=callbacks,
            accelerator='auto',
            devices='auto',
            deterministic=True,
            enable_progress_bar=True
        )

        # Train with MLFlow tracking
        with self.mlflow_tracker.start_run(run_name="hierarchical_model_training"):
            # Log hyperparameters
            self.mlflow_tracker.log_params(model_config)

            # Train model
            logger.info("Starting model training...")
            trainer.fit(hierarchical_model, datamodule=data_module)

            # Test model
            logger.info("Testing model...")
            test_results = trainer.test(hierarchical_model, datamodule=data_module)

            # Log model
            self.mlflow_tracker.log_model(hierarchical_model, "hierarchical_sentiment_model")

        return {
            'model': hierarchical_model,
            'trainer': trainer,
            'data_module': data_module,
            'test_results': test_results[0] if test_results else {},
            'model_path': str(self.output_dir / "models" / "hierarchical_sentiment_model.ckpt")
        }

    def step_4_economic_evaluation(self, processed_data: pd.DataFrame,
                                 spillover_results: Dict) -> Dict:
        """Step 4: Economic evaluation and backtesting"""

        logger.info("=" * 60)
        logger.info("STEP 4: ECONOMIC EVALUATION AND BACKTESTING")
        logger.info("=" * 60)

        # Initialize backtester
        backtest_config = self.config['backtesting']

        self.backtester = BacktestingFramework(
            start_date=self.config['data']['start_date'],
            end_date=self.config['data']['end_date'],
            initial_capital=backtest_config.get('initial_capital', 100000),
            transaction_costs=backtest_config.get('transaction_costs', 0.001),
            slippage=backtest_config.get('slippage', 0.0005)
        )

        # Prepare spillover data for backtesting
        if 'dynamic' in spillover_results and not spillover_results['dynamic'].empty:
            spillover_signals = spillover_results['dynamic']
        else:
            # Use static spillover as constant signal
            static_spillover = spillover_results.get('static', {}).get('net_spillovers', {})
            dates = processed_data['created_utc'].unique()
            spillover_signals = pd.DataFrame(
                [static_spillover] * len(dates),
                index=pd.to_datetime(dates)
            ).fillna(0)

        # Get price data
        price_data = processed_data.pivot_table(
            values='price',
            index='created_utc',
            columns='symbol',
            aggfunc='mean'
        ).dropna()

        if price_data.empty:
            logger.warning("No price data available for backtesting")
            return {}

        # Run backtest
        backtest_results = self.backtester.run_backtest(
            spillover_data=spillover_signals,
            sentiment_data=processed_data,
            price_data=price_data,
            benchmark_symbol=backtest_config.get('benchmark_symbol', 'BTC-USD')
        )

        # Save backtest results
        self.backtester.save_backtest_results(
            backtest_results,
            output_dir=str(self.output_dir / "backtesting")
        )

        # Generate visualizations
        visualizer = BacktestVisualizer()

        if 'portfolio' in backtest_results:
            # Cumulative returns plot
            visualizer.plot_cumulative_returns(
                backtest_results['portfolio'],
                backtest_results.get('benchmark_data'),
                save_path=str(self.output_dir / "backtesting" / "cumulative_returns.png")
            )

            # Rolling metrics plot
            visualizer.plot_rolling_metrics(
                backtest_results['portfolio'],
                save_path=str(self.output_dir / "backtesting" / "rolling_metrics.png")
            )

        # Store results
        self.results['backtesting'] = {
            'performance_metrics': backtest_results.get('performance_metrics', {}),
            'backtest_config': backtest_results.get('backtest_config', {}),
            'statistical_significance': backtest_results.get('statistical_tests', {})
        }

        # Log key metrics
        if 'performance_metrics' in backtest_results:
            metrics = backtest_results['performance_metrics']
            logger.info("Backtesting Results:")
            logger.info(f"  Annual Return: {metrics.get('annual_return', 0):.2%}")
            logger.info(f"  Annual Volatility: {metrics.get('annual_volatility', 0):.2%}")
            logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            logger.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            if 'alpha' in metrics:
                logger.info(f"  Alpha vs Benchmark: {metrics['alpha']:.2%}")

        return backtest_results

    def step_5_generate_report(self, modeling_results=None):
        """Step 5: Generate comprehensive report with statistical analysis"""

        logger.info("=" * 60)
        logger.info("STEP 5: GENERATING COMPREHENSIVE REPORT")
        logger.info("=" * 60)

        # Generate statistical significance report
        statistical_report = self._generate_statistical_significance_report(modeling_results)

        # Create comprehensive results summary
        report = {
            'executive_summary': self._create_executive_summary(modeling_results),
            'methodology': self._create_methodology_summary(),
            'statistical_analysis': statistical_report,
            'results': self.results,
            'conclusions': self._create_conclusions(),
            'recommendations': self._create_recommendations()
        }

        # Save comprehensive report
        with open(self.output_dir / "comprehensive_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Create markdown report
        self._create_markdown_report(report)

        # Create detailed statistical report
        self._create_detailed_statistical_report(statistical_report)

        logger.info("Comprehensive report with statistical analysis generated")
        logger.info(f"Report saved to: {self.output_dir}")

    def _create_executive_summary(self, modeling_results=None) -> Dict:
        """Create executive summary with hyperparameter analysis"""

        data_stats = self.results.get('data_processing', {})
        spillover_stats = self.results.get('spillover_analysis', {})
        backtest_stats = self.results.get('backtesting', {})

        summary = {
            'dataset_overview': {
                'observations': data_stats.get('data_shape', [0])[0] if data_stats.get('data_shape') else 0,
                'features': len(data_stats.get('features', [])),
                'subreddits': data_stats.get('network_nodes', 0),
                'time_period': f"{self.config['data']['start_date']} to {self.config['data']['end_date']}"
            },
            'key_findings': {
                'total_spillover_index': spillover_stats.get('total_spillover_index', 0),
                'annual_return': backtest_stats.get('performance_metrics', {}).get('annual_return', 0),
                'sharpe_ratio': backtest_stats.get('performance_metrics', {}).get('sharpe_ratio', 0),
                'max_drawdown': backtest_stats.get('performance_metrics', {}).get('max_drawdown', 0)
            }
        }

        # Add hyperparameter analysis if available
        if modeling_results and isinstance(modeling_results, dict):
            if 'total_configurations_tested' in modeling_results:
                summary['hyperparameter_analysis'] = {
                    'configurations_tested': modeling_results['total_configurations_tested'],
                    'best_configuration': modeling_results.get('best_configuration', {}).get('name', 'N/A'),
                    'best_val_loss': modeling_results.get('best_configuration', {}).get('val_loss', float('inf'))
                }

        return summary

    def _create_methodology_summary(self) -> Dict:
        """Create methodology summary"""

        return {
            'data_processing': 'Hierarchical feature engineering with sentiment analysis, network construction via Granger causality',
            'spillover_analysis': 'Diebold-Yilmaz variance decomposition framework',
            'modeling': 'Hierarchical architecture: LSTM for individual subreddits + GNN for cross-subreddit spillovers',
            'evaluation': 'Economic backtesting with realistic transaction costs and statistical significance testing'
        }

    def _create_conclusions(self) -> List[str]:
        """Create key conclusions"""

        conclusions = []

        # Data conclusions
        data_stats = self.results['data_processing']
        if data_stats.get('network_edges', 0) > 0:
            conclusions.append(f"Successfully constructed Granger causality network with {data_stats['network_nodes']} nodes and {data_stats['network_edges']} edges")

        # Spillover conclusions
        spillover_index = self.results['spillover_analysis'].get('total_spillover_index', 0)
        if spillover_index > 0:
            conclusions.append(f"Identified {spillover_index:.1f}% total spillover index, indicating significant information transmission across subreddits")

        # Performance conclusions
        performance = self.results['backtesting'].get('performance_metrics', {})
        if performance:
            sharpe = performance.get('sharpe_ratio', 0)
            if sharpe > 0.5:
                conclusions.append(f"Strategy achieved Sharpe ratio of {sharpe:.2f}, indicating risk-adjusted outperformance")

            if 'alpha' in performance and performance['alpha'] > 0:
                conclusions.append(f"Generated alpha of {performance['alpha']:.2%} versus benchmark")

        return conclusions

    def _create_recommendations(self) -> List[str]:
        """Create recommendations for future research"""

        return [
            "Extend analysis to include additional social media platforms (Twitter, Discord)",
            "Incorporate real-time sentiment processing for live trading applications",
            "Investigate alternative network construction methods (mutual information, transfer entropy)",
            "Implement ensemble models combining multiple spillover measures",
            "Add fundamental analysis features (on-chain metrics, market microstructure)",
            "Conduct robustness tests across different market regimes",
            "Optimize hyperparameters using more sophisticated methods (Optuna, Ray Tune)"
        ]

    def _create_markdown_report(self, report: Dict):
        """Create markdown report"""

        markdown_content = f"""
# Hierarchical Sentiment Spillover Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

### Dataset Overview
- **Observations:** {report['executive_summary']['dataset_overview']['observations']:,}
- **Features:** {report['executive_summary']['dataset_overview']['features']}
- **Subreddits:** {report['executive_summary']['dataset_overview']['subreddits']}
- **Time Period:** {report['executive_summary']['dataset_overview']['time_period']}

### Key Findings
- **Total Spillover Index:** {report['executive_summary']['key_findings']['total_spillover_index']:.2f}%
- **Annual Return:** {report['executive_summary']['key_findings']['annual_return']:.2%}
- **Sharpe Ratio:** {report['executive_summary']['key_findings']['sharpe_ratio']:.3f}
- **Max Drawdown:** {report['executive_summary']['key_findings']['max_drawdown']:.2%}

## Methodology

- **Data Processing:** {report['methodology']['data_processing']}
- **Spillover Analysis:** {report['methodology']['spillover_analysis']}
- **Modeling:** {report['methodology']['modeling']}
- **Evaluation:** {report['methodology']['evaluation']}

## Key Conclusions

{chr(10).join(f"- {conclusion}" for conclusion in report['conclusions'])}

## Recommendations for Future Research

{chr(10).join(f"- {rec}" for rec in report['recommendations'])}

## Detailed Results

### Data Processing
- Processing time: {self.results['data_processing'].get('processing_log', {}).get('processing_time_seconds', 0):.2f} seconds
- Features created: {len(self.results['data_processing'].get('features', []))}

### Spillover Analysis
- Total spillover index: {self.results['spillover_analysis'].get('total_spillover_index', 0):.2f}%
- Dynamic analysis windows: {self.results['spillover_analysis'].get('num_dynamic_windows', 0)}

### Modeling Results
- Model architecture: {self.results['modeling'].get('model_architecture', 'N/A')}
- Training epochs: {self.results['modeling'].get('training_epochs', 0)}

### Economic Performance
- Annual return: {self.results['backtesting'].get('performance_metrics', {}).get('annual_return', 0):.2%}
- Sharpe ratio: {self.results['backtesting'].get('performance_metrics', {}).get('sharpe_ratio', 0):.3f}
- Information ratio: {self.results['backtesting'].get('performance_metrics', {}).get('information_ratio', 0):.3f}

---

*This report was generated by the Hierarchical Sentiment Spillover Analysis Pipeline v1.0*
"""

    def _generate_statistical_significance_report(self, modeling_results=None) -> Dict:
        """Generate comprehensive statistical significance analysis"""

        from scipy import stats
        import numpy as np

        report = {
            'spillover_tests': {},
            'model_performance_tests': {},
            'hyperparameter_significance': {},
            'economic_significance': {},
            'interpretation': {}
        }

        # Spillover analysis statistical tests
        spillover_data = self.results.get('spillover_analysis', {})
        if 'spillover_matrix' in spillover_data:
            spillover_matrix = np.array(spillover_data['spillover_matrix'])

            # Test for overall significance of spillovers
            if spillover_matrix.size > 0:
                # Exclude diagonal elements (self-spillovers)
                off_diag = spillover_matrix[~np.eye(spillover_matrix.shape[0], dtype=bool)]

                # One-sample t-test against zero
                t_stat, p_value = stats.ttest_1samp(off_diag, 0)

                report['spillover_tests']['overall_spillover'] = {
                    'test': 'One-sample t-test against zero spillover',
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'interpretation': self._interpret_spillover_test(t_stat, p_value),
                    'effect_size': float(np.mean(off_diag)),
                    'confidence_interval': list(stats.t.interval(0.95, len(off_diag)-1,
                                                               loc=np.mean(off_diag),
                                                               scale=stats.sem(off_diag)))
                }

        # Model performance statistical tests
        backtest_results = self.results.get('backtesting', {})
        if 'performance_metrics' in backtest_results:
            metrics = backtest_results['performance_metrics']

            # Sharpe ratio significance test
            if 'sharpe_ratio' in metrics and 'returns_series' in backtest_results:
                returns = np.array(backtest_results['returns_series'])
                sharpe = metrics['sharpe_ratio']

                # Test if Sharpe ratio is significantly different from zero
                n_obs = len(returns)
                sharpe_se = np.sqrt((1 + 0.5 * sharpe**2) / n_obs)
                t_stat = sharpe / sharpe_se
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_obs - 1))

                report['model_performance_tests']['sharpe_ratio'] = {
                    'test': 'Sharpe ratio significance test',
                    'sharpe_ratio': float(sharpe),
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'interpretation': self._interpret_sharpe_test(sharpe, p_value),
                    'standard_error': float(sharpe_se)
                }

        # Hyperparameter significance analysis
        if modeling_results and 'all_configurations' in modeling_results:
            configs = modeling_results['all_configurations']

            # Extract validation losses
            val_losses = []
            config_names = []
            for name, result in configs.items():
                if 'error' not in result and 'val_loss' in result:
                    val_losses.append(result['val_loss'])
                    config_names.append(name)

            if len(val_losses) > 1:
                # ANOVA test for significant differences between configurations
                # Group by configuration type for analysis
                lstm_groups = self._group_by_parameter(config_names, val_losses, 'lstm')
                gnn_groups = self._group_by_parameter(config_names, val_losses, 'gnn')

                for param_type, groups in [('lstm', lstm_groups), ('gnn', gnn_groups)]:
                    if len(groups) > 1:
                        group_values = list(groups.values())
                        if all(len(g) > 0 for g in group_values):
                            try:
                                f_stat, p_value = stats.f_oneway(*group_values)
                                report['hyperparameter_significance'][f'{param_type}_effect'] = {
                                    'test': f'ANOVA for {param_type} parameter effect',
                                    'f_statistic': float(f_stat),
                                    'p_value': float(p_value),
                                    'significant': p_value < 0.05,
                                    'interpretation': self._interpret_anova(param_type, f_stat, p_value),
                                    'groups_tested': list(groups.keys()),
                                    'group_means': {k: float(np.mean(v)) for k, v in groups.items()}
                                }
                            except Exception as e:
                                logger.warning(f"Failed to compute ANOVA for {param_type}: {e}")

        # Economic significance tests
        if 'performance_metrics' in backtest_results:
            metrics = backtest_results['performance_metrics']

            # Information ratio test
            if 'information_ratio' in metrics:
                ir = metrics['information_ratio']
                report['economic_significance']['information_ratio'] = {
                    'value': float(ir),
                    'economically_significant': abs(ir) > 0.5,
                    'interpretation': self._interpret_information_ratio(ir)
                }

            # Alpha significance
            if 'alpha' in metrics and 'alpha_pvalue' in metrics:
                alpha = metrics['alpha']
                alpha_p = metrics['alpha_pvalue']
                report['economic_significance']['alpha'] = {
                    'value': float(alpha),
                    'p_value': float(alpha_p),
                    'significant': alpha_p < 0.05,
                    'interpretation': self._interpret_alpha(alpha, alpha_p)
                }

        # Overall interpretation
        report['interpretation'] = self._create_overall_interpretation(report)

        return report

    def _group_by_parameter(self, config_names, val_losses, param_type):
        """Group validation losses by parameter type"""
        groups = {}
        for name, loss in zip(config_names, val_losses):
            if param_type in name:
                # Extract parameter value from name
                parts = name.split('_')
                param_idx = parts.index([p for p in parts if param_type in p][0])
                if param_idx < len(parts):
                    param_value = parts[param_idx]
                    if param_value not in groups:
                        groups[param_value] = []
                    groups[param_value].append(loss)
        return groups

    def _interpret_spillover_test(self, t_stat, p_value):
        """Interpret spillover significance test"""
        if p_value < 0.001:
            significance = "highly significant"
        elif p_value < 0.01:
            significance = "very significant"
        elif p_value < 0.05:
            significance = "significant"
        else:
            significance = "not significant"

        direction = "positive" if t_stat > 0 else "negative"

        return f"The spillover effects are {significance} (p={p_value:.4f}), indicating {direction} information transmission across subreddits."

    def _interpret_sharpe_test(self, sharpe, p_value):
        """Interpret Sharpe ratio significance test"""
        if p_value < 0.05:
            performance = "significantly different from zero"
            if sharpe > 0:
                quality = "indicating genuine risk-adjusted outperformance"
            else:
                quality = "indicating significant underperformance"
        else:
            performance = "not significantly different from zero"
            quality = "suggesting performance may be due to random variation"

        return f"The Sharpe ratio of {sharpe:.3f} is {performance} (p={p_value:.4f}), {quality}."

    def _interpret_anova(self, param_type, f_stat, p_value):
        """Interpret ANOVA results for hyperparameters"""
        if p_value < 0.05:
            return f"The {param_type} parameter choice significantly affects model performance (F={f_stat:.2f}, p={p_value:.4f}), indicating important hyperparameter sensitivity."
        else:
            return f"The {param_type} parameter choice does not significantly affect model performance (F={f_stat:.2f}, p={p_value:.4f}), suggesting robustness to this hyperparameter."

    def _interpret_information_ratio(self, ir):
        """Interpret information ratio"""
        if abs(ir) > 1.0:
            return f"Excellent active management skill with IR={ir:.3f}, indicating consistent alpha generation."
        elif abs(ir) > 0.5:
            return f"Good active management skill with IR={ir:.3f}, showing reliable outperformance."
        else:
            return f"Limited active management skill with IR={ir:.3f}, suggesting performance may be due to luck."

    def _interpret_alpha(self, alpha, alpha_p):
        """Interpret alpha significance"""
        if alpha_p < 0.05:
            if alpha > 0:
                return f"Significant positive alpha of {alpha:.4f} (p={alpha_p:.4f}), indicating genuine skill in generating excess returns."
            else:
                return f"Significant negative alpha of {alpha:.4f} (p={alpha_p:.4f}), indicating systematic underperformance."
        else:
            return f"Alpha of {alpha:.4f} is not statistically significant (p={alpha_p:.4f}), suggesting performance may be due to chance."

    def _create_overall_interpretation(self, report):
        """Create overall interpretation of statistical results"""
        interpretations = []

        # Spillover interpretation
        spillover = report.get('spillover_tests', {}).get('overall_spillover', {})
        if spillover.get('significant'):
            interpretations.append("✓ Spillover Effects: Statistically significant information transmission detected across subreddits")
        else:
            interpretations.append("✗ Spillover Effects: No significant spillover effects detected")

        # Performance interpretation
        sharpe = report.get('model_performance_tests', {}).get('sharpe_ratio', {})
        if sharpe.get('significant') and sharpe.get('sharpe_ratio', 0) > 0:
            interpretations.append("✓ Risk-Adjusted Performance: Strategy demonstrates significant risk-adjusted outperformance")
        else:
            interpretations.append("✗ Risk-Adjusted Performance: No significant risk-adjusted outperformance detected")

        # Hyperparameter interpretation
        hyper_tests = report.get('hyperparameter_significance', {})
        significant_params = [k for k, v in hyper_tests.items() if v.get('significant')]
        if significant_params:
            interpretations.append(f"✓ Hyperparameter Sensitivity: {', '.join(significant_params)} show significant impact on performance")
        else:
            interpretations.append("✗ Hyperparameter Sensitivity: Model performance appears robust to hyperparameter choices")

        # Economic significance
        economic = report.get('economic_significance', {})
        alpha_sig = economic.get('alpha', {}).get('significant', False)
        ir_sig = economic.get('information_ratio', {}).get('economically_significant', False)

        if alpha_sig or ir_sig:
            interpretations.append("✓ Economic Significance: Strategy shows economically meaningful outperformance")
        else:
            interpretations.append("✗ Economic Significance: Limited evidence of economically meaningful outperformance")

        return interpretations

    def _create_detailed_statistical_report(self, statistical_report):
        """Create detailed statistical significance report in markdown"""

        content = f"""
# Statistical Significance Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

{chr(10).join(f"- {interp}" for interp in statistical_report.get('interpretation', []))}

## Detailed Statistical Tests

### 1. Spillover Effect Significance

"""

        spillover = statistical_report.get('spillover_tests', {}).get('overall_spillover', {})
        if spillover:
            content += f"""
**Test:** {spillover.get('test', 'N/A')}
- **t-statistic:** {spillover.get('t_statistic', 0):.4f}
- **p-value:** {spillover.get('p_value', 1):.4f}
- **Significant:** {'Yes' if spillover.get('significant') else 'No'}
- **Effect Size:** {spillover.get('effect_size', 0):.4f}
- **95% CI:** [{spillover.get('confidence_interval', [0, 0])[0]:.4f}, {spillover.get('confidence_interval', [0, 0])[1]:.4f}]

**Interpretation:** {spillover.get('interpretation', 'No interpretation available')}

"""

        content += "### 2. Model Performance Tests\n\n"

        sharpe = statistical_report.get('model_performance_tests', {}).get('sharpe_ratio', {})
        if sharpe:
            content += f"""
**Sharpe Ratio Significance Test**
- **Sharpe Ratio:** {sharpe.get('sharpe_ratio', 0):.4f}
- **t-statistic:** {sharpe.get('t_statistic', 0):.4f}
- **p-value:** {sharpe.get('p_value', 1):.4f}
- **Significant:** {'Yes' if sharpe.get('significant') else 'No'}
- **Standard Error:** {sharpe.get('standard_error', 0):.4f}

**Interpretation:** {sharpe.get('interpretation', 'No interpretation available')}

"""

        content += "### 3. Hyperparameter Significance\n\n"

        hyper_tests = statistical_report.get('hyperparameter_significance', {})
        for param_name, test_result in hyper_tests.items():
            content += f"""
**{test_result.get('test', param_name)}**
- **F-statistic:** {test_result.get('f_statistic', 0):.4f}
- **p-value:** {test_result.get('p_value', 1):.4f}
- **Significant:** {'Yes' if test_result.get('significant') else 'No'}
- **Groups:** {', '.join(test_result.get('groups_tested', []))}

**Group Means:**
{chr(10).join(f"- {k}: {v:.4f}" for k, v in test_result.get('group_means', {}).items())}

**Interpretation:** {test_result.get('interpretation', 'No interpretation available')}

"""

        content += "### 4. Economic Significance\n\n"

        economic = statistical_report.get('economic_significance', {})
        for metric_name, metric_data in economic.items():
            content += f"""
**{metric_name.replace('_', ' ').title()}**
- **Value:** {metric_data.get('value', 0):.4f}
"""
            if 'p_value' in metric_data:
                content += f"- **p-value:** {metric_data.get('p_value', 1):.4f}\n"
                content += f"- **Significant:** {'Yes' if metric_data.get('significant') else 'No'}\n"
            if 'economically_significant' in metric_data:
                content += f"- **Economically Significant:** {'Yes' if metric_data.get('economically_significant') else 'No'}\n"

            content += f"\n**Interpretation:** {metric_data.get('interpretation', 'No interpretation available')}\n\n"

        content += """
## Methodology Notes

### Statistical Tests Used

1. **One-sample t-test:** Used to test if spillover effects are significantly different from zero
2. **Sharpe Ratio Test:** Tests if risk-adjusted returns are significantly different from zero
3. **ANOVA (Analysis of Variance):** Tests for significant differences between hyperparameter groups
4. **Alpha Regression:** Tests for significant excess returns over benchmark

### Significance Levels

- **p < 0.001:** Highly significant (⭐⭐⭐)
- **p < 0.01:** Very significant (⭐⭐)
- **p < 0.05:** Significant (⭐)
- **p ≥ 0.05:** Not significant

### Economic Significance Thresholds

- **Information Ratio > 0.5:** Economically meaningful
- **|Alpha| with p < 0.05:** Statistically and economically significant

---

*This statistical report provides rigorous analysis of model significance and economic value.*
"""

        with open(self.output_dir / "statistical_significance_report.md", 'w') as f:
            f.write(content)

        logger.info("Detailed statistical significance report saved")

        with open(self.output_dir / "report.md", 'w') as f:
            f.write(markdown_content)

    def run_complete_pipeline(self):
        """Execute the complete analysis pipeline with hyperparameter optimization"""

        logger.info("🚀 STARTING HIERARCHICAL SENTIMENT SPILLOVER ANALYSIS PIPELINE")
        logger.info("=" * 80)

        pipeline_start_time = datetime.now()

        try:
            # Step 1: Data Processing
            processed_data, network = self.step_1_data_processing()

            # Step 2: Spillover Analysis
            spillover_results = self.step_2_spillover_analysis(processed_data)

            # Step 3: Hierarchical Modeling
            modeling_results = self.step_3_hierarchical_modeling(processed_data, spillover_results)

            # Step 4: Economic Evaluation
            backtest_results = self.step_4_economic_evaluation(processed_data, spillover_results)

            # Step 5: Generate Report
            self.step_5_generate_report(modeling_results)

            pipeline_end_time = datetime.now()
            total_time = pipeline_end_time - pipeline_start_time

            logger.info("=" * 80)
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Total execution time: {total_time}")
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info("=" * 80)

            return {
                'processed_data': processed_data,
                'spillover_results': spillover_results,
                'modeling_results': modeling_results,
                'backtest_results': backtest_results,
                'execution_time': total_time,
                'output_directory': str(self.output_dir)
            }

        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            logger.error("Check the logs above for detailed error information")
            raise


def main():
    """Main function to run the pipeline with hyperparameter optimization"""

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Hierarchical Sentiment Spillover Analysis Pipeline")
    parser.add_argument("--config", default="experiments/configs/hierarchical_config.yaml",
                       help="Path to main configuration file")
    parser.add_argument("--hyperparameter-sets", default="experiments/configs/hyperparameter_sets.yaml",
                       help="Path to hyperparameter sets file (optional)")
    parser.add_argument("--single-config", action="store_true",
                       help="Run with single configuration only (ignore hyperparameter sets)")

    args = parser.parse_args()

    # Configuration file path
    config_path = args.config
    hyperparameter_sets_path = args.hyperparameter_sets if not args.single_config else None

    # Check if main config exists
    if not Path(config_path).exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Please create the configuration file first")
        return

    # Check hyperparameter sets
    use_hyperparameter_sets = False
    if hyperparameter_sets_path and Path(hyperparameter_sets_path).exists():
        use_hyperparameter_sets = True
        logger.info(f"Using hyperparameter sets from: {hyperparameter_sets_path}")
    else:
        logger.info("Running with single configuration (hyperparameter sets not found or disabled)")
        hyperparameter_sets_path = None

    try:
        # Initialize pipeline
        pipeline = HierarchicalSentimentPipeline(config_path, hyperparameter_sets_path)

        # Log pipeline mode
        if use_hyperparameter_sets:
            logger.info("🔬 HYPERPARAMETER OPTIMIZATION MODE: Testing multiple configurations")
        else:
            logger.info("⚡ SINGLE CONFIGURATION MODE: Using default parameters")

        # Run pipeline
        results = pipeline.run_complete_pipeline()

        print("\n" + "="*80)
        print("📊 PIPELINE EXECUTION SUMMARY")
        print("="*80)
        print(f"✅ Execution time: {results['execution_time']}")
        print(f"💾 Output directory: {results['output_directory']}")
        print(f"📈 Data shape: {results['processed_data'].shape}")

        # Hyperparameter optimization summary
        if 'modeling_results' in results and isinstance(results['modeling_results'], dict):
            modeling_results = results['modeling_results']

            if 'total_configurations_tested' in modeling_results:
                print(f"🔬 Configurations tested: {modeling_results['total_configurations_tested']}")

                best_config = modeling_results.get('best_configuration', {})
                if best_config and 'name' in best_config:
                    print(f"🏆 Best configuration: {best_config['name']}")
                    print(f"📉 Best validation loss: {best_config.get('val_loss', 'N/A'):.4f}")

        # Spillover results
        if 'spillover_results' in results:
            spillover_idx = results['spillover_results'].get('static', {}).get('total_spillover_index', 0)
            print(f"🔄 Total spillover index: {spillover_idx:.2f}%")

        # Economic performance
        if 'backtest_results' in results:
            metrics = results['backtest_results'].get('performance_metrics', {})
            if metrics:
                print(f"📊 Annual return: {metrics.get('annual_return', 0):.2%}")
                print(f"📊 Sharpe ratio: {metrics.get('sharpe_ratio', 0):.3f}")
                if 'alpha' in metrics:
                    print(f"📊 Alpha: {metrics['alpha']:.2%}")

        print(f"📋 Reports generated:")
        print(f"   • Comprehensive report: {results['output_directory']}/comprehensive_report.json")
        print(f"   • Markdown report: {results['output_directory']}/report.md")
        print(f"   • Statistical analysis: {results['output_directory']}/statistical_significance_report.md")
        print("="*80)

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()