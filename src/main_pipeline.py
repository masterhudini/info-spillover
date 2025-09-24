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

        # Initialize data processor
        self.data_processor = HierarchicalDataProcessor()

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
            nx.write_gml(network, data_output_dir / "granger_causality_network.gml")

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

        # Prepare time series data
        spillover_data = processed_data.pivot_table(
            values='compound_sentiment',
            index='created_utc',
            columns='subreddit',
            aggfunc='mean'
        ).resample('1H').mean().dropna()

        if spillover_data.shape[1] < 3:
            logger.warning("Insufficient subreddits for spillover analysis")
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

    def step_5_generate_report(self):
        """Step 5: Generate comprehensive report"""

        logger.info("=" * 60)
        logger.info("STEP 5: GENERATING COMPREHENSIVE REPORT")
        logger.info("=" * 60)

        # Create comprehensive results summary
        report = {
            'executive_summary': self._create_executive_summary(),
            'methodology': self._create_methodology_summary(),
            'results': self.results,
            'conclusions': self._create_conclusions(),
            'recommendations': self._create_recommendations()
        }

        # Save comprehensive report
        with open(self.output_dir / "comprehensive_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Create markdown report
        self._create_markdown_report(report)

        logger.info("Comprehensive report generated")
        logger.info(f"Report saved to: {self.output_dir}")

    def _create_executive_summary(self) -> Dict:
        """Create executive summary"""

        data_stats = self.results['data_processing']
        spillover_stats = self.results['spillover_analysis']
        backtest_stats = self.results['backtesting']

        return {
            'dataset_overview': {
                'observations': data_stats.get('data_shape', [0])[0],
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

        with open(self.output_dir / "report.md", 'w') as f:
            f.write(markdown_content)

    def run_complete_pipeline(self):
        """Execute the complete analysis pipeline"""

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
            self.step_5_generate_report()

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
    """Main function to run the pipeline"""

    # Configuration file path
    config_path = "experiments/configs/hierarchical_config.yaml"

    # Check if config exists
    if not Path(config_path).exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Please create the configuration file first")
        return

    try:
        # Initialize and run pipeline
        pipeline = HierarchicalSentimentPipeline(config_path)
        results = pipeline.run_complete_pipeline()

        print("\n" + "="*80)
        print("📊 PIPELINE EXECUTION SUMMARY")
        print("="*80)
        print(f"✅ Execution time: {results['execution_time']}")
        print(f"💾 Output directory: {results['output_directory']}")
        print(f"📈 Data shape: {results['processed_data'].shape}")

        if 'spillover_results' in results:
            spillover_idx = results['spillover_results'].get('static', {}).get('total_spillover_index', 0)
            print(f"🔄 Total spillover index: {spillover_idx:.2f}%")

        if 'backtest_results' in results:
            metrics = results['backtest_results'].get('performance_metrics', {})
            if metrics:
                print(f"📊 Annual return: {metrics.get('annual_return', 0):.2%}")
                print(f"📊 Sharpe ratio: {metrics.get('sharpe_ratio', 0):.3f}")

        print("="*80)

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()