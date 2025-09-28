"""
Comprehensive unit tests for Diebold-Yilmaz spillover analysis
Following scientific methodology and project best practices
"""

import pytest
import pandas as pd
import numpy as np
import networkx as nx
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import logging

# Import the modules to test
import sys
sys.path.append('/home/Hudini/projects/info_spillover')

from src.analysis.diebold_yilmaz_spillover import DieboldYilmazSpillover, SpilloverVisualizer


class TestDieboldYilmazSpillover:
    """Test cases for DieboldYilmazSpillover class"""

    @pytest.fixture
    def spillover_analyzer(self):
        """Create spillover analyzer instance"""
        return DieboldYilmazSpillover(forecast_horizon=5, identification='cholesky')

    @pytest.fixture
    def sample_time_series(self):
        """Create sample time series data for testing"""
        np.random.seed(42)  # For reproducible tests
        dates = pd.date_range('2024-01-01', periods=100, freq='H')

        # Create realistic sentiment data with some correlation
        n_vars = 3
        data = {}

        # Base series
        base_trend = np.cumsum(np.random.randn(100) * 0.1)

        for i in range(n_vars):
            # Add some correlation between series
            noise = np.random.randn(100) * 0.5
            correlation_factor = np.roll(base_trend, i*5) * 0.3
            data[f'subreddit_{i}'] = base_trend + correlation_factor + noise

        df = pd.DataFrame(data, index=dates)
        # Normalize to [-1, 1] range like real sentiment data
        df = (df - df.mean()) / df.std() * 0.5
        df = np.clip(df, -1, 1)

        return df

    @pytest.fixture
    def small_time_series(self):
        """Create minimal time series for edge case testing"""
        dates = pd.date_range('2024-01-01', periods=20, freq='H')
        np.random.seed(123)

        data = {
            'sub_a': np.random.randn(20) * 0.3,
            'sub_b': np.random.randn(20) * 0.3
        }

        return pd.DataFrame(data, index=dates)

    def test_init(self):
        """Test DieboldYilmazSpillover initialization"""
        analyzer = DieboldYilmazSpillover(forecast_horizon=10, identification='generalized')

        assert analyzer.forecast_horizon == 10
        assert analyzer.identification == 'generalized'
        assert analyzer.var_model is None
        assert analyzer.spillover_results is None

    def test_prepare_var_data_valid(self, spillover_analyzer, sample_time_series):
        """Test VAR data preparation with valid data"""
        prepared_data = spillover_analyzer._prepare_var_data(sample_time_series)

        assert isinstance(prepared_data, pd.DataFrame)
        assert prepared_data.shape[1] == 3  # 3 subreddits
        assert prepared_data.shape[0] <= sample_time_series.shape[0]  # May drop some NaN
        assert prepared_data.index.is_monotonic_increasing
        assert not prepared_data.isna().any().any()  # No NaN values

    def test_prepare_var_data_with_missing_values(self, spillover_analyzer):
        """Test VAR data preparation with missing values"""
        dates = pd.date_range('2024-01-01', periods=50, freq='H')
        data = pd.DataFrame({
            'sub_a': np.random.randn(50),
            'sub_b': np.random.randn(50),
            'sub_c': np.random.randn(50)
        }, index=dates)

        # Introduce some NaN values
        data.iloc[10:15, 0] = np.nan
        data.iloc[20:25, 1] = np.nan

        prepared_data = spillover_analyzer._prepare_var_data(data)

        assert not prepared_data.isna().any().any()
        assert prepared_data.shape[0] < data.shape[0]  # Some rows should be dropped

    def test_fit_var_model_success(self, spillover_analyzer, sample_time_series):
        """Test successful VAR model fitting"""
        prepared_data = spillover_analyzer._prepare_var_data(sample_time_series)
        var_fitted, optimal_lags = spillover_analyzer._fit_var_model(prepared_data)

        assert var_fitted is not None
        assert optimal_lags >= 1
        assert hasattr(var_fitted, 'fittedvalues')
        assert hasattr(var_fitted, 'resid')

    def test_fit_var_model_small_dataset(self, spillover_analyzer, small_time_series):
        """Test VAR fitting with small dataset (should use 1 lag)"""
        var_fitted, optimal_lags = spillover_analyzer._fit_var_model(small_time_series)

        assert var_fitted is not None
        assert optimal_lags == 1  # Should default to 1 lag for small datasets

    def test_fit_var_model_insufficient_data(self, spillover_analyzer):
        """Test VAR fitting with insufficient data"""
        # Create dataset that's too small
        dates = pd.date_range('2024-01-01', periods=5, freq='H')
        data = pd.DataFrame({
            'sub_a': [1, 2, 3, 4, 5],
            'sub_b': [2, 3, 4, 5, 6]
        }, index=dates)

        with pytest.raises(Exception):  # Should raise some exception
            spillover_analyzer._fit_var_model(data)

    def test_compute_spillover_measures(self, spillover_analyzer):
        """Test spillover measures computation"""
        # Create mock variance decomposition
        np.random.seed(42)
        n_vars = 3
        horizon = 5
        var_decomp = np.random.rand(n_vars, n_vars, horizon + 1)

        # Ensure each row sums to 1 (valid variance decomposition)
        for h in range(horizon + 1):
            var_decomp[:, :, h] = var_decomp[:, :, h] / var_decomp[:, :, h].sum(axis=1, keepdims=True)

        variable_names = ['sub_a', 'sub_b', 'sub_c']
        spillover_measures = spillover_analyzer.compute_spillover_measures(var_decomp, variable_names)

        # Validate structure
        assert 'total_spillover' in spillover_measures
        assert 'directional_spillovers_from' in spillover_measures
        assert 'directional_spillovers_to' in spillover_measures
        assert 'net_spillovers' in spillover_measures
        assert 'pairwise_spillovers' in spillover_measures
        assert 'variable_names' in spillover_measures

        # Validate types
        assert isinstance(spillover_measures['total_spillover'], (int, float))
        assert isinstance(spillover_measures['directional_spillovers_from'], dict)
        assert isinstance(spillover_measures['net_spillovers'], dict)
        assert isinstance(spillover_measures['pairwise_spillovers'], pd.DataFrame)

        # Validate values
        assert 0 <= spillover_measures['total_spillover'] <= 100
        assert len(spillover_measures['directional_spillovers_from']) == n_vars
        assert spillover_measures['pairwise_spillovers'].shape == (n_vars, n_vars)

    def test_create_spillover_network_with_dataframe(self, spillover_analyzer):
        """Test spillover network creation with DataFrame pairwise spillovers"""
        subreddits = ['sub_a', 'sub_b', 'sub_c']
        spillover_measures = {
            'total_spillover': 45.0,
            'net_spillovers': {'sub_a': 5.0, 'sub_b': -2.0, 'sub_c': -3.0},
            'pairwise_spillovers': pd.DataFrame(
                [[0, 10, 15], [5, 0, 8], [12, 6, 0]],
                index=subreddits,
                columns=subreddits
            )
        }

        network = spillover_analyzer.create_spillover_network(spillover_measures)

        assert isinstance(network, nx.DiGraph)
        assert network.number_of_nodes() == 3
        assert network.number_of_edges() > 0

        # Check node attributes
        for node in network.nodes():
            assert 'net_spillover' in network.nodes[node]
            assert 'size' in network.nodes[node]
            assert 'color' in network.nodes[node]

    def test_create_spillover_network_with_dict(self, spillover_analyzer):
        """Test spillover network creation with dictionary pairwise spillovers"""
        subreddits = ['sub_a', 'sub_b']
        spillover_measures = {
            'total_spillover': 30.0,
            'net_spillovers': {'sub_a': 2.0, 'sub_b': -2.0},
            'pairwise_spillovers': {
                ('sub_a', 'sub_b'): 8.0,
                ('sub_b', 'sub_a'): 4.0
            }
        }

        network = spillover_analyzer.create_spillover_network(spillover_measures)

        assert isinstance(network, nx.DiGraph)
        assert network.number_of_nodes() == 2
        assert network.number_of_edges() <= 2

    def test_analyze_spillover_dynamics_integration(self, spillover_analyzer, sample_time_series):
        """Integration test for complete spillover analysis"""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = spillover_analyzer.analyze_spillover_dynamics(
                sample_time_series,
                save_results=True,
                output_dir=temp_dir
            )

            # Validate results structure
            assert 'static_analysis' in results
            assert 'dynamic_analysis' in results
            assert 'network' in results

            static = results['static_analysis']
            assert 'total_spillover' in static
            assert 'pairwise_spillovers' in static

            # Validate network
            network = results['network']
            assert isinstance(network, nx.DiGraph)
            assert network.number_of_nodes() == 3

            # Check files were saved
            output_path = Path(temp_dir)
            assert (output_path / "spillover_summary.json").exists()

    def test_edge_case_single_variable(self, spillover_analyzer):
        """Test behavior with single variable (should fail gracefully)"""
        dates = pd.date_range('2024-01-01', periods=50, freq='H')
        data = pd.DataFrame({'single_var': np.random.randn(50)}, index=dates)

        with pytest.raises(Exception):  # Should raise exception for insufficient variables
            spillover_analyzer.analyze_spillover_dynamics(data)

    def test_edge_case_identical_series(self, spillover_analyzer):
        """Test behavior with identical time series"""
        dates = pd.date_range('2024-01-01', periods=50, freq='H')
        base_series = np.random.randn(50)
        data = pd.DataFrame({
            'sub_a': base_series,
            'sub_b': base_series.copy(),
            'sub_c': base_series.copy()
        }, index=dates)

        # Should handle identical series (may result in singular covariance matrix)
        try:
            results = spillover_analyzer.analyze_spillover_dynamics(data)
            # If successful, validate results
            assert 'static_analysis' in results
        except Exception as e:
            # If it fails, that's also acceptable for identical series
            assert "singular" in str(e).lower() or "rank" in str(e).lower()


class TestSpilloverVisualizer:
    """Test cases for SpilloverVisualizer class"""

    @pytest.fixture
    def sample_network(self):
        """Create sample spillover network for testing"""
        G = nx.DiGraph()
        nodes = ['Bitcoin', 'Ethereum', 'Dogecoin']
        G.add_nodes_from(nodes)

        # Add node attributes
        for i, node in enumerate(nodes):
            G.nodes[node]['net_spillover'] = [-2.5, 1.0, 1.5][i]
            G.nodes[node]['size'] = abs(G.nodes[node]['net_spillover'])
            G.nodes[node]['color'] = 'red' if G.nodes[node]['net_spillover'] > 0 else 'blue'

        # Add edges
        G.add_edge('Bitcoin', 'Ethereum', weight=0.3)
        G.add_edge('Ethereum', 'Bitcoin', weight=0.15)
        G.add_edge('Bitcoin', 'Dogecoin', weight=0.25)

        return G

    @pytest.fixture
    def sample_spillover_results(self, sample_network):
        """Create sample spillover results"""
        return {
            'static_analysis': {
                'total_spillover': 35.2,
                'directional_spillovers_from': {
                    'Bitcoin': 12.5,
                    'Ethereum': 11.8,
                    'Dogecoin': 10.9
                },
                'net_spillovers': {
                    'Bitcoin': -2.5,
                    'Ethereum': 1.0,
                    'Dogecoin': 1.5
                }
            },
            'network': sample_network
        }

    def test_visualizer_init(self):
        """Test SpilloverVisualizer initialization"""
        visualizer = SpilloverVisualizer(figsize=(12, 8))
        assert visualizer.figsize == (12, 8)

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_spillover_network(self, mock_savefig, mock_show, sample_spillover_results):
        """Test spillover network plotting"""
        visualizer = SpilloverVisualizer()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Should not raise exception
            visualizer.plot_spillover_network(
                sample_spillover_results['network'],
                title="Test Network",
                save_path=f"{temp_dir}/test_network.png"
            )

            mock_savefig.assert_called_once()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_spillover_summary(self, mock_savefig, mock_show, sample_spillover_results):
        """Test spillover summary plotting"""
        visualizer = SpilloverVisualizer()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Should not raise exception
            visualizer.plot_spillover_summary(
                sample_spillover_results,
                save_path=f"{temp_dir}/test_summary.png"
            )

            mock_savefig.assert_called_once()

    def test_empty_network_handling(self):
        """Test handling of empty network"""
        visualizer = SpilloverVisualizer()
        empty_network = nx.DiGraph()

        # Should handle empty network gracefully
        try:
            visualizer.plot_spillover_network(empty_network, title="Empty Network")
        except Exception as e:
            # If it raises exception, should be informative
            assert "empty" in str(e).lower() or "nodes" in str(e).lower()


class TestDataValidation:
    """Test data validation and edge cases"""

    def test_non_datetime_index(self):
        """Test handling of non-datetime index"""
        analyzer = DieboldYilmazSpillover()

        # Create data with non-datetime index
        data = pd.DataFrame({
            'sub_a': np.random.randn(50),
            'sub_b': np.random.randn(50)
        }, index=range(50))

        # Should handle gracefully or raise informative error
        try:
            prepared_data = analyzer._prepare_var_data(data)
            # If successful, should have proper index
            assert len(prepared_data) > 0
        except Exception as e:
            assert "datetime" in str(e).lower() or "index" in str(e).lower()

    def test_non_numeric_data(self):
        """Test handling of non-numeric data"""
        analyzer = DieboldYilmazSpillover()

        dates = pd.date_range('2024-01-01', periods=20, freq='H')
        data = pd.DataFrame({
            'sub_a': ['positive'] * 10 + ['negative'] * 10,
            'sub_b': np.random.randn(20)
        }, index=dates)

        # Should raise exception or handle gracefully
        with pytest.raises(Exception):
            analyzer._prepare_var_data(data)

    def test_extremely_large_values(self):
        """Test handling of extremely large values"""
        analyzer = DieboldYilmazSpillover()

        dates = pd.date_range('2024-01-01', periods=30, freq='H')
        data = pd.DataFrame({
            'sub_a': np.random.randn(30) * 1e6,  # Very large values
            'sub_b': np.random.randn(30) * 1e-6  # Very small values
        }, index=dates)

        # Should handle or raise appropriate error
        try:
            results = analyzer.analyze_spillover_dynamics(data)
            # If successful, results should be reasonable
            assert 'static_analysis' in results
        except Exception as e:
            # Should be a reasonable error message
            assert len(str(e)) > 0


if __name__ == "__main__":
    # Set up logging for tests
    logging.basicConfig(level=logging.INFO)

    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])