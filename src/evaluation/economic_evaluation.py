"""
Economic Evaluation and Backtesting System
Based on academic finance methodology for trading strategy evaluation

Implements:
1. Portfolio construction based on sentiment spillovers
2. Economic performance metrics
3. Statistical significance tests
4. Risk-adjusted performance evaluation
5. Transaction cost modeling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import networkx as nx
from scipy import stats
from scipy.stats import jarque_bera, normaltest
import warnings
from statsmodels.stats.diagnostic import het_white
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import yfinance as yf  # For benchmark data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioConstructor:
    """
    Portfolio construction based on sentiment spillover signals
    Following Markowitz (1952) and modern portfolio theory
    """

    def __init__(self, transaction_costs: float = 0.001,
                 max_position_size: float = 0.2,
                 rebalancing_frequency: str = 'D'):
        self.transaction_costs = transaction_costs
        self.max_position_size = max_position_size
        self.rebalancing_frequency = rebalancing_frequency
        self.portfolio_history = []

    def sentiment_spillover_signal(self, spillover_data: pd.DataFrame,
                                 sentiment_data: pd.DataFrame,
                                 lookback_window: int = 5) -> pd.DataFrame:
        """
        Generate trading signals based on sentiment spillover analysis
        """
        logger.info("Generating sentiment spillover signals...")

        signals = pd.DataFrame(index=sentiment_data.index)

        # Get unique subreddits
        subreddits = sentiment_data['subreddit'].unique()

        for subreddit in subreddits:
            subreddit_data = sentiment_data[sentiment_data['subreddit'] == subreddit].copy()
            subreddit_data = subreddit_data.sort_values('created_utc').set_index('created_utc')

            if subreddit in spillover_data.columns:
                # Net spillover signal
                net_spillover = spillover_data[subreddit].rolling(lookback_window).mean()

                # Sentiment momentum
                sentiment_momentum = subreddit_data['compound_sentiment'].rolling(lookback_window).mean()

                # Combined signal: spillover direction * sentiment strength
                combined_signal = net_spillover * sentiment_momentum

                # Normalize signal to [-1, 1]
                signal_std = combined_signal.rolling(252).std()  # Annual rolling std
                normalized_signal = combined_signal / (2 * signal_std)  # 2-sigma normalization
                normalized_signal = np.clip(normalized_signal, -1, 1)

                # Align with common index
                aligned_signal = normalized_signal.reindex(signals.index, method='ffill')
                signals[subreddit] = aligned_signal.fillna(0)

            else:
                logger.warning(f"No spillover data available for {subreddit}")
                signals[subreddit] = 0.0

        # Risk management: limit individual position sizes
        signals = signals.clip(-self.max_position_size, self.max_position_size)

        # Ensure positions sum to reasonable leverage (max 1.0)
        position_sum = signals.abs().sum(axis=1)
        leverage_adjustment = np.minimum(1.0, 1.0 / position_sum)
        signals = signals.multiply(leverage_adjustment, axis=0)

        return signals.fillna(0)

    def construct_portfolio(self, signals: pd.DataFrame,
                          price_data: pd.DataFrame,
                          initial_capital: float = 100000) -> pd.DataFrame:
        """
        Construct portfolio based on signals and price data
        """
        logger.info("Constructing portfolio...")

        # Align signals and prices
        common_index = signals.index.intersection(price_data.index)
        signals_aligned = signals.loc[common_index]
        prices_aligned = price_data.loc[common_index]

        # Initialize portfolio
        portfolio = pd.DataFrame(index=common_index)
        portfolio['total_value'] = initial_capital
        portfolio['cash'] = initial_capital
        portfolio['positions_value'] = 0.0
        portfolio['daily_return'] = 0.0
        portfolio['cumulative_return'] = 0.0

        # Track individual positions
        for asset in signals.columns:
            portfolio[f'position_{asset}'] = 0.0
            portfolio[f'weight_{asset}'] = 0.0

        previous_positions = pd.Series(0.0, index=signals.columns)

        for i, date in enumerate(common_index):
            current_signals = signals_aligned.loc[date]
            current_prices = prices_aligned.loc[date]

            # Skip if prices are missing
            if current_prices.isna().any():
                continue

            # Calculate target positions (as portfolio weights)
            target_weights = current_signals / current_signals.abs().sum() if current_signals.abs().sum() > 0 else current_signals * 0

            # Convert weights to positions (number of shares)
            current_portfolio_value = portfolio.loc[date, 'total_value'] if i > 0 else initial_capital
            target_positions = (target_weights * current_portfolio_value) / current_prices

            # Calculate trades (change in positions)
            trades = target_positions - previous_positions

            # Apply transaction costs
            transaction_cost = (trades.abs() * current_prices * self.transaction_costs).sum()

            # Update cash
            cash_flow = -(trades * current_prices).sum() - transaction_cost
            new_cash = portfolio.loc[date, 'cash'] + cash_flow

            # Update positions
            new_positions = previous_positions + trades

            # Calculate new portfolio value
            positions_value = (new_positions * current_prices).sum()
            total_value = new_cash + positions_value

            # Calculate returns
            if i > 0:
                previous_value = portfolio.iloc[i-1]['total_value']
                daily_return = (total_value - previous_value) / previous_value
            else:
                daily_return = 0.0

            cumulative_return = (total_value - initial_capital) / initial_capital

            # Update portfolio DataFrame
            portfolio.loc[date, 'cash'] = new_cash
            portfolio.loc[date, 'positions_value'] = positions_value
            portfolio.loc[date, 'total_value'] = total_value
            portfolio.loc[date, 'daily_return'] = daily_return
            portfolio.loc[date, 'cumulative_return'] = cumulative_return

            # Update individual positions and weights
            for asset in signals.columns:
                portfolio.loc[date, f'position_{asset}'] = new_positions[asset]
                portfolio.loc[date, f'weight_{asset}'] = (new_positions[asset] * current_prices[asset]) / total_value if total_value > 0 else 0

            # Update for next iteration
            previous_positions = new_positions.copy()

        logger.info(f"Portfolio constructed: {len(portfolio)} periods")
        logger.info(f"Final portfolio value: ${portfolio['total_value'].iloc[-1]:,.2f}")
        logger.info(f"Total return: {portfolio['cumulative_return'].iloc[-1]:.2%}")

        return portfolio


class PerformanceEvaluator:
    """
    Performance evaluation with academic finance metrics
    """

    def __init__(self, risk_free_rate: float = 0.02):  # 2% annual risk-free rate
        self.risk_free_rate = risk_free_rate

    def calculate_performance_metrics(self, portfolio_returns: pd.Series,
                                    benchmark_returns: pd.Series = None,
                                    frequency: str = 'D') -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        logger.info("Calculating performance metrics...")

        # Annualization factor
        if frequency == 'D':
            periods_per_year = 252
        elif frequency == 'W':
            periods_per_year = 52
        elif frequency == 'M':
            periods_per_year = 12
        else:
            periods_per_year = 252

        # Remove NaN values
        returns = portfolio_returns.dropna()
        if returns.empty:
            logger.warning("No valid returns data")
            return {}

        # Basic statistics
        mean_return = returns.mean() * periods_per_year
        volatility = returns.std() * np.sqrt(periods_per_year)
        sharpe_ratio = (mean_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
        sortino_ratio = (mean_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

        # Tail risk measures
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean() if (returns <= var_95).any() else var_95

        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()

        # Hit rate (percentage of positive returns)
        hit_rate = (returns > 0).mean()

        # Calmar ratio (return/max drawdown)
        calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown != 0 else 0

        metrics = {
            'annual_return': mean_return,
            'annual_volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'hit_rate': hit_rate,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'total_periods': len(returns),
            'positive_periods': (returns > 0).sum(),
            'negative_periods': (returns < 0).sum(),
        }

        # Benchmark-relative metrics
        if benchmark_returns is not None:
            benchmark_aligned = benchmark_returns.reindex(returns.index, method='ffill').dropna()
            common_index = returns.index.intersection(benchmark_aligned.index)

            if len(common_index) > 0:
                port_aligned = returns.loc[common_index]
                bench_aligned = benchmark_aligned.loc[common_index]

                # Tracking error and information ratio
                active_returns = port_aligned - bench_aligned
                tracking_error = active_returns.std() * np.sqrt(periods_per_year)
                information_ratio = active_returns.mean() * periods_per_year / tracking_error if tracking_error > 0 else 0

                # Beta and alpha (CAPM)
                if bench_aligned.std() > 0:
                    beta = np.cov(port_aligned, bench_aligned)[0, 1] / np.var(bench_aligned)
                    alpha = mean_return - (self.risk_free_rate + beta * (bench_aligned.mean() * periods_per_year - self.risk_free_rate))
                else:
                    beta = 0
                    alpha = 0

                metrics.update({
                    'tracking_error': tracking_error,
                    'information_ratio': information_ratio,
                    'beta': beta,
                    'alpha': alpha,
                    'active_return': active_returns.mean() * periods_per_year
                })

        return metrics

    def statistical_tests(self, portfolio_returns: pd.Series,
                         benchmark_returns: pd.Series = None) -> Dict:
        """
        Perform statistical tests on returns
        """
        logger.info("Performing statistical tests...")

        returns = portfolio_returns.dropna()
        tests = {}

        if len(returns) < 30:
            logger.warning("Insufficient data for statistical tests")
            return tests

        # Normality tests
        try:
            # Jarque-Bera test
            jb_stat, jb_pvalue = jarque_bera(returns)
            tests['jarque_bera'] = {'statistic': jb_stat, 'p_value': jb_pvalue, 'normal': jb_pvalue > 0.05}

            # Shapiro-Wilk test (for smaller samples)
            if len(returns) <= 5000:
                sw_stat, sw_pvalue = stats.shapiro(returns)
                tests['shapiro_wilk'] = {'statistic': sw_stat, 'p_value': sw_pvalue, 'normal': sw_pvalue > 0.05}

        except Exception as e:
            logger.warning(f"Normality tests failed: {str(e)}")

        # Serial correlation test (Ljung-Box)
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(returns, lags=10, return_df=True)
            tests['ljung_box'] = {
                'statistic': lb_result['lb_stat'].iloc[-1],
                'p_value': lb_result['lb_pvalue'].iloc[-1],
                'serial_correlation': lb_result['lb_pvalue'].iloc[-1] < 0.05
            }
        except Exception as e:
            logger.warning(f"Ljung-Box test failed: {str(e)}")

        # Stationarity test
        try:
            adf_stat, adf_pvalue, adf_usedlag, adf_nobs, adf_critical, adf_icbest = adfuller(returns)
            tests['stationarity'] = {
                'adf_statistic': adf_stat,
                'p_value': adf_pvalue,
                'stationary': adf_pvalue < 0.05
            }
        except Exception as e:
            logger.warning(f"Stationarity test failed: {str(e)}")

        # Comparison with benchmark
        if benchmark_returns is not None:
            benchmark_aligned = benchmark_returns.reindex(returns.index, method='ffill').dropna()
            common_index = returns.index.intersection(benchmark_aligned.index)

            if len(common_index) > 10:
                port_aligned = returns.loc[common_index]
                bench_aligned = benchmark_aligned.loc[common_index]

                # Paired t-test
                try:
                    t_stat, t_pvalue = stats.ttest_rel(port_aligned, bench_aligned)
                    tests['paired_t_test'] = {
                        'statistic': t_stat,
                        'p_value': t_pvalue,
                        'significantly_different': t_pvalue < 0.05
                    }
                except Exception as e:
                    logger.warning(f"Paired t-test failed: {str(e)}")

                # Wilcoxon signed-rank test (non-parametric)
                try:
                    w_stat, w_pvalue = stats.wilcoxon(port_aligned, bench_aligned)
                    tests['wilcoxon_test'] = {
                        'statistic': w_stat,
                        'p_value': w_pvalue,
                        'significantly_different': w_pvalue < 0.05
                    }
                except Exception as e:
                    logger.warning(f"Wilcoxon test failed: {str(e)}")

        return tests

    def diebold_mariano_test(self, forecast_errors_1: pd.Series,
                           forecast_errors_2: pd.Series) -> Dict:
        """
        Diebold-Mariano test for forecast comparison
        Based on Diebold & Mariano (1995)
        """
        logger.info("Performing Diebold-Mariano test...")

        # Align the series
        common_index = forecast_errors_1.index.intersection(forecast_errors_2.index)
        e1 = forecast_errors_1.loc[common_index].dropna()
        e2 = forecast_errors_2.loc[common_index].dropna()

        if len(e1) != len(e2) or len(e1) < 10:
            logger.warning("Insufficient aligned data for Diebold-Mariano test")
            return {}

        # Loss differential (squared errors)
        d = e1**2 - e2**2

        # Mean loss differential
        d_mean = d.mean()

        # Variance of loss differential (with Newey-West adjustment for autocorrelation)
        try:
            from statsmodels.stats.sandwich_covariance import cov_hac
            from statsmodels.regression.linear_model import OLS
            import statsmodels.api as sm

            # Simple variance estimation
            d_var = d.var()

            # Newey-West HAC standard error
            if len(d) > 20:
                # Create dummy regression for HAC estimation
                y = d.values.reshape(-1, 1)
                x = np.ones_like(y)
                model = OLS(y, x)
                results = model.fit(cov_type='HAC', cov_kwds={'maxlags': min(5, len(d)//4)})
                d_var = results.cov_params().iloc[0, 0]

            # Test statistic
            dm_stat = d_mean / np.sqrt(d_var / len(d))

            # P-value (two-sided test)
            p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

            return {
                'dm_statistic': dm_stat,
                'p_value': p_value,
                'mean_loss_differential': d_mean,
                'significantly_different': p_value < 0.05,
                'model1_better': dm_stat < 0 and p_value < 0.05,
                'model2_better': dm_stat > 0 and p_value < 0.05
            }

        except Exception as e:
            logger.warning(f"Diebold-Mariano test failed: {str(e)}")
            return {}


class BacktestingFramework:
    """
    Comprehensive backtesting framework with realistic constraints
    """

    def __init__(self, start_date: str, end_date: str,
                 initial_capital: float = 100000,
                 transaction_costs: float = 0.001,
                 slippage: float = 0.0005):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.transaction_costs = transaction_costs
        self.slippage = slippage

        self.portfolio_constructor = PortfolioConstructor(
            transaction_costs=transaction_costs,
            max_position_size=0.2,
            rebalancing_frequency='D'
        )

        self.evaluator = PerformanceEvaluator()

    def get_benchmark_data(self, benchmark_symbol: str = 'BTC-USD') -> pd.DataFrame:
        """
        Get benchmark data (e.g., Bitcoin for crypto strategies)
        """
        logger.info(f"Fetching benchmark data for {benchmark_symbol}")

        try:
            benchmark = yf.download(benchmark_symbol, start=self.start_date, end=self.end_date)
            benchmark['returns'] = benchmark['Adj Close'].pct_change()
            return benchmark
        except Exception as e:
            logger.warning(f"Failed to fetch benchmark data: {str(e)}")
            return pd.DataFrame()

    def run_backtest(self, spillover_data: pd.DataFrame,
                    sentiment_data: pd.DataFrame,
                    price_data: pd.DataFrame,
                    benchmark_symbol: str = 'BTC-USD') -> Dict:
        """
        Run complete backtest with economic evaluation
        """
        logger.info("Running comprehensive backtest...")

        results = {}

        # 1. Generate signals
        logger.info("1. Generating trading signals...")
        signals = self.portfolio_constructor.sentiment_spillover_signal(
            spillover_data, sentiment_data, lookback_window=5
        )

        # 2. Construct portfolio
        logger.info("2. Constructing portfolio...")
        portfolio = self.portfolio_constructor.construct_portfolio(
            signals, price_data, self.initial_capital
        )

        # 3. Get benchmark
        logger.info("3. Fetching benchmark data...")
        benchmark_data = self.get_benchmark_data(benchmark_symbol)

        # 4. Calculate performance metrics
        logger.info("4. Calculating performance metrics...")
        portfolio_returns = portfolio['daily_return']

        benchmark_returns = None
        if not benchmark_data.empty:
            # Align benchmark with portfolio dates
            benchmark_returns = benchmark_data['returns'].reindex(
                portfolio.index, method='ffill'
            ).dropna()

        performance_metrics = self.evaluator.calculate_performance_metrics(
            portfolio_returns, benchmark_returns
        )

        # 5. Statistical tests
        logger.info("5. Performing statistical tests...")
        statistical_tests = self.evaluator.statistical_tests(
            portfolio_returns, benchmark_returns
        )

        # 6. Compile results
        results = {
            'portfolio': portfolio,
            'signals': signals,
            'performance_metrics': performance_metrics,
            'statistical_tests': statistical_tests,
            'benchmark_data': benchmark_data if not benchmark_data.empty else None,
            'backtest_config': {
                'start_date': self.start_date,
                'end_date': self.end_date,
                'initial_capital': self.initial_capital,
                'transaction_costs': self.transaction_costs,
                'slippage': self.slippage,
                'benchmark': benchmark_symbol
            }
        }

        # 7. Summary statistics
        final_value = portfolio['total_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        logger.info("=== BACKTEST SUMMARY ===")
        logger.info(f"Period: {self.start_date} to {self.end_date}")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Final Value: ${final_value:,.2f}")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Annual Return: {performance_metrics.get('annual_return', 0):.2%}")
        logger.info(f"Annual Volatility: {performance_metrics.get('annual_volatility', 0):.2%}")
        logger.info(f"Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.3f}")
        logger.info(f"Max Drawdown: {performance_metrics.get('max_drawdown', 0):.2%}")

        if 'alpha' in performance_metrics:
            logger.info(f"Alpha vs Benchmark: {performance_metrics['alpha']:.2%}")
            logger.info(f"Beta vs Benchmark: {performance_metrics.get('beta', 0):.3f}")

        return results

    def save_backtest_results(self, results: Dict, output_dir: str = "results/backtest"):
        """Save backtest results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save portfolio data
        results['portfolio'].to_csv(output_path / "portfolio_performance.csv")

        # Save signals
        results['signals'].to_csv(output_path / "trading_signals.csv")

        # Save performance metrics
        with open(output_path / "performance_metrics.json", 'w') as f:
            json.dump(results['performance_metrics'], f, indent=2, default=str)

        # Save statistical tests
        with open(output_path / "statistical_tests.json", 'w') as f:
            json.dump(results['statistical_tests'], f, indent=2, default=str)

        # Save configuration
        with open(output_path / "backtest_config.json", 'w') as f:
            json.dump(results['backtest_config'], f, indent=2, default=str)

        logger.info(f"Backtest results saved to {output_dir}")


class BacktestVisualizer:
    """
    Visualization tools for backtest results
    """

    def __init__(self):
        self.style_config = {
            'figure_size': (15, 10),
            'dpi': 300
        }

    def plot_cumulative_returns(self, portfolio: pd.DataFrame,
                               benchmark_data: pd.DataFrame = None,
                               save_path: str = None):
        """Plot cumulative returns vs benchmark"""

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})

        # Cumulative returns
        portfolio_cumret = (1 + portfolio['daily_return']).cumprod()
        ax1.plot(portfolio.index, portfolio_cumret, label='Strategy', linewidth=2, color='blue')

        if benchmark_data is not None and not benchmark_data.empty:
            benchmark_aligned = benchmark_data['returns'].reindex(portfolio.index, method='ffill')
            benchmark_cumret = (1 + benchmark_aligned).cumprod()
            ax1.plot(portfolio.index, benchmark_cumret, label='Benchmark (BTC)', linewidth=2, color='orange')

        ax1.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Drawdown
        running_max = portfolio_cumret.expanding().max()
        drawdown = (portfolio_cumret - running_max) / running_max
        ax2.fill_between(portfolio.index, drawdown, 0, alpha=0.3, color='red')
        ax2.plot(portfolio.index, drawdown, color='red', linewidth=1)
        ax2.set_title('Drawdown', fontsize=12)
        ax2.set_ylabel('Drawdown', fontsize=10)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.style_config['dpi'], bbox_inches='tight')

        plt.show()

    def plot_rolling_metrics(self, portfolio: pd.DataFrame, window: int = 252, save_path: str = None):
        """Plot rolling performance metrics"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        returns = portfolio['daily_return']

        # Rolling Sharpe ratio
        rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
        axes[0, 0].plot(portfolio.index, rolling_sharpe, color='blue', linewidth=1.5)
        axes[0, 0].set_title(f'Rolling Sharpe Ratio ({window}D)', fontsize=12)
        axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 0].grid(True, alpha=0.3)

        # Rolling volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        axes[0, 1].plot(portfolio.index, rolling_vol, color='red', linewidth=1.5)
        axes[0, 1].set_title(f'Rolling Volatility ({window}D)', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)

        # Rolling returns
        rolling_ret = returns.rolling(window).mean() * 252
        axes[1, 0].plot(portfolio.index, rolling_ret, color='green', linewidth=1.5)
        axes[1, 0].set_title(f'Rolling Annual Return ({window}D)', fontsize=12)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)

        # Portfolio value
        axes[1, 1].plot(portfolio.index, portfolio['total_value'], color='purple', linewidth=1.5)
        axes[1, 1].set_title('Portfolio Value', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)

        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.style_config['dpi'], bbox_inches='tight')

        plt.show()


def main():
    """Example usage of economic evaluation framework"""

    logger.info("Economic Evaluation and Backtesting Framework - Example")

    # Initialize backtesting framework
    backtester = BacktestingFramework(
        start_date='2021-01-01',
        end_date='2023-12-31',
        initial_capital=100000,
        transaction_costs=0.001,  # 0.1%
        slippage=0.0005           # 0.05%
    )

    print("Economic evaluation framework ready!")
    print("Provide spillover analysis results and price data to run backtest")


if __name__ == "__main__":
    main()