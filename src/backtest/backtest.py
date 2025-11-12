"""
Backtesting module with walk-forward time-series cross-validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from src.strategy.strategy import MLTradingStrategy, PortfolioManager


class WalkForwardBacktest:
    """Walk-forward backtesting with time-series cross-validation."""
    
    def __init__(
        self,
        train_window: int = 1260,  # 5 years
        test_window: int = 252,    # 1 year
        step_size: int = 63,       # 3 months
        min_train_samples: int = 500
    ):
        """
        Initialize backtest.
        
        Args:
            train_window: Training window size in days
            test_window: Test window size in days
            step_size: Step size for rolling window
            min_train_samples: Minimum training samples required
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.min_train_samples = min_train_samples
        
    def create_splits(
        self,
        data: pd.DataFrame,
        date_col: str = 'Date'
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create walk-forward train/test splits.
        
        Args:
            data: Input DataFrame with date index or column
            date_col: Name of date column
            
        Returns:
            List of (train, test) DataFrame tuples
        """
        if date_col in data.columns:
            data = data.set_index(date_col)
        
        # Sort by date
        data = data.sort_index()
        dates = data.index.unique()
        
        splits = []
        start_idx = 0
        
        while start_idx + self.train_window + self.test_window <= len(dates):
            # Define windows
            train_end_idx = start_idx + self.train_window
            test_end_idx = train_end_idx + self.test_window
            
            # Get date ranges
            train_dates = dates[start_idx:train_end_idx]
            test_dates = dates[train_end_idx:test_end_idx]
            
            # Create splits
            train_data = data.loc[train_dates]
            test_data = data.loc[test_dates]
            
            # Check minimum samples
            if len(train_data) >= self.min_train_samples:
                splits.append((train_data, test_data))
            
            # Move window
            start_idx += self.step_size
        
        return splits


class PerformanceMetrics:
    """Calculate trading performance metrics."""
    
    @staticmethod
    def calculate_returns(equity_curve: pd.Series) -> pd.Series:
        """Calculate returns from equity curve."""
        return equity_curve.pct_change().fillna(0)
    
    @staticmethod
    def calculate_cagr(equity_curve: pd.Series) -> float:
        """Calculate Compound Annual Growth Rate."""
        if len(equity_curve) < 2:
            return 0.0
        
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        
        if years <= 0:
            return 0.0
        
        cagr = (1 + total_return) ** (1 / years) - 1
        return cagr
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        return sharpe
    
    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        sortino = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
        return sortino
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        return drawdown.min()
    
    @staticmethod
    def calculate_calmar_ratio(
        cagr: float,
        max_drawdown: float
    ) -> float:
        """Calculate Calmar ratio (CAGR / |Max Drawdown|)."""
        if max_drawdown == 0:
            return 0.0
        return cagr / abs(max_drawdown)
    
    @staticmethod
    def calculate_hit_rate(trades: pd.DataFrame) -> float:
        """Calculate hit rate (% winning trades)."""
        if len(trades) == 0:
            return 0.0
        return (trades['return'] > 0).mean()
    
    @staticmethod
    def calculate_turnover(
        trades: pd.DataFrame,
        portfolio_value: float
    ) -> float:
        """Calculate annualized turnover."""
        if len(trades) == 0:
            return 0.0
        
        total_traded = trades['shares'].sum() * trades['entry_price'].mean()
        days = (trades['exit_date'].max() - trades['entry_date'].min()).days
        years = days / 365.25 if days > 0 else 1
        
        annual_turnover = (total_traded / portfolio_value) / years
        return annual_turnover
    
    @staticmethod
    def get_all_metrics(
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        initial_capital: float
    ) -> Dict[str, float]:
        """Calculate all performance metrics."""
        returns = PerformanceMetrics.calculate_returns(equity_curve)
        cagr = PerformanceMetrics.calculate_cagr(equity_curve)
        max_dd = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        
        metrics = {
            'total_return': (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) > 0 else 0,
            'cagr': cagr,
            'sharpe_ratio': PerformanceMetrics.calculate_sharpe_ratio(returns),
            'sortino_ratio': PerformanceMetrics.calculate_sortino_ratio(returns),
            'max_drawdown': max_dd,
            'calmar_ratio': PerformanceMetrics.calculate_calmar_ratio(cagr, max_dd),
            'volatility': returns.std() * np.sqrt(252),
            'hit_rate': PerformanceMetrics.calculate_hit_rate(trades),
            'total_trades': len(trades),
            'avg_return_per_trade': trades['return'].mean() if len(trades) > 0 else 0,
            'turnover': PerformanceMetrics.calculate_turnover(trades, initial_capital)
        }
        
        return metrics


class Backtester:
    """Complete backtesting engine."""
    
    def __init__(
        self,
        strategy: MLTradingStrategy,
        initial_capital: float = 100000
    ):
        """
        Initialize backtester.
        
        Args:
            strategy: Trading strategy instance
            initial_capital: Starting capital
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.results = []
        
    def run_backtest(
        self,
        predictions: pd.DataFrame,
        prices: pd.DataFrame,
        volatilities: Optional[pd.DataFrame] = None
    ) -> Dict[str, any]:
        """
        Run backtest on predictions.
        
        Args:
            predictions: DataFrame with ['date', 'ticker', 'probability', 'expected_return']
            prices: DataFrame with price data (indexed by date, columns are tickers)
            volatilities: Optional DataFrame with volatility estimates
            
        Returns:
            Dictionary with backtest results
        """
        # Initialize portfolio
        portfolio = PortfolioManager(
            initial_capital=self.initial_capital,
            max_positions=self.strategy.max_positions
        )
        
        # Get unique dates and sort
        dates = sorted(predictions['date'].unique())
        
        # Track signals
        signals_taken = []
        
        for current_date in dates:
            # Get current predictions
            day_predictions = predictions[predictions['date'] == current_date]
            
            # Get current prices
            if current_date not in prices.index:
                continue
            
            current_prices = prices.loc[current_date].to_dict()
            
            # Update existing positions
            portfolio.update_positions(current_prices, current_date)
            
            # Check exits for existing positions
            for ticker in list(portfolio.positions.keys()):
                if ticker in current_prices:
                    position = portfolio.positions[ticker]
                    should_exit, reason = self.strategy.should_exit(
                        position['entry_price'],
                        current_prices[ticker],
                        position['max_price']
                    )
                    
                    if should_exit:
                        portfolio.close_position(
                            ticker,
                            current_prices[ticker],
                            current_date,
                            reason,
                            self.strategy.total_cost
                        )
            
            # Check entries for new positions
            for _, pred in day_predictions.iterrows():
                ticker = pred['ticker']
                
                # Skip if already have position
                if ticker in portfolio.positions:
                    continue
                
                # Check entry criteria
                if self.strategy.should_enter(
                    pred['probability'],
                    pred['expected_return'],
                    len(portfolio.positions)
                ):
                    # Get volatility
                    vol = 0.20  # Default
                    if volatilities is not None and ticker in volatilities.columns:
                        if current_date in volatilities.index:
                            vol = volatilities.loc[current_date, ticker]
                    
                    # Calculate position size
                    position_size = self.strategy.calculate_position_size(
                        pred['probability'],
                        pred['expected_return'],
                        vol,
                        portfolio.get_portfolio_value(current_prices)
                    )
                    
                    # Open position
                    if ticker in current_prices:
                        success = portfolio.open_position(
                            ticker,
                            current_prices[ticker],
                            position_size,
                            current_date,
                            self.strategy.total_cost
                        )
                        
                        if success:
                            signals_taken.append({
                                'date': current_date,
                                'ticker': ticker,
                                'probability': pred['probability'],
                                'expected_return': pred['expected_return'],
                                'position_size': position_size
                            })
        
        # Close any remaining positions at the end
        final_date = dates[-1]
        if final_date in prices.index:
            final_prices = prices.loc[final_date].to_dict()
            for ticker in list(portfolio.positions.keys()):
                if ticker in final_prices:
                    portfolio.close_position(
                        ticker,
                        final_prices[ticker],
                        final_date,
                        "end_of_period",
                        self.strategy.total_cost
                    )
        
        # Calculate metrics
        equity_df = pd.DataFrame({
            'date': portfolio.dates,
            'equity': portfolio.equity_curve
        }).set_index('date')
        
        trades_df = pd.DataFrame(portfolio.closed_positions)
        
        metrics = PerformanceMetrics.get_all_metrics(
            equity_df['equity'],
            trades_df,
            self.initial_capital
        )
        
        return {
            'metrics': metrics,
            'equity_curve': equity_df,
            'trades': trades_df,
            'signals': pd.DataFrame(signals_taken),
            'portfolio': portfolio
        }
    
    def plot_results(
        self,
        results: Dict[str, any],
        save_path: Optional[str] = None
    ):
        """
        Plot backtest results.
        
        Args:
            results: Results dictionary from run_backtest
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
        # Plot 1: Equity curve
        ax = axes[0, 0]
        equity = results['equity_curve']['equity']
        ax.plot(equity.index, equity.values, linewidth=2)
        ax.set_title('Equity Curve')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Drawdown
        ax = axes[0, 1]
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        ax.set_title('Drawdown')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Returns distribution
        ax = axes[1, 0]
        returns = equity.pct_change().dropna()
        ax.hist(returns * 100, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_title('Daily Returns Distribution')
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Trade returns distribution
        ax = axes[1, 1]
        if len(results['trades']) > 0:
            trade_returns = results['trades']['return'] * 100
            ax.hist(trade_returns, bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax.set_title('Trade Returns Distribution')
            ax.set_xlabel('Return (%)')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        # Plot 5: Metrics table
        ax = axes[2, 0]
        ax.axis('off')
        
        metrics = results['metrics']
        metrics_text = f"""
        Performance Metrics:
        
        Total Return: {metrics['total_return']*100:.2f}%
        CAGR: {metrics['cagr']*100:.2f}%
        Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
        Sortino Ratio: {metrics['sortino_ratio']:.2f}
        Max Drawdown: {metrics['max_drawdown']*100:.2f}%
        Calmar Ratio: {metrics['calmar_ratio']:.2f}
        Volatility: {metrics['volatility']*100:.2f}%
        
        Trading Stats:
        Total Trades: {metrics['total_trades']}
        Hit Rate: {metrics['hit_rate']*100:.1f}%
        Avg Return/Trade: {metrics['avg_return_per_trade']*100:.2f}%
        Turnover: {metrics['turnover']:.2f}x
        """
        
        ax.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace', verticalalignment='center')
        
        # Plot 6: Cumulative trades
        ax = axes[2, 1]
        if len(results['trades']) > 0:
            trades = results['trades'].copy()
            trades['cumulative_trades'] = range(1, len(trades) + 1)
            ax.plot(trades['exit_date'], trades['cumulative_trades'], linewidth=2)
            ax.set_title('Cumulative Trades')
            ax.set_xlabel('Date')
            ax.set_ylabel('Number of Trades')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results plot saved to {save_path}")
        
        plt.show()
