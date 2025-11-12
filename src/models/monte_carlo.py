"""
Monte Carlo simulation module for generating future price distributions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class MonteCarloSimulator:
    """Generate future price distributions using Monte Carlo simulation."""
    
    def __init__(
        self,
        n_simulations: int = 10000,
        method: str = "gbm",
        random_seed: int = 42
    ):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            n_simulations: Number of simulation paths
            method: Simulation method ('gbm' or 'bootstrap')
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.method = method
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def simulate_gbm(
        self,
        current_price: float,
        expected_return: float,
        volatility: float,
        horizon_days: int,
        drift_adjustment: float = 0.0
    ) -> np.ndarray:
        """
        Simulate price paths using Geometric Brownian Motion.
        
        Args:
            current_price: Current stock price
            expected_return: Expected return (drift)
            volatility: Annual volatility
            horizon_days: Number of days to simulate
            drift_adjustment: Additional drift adjustment from model
            
        Returns:
            Array of shape (n_simulations, horizon_days+1) with price paths
        """
        # Annualized parameters
        dt = 1/252  # Daily time step
        
        # Adjust drift
        drift = expected_return / horizon_days + drift_adjustment
        
        # Initialize price paths
        paths = np.zeros((self.n_simulations, horizon_days + 1))
        paths[:, 0] = current_price
        
        # Generate random shocks
        np.random.seed(self.random_seed)
        shocks = np.random.normal(0, 1, (self.n_simulations, horizon_days))
        
        # Simulate paths
        for t in range(1, horizon_days + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shocks[:, t-1]
            )
        
        return paths
    
    def simulate_bootstrap(
        self,
        current_price: float,
        historical_returns: pd.Series,
        horizon_days: int,
        expected_return: Optional[float] = None
    ) -> np.ndarray:
        """
        Simulate price paths using bootstrapped historical returns.
        
        Args:
            current_price: Current stock price
            historical_returns: Historical daily returns
            horizon_days: Number of days to simulate
            expected_return: Optional expected return to adjust bootstrap
            
        Returns:
            Array of shape (n_simulations, horizon_days+1) with price paths
        """
        # Initialize price paths
        paths = np.zeros((self.n_simulations, horizon_days + 1))
        paths[:, 0] = current_price
        
        # Bootstrap returns
        np.random.seed(self.random_seed)
        
        for i in range(self.n_simulations):
            # Sample returns with replacement
            sampled_returns = np.random.choice(
                historical_returns.dropna(),
                size=horizon_days,
                replace=True
            )
            
            # Adjust returns if expected return provided
            if expected_return is not None:
                adjustment = expected_return / horizon_days - sampled_returns.mean()
                sampled_returns = sampled_returns + adjustment
            
            # Generate price path
            for t in range(1, horizon_days + 1):
                paths[i, t] = paths[i, t-1] * (1 + sampled_returns[t-1])
        
        return paths
    
    def simulate(
        self,
        current_price: float,
        expected_return: float,
        volatility: float,
        horizon_days: int,
        historical_returns: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Simulate price paths using the configured method.
        
        Args:
            current_price: Current stock price
            expected_return: Expected return (from model)
            volatility: Volatility estimate
            horizon_days: Prediction horizon in days
            historical_returns: Historical returns (for bootstrap method)
            
        Returns:
            Array of simulated price paths
        """
        if self.method == "gbm":
            return self.simulate_gbm(
                current_price, expected_return, volatility, horizon_days
            )
        elif self.method == "bootstrap":
            if historical_returns is None:
                raise ValueError("historical_returns required for bootstrap method")
            return self.simulate_bootstrap(
                current_price, historical_returns, horizon_days, expected_return
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def calculate_statistics(
        self,
        paths: np.ndarray,
        current_price: float,
        percentiles: List[float] = [10, 25, 50, 75, 90]
    ) -> Dict[str, any]:
        """
        Calculate statistics from simulated paths.
        
        Args:
            paths: Simulated price paths
            current_price: Current price (entry price)
            percentiles: Percentiles to compute
            
        Returns:
            Dictionary of statistics
        """
        final_prices = paths[:, -1]
        
        stats = {
            'mean_price': np.mean(final_prices),
            'median_price': np.median(final_prices),
            'std_price': np.std(final_prices),
            'percentiles': {p: np.percentile(final_prices, p) for p in percentiles},
            'prob_loss': np.mean(final_prices < current_price),
            'prob_gain': np.mean(final_prices > current_price),
            'expected_return': (np.mean(final_prices) / current_price - 1),
            'expected_gain_if_up': np.mean(final_prices[final_prices > current_price] / current_price - 1),
            'expected_loss_if_down': np.mean(final_prices[final_prices < current_price] / current_price - 1)
        }
        
        # Calculate drawdown statistics
        max_drawdowns = []
        for path in paths:
            running_max = np.maximum.accumulate(path)
            drawdown = (path - running_max) / running_max
            max_drawdowns.append(drawdown.min())
        
        stats['expected_max_drawdown'] = np.mean(max_drawdowns)
        stats['worst_drawdown_95'] = np.percentile(max_drawdowns, 5)
        
        # Value at Risk (VaR) and Conditional VaR
        returns = final_prices / current_price - 1
        stats['VaR_95'] = np.percentile(returns, 5)
        stats['CVaR_95'] = np.mean(returns[returns <= stats['VaR_95']])
        
        return stats
    
    def plot_distribution(
        self,
        paths: np.ndarray,
        current_price: float,
        ticker: str,
        horizon_days: int,
        save_path: Optional[str] = None,
        show_n_paths: int = 100
    ):
        """
        Plot price distribution and sample paths.
        
        Args:
            paths: Simulated price paths
            current_price: Current price
            ticker: Stock ticker
            horizon_days: Prediction horizon
            save_path: Path to save plot
            show_n_paths: Number of sample paths to display
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Sample paths
        ax = axes[0, 0]
        sample_indices = np.random.choice(len(paths), min(show_n_paths, len(paths)), replace=False)
        for idx in sample_indices:
            ax.plot(paths[idx], alpha=0.1, color='blue')
        
        # Add percentile bands
        percentiles = [10, 50, 90]
        colors = ['red', 'green', 'red']
        labels = ['P10', 'P50 (Median)', 'P90']
        for p, color, label in zip(percentiles, colors, labels):
            p_path = np.percentile(paths, p, axis=0)
            ax.plot(p_path, color=color, linewidth=2, label=label)
        
        ax.axhline(y=current_price, color='black', linestyle='--', label='Current Price')
        ax.set_xlabel('Days')
        ax.set_ylabel('Price')
        ax.set_title(f'{ticker} - Simulated Price Paths ({self.n_simulations} simulations)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Final price distribution
        ax = axes[0, 1]
        final_prices = paths[:, -1]
        ax.hist(final_prices, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=current_price, color='red', linestyle='--', linewidth=2, label='Current Price')
        ax.axvline(x=np.mean(final_prices), color='green', linestyle='--', linewidth=2, label='Expected Price')
        ax.set_xlabel('Price')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of Prices at Day {horizon_days}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Return distribution
        ax = axes[1, 0]
        returns = (final_prices / current_price - 1) * 100
        ax.hist(returns, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Breakeven')
        ax.axvline(x=np.mean(returns), color='blue', linestyle='--', linewidth=2, label='Expected Return')
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of Returns at Day {horizon_days}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Statistics table
        ax = axes[1, 1]
        ax.axis('off')
        
        stats = self.calculate_statistics(paths, current_price)
        
        stats_text = f"""
        Simulation Statistics ({self.n_simulations} paths, {horizon_days} days)
        
        Price Statistics:
        - Current Price: {current_price:.2f}
        - Expected Price: {stats['mean_price']:.2f}
        - Median Price: {stats['median_price']:.2f}
        - Std Dev: {stats['std_price']:.2f}
        
        Return Statistics:
        - Expected Return: {stats['expected_return']*100:.2f}%
        - Probability of Gain: {stats['prob_gain']*100:.1f}%
        - Probability of Loss: {stats['prob_loss']*100:.1f}%
        
        Risk Metrics:
        - Expected Max Drawdown: {stats['expected_max_drawdown']*100:.2f}%
        - 95% Worst Drawdown: {stats['worst_drawdown_95']*100:.2f}%
        - VaR (95%): {stats['VaR_95']*100:.2f}%
        - CVaR (95%): {stats['CVaR_95']*100:.2f}%
        
        Percentiles:
        - P10: {stats['percentiles'][10]:.2f}
        - P25: {stats['percentiles'][25]:.2f}
        - P50: {stats['percentiles'][50]:.2f}
        - P75: {stats['percentiles'][75]:.2f}
        - P90: {stats['percentiles'][90]:.2f}
        """
        
        ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
        return stats


def generate_price_forecast(
    ticker: str,
    current_price: float,
    expected_return: float,
    volatility: float,
    horizon_days: int,
    n_simulations: int = 10000,
    method: str = "gbm",
    historical_returns: Optional[pd.Series] = None,
    save_path: Optional[str] = None
) -> Dict[str, any]:
    """
    Generate a complete price forecast with Monte Carlo simulation.
    
    Args:
        ticker: Stock ticker symbol
        current_price: Current stock price
        expected_return: Expected return from model
        volatility: Volatility estimate
        horizon_days: Prediction horizon in days
        n_simulations: Number of Monte Carlo simulations
        method: Simulation method ('gbm' or 'bootstrap')
        historical_returns: Historical returns (for bootstrap)
        save_path: Path to save visualization
        
    Returns:
        Dictionary with forecast statistics and paths
    """
    # Initialize simulator
    simulator = MonteCarloSimulator(
        n_simulations=n_simulations,
        method=method
    )
    
    # Run simulation
    paths = simulator.simulate(
        current_price=current_price,
        expected_return=expected_return,
        volatility=volatility,
        horizon_days=horizon_days,
        historical_returns=historical_returns
    )
    
    # Calculate statistics
    stats = simulator.calculate_statistics(paths, current_price)
    
    # Generate plot
    simulator.plot_distribution(
        paths=paths,
        current_price=current_price,
        ticker=ticker,
        horizon_days=horizon_days,
        save_path=save_path
    )
    
    return {
        'paths': paths,
        'statistics': stats
    }
