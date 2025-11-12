"""
Trading strategy implementation using ML model signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class MLTradingStrategy:
    """Trading strategy based on ML model predictions."""
    
    def __init__(
        self,
        min_probability: float = 0.60,
        min_expected_return: float = 0.05,
        max_drawdown: float = 0.15,
        profit_target: float = 0.20,
        position_sizing_method: str = "volatility_target",
        target_volatility: float = 0.15,
        max_position_size: float = 0.20,
        max_positions: int = 10,
        commission: float = 0.0025,
        slippage: float = 0.0015
    ):
        """
        Initialize trading strategy.
        
        Args:
            min_probability: Minimum P(up) for entry
            min_expected_return: Minimum expected return for entry
            max_drawdown: Maximum drawdown before exit
            profit_target: Profit target for exit
            position_sizing_method: Method for position sizing
            target_volatility: Target portfolio volatility
            max_position_size: Maximum position size as % of portfolio
            max_positions: Maximum number of concurrent positions
            commission: Commission rate per trade
            slippage: Slippage rate per trade
        """
        self.min_probability = min_probability
        self.min_expected_return = min_expected_return
        self.max_drawdown = max_drawdown
        self.profit_target = profit_target
        self.position_sizing_method = position_sizing_method
        self.target_volatility = target_volatility
        self.max_position_size = max_position_size
        self.max_positions = max_positions
        self.commission = commission
        self.slippage = slippage
        self.total_cost = commission + slippage
    
    def calculate_position_size(
        self,
        probability: float,
        expected_return: float,
        volatility: float,
        portfolio_value: float
    ) -> float:
        """
        Calculate position size based on strategy parameters.
        
        Args:
            probability: Predicted probability of up move
            expected_return: Expected return
            volatility: Stock volatility
            portfolio_value: Current portfolio value
            
        Returns:
            Position size in dollars
        """
        if self.position_sizing_method == "equal":
            # Equal weighting
            size = portfolio_value * self.max_position_size
            
        elif self.position_sizing_method == "volatility_target":
            # Inverse volatility weighting
            if volatility > 0:
                size = (self.target_volatility / volatility) * portfolio_value
                size = min(size, portfolio_value * self.max_position_size)
            else:
                size = portfolio_value * self.max_position_size
                
        elif self.position_sizing_method == "kelly":
            # Kelly criterion (simplified)
            # f = (p * (1 + expected_return) - 1) / expected_return
            if expected_return > 0:
                kelly_fraction = (probability * (1 + expected_return) - 1) / expected_return
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
                size = kelly_fraction * portfolio_value
            else:
                size = 0
        else:
            size = portfolio_value * self.max_position_size
        
        return size
    
    def should_enter(
        self,
        probability: float,
        expected_return: float,
        current_positions: int
    ) -> bool:
        """
        Determine if we should enter a position.
        
        Args:
            probability: Predicted probability of up move
            expected_return: Expected return
            current_positions: Current number of positions
            
        Returns:
            True if should enter, False otherwise
        """
        # Check position limit
        if current_positions >= self.max_positions:
            return False
        
        # Check probability threshold
        if probability < self.min_probability:
            return False
        
        # Check expected return threshold (must exceed costs)
        cost_adjusted_return = expected_return - (2 * self.total_cost)
        if cost_adjusted_return < self.min_expected_return:
            return False
        
        return True
    
    def should_exit(
        self,
        entry_price: float,
        current_price: float,
        max_price_since_entry: float
    ) -> Tuple[bool, str]:
        """
        Determine if we should exit a position.
        
        Args:
            entry_price: Entry price
            current_price: Current price
            max_price_since_entry: Maximum price since entry
            
        Returns:
            Tuple of (should_exit, reason)
        """
        # Calculate current return
        current_return = (current_price / entry_price) - 1
        
        # Check profit target
        if current_return >= self.profit_target:
            return True, "profit_target"
        
        # Check drawdown from peak
        drawdown_from_peak = (current_price / max_price_since_entry) - 1
        if drawdown_from_peak <= -self.max_drawdown:
            return True, "max_drawdown"
        
        # Check stop loss (loss exceeds max drawdown)
        if current_return <= -self.max_drawdown:
            return True, "stop_loss"
        
        return False, None
    
    def generate_signals(
        self,
        predictions_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        volatility_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate trading signals from model predictions.
        
        Args:
            predictions_df: DataFrame with columns ['date', 'ticker', 'probability', 'expected_return']
            prices_df: DataFrame with price data
            volatility_df: DataFrame with volatility estimates
            
        Returns:
            DataFrame with trading signals
        """
        signals = predictions_df.copy()
        
        # Calculate position sizes
        signals['position_size'] = signals.apply(
            lambda row: self.calculate_position_size(
                row['probability'],
                row['expected_return'],
                volatility_df.loc[row.name] if row.name in volatility_df.index else 0.20,
                100000  # Placeholder portfolio value
            ),
            axis=1
        )
        
        # Generate entry signals
        signals['signal'] = signals.apply(
            lambda row: 1 if self.should_enter(
                row['probability'],
                row['expected_return'],
                0  # Will be updated in backtest
            ) else 0,
            axis=1
        )
        
        return signals


class PortfolioManager:
    """Manage portfolio positions and track performance."""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        max_positions: int = 10
    ):
        """
        Initialize portfolio manager.
        
        Args:
            initial_capital: Starting capital
            max_positions: Maximum concurrent positions
        """
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.cash = initial_capital
        self.positions = {}  # ticker -> position dict
        self.closed_positions = []
        self.equity_curve = []
        self.dates = []
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value."""
        value = self.cash
        
        for ticker, position in self.positions.items():
            if ticker in current_prices:
                value += position['shares'] * current_prices[ticker]
        
        return value
    
    def open_position(
        self,
        ticker: str,
        entry_price: float,
        position_size: float,
        date: pd.Timestamp,
        cost_rate: float = 0.004
    ):
        """Open a new position."""
        if len(self.positions) >= self.max_positions:
            return False
        
        if ticker in self.positions:
            return False
        
        # Calculate shares (accounting for costs)
        total_cost = position_size * (1 + cost_rate)
        if total_cost > self.cash:
            position_size = self.cash / (1 + cost_rate)
            total_cost = position_size * (1 + cost_rate)
        
        shares = position_size / entry_price
        
        # Update cash
        self.cash -= total_cost
        
        # Create position
        self.positions[ticker] = {
            'ticker': ticker,
            'shares': shares,
            'entry_price': entry_price,
            'entry_date': date,
            'max_price': entry_price,
            'cost': total_cost - position_size  # Track costs
        }
        
        return True
    
    def close_position(
        self,
        ticker: str,
        exit_price: float,
        date: pd.Timestamp,
        reason: str = "signal",
        cost_rate: float = 0.004
    ):
        """Close an existing position."""
        if ticker not in self.positions:
            return False
        
        position = self.positions[ticker]
        
        # Calculate proceeds
        proceeds = position['shares'] * exit_price
        costs = proceeds * cost_rate
        net_proceeds = proceeds - costs
        
        # Update cash
        self.cash += net_proceeds
        
        # Calculate return
        total_cost = position['shares'] * position['entry_price'] + position['cost'] + costs
        total_return = (proceeds - total_cost) / (position['shares'] * position['entry_price'])
        
        # Record closed position
        self.closed_positions.append({
            'ticker': ticker,
            'entry_date': position['entry_date'],
            'exit_date': date,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'shares': position['shares'],
            'return': total_return,
            'pnl': net_proceeds - (position['shares'] * position['entry_price'] + position['cost']),
            'exit_reason': reason
        })
        
        # Remove position
        del self.positions[ticker]
        
        return True
    
    def update_positions(
        self,
        current_prices: Dict[str, float],
        date: pd.Timestamp
    ):
        """Update position tracking."""
        for ticker, position in self.positions.items():
            if ticker in current_prices:
                current_price = current_prices[ticker]
                position['max_price'] = max(position['max_price'], current_price)
        
        # Record equity curve
        portfolio_value = self.get_portfolio_value(current_prices)
        self.equity_curve.append(portfolio_value)
        self.dates.append(date)
    
    def get_summary(self) -> Dict[str, any]:
        """Get portfolio performance summary."""
        if not self.closed_positions:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'cagr': 0
            }
        
        closed_df = pd.DataFrame(self.closed_positions)
        
        total_return = (self.equity_curve[-1] / self.initial_capital - 1) if self.equity_curve else 0
        
        # Calculate CAGR
        days = (self.dates[-1] - self.dates[0]).days if len(self.dates) > 1 else 1
        years = days / 365.25
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        return {
            'total_trades': len(self.closed_positions),
            'win_rate': (closed_df['return'] > 0).mean(),
            'avg_return': closed_df['return'].mean(),
            'total_return': total_return,
            'cagr': cagr,
            'total_pnl': closed_df['pnl'].sum(),
            'best_trade': closed_df['return'].max() if len(closed_df) > 0 else 0,
            'worst_trade': closed_df['return'].min() if len(closed_df) > 0 else 0
        }
