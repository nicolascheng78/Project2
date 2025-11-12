"""
Feature engineering module for Indonesian stock prediction.
Implements technical, momentum, volatility, volume, and relative features.
"""

import numpy as np
import pandas as pd
from typing import List, Optional


class FeatureEngineer:
    """Generate features for stock prediction with proper time-series handling."""
    
    def __init__(
        self,
        technical_windows: List[int] = [5, 10, 20, 50, 120],
        momentum_windows: List[int] = [5, 10, 20, 60],
        volatility_windows: List[int] = [10, 20, 60],
        volume_windows: List[int] = [5, 20, 60],
        shift_periods: int = 1
    ):
        """
        Initialize feature engineer.
        
        Args:
            technical_windows: Window sizes for technical indicators
            momentum_windows: Window sizes for momentum features
            volatility_windows: Window sizes for volatility features
            volume_windows: Window sizes for volume features
            shift_periods: Number of periods to shift features (avoid leakage)
        """
        self.technical_windows = technical_windows
        self.momentum_windows = momentum_windows
        self.volatility_windows = volatility_windows
        self.volume_windows = volume_windows
        self.shift_periods = shift_periods
    
    def add_technical_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'Adj Close'
    ) -> pd.DataFrame:
        """
        Add technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands.
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price
            
        Returns:
            DataFrame with added technical features
        """
        result = df.copy()
        
        # Simple and Exponential Moving Averages
        for window in self.technical_windows:
            result[f'SMA_{window}'] = result[price_col].rolling(window=window).mean()
            result[f'EMA_{window}'] = result[price_col].ewm(span=window, adjust=False).mean()
            
            # Price relative to moving average
            result[f'Price_to_SMA_{window}'] = result[price_col] / result[f'SMA_{window}'] - 1
            result[f'Price_to_EMA_{window}'] = result[price_col] / result[f'EMA_{window}'] - 1
        
        # RSI (Relative Strength Index)
        for window in [14, 30]:  # Standard RSI periods
            delta = result[price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            result[f'RSI_{window}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = result[price_col].ewm(span=12, adjust=False).mean()
        ema_26 = result[price_col].ewm(span=26, adjust=False).mean()
        result['MACD'] = ema_12 - ema_26
        result['MACD_Signal'] = result['MACD'].ewm(span=9, adjust=False).mean()
        result['MACD_Hist'] = result['MACD'] - result['MACD_Signal']
        
        # Bollinger Bands
        for window in [20]:
            sma = result[price_col].rolling(window=window).mean()
            std = result[price_col].rolling(window=window).std()
            result[f'BB_Upper_{window}'] = sma + (2 * std)
            result[f'BB_Lower_{window}'] = sma - (2 * std)
            result[f'BB_Width_{window}'] = (result[f'BB_Upper_{window}'] - result[f'BB_Lower_{window}']) / sma
            result[f'BB_Position_{window}'] = (result[price_col] - result[f'BB_Lower_{window}']) / (
                result[f'BB_Upper_{window}'] - result[f'BB_Lower_{window}']
            )
        
        return result
    
    def add_momentum_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'Adj Close'
    ) -> pd.DataFrame:
        """
        Add momentum features: returns, momentum, rate of change.
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price
            
        Returns:
            DataFrame with added momentum features
        """
        result = df.copy()
        
        # Returns over different periods
        for window in self.momentum_windows:
            result[f'Return_{window}d'] = result[price_col].pct_change(periods=window)
            
            # Log returns
            result[f'LogReturn_{window}d'] = np.log(result[price_col] / result[price_col].shift(window))
        
        # Rate of Change (ROC)
        for window in self.momentum_windows:
            result[f'ROC_{window}d'] = (
                (result[price_col] - result[price_col].shift(window)) / result[price_col].shift(window)
            ) * 100
        
        # Momentum (current price - price N days ago)
        for window in self.momentum_windows:
            result[f'Momentum_{window}d'] = result[price_col] - result[price_col].shift(window)
        
        return result
    
    def add_volatility_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'Adj Close'
    ) -> pd.DataFrame:
        """
        Add volatility features: historical volatility, ATR, parkinson volatility.
        
        Args:
            df: DataFrame with OHLC data
            price_col: Column name for price
            
        Returns:
            DataFrame with added volatility features
        """
        result = df.copy()
        
        # Historical volatility (rolling std of returns)
        for window in self.volatility_windows:
            returns = result[price_col].pct_change()
            result[f'Volatility_{window}d'] = returns.rolling(window=window).std() * np.sqrt(252)
        
        # Average True Range (ATR)
        if all(col in result.columns for col in ['High', 'Low', 'Close']):
            high_low = result['High'] - result['Low']
            high_close = np.abs(result['High'] - result['Close'].shift())
            low_close = np.abs(result['Low'] - result['Close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            for window in [14, 20]:
                result[f'ATR_{window}'] = true_range.rolling(window=window).mean()
                result[f'ATR_Pct_{window}'] = result[f'ATR_{window}'] / result['Close']
        
        # Parkinson volatility (uses High and Low)
        if all(col in result.columns for col in ['High', 'Low']):
            for window in self.volatility_windows:
                hl_ratio = np.log(result['High'] / result['Low'])
                result[f'Parkinson_Vol_{window}d'] = (
                    np.sqrt(hl_ratio.rolling(window=window).var() / (4 * np.log(2))) * np.sqrt(252)
                )
        
        return result
    
    def add_volume_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'Adj Close'
    ) -> pd.DataFrame:
        """
        Add volume-based features: volume ratios, VWAP, OBV.
        
        Args:
            df: DataFrame with volume data
            price_col: Column name for price
            
        Returns:
            DataFrame with added volume features
        """
        result = df.copy()
        
        if 'Volume' not in result.columns:
            return result
        
        # Volume moving averages and ratios
        for window in self.volume_windows:
            result[f'Volume_SMA_{window}'] = result['Volume'].rolling(window=window).mean()
            result[f'Volume_Ratio_{window}'] = result['Volume'] / result[f'Volume_SMA_{window}']
        
        # Dollar volume (liquidity)
        result['Dollar_Volume'] = result[price_col] * result['Volume']
        for window in self.volume_windows:
            result[f'Dollar_Volume_SMA_{window}'] = result['Dollar_Volume'].rolling(window=window).mean()
        
        # On-Balance Volume (OBV)
        price_change = result[price_col].diff()
        volume_direction = np.where(price_change > 0, result['Volume'], 
                                    np.where(price_change < 0, -result['Volume'], 0))
        result['OBV'] = volume_direction.cumsum()
        
        # OBV moving averages
        for window in [20, 50]:
            result[f'OBV_SMA_{window}'] = result['OBV'].rolling(window=window).mean()
        
        # Volume-Weighted Average Price (VWAP)
        if all(col in result.columns for col in ['High', 'Low', 'Close', 'Volume']):
            typical_price = (result['High'] + result['Low'] + result['Close']) / 3
            result['VWAP'] = (typical_price * result['Volume']).cumsum() / result['Volume'].cumsum()
            
            # Rolling VWAP
            for window in [20]:
                result[f'VWAP_{window}'] = (
                    (typical_price * result['Volume']).rolling(window=window).sum() /
                    result['Volume'].rolling(window=window).sum()
                )
                result[f'Price_to_VWAP_{window}'] = result[price_col] / result[f'VWAP_{window}'] - 1
        
        return result
    
    def add_relative_features(
        self,
        df: pd.DataFrame,
        index_df: pd.DataFrame,
        price_col: str = 'Adj Close'
    ) -> pd.DataFrame:
        """
        Add features relative to market index.
        
        Args:
            df: Stock DataFrame
            index_df: Index DataFrame
            price_col: Column name for price
            
        Returns:
            DataFrame with relative features
        """
        result = df.copy()
        
        # Align dates
        common_dates = result.index.intersection(index_df.index)
        result = result.loc[common_dates]
        aligned_index = index_df.loc[common_dates]
        
        # Relative strength vs index
        stock_returns = result[price_col].pct_change()
        index_returns = aligned_index['Close'].pct_change()
        
        for window in [5, 20, 60]:
            stock_perf = (1 + stock_returns).rolling(window=window).apply(np.prod, raw=True) - 1
            index_perf = (1 + index_returns).rolling(window=window).apply(np.prod, raw=True) - 1
            result[f'Relative_Strength_{window}d'] = stock_perf - index_perf
        
        # Beta (rolling)
        for window in [60, 120]:
            covariance = stock_returns.rolling(window=window).cov(index_returns)
            variance = index_returns.rolling(window=window).var()
            result[f'Beta_{window}d'] = covariance / variance
        
        # Correlation with index
        for window in [20, 60]:
            result[f'Correlation_Index_{window}d'] = stock_returns.rolling(window=window).corr(index_returns)
        
        return result
    
    def add_market_regime_features(
        self,
        df: pd.DataFrame,
        index_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add market regime indicators.
        
        Args:
            df: Stock DataFrame
            index_df: Index DataFrame
            
        Returns:
            DataFrame with regime features
        """
        result = df.copy()
        
        # Align dates
        common_dates = result.index.intersection(index_df.index)
        aligned_index = index_df.loc[common_dates]
        
        # Market trend (index moving averages)
        index_close = aligned_index['Close']
        for window in [50, 200]:
            ma = index_close.rolling(window=window).mean()
            result[f'Market_Above_MA{window}'] = (index_close > ma).astype(int)
        
        # Market volatility regime
        index_returns = index_close.pct_change()
        for window in [20, 60]:
            vol = index_returns.rolling(window=window).std() * np.sqrt(252)
            vol_ma = vol.rolling(window=60).mean()
            result[f'Market_High_Vol_{window}d'] = (vol > vol_ma).astype(int)
        
        # VIX-like indicator (not real VIX for Indonesia, but similar concept)
        result['Market_Volatility_20d'] = index_returns.rolling(window=20).std() * np.sqrt(252) * 100
        
        return result
    
    def create_target_variables(
        self,
        df: pd.DataFrame,
        horizons: List[int] = [21, 63, 126],
        price_col: str = 'Adj Close'
    ) -> pd.DataFrame:
        """
        Create target variables for prediction (future returns and direction).
        
        Args:
            df: DataFrame with price data
            horizons: List of forward-looking periods (trading days)
            price_col: Column name for price
            
        Returns:
            DataFrame with target variables
        """
        result = df.copy()
        
        for horizon in horizons:
            # Future price
            result[f'Future_Price_{horizon}d'] = result[price_col].shift(-horizon)
            
            # Future return
            result[f'Future_Return_{horizon}d'] = (
                result[f'Future_Price_{horizon}d'] / result[price_col] - 1
            )
            
            # Future log return
            result[f'Future_LogReturn_{horizon}d'] = np.log(
                result[f'Future_Price_{horizon}d'] / result[price_col]
            )
            
            # Direction (Up=1, Down=0)
            result[f'Direction_{horizon}d'] = (result[f'Future_Return_{horizon}d'] > 0).astype(int)
        
        return result
    
    def shift_features(
        self,
        df: pd.DataFrame,
        exclude_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Shift features to avoid lookahead bias.
        
        Args:
            df: DataFrame with features
            exclude_cols: Columns to exclude from shifting (e.g., targets, identifiers)
            
        Returns:
            DataFrame with shifted features
        """
        result = df.copy()
        
        if exclude_cols is None:
            exclude_cols = []
        
        # Add typical columns to exclude
        exclude_patterns = ['Future_', 'Direction_', 'Target_', 'Ticker', 'Date']
        
        # Determine which columns to shift
        cols_to_shift = []
        for col in result.columns:
            if col in exclude_cols:
                continue
            if any(pattern in col for pattern in exclude_patterns):
                continue
            if col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']:
                continue  # Keep raw OHLCV unshifted for reference
            cols_to_shift.append(col)
        
        # Shift the features
        for col in cols_to_shift:
            result[col] = result[col].shift(self.shift_periods)
        
        return result
    
    def create_all_features(
        self,
        df: pd.DataFrame,
        index_df: Optional[pd.DataFrame] = None,
        horizons: List[int] = [21, 63, 126],
        price_col: str = 'Adj Close'
    ) -> pd.DataFrame:
        """
        Create all features in one pipeline.
        
        Args:
            df: Stock DataFrame with OHLCV data
            index_df: Optional index DataFrame for relative features
            horizons: Prediction horizons
            price_col: Price column name
            
        Returns:
            DataFrame with all features
        """
        result = df.copy()
        
        # Add all feature groups
        result = self.add_technical_features(result, price_col)
        result = self.add_momentum_features(result, price_col)
        result = self.add_volatility_features(result, price_col)
        result = self.add_volume_features(result, price_col)
        
        # Add relative features if index data provided
        if index_df is not None:
            result = self.add_relative_features(result, index_df, price_col)
            result = self.add_market_regime_features(result, index_df)
        
        # Create targets (these should NOT be shifted)
        result = self.create_target_variables(result, horizons, price_col)
        
        # Shift features to avoid leakage
        result = self.shift_features(result)
        
        return result


def get_feature_columns(df: pd.DataFrame, exclude_patterns: List[str] = None) -> List[str]:
    """
    Get list of feature columns, excluding targets and metadata.
    
    Args:
        df: DataFrame with features
        exclude_patterns: Patterns to exclude from features
        
    Returns:
        List of feature column names
    """
    if exclude_patterns is None:
        exclude_patterns = ['Future_', 'Direction_', 'Target_', 'Ticker']
    
    # Also exclude raw OHLCV if present
    exclude_exact = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Date']
    
    feature_cols = []
    for col in df.columns:
        # Skip if matches exclude pattern
        if any(pattern in col for pattern in exclude_patterns):
            continue
        # Skip if in exact exclude list
        if col in exclude_exact:
            continue
        feature_cols.append(col)
    
    return feature_cols
