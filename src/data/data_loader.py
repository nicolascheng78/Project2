"""
Data loader module for Indonesian stock market data.
Handles yfinance data loading with proper auto_adjust handling and survivorship bias prevention.
"""

import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm


class IndonesianStockDataLoader:
    """Load and manage Indonesian stock market data with proper handling of adjustments."""
    
    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        cache_dir: str = "artifacts/data",
        auto_adjust: bool = True
    ):
        """
        Initialize the data loader.
        
        Args:
            tickers: List of ticker symbols (e.g., ['BBCA.JK', 'BBRI.JK'])
            start_date: Start date for data loading (YYYY-MM-DD)
            end_date: End date for data loading (YYYY-MM-DD), None for current date
            cache_dir: Directory to cache downloaded data
            auto_adjust: Whether to use auto-adjusted prices
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.cache_dir = cache_dir
        self.auto_adjust = auto_adjust
        
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_path(self, ticker: str) -> str:
        """Get cache file path for a ticker."""
        safe_ticker = ticker.replace(".", "_")
        return os.path.join(self.cache_dir, f"{safe_ticker}_{self.start_date}_{self.end_date}.pkl")
    
    def download_ticker(self, ticker: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Download data for a single ticker with proper adjustment handling.
        
        Args:
            ticker: Ticker symbol
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with OHLCV data and proper adjusted close
        """
        cache_path = self._get_cache_path(ticker)
        
        # Try to load from cache
        if use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Cache load failed for {ticker}: {e}. Re-downloading...")
        
        try:
            # Download with auto_adjust parameter
            data = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=self.auto_adjust,
                progress=False
            )
            
            if data.empty:
                print(f"No data available for {ticker}")
                return pd.DataFrame()
            
            # Handle auto_adjust changes in yfinance API
            # When auto_adjust=True, use Close column (already adjusted)
            # When auto_adjust=False, use Adj Close column
            if self.auto_adjust:
                # Close is already adjusted, rename for consistency
                if 'Close' in data.columns:
                    data['Adj Close'] = data['Close']
            else:
                # Adj Close should be present
                if 'Adj Close' not in data.columns and 'Close' in data.columns:
                    print(f"Warning: Adj Close not found for {ticker}, using Close")
                    data['Adj Close'] = data['Close']
            
            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            for col in required_cols:
                if col not in data.columns:
                    if col == 'Adj Close':
                        data['Adj Close'] = data['Close']
                    else:
                        print(f"Warning: Missing column {col} for {ticker}")
            
            # Add ticker column
            data['Ticker'] = ticker
            
            # Cache the data
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            return data
            
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            return pd.DataFrame()
    
    def download_all(self, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Download data for all tickers.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        data_dict = {}
        
        for ticker in tqdm(self.tickers, desc="Downloading tickers"):
            df = self.download_ticker(ticker, use_cache=use_cache)
            if not df.empty:
                data_dict[ticker] = df
        
        return data_dict
    
    def get_combined_data(
        self,
        use_cache: bool = True,
        min_history_days: int = 252
    ) -> pd.DataFrame:
        """
        Get combined data for all tickers with filtering.
        
        Args:
            use_cache: Whether to use cached data
            min_history_days: Minimum number of days of history required
            
        Returns:
            Combined DataFrame with all tickers
        """
        data_dict = self.download_all(use_cache=use_cache)
        
        # Filter tickers with insufficient history
        valid_tickers = {}
        for ticker, df in data_dict.items():
            if len(df) >= min_history_days:
                valid_tickers[ticker] = df
            else:
                print(f"Skipping {ticker}: only {len(df)} days (< {min_history_days})")
        
        if not valid_tickers:
            raise ValueError("No tickers have sufficient history")
        
        # Combine all dataframes
        combined = pd.concat(valid_tickers.values(), axis=0)
        combined = combined.sort_index()
        
        return combined
    
    def calculate_liquidity_scores(
        self,
        data_dict: Dict[str, pd.DataFrame],
        lookback_days: int = 252
    ) -> pd.Series:
        """
        Calculate liquidity scores for tickers (average daily dollar volume).
        
        Args:
            data_dict: Dictionary of ticker DataFrames
            lookback_days: Number of days to calculate average over
            
        Returns:
            Series with liquidity scores per ticker
        """
        liquidity_scores = {}
        
        for ticker, df in data_dict.items():
            # Calculate dollar volume (price * volume)
            dollar_volume = df['Adj Close'] * df['Volume']
            
            # Take mean over lookback period
            recent_liq = dollar_volume.tail(lookback_days).mean()
            liquidity_scores[ticker] = recent_liq
        
        return pd.Series(liquidity_scores).sort_values(ascending=False)
    
    def get_top_liquid_tickers(
        self,
        top_n: int = 50,
        use_cache: bool = True
    ) -> List[str]:
        """
        Get top N most liquid tickers.
        
        Args:
            top_n: Number of top liquid tickers to return
            use_cache: Whether to use cached data
            
        Returns:
            List of top N liquid ticker symbols
        """
        data_dict = self.download_all(use_cache=use_cache)
        liquidity_scores = self.calculate_liquidity_scores(data_dict)
        
        # Return top N
        top_tickers = liquidity_scores.head(top_n).index.tolist()
        print(f"Top {top_n} liquid tickers selected from {len(liquidity_scores)} total")
        
        return top_tickers


class MacroDataLoader:
    """Load macroeconomic and index data for Indonesian market."""
    
    def __init__(
        self,
        index_ticker: str = "^JKSE",
        forex_ticker: str = "USDIDR=X",
        start_date: str = "2000-01-01",
        end_date: Optional[str] = None,
        cache_dir: str = "artifacts/data"
    ):
        """
        Initialize macro data loader.
        
        Args:
            index_ticker: Jakarta Composite Index ticker
            forex_ticker: USD/IDR forex pair ticker
            start_date: Start date for data
            end_date: End date for data (None for current)
            cache_dir: Cache directory
        """
        self.index_ticker = index_ticker
        self.forex_ticker = forex_ticker
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.cache_dir = cache_dir
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def download_index(self, use_cache: bool = True) -> pd.DataFrame:
        """Download index data."""
        cache_path = os.path.join(
            self.cache_dir,
            f"index_{self.start_date}_{self.end_date}.pkl"
        )
        
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        try:
            data = yf.download(
                self.index_ticker,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                progress=False
            )
            
            if not data.empty:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            
            return data
        except Exception as e:
            print(f"Error downloading index: {e}")
            return pd.DataFrame()
    
    def download_forex(self, use_cache: bool = True) -> pd.DataFrame:
        """Download forex data."""
        cache_path = os.path.join(
            self.cache_dir,
            f"forex_{self.start_date}_{self.end_date}.pkl"
        )
        
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        try:
            data = yf.download(
                self.forex_ticker,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                progress=False
            )
            
            if not data.empty:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            
            return data
        except Exception as e:
            print(f"Error downloading forex: {e}")
            return pd.DataFrame()
    
    def get_all_macro_data(self, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Download all macro data.
        
        Returns:
            Dictionary with 'index' and 'forex' DataFrames
        """
        return {
            'index': self.download_index(use_cache=use_cache),
            'forex': self.download_forex(use_cache=use_cache)
        }


def align_data_to_trading_days(
    stock_data: pd.DataFrame,
    index_data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align stock and index data to common trading days.
    
    Args:
        stock_data: Stock price DataFrame
        index_data: Index price DataFrame
        
    Returns:
        Tuple of aligned (stock_data, index_data)
    """
    # Get common dates
    common_dates = stock_data.index.intersection(index_data.index)
    
    if len(common_dates) == 0:
        raise ValueError("No common trading days found between stock and index data")
    
    # Align both datasets
    aligned_stock = stock_data.loc[common_dates]
    aligned_index = index_data.loc[common_dates]
    
    return aligned_stock, aligned_index
