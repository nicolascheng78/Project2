"""
Unit tests for data loader module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.data.data_loader import IndonesianStockDataLoader, MacroDataLoader


class TestIndonesianStockDataLoader:
    """Tests for Indonesian stock data loader."""
    
    def test_initialization(self):
        """Test loader initialization."""
        loader = IndonesianStockDataLoader(
            tickers=['BBCA.JK'],
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        
        assert loader.tickers == ['BBCA.JK']
        assert loader.start_date == '2020-01-01'
        assert loader.end_date == '2020-12-31'
    
    def test_cache_path(self):
        """Test cache path generation."""
        loader = IndonesianStockDataLoader(
            tickers=['BBCA.JK'],
            start_date='2020-01-01',
            cache_dir='/tmp/test_cache'
        )
        
        cache_path = loader._get_cache_path('BBCA.JK')
        assert 'BBCA_JK' in cache_path
        assert '2020-01-01' in cache_path


class TestMacroDataLoader:
    """Tests for macro data loader."""
    
    def test_initialization(self):
        """Test macro loader initialization."""
        loader = MacroDataLoader(
            index_ticker='^JKSE',
            forex_ticker='USDIDR=X',
            start_date='2020-01-01'
        )
        
        assert loader.index_ticker == '^JKSE'
        assert loader.forex_ticker == 'USDIDR=X'
        assert loader.start_date == '2020-01-01'


def test_align_data_to_trading_days():
    """Test data alignment function."""
    from src.data.data_loader import align_data_to_trading_days
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    
    stock_data = pd.DataFrame({
        'Close': np.random.randn(10)
    }, index=dates)
    
    index_data = pd.DataFrame({
        'Close': np.random.randn(10)
    }, index=dates)
    
    aligned_stock, aligned_index = align_data_to_trading_days(stock_data, index_data)
    
    assert len(aligned_stock) == len(aligned_index)
    assert (aligned_stock.index == aligned_index.index).all()
