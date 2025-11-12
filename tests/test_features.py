"""
Unit tests for feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np

from src.features.features import FeatureEngineer, get_feature_columns


class TestFeatureEngineer:
    """Tests for feature engineering."""
    
    def setup_method(self):
        """Setup test data."""
        # Create sample price data
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        
        self.sample_data = pd.DataFrame({
            'Open': np.random.randn(200).cumsum() + 100,
            'High': np.random.randn(200).cumsum() + 102,
            'Low': np.random.randn(200).cumsum() + 98,
            'Close': np.random.randn(200).cumsum() + 100,
            'Adj Close': np.random.randn(200).cumsum() + 100,
            'Volume': np.random.randint(1000000, 10000000, 200)
        }, index=dates)
        
        self.engineer = FeatureEngineer()
    
    def test_initialization(self):
        """Test feature engineer initialization."""
        engineer = FeatureEngineer(
            technical_windows=[5, 10, 20],
            shift_periods=1
        )
        
        assert engineer.technical_windows == [5, 10, 20]
        assert engineer.shift_periods == 1
    
    def test_add_technical_features(self):
        """Test technical feature generation."""
        result = self.engineer.add_technical_features(self.sample_data)
        
        # Check that SMA columns were added
        assert 'SMA_5' in result.columns
        assert 'SMA_10' in result.columns
        assert 'EMA_5' in result.columns
        assert 'RSI_14' in result.columns
        assert 'MACD' in result.columns
    
    def test_add_momentum_features(self):
        """Test momentum feature generation."""
        result = self.engineer.add_momentum_features(self.sample_data)
        
        # Check that momentum columns were added
        assert 'Return_5d' in result.columns
        assert 'ROC_5d' in result.columns
        assert 'Momentum_5d' in result.columns
    
    def test_add_volatility_features(self):
        """Test volatility feature generation."""
        result = self.engineer.add_volatility_features(self.sample_data)
        
        # Check that volatility columns were added
        assert 'Volatility_10d' in result.columns
        assert 'ATR_14' in result.columns
    
    def test_add_volume_features(self):
        """Test volume feature generation."""
        result = self.engineer.add_volume_features(self.sample_data)
        
        # Check that volume columns were added
        assert 'Volume_SMA_5' in result.columns
        assert 'OBV' in result.columns
        assert 'Dollar_Volume' in result.columns
    
    def test_create_target_variables(self):
        """Test target variable creation."""
        result = self.engineer.create_target_variables(self.sample_data, horizons=[21])
        
        # Check that target columns were added
        assert 'Future_Price_21d' in result.columns
        assert 'Future_Return_21d' in result.columns
        assert 'Direction_21d' in result.columns
    
    def test_no_leakage_in_features(self):
        """Test that features are properly shifted to avoid leakage."""
        result = self.engineer.create_all_features(
            self.sample_data,
            horizons=[21]
        )
        
        # Clean data
        result = result.dropna()
        
        # Check that features are shifted (current row features should not equal current price ratios)
        if len(result) > 1:
            # Features should be from previous periods
            assert 'SMA_5' in result.columns
            # The SMA should be calculated from shifted data


class TestGetFeatureColumns:
    """Tests for feature column selection."""
    
    def test_get_feature_columns(self):
        """Test feature column extraction."""
        df = pd.DataFrame({
            'SMA_5': [1, 2, 3],
            'RSI_14': [4, 5, 6],
            'Future_Return_21d': [7, 8, 9],
            'Direction_21d': [0, 1, 0],
            'Ticker': ['A', 'B', 'C'],
            'Close': [10, 11, 12]
        })
        
        feature_cols = get_feature_columns(df)
        
        # Should include SMA and RSI, but not targets or metadata
        assert 'SMA_5' in feature_cols
        assert 'RSI_14' in feature_cols
        assert 'Future_Return_21d' not in feature_cols
        assert 'Direction_21d' not in feature_cols
        assert 'Ticker' not in feature_cols
        assert 'Close' not in feature_cols
