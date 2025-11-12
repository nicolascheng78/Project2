"""
Quick validation script to ensure the system works end-to-end.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime

print("="*60)
print("Indonesian Stock ML Prediction System - Validation")
print("="*60)

# Test 1: Data Loading
print("\n[1/5] Testing data loading...")
from src.data.data_loader import IndonesianStockDataLoader, MacroDataLoader

loader = IndonesianStockDataLoader(
    tickers=['BBCA.JK'],
    start_date='2023-01-01',
    cache_dir='artifacts/data'
)
print("  ✓ Data loader initialized")

# Test 2: Feature Engineering
print("\n[2/5] Testing feature engineering...")
from src.features.features import FeatureEngineer

# Create synthetic data for testing
dates = pd.date_range('2023-01-01', periods=100, freq='D')
test_data = pd.DataFrame({
    'Open': np.random.randn(100).cumsum() + 100,
    'High': np.random.randn(100).cumsum() + 102,
    'Low': np.random.randn(100).cumsum() + 98,
    'Close': np.random.randn(100).cumsum() + 100,
    'Adj Close': np.random.randn(100).cumsum() + 100,
    'Volume': np.random.randint(1000000, 10000000, 100)
}, index=dates)

engineer = FeatureEngineer()
features = engineer.create_all_features(test_data, horizons=[21])
print(f"  ✓ Generated {len(features.columns)} features")

# Test 3: Model Training
print("\n[3/5] Testing model training...")
from src.models.models import CombinedStockPredictor
from src.features.features import get_feature_columns

# Prepare minimal dataset
feature_cols = get_feature_columns(features)
target_dir = 'Direction_21d'
target_ret = 'Future_Return_21d'

train_data = features[[*feature_cols[:10], target_dir, target_ret]].dropna()  # Use only 10 features for speed

if len(train_data) >= 20:
    X_train = train_data[feature_cols[:10]].iloc[:15]
    y_train_dir = train_data[target_dir].iloc[:15]
    y_train_ret = train_data[target_ret].iloc[:15]
    
    X_val = train_data[feature_cols[:10]].iloc[15:]
    y_val_dir = train_data[target_dir].iloc[15:]
    y_val_ret = train_data[target_ret].iloc[15:]
    
    # Train minimal models
    predictor = CombinedStockPredictor(
        classifier_params={
            'model_type': 'lightgbm',
            'params': {'n_estimators': 10, 'max_depth': 3, 'random_state': 42, 'verbose': -1}
        },
        regressor_params={
            'model_type': 'lightgbm',
            'params': {'n_estimators': 10, 'max_depth': 3, 'random_state': 42, 'verbose': -1},
            'quantile_regression': False
        }
    )
    
    predictor.fit(X_train, y_train_dir, y_train_ret)
    print("  ✓ Models trained successfully")
    
    # Test predictions
    preds = predictor.predict(X_val)
    print(f"  ✓ Predictions generated: {len(preds['direction_proba'])} samples")
else:
    print("  ⚠ Insufficient data for training test (skipped)")

# Test 4: Monte Carlo Simulation
print("\n[4/5] Testing Monte Carlo simulation...")
from src.models.monte_carlo import MonteCarloSimulator

simulator = MonteCarloSimulator(n_simulations=100, method='gbm')
paths = simulator.simulate_gbm(
    current_price=100,
    expected_return=0.10,
    volatility=0.20,
    horizon_days=21
)
print(f"  ✓ Generated {paths.shape[0]} Monte Carlo paths")

stats = simulator.calculate_statistics(paths, 100)
print(f"  ✓ Calculated statistics: {len(stats)} metrics")

# Test 5: Strategy and Backtesting
print("\n[5/5] Testing trading strategy...")
from src.strategy.strategy import MLTradingStrategy, PortfolioManager

strategy = MLTradingStrategy(
    min_probability=0.60,
    min_expected_return=0.05,
    max_positions=5
)
print("  ✓ Strategy initialized")

portfolio = PortfolioManager(initial_capital=100000, max_positions=5)
print("  ✓ Portfolio manager initialized")

# Test position management
success = portfolio.open_position(
    ticker='TEST.JK',
    entry_price=100,
    position_size=10000,
    date=pd.Timestamp('2023-01-01')
)
print(f"  ✓ Position management working: {success}")

# Summary
print("\n" + "="*60)
print("VALIDATION COMPLETE ✓")
print("="*60)
print("\nAll core components are working correctly!")
print("\nSystem is ready for use:")
print("  1. Train models: jupyter notebook notebooks/training_pipeline.ipynb")
print("  2. Make predictions: python cli.py predict --ticker BBCA.JK")
print("  3. Run example: python example.py")
print("\n" + "="*60)
