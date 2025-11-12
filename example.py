"""
Example script demonstrating the Indonesian Stock ML Prediction System.

This script shows how to:
1. Load data for Indonesian stocks
2. Generate features
3. Train models
4. Make predictions
5. Run Monte Carlo simulations
"""

import warnings
warnings.filterwarnings('ignore')

import yaml
import pandas as pd
import numpy as np
from datetime import datetime

from src.data.data_loader import IndonesianStockDataLoader, MacroDataLoader
from src.features.features import FeatureEngineer, get_feature_columns
from src.models.models import CombinedStockPredictor
from src.models.monte_carlo import generate_price_forecast

# Load configuration
print("Loading configuration...")
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Select a few tickers for demo
DEMO_TICKERS = ['BBCA.JK', 'BBRI.JK', 'TLKM.JK']
HORIZON = 21  # 1 month

print(f"\nDemo: Training models for {DEMO_TICKERS}")
print(f"Prediction horizon: {HORIZON} days\n")

# Step 1: Load Data
print("Step 1: Loading stock data...")
stock_loader = IndonesianStockDataLoader(
    tickers=DEMO_TICKERS,
    start_date='2020-01-01',  # Use recent data for faster demo
    cache_dir=config['data']['cache_dir']
)

stock_data_dict = stock_loader.download_all(use_cache=True)
print(f"Loaded {len(stock_data_dict)} tickers")

# Load index data
print("\nLoading index data...")
macro_loader = MacroDataLoader(
    index_ticker=config['data']['macro_tickers']['index'],
    start_date='2020-01-01',
    cache_dir=config['data']['cache_dir']
)

index_data = macro_loader.download_index(use_cache=True)
print(f"Index data: {len(index_data)} days")

# Step 2: Generate Features
print("\nStep 2: Generating features...")
feature_engineer = FeatureEngineer(
    technical_windows=config['features']['technical_windows'],
    momentum_windows=config['features']['momentum_windows'],
    volatility_windows=config['features']['volatility_windows'],
    volume_windows=config['features']['volume_windows'],
    shift_periods=config['features']['shift_periods']
)

all_features = []
for ticker, data in stock_data_dict.items():
    print(f"  Processing {ticker}...")
    try:
        features = feature_engineer.create_all_features(
            data,
            index_data,
            horizons=[HORIZON]
        )
        features['Ticker'] = ticker
        all_features.append(features)
    except Exception as e:
        print(f"  Error processing {ticker}: {e}")

# Combine features
features_df = pd.concat(all_features, axis=0)
features_df = features_df.sort_index()
print(f"\nCombined features shape: {features_df.shape}")

# Step 3: Prepare Training Data
print("\nStep 3: Preparing training data...")
feature_cols = get_feature_columns(features_df)
target_direction_col = f'Direction_{HORIZON}d'
target_return_col = f'Future_Return_{HORIZON}d'

# Remove rows with missing targets
train_data = features_df[[*feature_cols, target_direction_col, target_return_col, 'Ticker']].dropna()
print(f"Training data: {train_data.shape}")

# Split data (80/20)
train_size = int(len(train_data) * 0.8)
train_df = train_data.iloc[:train_size]
val_df = train_data.iloc[train_size:]

X_train = train_df[feature_cols]
y_train_dir = train_df[target_direction_col]
y_train_ret = train_df[target_return_col]

X_val = val_df[feature_cols]
y_val_dir = val_df[target_direction_col]
y_val_ret = val_df[target_return_col]

print(f"Train: {len(X_train)}, Validation: {len(X_val)}")

# Step 4: Train Models
print("\nStep 4: Training ML models...")
predictor = CombinedStockPredictor(
    classifier_params={
        'model_type': 'lightgbm',
        'params': {
            'n_estimators': 100,  # Reduced for demo
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': -1
        },
        'calibration_method': 'isotonic'
    },
    regressor_params={
        'model_type': 'lightgbm',
        'params': {
            'n_estimators': 100,  # Reduced for demo
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': -1
        },
        'quantile_regression': True,
        'quantiles': [0.1, 0.5, 0.9]
    }
)

predictor.fit(
    X_train, y_train_dir, y_train_ret,
    X_val, y_val_dir, y_val_ret
)

# Step 5: Evaluate Models
print("\nStep 5: Evaluating models...")
print("\nClassification Performance:")
val_metrics = predictor.classifier.evaluate(X_val, y_val_dir)
for metric, value in val_metrics.items():
    print(f"  {metric}: {value:.4f}")

print("\nRegression Performance:")
val_metrics = predictor.regressor.evaluate(X_val, y_val_ret)
for metric, value in val_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Step 6: Make Predictions
print("\nStep 6: Making predictions for latest data...")
for ticker in DEMO_TICKERS:
    if ticker not in stock_data_dict:
        continue
    
    # Get latest data
    ticker_features = features_df[features_df['Ticker'] == ticker]
    ticker_features = ticker_features[feature_cols].dropna()
    
    if len(ticker_features) == 0:
        continue
    
    latest = ticker_features.iloc[[-1]]
    current_price = stock_data_dict[ticker]['Adj Close'].iloc[-1]
    
    # Predict
    pred = predictor.predict(latest)
    
    prob_up = pred['direction_proba'][0]
    expected_return = pred['expected_return'][0]
    target_price = current_price * (1 + expected_return)
    
    print(f"\n{ticker}:")
    print(f"  Current Price: {current_price:.2f}")
    print(f"  Direction: {'UP' if prob_up > 0.5 else 'DOWN'}")
    print(f"  Probability (Up): {prob_up*100:.1f}%")
    print(f"  Expected Return: {expected_return*100:.2f}%")
    print(f"  Target Price: {target_price:.2f}")

# Step 7: Monte Carlo Simulation Example
print("\nStep 7: Running Monte Carlo simulation for", DEMO_TICKERS[0])
ticker = DEMO_TICKERS[0]

if ticker in stock_data_dict:
    data = stock_data_dict[ticker]
    current_price = data['Adj Close'].iloc[-1]
    returns = data['Adj Close'].pct_change().dropna()
    volatility = returns.tail(60).std() * np.sqrt(252)
    
    # Get prediction
    ticker_features = features_df[features_df['Ticker'] == ticker]
    ticker_features = ticker_features[feature_cols].dropna()
    
    if len(ticker_features) > 0:
        latest = ticker_features.iloc[[-1]]
        pred = predictor.predict(latest)
        expected_return = pred['expected_return'][0]
        
        print(f"  Current Price: {current_price:.2f}")
        print(f"  Expected Return: {expected_return*100:.2f}%")
        print(f"  Volatility: {volatility*100:.2f}%")
        print(f"  Running simulation...")
        
        mc_results = generate_price_forecast(
            ticker=ticker,
            current_price=current_price,
            expected_return=expected_return,
            volatility=volatility,
            horizon_days=HORIZON,
            n_simulations=1000,  # Reduced for demo
            method='gbm',
            save_path=f'artifacts/plots/{ticker}_demo_forecast.png'
        )
        
        stats = mc_results['statistics']
        print(f"\n  Monte Carlo Results:")
        print(f"    Expected Price: {stats['mean_price']:.2f}")
        print(f"    P10: {stats['percentiles'][10]:.2f}")
        print(f"    P90: {stats['percentiles'][90]:.2f}")
        print(f"    Probability of Loss: {stats['prob_loss']*100:.1f}%")
        print(f"    Expected Max Drawdown: {stats['expected_max_drawdown']*100:.2f}%")

print("\n" + "="*60)
print("Demo completed successfully!")
print("="*60)
print("\nNext steps:")
print("1. Review the generated plots in artifacts/plots/")
print("2. Run the full training notebook: jupyter notebook notebooks/training_pipeline.ipynb")
print("3. Use the CLI for predictions: python cli.py predict --ticker BBCA.JK")
