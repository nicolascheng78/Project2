"""
Command-line interface for stock prediction system.
"""

import click
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime

from src.data.data_loader import IndonesianStockDataLoader, MacroDataLoader
from src.features.features import FeatureEngineer
from src.models.models import CombinedStockPredictor
from src.models.monte_carlo import generate_price_forecast


@click.group()
def cli():
    """Indonesian Stock ML Prediction System CLI."""
    pass


@cli.command()
@click.option('--ticker', '-t', required=True, help='Stock ticker (e.g., BBCA.JK)')
@click.option('--horizon', '-h', type=int, default=21, help='Prediction horizon in days (default: 21)')
@click.option('--config', '-c', default='config.yaml', help='Config file path')
@click.option('--models-dir', '-m', default='artifacts/models', help='Models directory')
def predict(ticker: str, horizon: int, config: str, models_dir: str):
    """
    Make prediction for a specific ticker.
    
    Example:
        python cli.py predict --ticker BBCA.JK --horizon 21
    """
    click.echo(f"\n{'='*60}")
    click.echo(f"Indonesian Stock Prediction System")
    click.echo(f"{'='*60}\n")
    
    # Load config
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    click.echo(f"Loading data for {ticker}...")
    
    # Load stock data
    loader = IndonesianStockDataLoader(
        tickers=[ticker],
        start_date=cfg['data']['start_date'],
        cache_dir=cfg['data']['cache_dir']
    )
    
    stock_data = loader.download_ticker(ticker)
    
    if stock_data.empty:
        click.echo(f"Error: No data available for {ticker}")
        return
    
    # Load index data
    macro_loader = MacroDataLoader(
        index_ticker=cfg['data']['macro_tickers']['index'],
        start_date=cfg['data']['start_date'],
        cache_dir=cfg['data']['cache_dir']
    )
    
    index_data = macro_loader.download_index()
    
    # Generate features
    click.echo("Generating features...")
    
    feature_engineer = FeatureEngineer(
        technical_windows=cfg['features']['technical_windows'],
        momentum_windows=cfg['features']['momentum_windows'],
        volatility_windows=cfg['features']['volatility_windows'],
        volume_windows=cfg['features']['volume_windows'],
        shift_periods=cfg['features']['shift_periods']
    )
    
    features_df = feature_engineer.create_all_features(
        stock_data,
        index_data if not index_data.empty else None,
        horizons=[horizon]
    )
    
    # Get latest data point
    features_df = features_df.dropna()
    
    if len(features_df) == 0:
        click.echo("Error: Not enough data to generate features")
        return
    
    latest_data = features_df.iloc[[-1]]
    current_price = stock_data['Adj Close'].iloc[-1]
    current_date = stock_data.index[-1]
    
    # Load models
    click.echo("Loading models...")
    
    classifier_path = os.path.join(models_dir, f'classifier_{horizon}d.pkl')
    regressor_path = os.path.join(models_dir, f'regressor_{horizon}d.pkl')
    
    if not os.path.exists(classifier_path) or not os.path.exists(regressor_path):
        click.echo(f"Error: Models not found for horizon {horizon} days")
        click.echo(f"Please train models first using the training notebook")
        return
    
    predictor = CombinedStockPredictor()
    predictor.load(classifier_path, regressor_path)
    
    # Get feature columns
    from src.features.features import get_feature_columns
    feature_cols = get_feature_columns(latest_data)
    X = latest_data[feature_cols]
    
    # Make predictions
    click.echo("Making predictions...")
    
    predictions = predictor.predict(X)
    
    # Extract results
    prob_up = predictions['direction_proba'][0]
    expected_return = predictions['expected_return'][0]
    direction = "UP" if prob_up > 0.5 else "DOWN"
    
    # Calculate target price
    target_price = current_price * (1 + expected_return)
    
    # Calculate volatility for Monte Carlo
    returns = stock_data['Adj Close'].pct_change().dropna()
    volatility = returns.tail(60).std() * np.sqrt(252)
    
    # Run Monte Carlo simulation
    click.echo("Running Monte Carlo simulation...")
    
    mc_results = generate_price_forecast(
        ticker=ticker,
        current_price=current_price,
        expected_return=expected_return,
        volatility=volatility,
        horizon_days=horizon,
        n_simulations=cfg['monte_carlo']['n_simulations'],
        method=cfg['monte_carlo']['method'],
        save_path=os.path.join(cfg['output']['plots_dir'], f'{ticker}_{horizon}d_forecast.png')
    )
    
    stats = mc_results['statistics']
    
    # Display results
    click.echo(f"\n{'='*60}")
    click.echo(f"PREDICTION RESULTS FOR {ticker}")
    click.echo(f"{'='*60}\n")
    
    click.echo(f"Current Information:")
    click.echo(f"  Date: {current_date.strftime('%Y-%m-%d')}")
    click.echo(f"  Current Price: {current_price:.2f} IDR")
    click.echo(f"  Horizon: {horizon} trading days\n")
    
    click.echo(f"Direction Prediction:")
    click.echo(f"  Direction: {direction}")
    click.echo(f"  Probability (Up): {prob_up*100:.1f}%")
    click.echo(f"  Probability (Down): {(1-prob_up)*100:.1f}%\n")
    
    click.echo(f"Return & Price Prediction:")
    click.echo(f"  Expected Return: {expected_return*100:.2f}%")
    click.echo(f"  Target Price: {target_price:.2f} IDR\n")
    
    click.echo(f"Risk Assessment:")
    click.echo(f"  Probability of Loss: {stats['prob_loss']*100:.1f}%")
    click.echo(f"  Expected Max Drawdown: {stats['expected_max_drawdown']*100:.2f}%")
    click.echo(f"  95% Worst Drawdown: {stats['worst_drawdown_95']*100:.2f}%")
    click.echo(f"  VaR (95%): {stats['VaR_95']*100:.2f}%")
    click.echo(f"  CVaR (95%): {stats['CVaR_95']*100:.2f}%\n")
    
    click.echo(f"Monte Carlo Price Distribution:")
    click.echo(f"  Expected Price: {stats['mean_price']:.2f} IDR")
    click.echo(f"  Median Price: {stats['median_price']:.2f} IDR")
    click.echo(f"  P10: {stats['percentiles'][10]:.2f} IDR")
    click.echo(f"  P90: {stats['percentiles'][90]:.2f} IDR\n")
    
    # Generate recommendation
    click.echo(f"{'='*60}")
    click.echo(f"RECOMMENDATION")
    click.echo(f"{'='*60}\n")
    
    # Determine confidence level
    if prob_up >= 0.70:
        confidence = "HIGH"
    elif prob_up >= 0.60:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    
    # Generate recommendation
    if prob_up >= 0.60 and expected_return > 0.05:
        recommendation = "BUY"
        reason = f"High probability of upward movement ({prob_up*100:.1f}%) with positive expected return"
    elif prob_up < 0.40 or expected_return < -0.05:
        recommendation = "SELL/AVOID"
        reason = f"Low probability of upward movement ({prob_up*100:.1f}%) or negative expected return"
    else:
        recommendation = "HOLD/NEUTRAL"
        reason = "Uncertain direction or insufficient expected return"
    
    click.echo(f"  Recommendation: {recommendation}")
    click.echo(f"  Confidence: {confidence}")
    click.echo(f"  Reason: {reason}\n")
    
    click.echo(f"  Note: This is a model prediction and should not be considered")
    click.echo(f"        as financial advice. Always do your own research and")
    click.echo(f"        consider your risk tolerance.\n")
    
    click.echo(f"Monte Carlo visualization saved to:")
    click.echo(f"  {os.path.join(cfg['output']['plots_dir'], f'{ticker}_{horizon}d_forecast.png')}\n")


@cli.command()
@click.option('--config', '-c', default='config.yaml', help='Config file path')
def list_tickers(config: str):
    """List available tickers in the universe."""
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    click.echo("\nConfigured Tickers:")
    click.echo("=" * 40)
    
    for i, ticker in enumerate(cfg['data']['universe'], 1):
        click.echo(f"{i:2d}. {ticker}")
    
    click.echo(f"\nTotal: {len(cfg['data']['universe'])} tickers\n")


@cli.command()
@click.option('--config', '-c', default='config.yaml', help='Config file path')
def show_config(config: str):
    """Display current configuration."""
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    click.echo("\nCurrent Configuration:")
    click.echo("=" * 60)
    click.echo(yaml.dump(cfg, default_flow_style=False))


if __name__ == '__main__':
    cli()
