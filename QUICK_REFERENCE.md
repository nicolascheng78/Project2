# Quick Reference Guide

## Installation

```bash
# Clone repository
git clone https://github.com/nicolascheng78/Project2.git
cd Project2

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p artifacts/{data,models,plots,results}
```

## Common Commands

### 1. Validate System
```bash
python validate.py
```
Runs system-wide checks to ensure everything works.

### 2. List Available Tickers
```bash
python cli.py list-tickers
```

### 3. Show Configuration
```bash
python cli.py show-config
```

### 4. Make Prediction
```bash
# Basic prediction (1-month horizon)
python cli.py predict --ticker BBCA.JK

# Custom horizon (days)
python cli.py predict --ticker BBCA.JK --horizon 63

# With custom config
python cli.py predict --ticker BBCA.JK --config my_config.yaml
```

### 5. Train Models
```bash
# Open Jupyter notebook
jupyter notebook notebooks/training_pipeline.ipynb

# Or run example script
python example.py
```

### 6. Run Tests
```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_features.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## File Locations

- **Config**: `config.yaml`
- **Models**: `artifacts/models/`
- **Data Cache**: `artifacts/data/`
- **Plots**: `artifacts/plots/`
- **Results**: `artifacts/results/`

## Key Configuration Parameters

Edit `config.yaml` to customize:

### Data
- `data.universe`: List of tickers to analyze
- `data.start_date`: Historical data start date
- `data.top_n_liquid`: Number of liquid stocks to select

### Models
- `models.classification.model_type`: 'lightgbm' or 'xgboost'
- `models.classification.params`: Model hyperparameters
- `models.regression.quantile_regression`: Enable uncertainty estimation

### Strategy
- `strategy.entry.min_probability`: Entry threshold (default: 0.60)
- `strategy.entry.min_expected_return`: Min return threshold (default: 0.05)
- `strategy.position_sizing.method`: 'equal', 'volatility_target', or 'kelly'
- `strategy.max_positions`: Maximum concurrent positions

### Backtesting
- `backtest.train_window`: Training window in days (default: 1260)
- `backtest.test_window`: Test window in days (default: 252)
- `backtest.costs.total_roundtrip`: Transaction costs (default: 0.004)

## Prediction Output Explained

When you run `python cli.py predict --ticker BBCA.JK`, you get:

### 1. Direction Prediction
- **Direction**: UP or DOWN
- **Probability (Up)**: Calibrated probability of upward movement
- Interpretation: >60% = Strong signal, 50-60% = Weak signal, <50% = Down

### 2. Return & Price Prediction
- **Expected Return**: Model's predicted return
- **Target Price**: Current price × (1 + expected return)

### 3. Risk Assessment
- **Probability of Loss**: Chance price falls below entry
- **Expected Max Drawdown**: Average worst drawdown in simulations
- **95% Worst Drawdown**: 95th percentile worst case
- **VaR (95%)**: Value at Risk at 95% confidence
- **CVaR (95%)**: Conditional VaR (expected loss when VaR is exceeded)

### 4. Monte Carlo Distribution
- **P10/P90**: 10th and 90th percentile price outcomes
- Gives you range of possible outcomes

### 5. Recommendation
- **BUY**: High probability + positive expected return
- **SELL/AVOID**: Low probability or negative return
- **HOLD/NEUTRAL**: Uncertain or insufficient edge

## Performance Metrics Explained

### CAGR (Compound Annual Growth Rate)
Annual return if compounded. Target: >15%

### Sharpe Ratio
Risk-adjusted return (return per unit of volatility). Target: >1.0

### Sortino Ratio
Like Sharpe but only penalizes downside volatility. Higher is better.

### Max Drawdown
Worst peak-to-trough decline. Target: <20%

### Hit Rate
Percentage of winning trades. Target: >55%

### Calmar Ratio
CAGR / |Max Drawdown|. Higher is better.

## Troubleshooting

### "No data available for ticker"
- Check ticker format (should be TICKER.JK for Indonesian stocks)
- Ensure internet connection for yfinance
- Try different start date in config

### "Models not found"
- Train models first using notebook or example.py
- Check models are saved in `artifacts/models/`

### "Insufficient data"
- Reduce `min_history_days` in data loader
- Use more recent start_date in config

### Import errors
- Run `pip install -r requirements.txt`
- Ensure you're in project root directory

## Tips for Best Results

1. **Data Quality**: Use at least 3 years of data for training
2. **Retraining**: Retrain models monthly or quarterly
3. **Validation**: Always check model metrics before using
4. **Risk Management**: Never risk more than you can afford to lose
5. **Diversification**: Don't put all capital in one prediction
6. **Cost Awareness**: Transaction costs matter - don't overtrade

## Advanced Usage

### Custom Feature Engineering
Edit `src/features/features.py` to add new features:
```python
def add_custom_features(self, df):
    df['my_feature'] = ...  # Your calculation
    return df
```

### Hyperparameter Optimization
Modify `config.yaml` → `hpo.enabled: true` and run backtesting

### Walk-Forward Optimization
Use `src/backtest/backtest.py` WalkForwardBacktest class

### Multiple Horizons
Train separate models for different horizons (21, 63, 126 days)

## Support

- **Documentation**: See README.md and PROJECT_SUMMARY.md
- **Issues**: Open GitHub issue
- **Tests**: Run `pytest tests/ -v` to verify installation

## Disclaimer

⚠️ This is NOT financial advice. For educational purposes only.
Past performance does not guarantee future results. Always do your own research.
