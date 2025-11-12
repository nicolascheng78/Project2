# Indonesian Stock ML Prediction System

A state-of-the-art machine learning system for predicting Indonesian stock market movements, complete with risk assessment, Monte Carlo simulations, and backtested trading strategies.

## 🎯 Overview

This system provides comprehensive stock predictions for Indonesian (.JK) tickers using advanced machine learning techniques:

- **Directional Prediction**: Up/Down classification with calibrated probabilities
- **Return Forecasting**: Expected returns and target price predictions
- **Risk Assessment**: Probability of loss, expected drawdown, VaR/CVaR
- **Monte Carlo Simulation**: Future price distributions with percentile bands
- **Trading Strategy**: Backtested strategy targeting >15% CAGR after transaction costs

## 🌟 Key Features

### Data Pipeline
- ✅ Robust yfinance integration with proper `auto_adjust` handling
- ✅ Support for .JK (Indonesian) tickers with OHLCV data
- ✅ Macro/exogenous data: Jakarta Composite Index (^JKSE), USDIDR forex
- ✅ Top 50-100 liquid stocks to ensure tradability
- ✅ Historical membership tracking to avoid survivorship bias
- ✅ Data caching for efficient reruns

### Feature Engineering
- ✅ Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands
- ✅ Momentum features: Returns, ROC, momentum across multiple windows
- ✅ Volatility features: Historical volatility, ATR, Parkinson volatility
- ✅ Volume features: Volume ratios, VWAP, OBV
- ✅ Relative features: Beta, correlation, relative strength vs index
- ✅ Market regime indicators: Trend, volatility regime
- ✅ Proper time-series handling with lag shifting to prevent leakage

### Machine Learning Models
- ✅ **Classification Head**: Direction prediction (Up/Down) with probability calibration (isotonic/Platt)
- ✅ **Regression Head**: Forward return prediction with quantile regression for uncertainty
- ✅ Gradient Boosting: LightGBM and XGBoost implementations
- ✅ Feature importance analysis for interpretability
- ✅ Time-series cross-validation for robust evaluation

### Monte Carlo Simulation
- ✅ Geometric Brownian Motion (GBM) for price path generation
- ✅ Bootstrap method using historical residuals
- ✅ Distribution visualization: P10/P50/P90 percentile bands
- ✅ Risk metrics: Loss probability, expected max drawdown, VaR, CVaR

### Trading Strategy & Backtesting
- ✅ Signal generation: Probability and expected return thresholds
- ✅ Position sizing: Volatility targeting, Kelly criterion
- ✅ Risk management: Stop loss, profit targets, drawdown limits
- ✅ Realistic costs: 40 bps total (commission + slippage)
- ✅ Walk-forward time-series backtesting
- ✅ Performance metrics: CAGR, Sharpe, Sortino, max drawdown, hit rate

### User Interface
- ✅ Command-line interface for quick predictions
- ✅ Jupyter notebook for interactive analysis
- ✅ Visualization utilities for results and diagnostics

## 📁 Project Structure

```
Project2/
├── config.yaml                 # System configuration
├── requirements.txt           # Python dependencies
├── cli.py                     # Command-line interface
├── README.md                  # This file
│
├── src/                       # Source code
│   ├── data/
│   │   └── data_loader.py    # Data loading with yfinance
│   ├── features/
│   │   └── features.py       # Feature engineering
│   ├── models/
│   │   ├── models.py         # ML models (classification & regression)
│   │   └── monte_carlo.py    # Monte Carlo simulation
│   ├── strategy/
│   │   └── strategy.py       # Trading strategy logic
│   └── backtest/
│       └── backtest.py       # Backtesting engine
│
├── notebooks/
│   └── training_pipeline.ipynb  # End-to-end training notebook
│
├── tests/                     # Unit tests (to be added)
│
└── artifacts/                 # Generated outputs
    ├── data/                  # Cached data
    ├── models/                # Trained models
    ├── plots/                 # Visualizations
    └── results/               # Backtest results
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda for package management

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/nicolascheng78/Project2.git
cd Project2
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Create necessary directories:**
```bash
mkdir -p artifacts/{data,models,plots,results}
```

### Quick Start

#### 1. Train Models

Run the Jupyter notebook to train the prediction models:

```bash
jupyter notebook notebooks/training_pipeline.ipynb
```

The notebook will:
- Download historical data for Indonesian stocks
- Generate technical and fundamental features
- Train classification and regression models
- Evaluate model performance
- Save trained models to `artifacts/models/`

#### 2. Make Predictions

Use the CLI to get predictions for any ticker:

```bash
# Basic prediction
python cli.py predict --ticker BBCA.JK --horizon 21

# Custom horizon (days)
python cli.py predict --ticker TLKM.JK --horizon 63

# See all available options
python cli.py predict --help
```

**Output includes:**
- Direction prediction (Up/Down) with probability
- Target price and expected return
- Risk metrics (loss probability, expected drawdown)
- Monte Carlo price distribution
- Trading recommendation with confidence level

#### 3. View Configuration

```bash
# List configured tickers
python cli.py list-tickers

# Show full configuration
python cli.py show-config
```

## 📊 Example Usage

### Prediction Example

```bash
python cli.py predict --ticker BBCA.JK --horizon 21
```

**Sample Output:**
```
============================================================
Indonesian Stock Prediction System
============================================================

Loading data for BBCA.JK...
Generating features...
Loading models...
Making predictions...
Running Monte Carlo simulation...

============================================================
PREDICTION RESULTS FOR BBCA.JK
============================================================

Current Information:
  Date: 2024-01-15
  Current Price: 8,750.00 IDR
  Horizon: 21 trading days

Direction Prediction:
  Direction: UP
  Probability (Up): 67.5%
  Probability (Down): 32.5%

Return & Price Prediction:
  Expected Return: 8.20%
  Target Price: 9,467.50 IDR

Risk Assessment:
  Probability of Loss: 28.3%
  Expected Max Drawdown: -4.2%
  95% Worst Drawdown: -12.8%
  VaR (95%): -6.5%
  CVaR (95%): -9.2%

Monte Carlo Price Distribution:
  Expected Price: 9,420.00 IDR
  Median Price: 9,390.00 IDR
  P10: 8,100.00 IDR
  P90: 10,850.00 IDR

============================================================
RECOMMENDATION
============================================================

  Recommendation: BUY
  Confidence: MEDIUM
  Reason: High probability of upward movement (67.5%) with positive expected return

  Note: This is a model prediction and should not be considered
        as financial advice. Always do your own research and
        consider your risk tolerance.
```

## 🎛️ Configuration

Edit `config.yaml` to customize:

### Data Configuration
- Ticker universe (stocks to analyze)
- Date range for historical data
- Top N liquid stocks to focus on

### Feature Engineering
- Technical indicator windows
- Momentum/volatility/volume windows
- Feature lag periods

### Model Configuration
- Model type (LightGBM, XGBoost)
- Hyperparameters
- Calibration method
- Quantile regression settings

### Trading Strategy
- Entry thresholds (probability, expected return)
- Exit rules (profit target, stop loss)
- Position sizing method
- Maximum positions

### Backtesting
- Train/test windows
- Transaction costs
- Performance targets

## 📈 Model Performance

The system is designed to achieve:
- **Target CAGR**: >15% after transaction costs
- **Sharpe Ratio**: >1.0
- **Hit Rate**: >55%
- **Max Drawdown**: <20%

Performance may vary based on market conditions and selected parameters.

## 🔬 Technical Details

### Data Handling
- **Auto-adjust handling**: Uses `Close` when `auto_adjust=True`, otherwise `Adj Close`
- **Survivorship bias**: Tracks historical liquidity to avoid bias
- **Missing data**: Forward-fill for infrequent trading days
- **Alignment**: Ensures features and targets are properly aligned

### Feature Engineering
- **Leakage prevention**: All features shifted by 1 period
- **Rolling windows**: Ensures no future information
- **Normalization**: StandardScaler for stable training

### Model Training
- **Time-series split**: No shuffling, respects temporal order
- **Early stopping**: Prevents overfitting on validation set
- **Calibration**: Isotonic regression for well-calibrated probabilities
- **Uncertainty**: Quantile regression at 10th, 50th, 90th percentiles

### Backtesting
- **Walk-forward**: Rolling train/test windows
- **Realistic costs**: 25 bps commission + 15 bps slippage
- **Liquidity constraints**: Max 5% of ADV
- **Position sizing**: Volatility targeting or Kelly criterion

## 🛠️ Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

```bash
# Format code
black src/ cli.py

# Lint
flake8 src/ cli.py
```

## 📝 Assumptions and Limitations

### Assumptions
1. **Market efficiency**: Predictable patterns exist in Indonesian stocks
2. **Transaction costs**: Realistic estimate at 40 bps round-trip
3. **Liquidity**: Focus on top 50-100 liquid names ensures tradability
4. **Data quality**: yfinance data is accurate and reliable
5. **Stationarity**: Features maintain predictive power over time

### Limitations
1. **Fundamental data**: Limited fundamental data integration (P/E, P/B not included)
2. **News/sentiment**: Does not incorporate news or sentiment analysis
3. **Macro events**: May not capture sudden regime changes or crises
4. **Forex risk**: USDIDR included but not hedged in strategy
5. **Slippage**: Actual slippage may vary with order size
6. **Model decay**: Requires periodic retraining as market evolves

### Risk Warnings
⚠️ **This system is for educational and research purposes only.**
- Past performance does not guarantee future results
- All trading involves risk of loss
- This is NOT financial advice
- Always do your own research
- Consider consulting a licensed financial advisor

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📄 License

This project is provided as-is for educational purposes.

## 📧 Contact

For questions or issues, please open a GitHub issue.

## 🙏 Acknowledgments

- yfinance for data access
- LightGBM and XGBoost teams for excellent ML libraries
- Indonesian Stock Exchange for market data availability

---

**Disclaimer**: This software is provided for educational and research purposes only. It is not intended as financial advice. Trading stocks involves risk, and you should never invest more than you can afford to lose. Always conduct your own research and consider seeking advice from a licensed financial advisor.