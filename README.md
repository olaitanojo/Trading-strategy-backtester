# 📈 Trading Strategy Backtester

Comprehensive backtesting framework with multiple strategies, Monte Carlo simulations, and risk-adjusted performance analysis.

## 🚀 Features

- **🔄 Multiple Strategies**: MA Crossover, RSI, MACD, Bollinger Bands
- **🎲 Monte Carlo Simulation**: Statistical performance analysis
- **📊 Risk Metrics**: Sharpe ratio, max drawdown, Calmar ratio
- **📈 Interactive Charts**: Multi-panel performance visualization
- **⚡ Real-time Comparison**: Side-by-side strategy analysis

## 🛠️ Installation

```bash
git clone https://github.com/olaitanojo/trading-strategy-backtester.git
cd trading-strategy-backtester
pip install pandas numpy yfinance plotly
python backtester.py
```

## 📊 Sample Output

```
STRATEGY COMPARISON
Strategy              Total Return    Sharpe Ratio    Max Drawdown
MA_Cross_20_50       12.45%          0.89           -8.23%
RSI_MeanRev_14       15.67%          1.12           -6.45%
MACD_12_26_9         9.87%           0.76           -9.12%
BB_20_2              11.23%          0.95           -7.89%
```

## 🎯 Strategies Included

- **Moving Average Crossover**: Trend-following strategy
- **RSI Mean Reversion**: Overbought/oversold signals
- **MACD Strategy**: Momentum-based trading
- **Bollinger Bands**: Volatility-based entries

---
Created by [olaitanojo](https://github.com/olaitanojo)
