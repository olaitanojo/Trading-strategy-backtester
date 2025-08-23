#!/usr/bin/env python3
"""
Trading Strategy Backtester
A comprehensive backtesting framework with multiple strategies, Monte Carlo simulations,
risk-adjusted returns, and interactive visualization.

Author: olaitanojo
"""

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

@dataclass
class BacktestResult:
    """Results from a backtest run"""
    strategy_name: str
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    total_trades: int
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: Dict

class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals (-1: sell, 0: hold, 1: buy)"""
        raise NotImplementedError

class MovingAverageCrossover(TradingStrategy):
    """Moving Average Crossover Strategy"""
    
    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        super().__init__(f"MA_Cross_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on MA crossover"""
        fast_ma = data['Close'].rolling(window=self.fast_period).mean()
        slow_ma = data['Close'].rolling(window=self.slow_period).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[fast_ma > slow_ma] = 1
        signals[fast_ma < slow_ma] = -1
        
        return signals

class RSIMeanReversion(TradingStrategy):
    """RSI Mean Reversion Strategy"""
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__(f"RSI_MeanRev_{rsi_period}")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on RSI levels"""
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=data.index)
        signals[rsi < self.oversold] = 1   # Buy when oversold
        signals[rsi > self.overbought] = -1 # Sell when overbought
        
        return signals

class MACDStrategy(TradingStrategy):
    """MACD Strategy"""
    
    def __init__(self, fast_ema: int = 12, slow_ema: int = 26, signal_ema: int = 9):
        super().__init__(f"MACD_{fast_ema}_{slow_ema}_{signal_ema}")
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.signal_ema = signal_ema
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on MACD crossover"""
        ema_fast = data['Close'].ewm(span=self.fast_ema).mean()
        ema_slow = data['Close'].ewm(span=self.slow_ema).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=self.signal_ema).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[macd > macd_signal] = 1
        signals[macd < macd_signal] = -1
        
        return signals

class BollingerBandsStrategy(TradingStrategy):
    """Bollinger Bands Strategy"""
    
    def __init__(self, period: int = 20, std_dev: float = 2):
        super().__init__(f"BB_{period}_{std_dev}")
        self.period = period
        self.std_dev = std_dev
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on Bollinger Bands"""
        middle = data['Close'].rolling(window=self.period).mean()
        std = data['Close'].rolling(window=self.period).std()
        upper = middle + (std * self.std_dev)
        lower = middle - (std * self.std_dev)
        
        signals = pd.Series(0, index=data.index)
        signals[data['Close'] < lower] = 1   # Buy at lower band
        signals[data['Close'] > upper] = -1  # Sell at upper band
        
        return signals

class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
    
    def run_backtest(self, data: pd.DataFrame, strategy: TradingStrategy, 
                    start_date: str = None, end_date: str = None) -> BacktestResult:
        """Run backtest for a given strategy"""
        
        # Filter data by date range
        if start_date or end_date:
            mask = pd.Series(True, index=data.index)
            if start_date:
                mask = mask & (data.index >= start_date)
            if end_date:
                mask = mask & (data.index <= end_date)
            data = data[mask]
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Calculate positions and returns
        positions = signals.shift(1).fillna(0)  # Enter position next day
        returns = data['Close'].pct_change()
        
        # Calculate portfolio returns
        portfolio_returns = positions * returns
        
        # Account for commissions
        trades = positions.diff().abs()
        commission_costs = trades * self.commission
        portfolio_returns -= commission_costs
        
        # Calculate equity curve
        equity_curve = (1 + portfolio_returns).cumprod() * self.initial_capital
        
        # Calculate metrics
        total_return = (equity_curve.iloc[-1] / self.initial_capital) - 1
        annual_return = (1 + total_return) ** (252 / len(data)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Calculate max drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calculate Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Calculate win rate
        winning_trades = portfolio_returns[portfolio_returns > 0]
        total_trades = len(trades[trades > 0])
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Create trades DataFrame
        trade_signals = signals[signals != 0].copy()
        trades_df = pd.DataFrame({
            'Date': trade_signals.index,
            'Signal': trade_signals.values,
            'Price': data.loc[trade_signals.index, 'Close']
        })
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'total_trades': total_trades
        }
        
        return BacktestResult(
            strategy_name=strategy.name,
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            total_trades=total_trades,
            equity_curve=equity_curve,
            trades=trades_df,
            metrics=metrics
        )
    
    def monte_carlo_simulation(self, data: pd.DataFrame, strategy: TradingStrategy, 
                              num_simulations: int = 1000) -> Dict:
        """Run Monte Carlo simulation on strategy"""
        
        signals = strategy.generate_signals(data)
        returns = data['Close'].pct_change()
        strategy_returns = signals.shift(1) * returns
        
        # Bootstrap returns for Monte Carlo
        mc_results = []
        
        for _ in range(num_simulations):
            # Sample returns with replacement
            sampled_returns = np.random.choice(
                strategy_returns.dropna(), 
                size=len(strategy_returns.dropna()), 
                replace=True
            )
            
            # Calculate cumulative return
            cumulative_return = (1 + pd.Series(sampled_returns)).prod() - 1
            mc_results.append(cumulative_return)
        
        mc_results = np.array(mc_results)
        
        return {
            'mean_return': np.mean(mc_results),
            'std_return': np.std(mc_results),
            'percentile_5': np.percentile(mc_results, 5),
            'percentile_95': np.percentile(mc_results, 95),
            'probability_positive': np.sum(mc_results > 0) / len(mc_results),
            'all_results': mc_results
        }
    
    def compare_strategies(self, data: pd.DataFrame, strategies: List[TradingStrategy]) -> pd.DataFrame:
        """Compare multiple strategies"""
        results = []
        
        for strategy in strategies:
            result = self.run_backtest(data, strategy)
            results.append({
                'Strategy': result.strategy_name,
                'Total Return': f"{result.total_return:.2%}",
                'Annual Return': f"{result.annual_return:.2%}",
                'Volatility': f"{result.volatility:.2%}",
                'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
                'Max Drawdown': f"{result.max_drawdown:.2%}",
                'Calmar Ratio': f"{result.calmar_ratio:.2f}",
                'Win Rate': f"{result.win_rate:.2%}",
                'Total Trades': result.total_trades
            })
        
        return pd.DataFrame(results)
    
    def create_performance_chart(self, results: List[BacktestResult]) -> go.Figure:
        """Create interactive performance comparison chart"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Equity Curves', 'Return Distribution', 
                          'Drawdown', 'Risk-Return Scatter'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, result in enumerate(results):
            color = colors[i % len(colors)]
            
            # Equity curves
            fig.add_trace(
                go.Scatter(
                    x=result.equity_curve.index,
                    y=result.equity_curve,
                    name=result.strategy_name,
                    line=dict(color=color),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Return distribution (histogram)
            returns = result.equity_curve.pct_change().dropna()
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    name=f"{result.strategy_name} Returns",
                    opacity=0.7,
                    nbinsx=30,
                    showlegend=False,
                    marker_color=color
                ),
                row=1, col=2
            )
            
            # Drawdown
            peak = result.equity_curve.expanding().max()
            drawdown = (result.equity_curve - peak) / peak
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown,
                    name=f"{result.strategy_name} Drawdown",
                    line=dict(color=color),
                    fill='tonexty' if i == 0 else None,
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Risk-Return scatter
            fig.add_trace(
                go.Scatter(
                    x=[result.volatility],
                    y=[result.annual_return],
                    mode='markers',
                    name=result.strategy_name,
                    marker=dict(size=10, color=color),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Strategy Performance Comparison",
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        
        fig.update_xaxes(title_text="Daily Returns", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        fig.update_xaxes(title_text="Volatility", row=2, col=2)
        fig.update_yaxes(title_text="Annual Return", row=2, col=2)
        
        return fig

def main():
    """Main function to demonstrate the backtesting framework"""
    print("="*60)
    print("TRADING STRATEGY BACKTESTER")
    print("="*60)
    
    # Fetch sample data
    ticker = "SPY"
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Initialize strategies
    strategies = [
        MovingAverageCrossover(20, 50),
        RSIMeanReversion(14, 30, 70),
        MACDStrategy(12, 26, 9),
        BollingerBandsStrategy(20, 2)
    ]
    
    # Initialize backtesting engine
    engine = BacktestEngine(initial_capital=100000, commission=0.001)
    
    # Run backtests
    print("\nRunning backtests...")
    results = []
    for strategy in strategies:
        result = engine.run_backtest(data, strategy)
        results.append(result)
        print(f"✓ {strategy.name} completed")
    
    # Display comparison table
    comparison_df = engine.compare_strategies(data, strategies)
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    # Run Monte Carlo simulation for best strategy
    best_strategy = max(results, key=lambda x: x.sharpe_ratio)
    print(f"\n📊 Monte Carlo Simulation for {best_strategy.strategy_name}")
    print("-" * 50)
    
    mc_results = engine.monte_carlo_simulation(data, 
                                             next(s for s in strategies if s.name == best_strategy.strategy_name),
                                             num_simulations=1000)
    
    print(f"Mean Return: {mc_results['mean_return']:.2%}")
    print(f"Standard Deviation: {mc_results['std_return']:.2%}")
    print(f"5th Percentile: {mc_results['percentile_5']:.2%}")
    print(f"95th Percentile: {mc_results['percentile_95']:.2%}")
    print(f"Probability of Positive Return: {mc_results['probability_positive']:.1%}")
    
    # Create and save performance chart
    print(f"\n📈 Creating performance visualization...")
    fig = engine.create_performance_chart(results)
    fig.write_html("strategy_performance_comparison.html")
    print("✓ Performance chart saved as 'strategy_performance_comparison.html'")
    
    # Display individual strategy details
    print(f"\n🏆 BEST STRATEGY: {best_strategy.strategy_name}")
    print("-" * 50)
    print(f"Total Return: {best_strategy.total_return:.2%}")
    print(f"Annual Return: {best_strategy.annual_return:.2%}")
    print(f"Volatility: {best_strategy.volatility:.2%}")
    print(f"Sharpe Ratio: {best_strategy.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {best_strategy.max_drawdown:.2%}")
    print(f"Calmar Ratio: {best_strategy.calmar_ratio:.2f}")
    print(f"Win Rate: {best_strategy.win_rate:.2%}")
    print(f"Total Trades: {best_strategy.total_trades}")
    
    print(f"\nBacktesting completed! Check 'strategy_performance_comparison.html' for detailed charts.")

if __name__ == "__main__":
    main()
