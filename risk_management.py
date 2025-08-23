#!/usr/bin/env python3
"""
Risk Management Module
Advanced risk management tools including position sizing, stop-loss/take-profit,
portfolio allocation, and sophisticated risk metrics.

Author: olaitanojo
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RiskMetrics:
    """Container for comprehensive risk metrics"""
    var_95: float  # Value at Risk (95%)
    var_99: float  # Value at Risk (99%)
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    maximum_drawdown: float
    calmar_ratio: float
    sortino_ratio: float
    sharpe_ratio: float
    volatility: float
    skewness: float
    kurtosis: float
    beta: Optional[float] = None
    alpha: Optional[float] = None
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None


class PositionSizer:
    """Advanced position sizing algorithms"""
    
    @staticmethod
    def fixed_fractional(capital: float, risk_per_trade: float, 
                        entry_price: float, stop_loss: float) -> int:
        """Fixed fractional position sizing based on risk per trade"""
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0
        
        risk_amount = capital * risk_per_trade
        position_size = int(risk_amount / risk_per_share)
        return max(0, position_size)
    
    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Kelly Criterion for optimal position sizing"""
        if avg_loss == 0:
            return 0
        
        b = avg_win / avg_loss  # Win/loss ratio
        p = win_rate  # Probability of winning
        q = 1 - p     # Probability of losing
        
        kelly_f = (b * p - q) / b
        return max(0, min(kelly_f, 0.25))  # Cap at 25% for safety
    
    @staticmethod
    def volatility_targeting(returns: pd.Series, target_vol: float = 0.15,
                           lookback: int = 252) -> pd.Series:
        """Volatility targeting position sizing"""
        realized_vol = returns.rolling(window=lookback).std() * np.sqrt(252)
        vol_scalar = target_vol / realized_vol
        return vol_scalar.fillna(1.0).clip(0, 2)  # Limit leverage to 2x
    
    @staticmethod
    def atr_position_sizing(prices: pd.Series, atr: pd.Series, 
                          risk_per_trade: float, capital: float) -> pd.Series:
        """Position sizing based on Average True Range"""
        risk_amount = capital * risk_per_trade
        position_size = risk_amount / (atr * 2)  # 2x ATR stop
        return position_size.fillna(0)


class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, initial_capital: float = 100000, max_risk_per_trade: float = 0.02,
                 max_portfolio_risk: float = 0.06, max_correlation: float = 0.8):
        self.initial_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlation = max_correlation
        self.position_sizer = PositionSizer()
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk using historical simulation"""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(returns) == 0:
            return 0.0
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def calculate_maximum_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown from equity curve"""
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min()
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_std = downside_returns.std() * np.sqrt(252)
        if downside_std == 0:
            return np.inf
        
        return (excess_returns.mean() * 252) / downside_std
    
    def calculate_comprehensive_metrics(self, returns: pd.Series, 
                                      benchmark_returns: Optional[pd.Series] = None,
                                      risk_free_rate: float = 0.02) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        if len(returns) == 0:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Basic metrics
        annual_returns = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        
        # Risk metrics
        var_95 = self.calculate_var(returns, 0.95)
        var_99 = self.calculate_var(returns, 0.99)
        cvar_95 = self.calculate_cvar(returns, 0.95)
        
        # Equity curve for drawdown
        equity_curve = (1 + returns).cumprod()
        max_dd = self.calculate_maximum_drawdown(equity_curve)
        
        # Ratios
        sharpe = (annual_returns - risk_free_rate) / volatility if volatility > 0 else 0
        sortino = self.calculate_sortino_ratio(returns, risk_free_rate)
        calmar = annual_returns / abs(max_dd) if max_dd != 0 else 0
        
        # Higher moments
        skewness = stats.skew(returns.dropna())
        kurtosis = stats.kurtosis(returns.dropna())
        
        # Benchmark-relative metrics
        beta = None
        alpha = None
        tracking_error = None
        information_ratio = None
        
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            # Align data
            common_index = returns.index.intersection(benchmark_returns.index)
            ret_aligned = returns.reindex(common_index)
            bench_aligned = benchmark_returns.reindex(common_index)
            
            if len(ret_aligned) > 1 and ret_aligned.std() > 0 and bench_aligned.std() > 0:
                beta = ret_aligned.cov(bench_aligned) / bench_aligned.var()
                alpha = (ret_aligned.mean() - bench_aligned.mean()) * 252 - beta * (bench_aligned.mean() * 252 - risk_free_rate)
                
                excess_ret = ret_aligned - bench_aligned
                tracking_error = excess_ret.std() * np.sqrt(252)
                information_ratio = (excess_ret.mean() * 252) / tracking_error if tracking_error > 0 else 0
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            maximum_drawdown=max_dd,
            calmar_ratio=calmar,
            sortino_ratio=sortino,
            sharpe_ratio=sharpe,
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis,
            beta=beta,
            alpha=alpha,
            tracking_error=tracking_error,
            information_ratio=information_ratio
        )
    
    def apply_stop_loss_take_profit(self, prices: pd.Series, entry_price: float,
                                   stop_loss_pct: float = 0.05, 
                                   take_profit_pct: float = 0.10,
                                   position_type: str = 'long') -> Tuple[pd.Series, pd.Series]:
        """Apply stop loss and take profit to price series"""
        
        stop_loss_level = entry_price * (1 - stop_loss_pct) if position_type == 'long' else entry_price * (1 + stop_loss_pct)
        take_profit_level = entry_price * (1 + take_profit_pct) if position_type == 'long' else entry_price * (1 - take_profit_pct)
        
        stop_signals = pd.Series(0, index=prices.index)
        profit_signals = pd.Series(0, index=prices.index)
        
        if position_type == 'long':
            stop_signals[prices <= stop_loss_level] = 1
            profit_signals[prices >= take_profit_level] = 1
        else:
            stop_signals[prices >= stop_loss_level] = 1
            profit_signals[prices <= take_profit_level] = 1
        
        return stop_signals, profit_signals
    
    def portfolio_correlation_check(self, returns_matrix: pd.DataFrame) -> pd.DataFrame:
        """Check correlation matrix for portfolio diversification"""
        correlation_matrix = returns_matrix.corr()
        
        # Flag high correlations
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > self.max_correlation:
                    high_corr_pairs.append({
                        'Asset1': correlation_matrix.columns[i],
                        'Asset2': correlation_matrix.columns[j],
                        'Correlation': corr
                    })
        
        return pd.DataFrame(high_corr_pairs)
    
    def calculate_portfolio_var(self, returns_matrix: pd.DataFrame, 
                               weights: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate portfolio Value at Risk using Monte Carlo simulation"""
        
        if len(returns_matrix) == 0:
            return 0.0
        
        # Calculate portfolio returns
        portfolio_returns = (returns_matrix * weights).sum(axis=1)
        
        return self.calculate_var(portfolio_returns, confidence_level)
    
    def optimize_kelly_sizing(self, strategy_results: List[Dict]) -> Dict[str, float]:
        """Optimize position sizing using Kelly criterion for multiple strategies"""
        
        kelly_weights = {}
        
        for result in strategy_results:
            trades = result.get('trades', [])
            if not trades:
                kelly_weights[result['strategy_name']] = 0.0
                continue
            
            # Calculate win rate and average win/loss
            profits = [trade['profit'] for trade in trades if 'profit' in trade]
            
            if not profits:
                kelly_weights[result['strategy_name']] = 0.0
                continue
            
            wins = [p for p in profits if p > 0]
            losses = [p for p in profits if p < 0]
            
            if len(wins) == 0 or len(losses) == 0:
                kelly_weights[result['strategy_name']] = 0.0
                continue
            
            win_rate = len(wins) / len(profits)
            avg_win = np.mean(wins)
            avg_loss = abs(np.mean(losses))
            
            kelly_f = self.position_sizer.kelly_criterion(win_rate, avg_win, avg_loss)
            kelly_weights[result['strategy_name']] = kelly_f
        
        # Normalize weights
        total_weight = sum(kelly_weights.values())
        if total_weight > 0:
            kelly_weights = {k: v/total_weight for k, v in kelly_weights.items()}
        
        return kelly_weights
    
    def stress_test_portfolio(self, returns: pd.Series, scenarios: Dict[str, float]) -> Dict[str, float]:
        """Stress test portfolio under various market scenarios"""
        
        stress_results = {}
        
        # Historical scenarios
        stress_results['2008_Crisis'] = np.percentile(returns, 0.1)  # Worst 0.1%
        stress_results['COVID_Crash'] = np.percentile(returns, 1)     # Worst 1%
        stress_results['Normal_Stress'] = np.percentile(returns, 5)   # Worst 5%
        
        # Custom scenarios
        for scenario_name, shock_magnitude in scenarios.items():
            shocked_returns = returns * (1 + shock_magnitude)
            stress_results[f'Custom_{scenario_name}'] = shocked_returns.sum()
        
        return stress_results


class DynamicHedging:
    """Dynamic hedging strategies for risk management"""
    
    def __init__(self, hedge_ratio: float = 0.5, rebalance_threshold: float = 0.05):
        self.hedge_ratio = hedge_ratio
        self.rebalance_threshold = rebalance_threshold
    
    def calculate_hedge_ratio(self, portfolio_returns: pd.Series, 
                            hedge_returns: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling hedge ratio using regression"""
        
        hedge_ratios = []
        
        for i in range(window, len(portfolio_returns)):
            port_window = portfolio_returns.iloc[i-window:i]
            hedge_window = hedge_returns.iloc[i-window:i]
            
            if port_window.std() > 0 and hedge_window.std() > 0:
                beta = port_window.cov(hedge_window) / hedge_window.var()
                hedge_ratios.append(beta)
            else:
                hedge_ratios.append(0)
        
        # Pad with initial hedge ratio
        full_ratios = [self.hedge_ratio] * window + hedge_ratios
        return pd.Series(full_ratios, index=portfolio_returns.index)
    
    def generate_hedge_signals(self, current_hedge: float, 
                             target_hedge: float) -> int:
        """Generate rebalancing signals for hedge positions"""
        
        difference = abs(current_hedge - target_hedge)
        
        if difference > self.rebalance_threshold:
            return 1 if target_hedge > current_hedge else -1
        
        return 0


def demo_risk_management():
    """Demonstration of risk management capabilities"""
    print("=" * 60)
    print("RISK MANAGEMENT SYSTEM DEMO")
    print("=" * 60)
    
    # Generate sample return data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # Portfolio returns with some volatility clustering
    returns = []
    volatility = 0.02
    
    for i in range(1000):
        # GARCH-like volatility clustering
        volatility = 0.8 * volatility + 0.2 * 0.02 + 0.1 * abs(np.random.randn()) * 0.01
        ret = np.random.randn() * volatility
        returns.append(ret)
    
    portfolio_returns = pd.Series(returns, index=dates)
    
    # Benchmark returns (less volatile)
    benchmark_returns = pd.Series(np.random.randn(1000) * 0.015, index=dates)
    
    # Initialize risk manager
    risk_manager = RiskManager(initial_capital=100000)
    
    # Calculate comprehensive risk metrics
    print("📊 Calculating comprehensive risk metrics...")
    metrics = risk_manager.calculate_comprehensive_metrics(
        portfolio_returns, benchmark_returns
    )
    
    print(f"\n🔍 RISK METRICS SUMMARY")
    print("-" * 40)
    print(f"Value at Risk (95%): {metrics.var_95:.4f}")
    print(f"Value at Risk (99%): {metrics.var_99:.4f}")
    print(f"Conditional VaR (95%): {metrics.cvar_95:.4f}")
    print(f"Maximum Drawdown: {metrics.maximum_drawdown:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"Calmar Ratio: {metrics.calmar_ratio:.2f}")
    print(f"Volatility: {metrics.volatility:.2%}")
    print(f"Skewness: {metrics.skewness:.3f}")
    print(f"Kurtosis: {metrics.kurtosis:.3f}")
    
    if metrics.beta is not None:
        print(f"\n📈 BENCHMARK-RELATIVE METRICS")
        print("-" * 40)
        print(f"Beta: {metrics.beta:.3f}")
        print(f"Alpha: {metrics.alpha:.4f}")
        print(f"Tracking Error: {metrics.tracking_error:.4f}")
        print(f"Information Ratio: {metrics.information_ratio:.3f}")
    
    # Position sizing examples
    print(f"\n💰 POSITION SIZING EXAMPLES")
    print("-" * 40)
    
    # Fixed fractional
    pos_size = PositionSizer.fixed_fractional(
        capital=100000, risk_per_trade=0.02, 
        entry_price=100, stop_loss=95
    )
    print(f"Fixed Fractional Position Size: {pos_size} shares")
    
    # Kelly criterion
    kelly_f = PositionSizer.kelly_criterion(
        win_rate=0.55, avg_win=0.03, avg_loss=0.02
    )
    print(f"Kelly Criterion Fraction: {kelly_f:.1%}")
    
    # Volatility targeting
    vol_scalar = PositionSizer.volatility_targeting(portfolio_returns)
    print(f"Current Vol Targeting Scalar: {vol_scalar.iloc[-1]:.2f}")
    
    # Stress testing
    print(f"\n🚨 STRESS TEST RESULTS")
    print("-" * 40)
    
    stress_scenarios = {
        'Market_Crash': -0.30,
        'High_Inflation': -0.15,
        'Interest_Rate_Spike': -0.10
    }
    
    stress_results = risk_manager.stress_test_portfolio(portfolio_returns, stress_scenarios)
    for scenario, result in stress_results.items():
        print(f"{scenario}: {result:.4f}")
    
    # Portfolio correlation analysis
    print(f"\n🔗 PORTFOLIO CORRELATION ANALYSIS")
    print("-" * 40)
    
    # Create mock multi-asset returns
    asset_returns = pd.DataFrame({
        'Asset_A': portfolio_returns,
        'Asset_B': portfolio_returns * 0.8 + np.random.randn(1000) * 0.01,
        'Asset_C': benchmark_returns,
        'Asset_D': -portfolio_returns * 0.3 + np.random.randn(1000) * 0.005
    })
    
    high_corr = risk_manager.portfolio_correlation_check(asset_returns)
    if not high_corr.empty:
        print("High correlation pairs found:")
        print(high_corr.to_string(index=False))
    else:
        print("No concerning correlations detected.")
    
    # Dynamic hedging example
    print(f"\n🛡️  DYNAMIC HEDGING EXAMPLE")
    print("-" * 40)
    
    hedger = DynamicHedging(hedge_ratio=0.5)
    hedge_ratios = hedger.calculate_hedge_ratio(portfolio_returns, benchmark_returns)
    current_ratio = hedge_ratios.iloc[-1]
    
    print(f"Current Hedge Ratio: {current_ratio:.3f}")
    print(f"Target Hedge Ratio: {hedger.hedge_ratio:.3f}")
    
    hedge_signal = hedger.generate_hedge_signals(current_ratio, hedger.hedge_ratio)
    signal_text = "Increase" if hedge_signal > 0 else "Decrease" if hedge_signal < 0 else "No change"
    print(f"Hedge Adjustment: {signal_text}")
    
    print(f"\n✅ Risk management analysis completed!")


if __name__ == "__main__":
    demo_risk_management()
