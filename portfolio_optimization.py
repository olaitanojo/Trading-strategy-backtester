#!/usr/bin/env python3
"""
Portfolio Optimization Module
Modern Portfolio Theory implementation with efficient frontier calculation,
multi-asset backtesting, and advanced optimization algorithms.

Author: olaitanojo
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PortfolioMetrics:
    """Container for portfolio performance metrics"""
    expected_return: float
    volatility: float
    sharpe_ratio: float
    weights: np.ndarray
    asset_names: List[str]


@dataclass
class EfficientFrontier:
    """Container for efficient frontier data"""
    returns: np.ndarray
    volatilities: np.ndarray
    sharpe_ratios: np.ndarray
    weights: np.ndarray
    max_sharpe_portfolio: PortfolioMetrics
    min_volatility_portfolio: PortfolioMetrics


class ModernPortfolioTheory:
    """Modern Portfolio Theory implementation"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def calculate_portfolio_metrics(self, weights: np.ndarray, 
                                  expected_returns: np.ndarray,
                                  cov_matrix: np.ndarray) -> Tuple[float, float, float]:
        """Calculate portfolio return, volatility, and Sharpe ratio"""
        
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def optimize_portfolio(self, expected_returns: np.ndarray, 
                         cov_matrix: np.ndarray,
                         objective: str = 'max_sharpe',
                         target_return: Optional[float] = None,
                         constraints: Optional[List] = None) -> np.ndarray:
        """Optimize portfolio weights based on objective"""
        
        n_assets = len(expected_returns)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Bounds for weights (0 to 1 for each asset, no short selling)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Constraint: weights sum to 1
        constraint_sum = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        constraint_list = [constraint_sum]
        
        # Add custom constraints if provided
        if constraints:
            constraint_list.extend(constraints)
        
        # Define objective functions
        def negative_sharpe(weights):
            ret, vol, sharpe = self.calculate_portfolio_metrics(weights, expected_returns, cov_matrix)
            return -sharpe
        
        def portfolio_volatility(weights):
            _, vol, _ = self.calculate_portfolio_metrics(weights, expected_returns, cov_matrix)
            return vol
        
        def negative_return(weights):
            ret, _, _ = self.calculate_portfolio_metrics(weights, expected_returns, cov_matrix)
            return -ret
        
        # Choose objective function based on optimization goal
        if objective == 'max_sharpe':
            objective_func = negative_sharpe
        elif objective == 'min_volatility':
            objective_func = portfolio_volatility
        elif objective == 'max_return':
            objective_func = negative_return
        elif objective == 'target_return' and target_return is not None:
            # Add return target constraint
            return_constraint = {'type': 'eq', 
                               'fun': lambda x: np.sum(x * expected_returns) - target_return}
            constraint_list.append(return_constraint)
            objective_func = portfolio_volatility
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Optimize
        result = optimize.minimize(
            objective_func,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if not result.success:
            print(f"Optimization warning: {result.message}")
        
        return result.x
    
    def generate_efficient_frontier(self, expected_returns: np.ndarray,
                                  cov_matrix: np.ndarray,
                                  asset_names: List[str],
                                  n_portfolios: int = 100) -> EfficientFrontier:
        """Generate the efficient frontier"""
        
        # Find the range of target returns
        min_ret = expected_returns.min()
        max_ret = expected_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)
        
        efficient_portfolios = []
        
        for target_ret in target_returns:
            try:
                weights = self.optimize_portfolio(
                    expected_returns, cov_matrix, 
                    objective='target_return',
                    target_return=target_ret
                )
                ret, vol, sharpe = self.calculate_portfolio_metrics(weights, expected_returns, cov_matrix)
                efficient_portfolios.append({
                    'return': ret,
                    'volatility': vol,
                    'sharpe': sharpe,
                    'weights': weights
                })
            except:
                continue
        
        if not efficient_portfolios:
            raise ValueError("Could not generate efficient frontier")
        
        # Extract data for efficient frontier
        returns = np.array([p['return'] for p in efficient_portfolios])
        volatilities = np.array([p['volatility'] for p in efficient_portfolios])
        sharpe_ratios = np.array([p['sharpe'] for p in efficient_portfolios])
        weights = np.array([p['weights'] for p in efficient_portfolios])
        
        # Find special portfolios
        max_sharpe_idx = np.argmax(sharpe_ratios)
        min_vol_idx = np.argmin(volatilities)
        
        max_sharpe_portfolio = PortfolioMetrics(
            expected_return=returns[max_sharpe_idx],
            volatility=volatilities[max_sharpe_idx],
            sharpe_ratio=sharpe_ratios[max_sharpe_idx],
            weights=weights[max_sharpe_idx],
            asset_names=asset_names
        )
        
        min_volatility_portfolio = PortfolioMetrics(
            expected_return=returns[min_vol_idx],
            volatility=volatilities[min_vol_idx],
            sharpe_ratio=sharpe_ratios[min_vol_idx],
            weights=weights[min_vol_idx],
            asset_names=asset_names
        )
        
        return EfficientFrontier(
            returns=returns,
            volatilities=volatilities,
            sharpe_ratios=sharpe_ratios,
            weights=weights,
            max_sharpe_portfolio=max_sharpe_portfolio,
            min_volatility_portfolio=min_volatility_portfolio
        )
    
    def black_litterman_optimization(self, returns_data: pd.DataFrame,
                                   views: Dict[str, float],
                                   confidence: Dict[str, float],
                                   market_caps: Optional[Dict[str, float]] = None,
                                   tau: float = 0.05,
                                   risk_aversion: float = 3.0) -> np.ndarray:
        """Black-Litterman portfolio optimization"""
        
        assets = returns_data.columns.tolist()
        n_assets = len(assets)
        
        # Calculate sample statistics
        mu = returns_data.mean().values * 252  # Annualized returns
        sigma = returns_data.cov().values * 252  # Annualized covariance
        
        # Market cap weights (equal if not provided)
        if market_caps is None:
            w_market = np.array([1/n_assets] * n_assets)
        else:
            cap_values = np.array([market_caps.get(asset, 1) for asset in assets])
            w_market = cap_values / cap_values.sum()
        
        # Implied equilibrium returns
        pi = risk_aversion * np.dot(sigma, w_market)
        
        # Build views matrix P and views vector Q
        view_assets = list(views.keys())
        n_views = len(view_assets)
        
        if n_views == 0:
            return w_market
        
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        
        for i, asset in enumerate(view_assets):
            asset_idx = assets.index(asset)
            P[i, asset_idx] = 1
            Q[i] = views[asset]
        
        # Uncertainty matrix Omega
        omega = np.zeros((n_views, n_views))
        for i, asset in enumerate(view_assets):
            conf = confidence.get(asset, 0.5)
            asset_idx = assets.index(asset)
            omega[i, i] = tau * sigma[asset_idx, asset_idx] / conf
        
        # Black-Litterman formula
        tau_sigma_inv = np.linalg.inv(tau * sigma)
        omega_inv = np.linalg.inv(omega)
        
        # New expected returns
        M1 = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
        M2 = tau_sigma_inv @ pi + P.T @ omega_inv @ Q
        mu_bl = M1 @ M2
        
        # New covariance matrix
        sigma_bl = M1
        
        # Optimize portfolio
        weights = self.optimize_portfolio(mu_bl, sigma_bl, objective='max_sharpe')
        
        return weights


class RiskParityOptimizer:
    """Risk Parity Portfolio Optimization"""
    
    def __init__(self):
        pass
    
    def calculate_risk_contributions(self, weights: np.ndarray, 
                                   cov_matrix: np.ndarray) -> np.ndarray:
        """Calculate risk contributions of each asset"""
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
        risk_contrib = weights * marginal_contrib
        return risk_contrib
    
    def optimize_risk_parity(self, cov_matrix: np.ndarray,
                           target_risk: Optional[np.ndarray] = None) -> np.ndarray:
        """Optimize for risk parity portfolio"""
        
        n_assets = cov_matrix.shape[0]
        
        if target_risk is None:
            target_risk = np.array([1/n_assets] * n_assets)
        
        # Objective function: minimize sum of squared deviations from target risk
        def objective(weights):
            risk_contrib = self.calculate_risk_contributions(weights, cov_matrix)
            total_risk = risk_contrib.sum()
            risk_percentages = risk_contrib / total_risk
            return np.sum((risk_percentages - target_risk) ** 2)
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0.001, 1) for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = optimize.minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        return result.x


class MultiAssetBacktester:
    """Multi-asset portfolio backtesting"""
    
    def __init__(self, rebalancing_frequency: str = 'monthly'):
        self.rebalancing_frequency = rebalancing_frequency
        self.mpt = ModernPortfolioTheory()
        self.risk_parity = RiskParityOptimizer()
    
    def backtest_portfolio(self, returns_data: pd.DataFrame,
                         optimization_method: str = 'max_sharpe',
                         lookback_window: int = 252,
                         rebalance_frequency: int = 21) -> Dict:
        """Backtest portfolio with periodic rebalancing"""
        
        assets = returns_data.columns.tolist()
        dates = returns_data.index
        n_assets = len(assets)
        
        # Initialize tracking arrays
        portfolio_weights = []
        portfolio_returns = []
        rebalance_dates = []
        
        # Start after lookback window
        start_idx = lookback_window
        
        for i in range(start_idx, len(returns_data), rebalance_frequency):
            
            # Historical data for optimization
            hist_data = returns_data.iloc[i-lookback_window:i]
            
            if len(hist_data) < 20:  # Need minimum data
                continue
            
            # Calculate expected returns and covariance
            expected_returns = hist_data.mean().values * 252
            cov_matrix = hist_data.cov().values * 252
            
            # Optimize portfolio
            try:
                if optimization_method == 'max_sharpe':
                    weights = self.mpt.optimize_portfolio(expected_returns, cov_matrix, 'max_sharpe')
                elif optimization_method == 'min_volatility':
                    weights = self.mpt.optimize_portfolio(expected_returns, cov_matrix, 'min_volatility')
                elif optimization_method == 'risk_parity':
                    weights = self.risk_parity.optimize_risk_parity(cov_matrix)
                elif optimization_method == 'equal_weight':
                    weights = np.array([1/n_assets] * n_assets)
                else:
                    raise ValueError(f"Unknown optimization method: {optimization_method}")
                
            except Exception as e:
                # Fallback to equal weights
                weights = np.array([1/n_assets] * n_assets)
            
            # Calculate returns until next rebalance
            end_idx = min(i + rebalance_frequency, len(returns_data))
            period_returns = returns_data.iloc[i:end_idx]
            
            if len(period_returns) > 0:
                # Portfolio returns for this period
                portfolio_period_returns = (period_returns * weights).sum(axis=1)
                
                portfolio_weights.append({
                    'date': dates[i],
                    'weights': weights,
                    'assets': assets
                })
                
                portfolio_returns.extend(portfolio_period_returns.tolist())
                rebalance_dates.append(dates[i])
        
        # Convert to pandas series
        portfolio_returns = pd.Series(portfolio_returns, 
                                    index=dates[start_idx:start_idx+len(portfolio_returns)])
        
        # Calculate performance metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annual_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        equity_curve = (1 + portfolio_returns).cumprod()
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        
        return {
            'portfolio_returns': portfolio_returns,
            'portfolio_weights': portfolio_weights,
            'rebalance_dates': rebalance_dates,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'optimization_method': optimization_method
        }
    
    def compare_strategies(self, returns_data: pd.DataFrame,
                         methods: List[str]) -> pd.DataFrame:
        """Compare multiple portfolio optimization strategies"""
        
        results = []
        
        for method in methods:
            try:
                result = self.backtest_portfolio(returns_data, optimization_method=method)
                results.append({
                    'Strategy': method.replace('_', ' ').title(),
                    'Total Return': f"{result['total_return']:.2%}",
                    'Annual Return': f"{result['annual_return']:.2%}",
                    'Volatility': f"{result['volatility']:.2%}",
                    'Sharpe Ratio': f"{result['sharpe_ratio']:.2f}",
                    'Max Drawdown': f"{result['max_drawdown']:.2%}"
                })
            except Exception as e:
                results.append({
                    'Strategy': method.replace('_', ' ').title(),
                    'Total Return': 'Error',
                    'Annual Return': 'Error',
                    'Volatility': 'Error',
                    'Sharpe Ratio': 'Error',
                    'Max Drawdown': 'Error'
                })
        
        return pd.DataFrame(results)


def demo_portfolio_optimization():
    """Demonstration of portfolio optimization capabilities"""
    print("=" * 60)
    print("PORTFOLIO OPTIMIZATION DEMO")
    print("=" * 60)
    
    # Generate sample multi-asset return data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # Create correlated asset returns
    n_assets = 4
    asset_names = ['Tech_Stock', 'Bond_Fund', 'Gold_ETF', 'Real_Estate']
    
    # Generate base factors
    market_factor = np.random.randn(500) * 0.015
    sector_factors = np.random.randn(500, 2) * 0.01
    
    returns_data = {}
    
    for i, asset in enumerate(asset_names):
        # Different factor loadings for each asset
        if 'Tech' in asset:
            returns = market_factor * 1.2 + sector_factors[:, 0] * 0.8 + np.random.randn(500) * 0.008
        elif 'Bond' in asset:
            returns = market_factor * 0.3 + np.random.randn(500) * 0.003
        elif 'Gold' in asset:
            returns = market_factor * -0.2 + np.random.randn(500) * 0.012
        else:  # Real Estate
            returns = market_factor * 0.7 + sector_factors[:, 1] * 0.6 + np.random.randn(500) * 0.010
        
        returns_data[asset] = returns
    
    returns_df = pd.DataFrame(returns_data, index=dates)
    
    print(f"📊 Generated returns for {len(asset_names)} assets over {len(dates)} days\n")
    
    # Initialize MPT
    mpt = ModernPortfolioTheory()
    
    # Calculate sample statistics
    expected_returns = returns_df.mean().values * 252
    cov_matrix = returns_df.cov().values * 252
    
    print("📈 ASSET STATISTICS")
    print("-" * 30)
    for i, asset in enumerate(asset_names):
        print(f"{asset}: {expected_returns[i]:.1%} return, {np.sqrt(cov_matrix[i,i]):.1%} volatility")
    
    # Generate efficient frontier
    print(f"\n🎯 GENERATING EFFICIENT FRONTIER")
    print("-" * 40)
    
    efficient_frontier = mpt.generate_efficient_frontier(expected_returns, cov_matrix, asset_names)
    
    print(f"✓ Generated {len(efficient_frontier.returns)} efficient portfolios")
    
    # Display key portfolios
    print(f"\n🏆 OPTIMAL PORTFOLIOS")
    print("-" * 30)
    
    print("Max Sharpe Portfolio:")
    max_sharpe = efficient_frontier.max_sharpe_portfolio
    for i, asset in enumerate(asset_names):
        print(f"  {asset}: {max_sharpe.weights[i]:.1%}")
    print(f"  Expected Return: {max_sharpe.expected_return:.1%}")
    print(f"  Volatility: {max_sharpe.volatility:.1%}")
    print(f"  Sharpe Ratio: {max_sharpe.sharpe_ratio:.2f}")
    
    print(f"\nMin Volatility Portfolio:")
    min_vol = efficient_frontier.min_volatility_portfolio
    for i, asset in enumerate(asset_names):
        print(f"  {asset}: {min_vol.weights[i]:.1%}")
    print(f"  Expected Return: {min_vol.expected_return:.1%}")
    print(f"  Volatility: {min_vol.volatility:.1%}")
    print(f"  Sharpe Ratio: {min_vol.sharpe_ratio:.2f}")
    
    # Risk Parity Portfolio
    print(f"\n⚖️  RISK PARITY OPTIMIZATION")
    print("-" * 30)
    
    risk_parity = RiskParityOptimizer()
    rp_weights = risk_parity.optimize_risk_parity(cov_matrix)
    
    print("Risk Parity Portfolio:")
    for i, asset in enumerate(asset_names):
        print(f"  {asset}: {rp_weights[i]:.1%}")
    
    # Calculate risk contributions
    risk_contrib = risk_parity.calculate_risk_contributions(rp_weights, cov_matrix)
    total_risk = risk_contrib.sum()
    
    print("Risk Contributions:")
    for i, asset in enumerate(asset_names):
        print(f"  {asset}: {(risk_contrib[i]/total_risk):.1%}")
    
    # Multi-asset backtesting
    print(f"\n🔄 MULTI-ASSET BACKTESTING")
    print("-" * 30)
    
    backtester = MultiAssetBacktester()
    methods = ['max_sharpe', 'min_volatility', 'risk_parity', 'equal_weight']
    
    comparison = backtester.compare_strategies(returns_df, methods)
    print(comparison.to_string(index=False))
    
    # Black-Litterman example
    print(f"\n🔮 BLACK-LITTERMAN OPTIMIZATION")
    print("-" * 35)
    
    # Express views on some assets
    views = {
        'Tech_Stock': 0.15,  # Expect 15% annual return
        'Bond_Fund': 0.03    # Expect 3% annual return
    }
    
    confidence = {
        'Tech_Stock': 0.7,   # 70% confidence
        'Bond_Fund': 0.8     # 80% confidence
    }
    
    bl_weights = mpt.black_litterman_optimization(returns_df, views, confidence)
    
    print("Black-Litterman Portfolio:")
    for i, asset in enumerate(asset_names):
        print(f"  {asset}: {bl_weights[i]:.1%}")
    
    print(f"\n✅ Portfolio optimization demonstration completed!")


if __name__ == "__main__":
    demo_portfolio_optimization()
