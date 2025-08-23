#!/usr/bin/env python3
"""
Walk-Forward Analysis Module
Implements walk-forward optimization and out-of-sample testing for robust 
strategy validation and parameter optimization.

Author: olaitanojo
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
from itertools import product
import warnings
warnings.filterwarnings('ignore')


@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis"""
    strategy_name: str
    in_sample_performance: Dict
    out_of_sample_performance: Dict
    parameter_stability: Dict
    optimization_periods: List[Dict]
    validation_periods: List[Dict]
    best_parameters: Dict
    robustness_metrics: Dict


class ParameterOptimizer:
    """Parameter optimization using various methods"""
    
    def __init__(self):
        self.optimization_methods = {
            'grid_search': self._grid_search,
            'random_search': self._random_search,
            'genetic_algorithm': self._genetic_algorithm
        }
    
    def _grid_search(self, objective_function: Callable, 
                    parameter_ranges: Dict[str, List], 
                    **kwargs) -> Tuple[Dict, float]:
        """Grid search optimization"""
        
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        best_params = None
        best_score = -np.inf
        
        # Generate all parameter combinations
        for param_combination in product(*param_values):
            params = dict(zip(param_names, param_combination))
            
            try:
                score = objective_function(params)
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            except:
                continue
        
        return best_params or {}, best_score
    
    def _random_search(self, objective_function: Callable,
                      parameter_ranges: Dict[str, List],
                      n_iterations: int = 100,
                      **kwargs) -> Tuple[Dict, float]:
        """Random search optimization"""
        
        best_params = None
        best_score = -np.inf
        
        for _ in range(n_iterations):
            # Random parameter selection
            params = {}
            for param_name, param_values in parameter_ranges.items():
                params[param_name] = np.random.choice(param_values)
            
            try:
                score = objective_function(params)
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            except:
                continue
        
        return best_params or {}, best_score
    
    def _genetic_algorithm(self, objective_function: Callable,
                          parameter_ranges: Dict[str, List],
                          population_size: int = 50,
                          generations: int = 20,
                          mutation_rate: float = 0.1,
                          **kwargs) -> Tuple[Dict, float]:
        """Simplified genetic algorithm optimization"""
        
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = {}
            for param_name, values in parameter_ranges.items():
                individual[param_name] = np.random.choice(values)
            population.append(individual)
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                try:
                    score = objective_function(individual)
                    fitness_scores.append(score)
                except:
                    fitness_scores.append(-np.inf)
            
            # Select best individuals (top 50%)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite_size = population_size // 2
            elite_population = [population[i] for i in sorted_indices[:elite_size]]
            
            # Create new generation
            new_population = elite_population.copy()
            
            while len(new_population) < population_size:
                # Crossover (simple average for numeric parameters)
                parent1, parent2 = np.random.choice(elite_population, 2, replace=False)
                child = {}
                
                for param_name in param_names:
                    if np.random.random() < 0.5:
                        child[param_name] = parent1[param_name]
                    else:
                        child[param_name] = parent2[param_name]
                
                # Mutation
                if np.random.random() < mutation_rate:
                    param_to_mutate = np.random.choice(param_names)
                    child[param_to_mutate] = np.random.choice(parameter_ranges[param_to_mutate])
                
                new_population.append(child)
            
            population = new_population
        
        # Return best individual
        final_scores = []
        for individual in population:
            try:
                score = objective_function(individual)
                final_scores.append(score)
            except:
                final_scores.append(-np.inf)
        
        best_idx = np.argmax(final_scores)
        return population[best_idx], final_scores[best_idx]
    
    def optimize(self, objective_function: Callable,
                parameter_ranges: Dict[str, List],
                method: str = 'grid_search',
                **kwargs) -> Tuple[Dict, float]:
        """Optimize parameters using specified method"""
        
        if method not in self.optimization_methods:
            raise ValueError(f"Unknown optimization method: {method}")
        
        return self.optimization_methods[method](
            objective_function, parameter_ranges, **kwargs
        )


class WalkForwardAnalyzer:
    """Walk-forward analysis implementation"""
    
    def __init__(self, optimization_window: int = 252,
                 validation_window: int = 63,
                 step_size: int = 21):
        self.optimization_window = optimization_window
        self.validation_window = validation_window
        self.step_size = step_size
        self.optimizer = ParameterOptimizer()
    
    def run_walk_forward_analysis(self, 
                                 data: pd.DataFrame,
                                 strategy_class: Any,
                                 parameter_ranges: Dict[str, List],
                                 backtest_engine: Any,
                                 optimization_metric: str = 'sharpe_ratio',
                                 optimization_method: str = 'grid_search',
                                 min_trades: int = 10) -> WalkForwardResult:
        """Run complete walk-forward analysis"""
        
        optimization_periods = []
        validation_periods = []
        parameter_history = []
        
        # Calculate walk-forward windows
        total_length = len(data)
        start_idx = self.optimization_window
        
        while start_idx + self.validation_window <= total_length:
            # Define windows
            opt_start = start_idx - self.optimization_window
            opt_end = start_idx
            val_start = start_idx
            val_end = min(start_idx + self.validation_window, total_length)
            
            # Extract data
            optimization_data = data.iloc[opt_start:opt_end]
            validation_data = data.iloc[val_start:val_end]
            
            # Define objective function for this period
            def objective_function(params):
                try:
                    # Create strategy with parameters
                    strategy = strategy_class(**params)
                    
                    # Run backtest on optimization period
                    result = backtest_engine.run_backtest(optimization_data, strategy)
                    
                    # Check minimum trades requirement
                    if result.total_trades < min_trades:
                        return -np.inf
                    
                    # Return optimization metric
                    return getattr(result, optimization_metric, -np.inf)
                except:
                    return -np.inf
            
            # Optimize parameters
            best_params, best_score = self.optimizer.optimize(
                objective_function, parameter_ranges, method=optimization_method
            )
            
            if not best_params:
                # Skip this period if optimization failed
                start_idx += self.step_size
                continue
            
            # Test on validation period
            validation_strategy = strategy_class(**best_params)
            validation_result = backtest_engine.run_backtest(validation_data, validation_strategy)
            
            # Store results
            optimization_periods.append({
                'start_date': optimization_data.index[0],
                'end_date': optimization_data.index[-1],
                'best_params': best_params,
                'best_score': best_score,
                'data_length': len(optimization_data)
            })
            
            validation_periods.append({
                'start_date': validation_data.index[0],
                'end_date': validation_data.index[-1],
                'params_used': best_params,
                'performance': {
                    'total_return': validation_result.total_return,
                    'annual_return': validation_result.annual_return,
                    'volatility': validation_result.volatility,
                    'sharpe_ratio': validation_result.sharpe_ratio,
                    'max_drawdown': validation_result.max_drawdown,
                    'total_trades': validation_result.total_trades
                }
            })
            
            parameter_history.append(best_params)
            
            # Move to next window
            start_idx += self.step_size
        
        if not optimization_periods:
            raise ValueError("No valid optimization periods found")
        
        # Calculate aggregate performance
        in_sample_returns = []
        out_of_sample_returns = []
        
        for opt_period in optimization_periods:
            if opt_period['best_score'] != -np.inf:
                in_sample_returns.append(opt_period['best_score'])
        
        for val_period in validation_periods:
            out_of_sample_returns.append(val_period['performance']['sharpe_ratio'])
        
        # Aggregate in-sample performance
        in_sample_performance = {
            'mean_metric': np.mean(in_sample_returns) if in_sample_returns else 0,
            'std_metric': np.std(in_sample_returns) if in_sample_returns else 0,
            'periods_count': len(in_sample_returns)
        }
        
        # Aggregate out-of-sample performance
        oos_metrics = self._calculate_aggregate_performance(validation_periods)
        
        # Parameter stability analysis
        parameter_stability = self._analyze_parameter_stability(parameter_history, parameter_ranges)
        
        # Robustness metrics
        robustness_metrics = self._calculate_robustness_metrics(
            in_sample_returns, out_of_sample_returns
        )
        
        # Find most common parameter combination
        best_parameters = self._find_most_stable_parameters(parameter_history, parameter_ranges)
        
        return WalkForwardResult(
            strategy_name=strategy_class.__name__,
            in_sample_performance=in_sample_performance,
            out_of_sample_performance=oos_metrics,
            parameter_stability=parameter_stability,
            optimization_periods=optimization_periods,
            validation_periods=validation_periods,
            best_parameters=best_parameters,
            robustness_metrics=robustness_metrics
        )
    
    def _calculate_aggregate_performance(self, validation_periods: List[Dict]) -> Dict:
        """Calculate aggregate out-of-sample performance"""
        
        if not validation_periods:
            return {}
        
        # Extract all performance metrics
        total_returns = [p['performance']['total_return'] for p in validation_periods]
        annual_returns = [p['performance']['annual_return'] for p in validation_periods]
        volatilities = [p['performance']['volatility'] for p in validation_periods]
        sharpe_ratios = [p['performance']['sharpe_ratio'] for p in validation_periods]
        max_drawdowns = [p['performance']['max_drawdown'] for p in validation_periods]
        total_trades = [p['performance']['total_trades'] for p in validation_periods]
        
        return {
            'mean_total_return': np.mean(total_returns),
            'mean_annual_return': np.mean(annual_returns),
            'mean_volatility': np.mean(volatilities),
            'mean_sharpe_ratio': np.mean(sharpe_ratios),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'total_trades': sum(total_trades),
            'std_sharpe_ratio': np.std(sharpe_ratios),
            'win_rate': len([r for r in total_returns if r > 0]) / len(total_returns),
            'periods_count': len(validation_periods)
        }
    
    def _analyze_parameter_stability(self, parameter_history: List[Dict],
                                   parameter_ranges: Dict[str, List]) -> Dict:
        """Analyze stability of optimized parameters"""
        
        if not parameter_history:
            return {}
        
        stability_metrics = {}
        
        for param_name in parameter_ranges.keys():
            param_values = [params.get(param_name) for params in parameter_history if param_name in params]
            
            if not param_values:
                continue
            
            # Calculate stability metrics
            unique_values = len(set(param_values))
            total_values = len(param_values)
            most_common_value = max(set(param_values), key=param_values.count)
            most_common_frequency = param_values.count(most_common_value)
            
            stability_metrics[param_name] = {
                'unique_count': unique_values,
                'stability_ratio': most_common_frequency / total_values,
                'most_common_value': most_common_value,
                'value_distribution': {val: param_values.count(val) for val in set(param_values)}
            }
        
        return stability_metrics
    
    def _calculate_robustness_metrics(self, in_sample_scores: List[float],
                                    out_of_sample_scores: List[float]) -> Dict:
        """Calculate robustness and overfitting metrics"""
        
        robustness_metrics = {}
        
        if in_sample_scores and out_of_sample_scores:
            # Performance degradation
            is_mean = np.mean(in_sample_scores)
            oos_mean = np.mean(out_of_sample_scores)
            
            robustness_metrics['performance_degradation'] = (is_mean - oos_mean) / is_mean if is_mean != 0 else 0
            
            # Consistency (correlation between IS and OOS)
            if len(in_sample_scores) == len(out_of_sample_scores):
                correlation = np.corrcoef(in_sample_scores, out_of_sample_scores)[0, 1]
                robustness_metrics['is_oos_correlation'] = correlation if not np.isnan(correlation) else 0
            
            # Overfitting indicator
            robustness_metrics['overfitting_ratio'] = max(0, (is_mean - oos_mean) / np.std(in_sample_scores)) if np.std(in_sample_scores) > 0 else 0
        
        # Stability metrics
        if out_of_sample_scores:
            robustness_metrics['oos_consistency'] = 1 - (np.std(out_of_sample_scores) / abs(np.mean(out_of_sample_scores))) if np.mean(out_of_sample_scores) != 0 else 0
            robustness_metrics['positive_periods_ratio'] = len([s for s in out_of_sample_scores if s > 0]) / len(out_of_sample_scores)
        
        return robustness_metrics
    
    def _find_most_stable_parameters(self, parameter_history: List[Dict],
                                   parameter_ranges: Dict[str, List]) -> Dict:
        """Find the most stable parameter combination"""
        
        if not parameter_history:
            return {}
        
        best_parameters = {}
        
        for param_name in parameter_ranges.keys():
            param_values = [params.get(param_name) for params in parameter_history if param_name in params]
            
            if param_values:
                # Use most frequent value as the most stable
                best_parameters[param_name] = max(set(param_values), key=param_values.count)
        
        return best_parameters
    
    def compare_strategies_walk_forward(self, 
                                      data: pd.DataFrame,
                                      strategy_configs: List[Dict],
                                      backtest_engine: Any) -> pd.DataFrame:
        """Compare multiple strategies using walk-forward analysis"""
        
        results = []
        
        for config in strategy_configs:
            try:
                wf_result = self.run_walk_forward_analysis(
                    data=data,
                    strategy_class=config['strategy_class'],
                    parameter_ranges=config['parameter_ranges'],
                    backtest_engine=backtest_engine,
                    optimization_metric=config.get('optimization_metric', 'sharpe_ratio')
                )
                
                results.append({
                    'Strategy': wf_result.strategy_name,
                    'OOS Sharpe Ratio': f"{wf_result.out_of_sample_performance.get('mean_sharpe_ratio', 0):.3f}",
                    'OOS Annual Return': f"{wf_result.out_of_sample_performance.get('mean_annual_return', 0):.2%}",
                    'Performance Degradation': f"{wf_result.robustness_metrics.get('performance_degradation', 0):.2%}",
                    'Consistency': f"{wf_result.robustness_metrics.get('oos_consistency', 0):.3f}",
                    'Win Rate': f"{wf_result.out_of_sample_performance.get('win_rate', 0):.1%}",
                    'Periods Tested': wf_result.out_of_sample_performance.get('periods_count', 0)
                })
                
            except Exception as e:
                results.append({
                    'Strategy': config.get('name', 'Unknown'),
                    'OOS Sharpe Ratio': 'Error',
                    'OOS Annual Return': 'Error', 
                    'Performance Degradation': 'Error',
                    'Consistency': 'Error',
                    'Win Rate': 'Error',
                    'Periods Tested': 0
                })
        
        return pd.DataFrame(results)


class MonteCarloValidator:
    """Monte Carlo validation for strategy robustness"""
    
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations
    
    def bootstrap_validation(self, returns: pd.Series, 
                           strategy_function: Callable,
                           n_bootstrap: int = 500) -> Dict:
        """Bootstrap validation of strategy performance"""
        
        bootstrap_results = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_sample = returns.sample(n=len(returns), replace=True)
            bootstrap_sample.index = returns.index  # Maintain time structure
            
            try:
                # Apply strategy to bootstrap sample
                result = strategy_function(bootstrap_sample)
                bootstrap_results.append(result)
            except:
                continue
        
        if not bootstrap_results:
            return {}
        
        return {
            'mean_performance': np.mean(bootstrap_results),
            'std_performance': np.std(bootstrap_results),
            'confidence_95_lower': np.percentile(bootstrap_results, 2.5),
            'confidence_95_upper': np.percentile(bootstrap_results, 97.5),
            'success_rate': len([r for r in bootstrap_results if r > 0]) / len(bootstrap_results)
        }
    
    def parameter_sensitivity_analysis(self, 
                                     data: pd.DataFrame,
                                     strategy_class: Any,
                                     base_parameters: Dict,
                                     backtest_engine: Any,
                                     sensitivity_range: float = 0.2) -> Dict:
        """Analyze parameter sensitivity"""
        
        sensitivity_results = {}
        
        # Test each parameter independently
        for param_name, param_value in base_parameters.items():
            if isinstance(param_value, (int, float)):
                # Create parameter variations
                if isinstance(param_value, int):
                    variations = [
                        max(1, int(param_value * (1 - sensitivity_range))),
                        param_value,
                        int(param_value * (1 + sensitivity_range))
                    ]
                else:
                    variations = [
                        param_value * (1 - sensitivity_range),
                        param_value,
                        param_value * (1 + sensitivity_range)
                    ]
                
                param_results = []
                
                for variation in variations:
                    test_params = base_parameters.copy()
                    test_params[param_name] = variation
                    
                    try:
                        strategy = strategy_class(**test_params)
                        result = backtest_engine.run_backtest(data, strategy)
                        param_results.append({
                            'parameter_value': variation,
                            'sharpe_ratio': result.sharpe_ratio,
                            'total_return': result.total_return
                        })
                    except:
                        continue
                
                if len(param_results) >= 2:
                    # Calculate sensitivity
                    sharpe_values = [r['sharpe_ratio'] for r in param_results]
                    param_values = [r['parameter_value'] for r in param_results]
                    
                    if len(set(sharpe_values)) > 1 and len(set(param_values)) > 1:
                        sensitivity = np.std(sharpe_values) / np.std(param_values)
                        sensitivity_results[param_name] = {
                            'sensitivity': sensitivity,
                            'results': param_results
                        }
        
        return sensitivity_results


def demo_walk_forward_analysis():
    """Demonstration of walk-forward analysis"""
    print("=" * 60)
    print("WALK-FORWARD ANALYSIS DEMO")
    print("=" * 60)
    
    # This is a simplified demo - in practice, you would use real strategies and data
    print("📊 Walk-forward analysis provides robust validation by:")
    print("   • Optimizing parameters on historical data")
    print("   • Testing on out-of-sample periods")
    print("   • Analyzing parameter stability")
    print("   • Measuring overfitting")
    
    print(f"\n🔄 WALK-FORWARD PROCESS")
    print("-" * 30)
    print("1. Split data into optimization and validation windows")
    print("2. Optimize parameters on each training window")
    print("3. Test optimized parameters on validation periods") 
    print("4. Analyze performance degradation and consistency")
    print("5. Identify most stable parameter combinations")
    
    print(f"\n📈 OPTIMIZATION METHODS AVAILABLE")
    print("-" * 35)
    print("• Grid Search - Exhaustive parameter search")
    print("• Random Search - Random parameter sampling")
    print("• Genetic Algorithm - Evolutionary optimization")
    
    print(f"\n🎯 ROBUSTNESS METRICS")
    print("-" * 25)
    print("• Performance Degradation - IS vs OOS difference")
    print("• Parameter Stability - Consistency across periods")
    print("• Overfitting Ratio - Risk of curve fitting")
    print("• OOS Consistency - Stability of out-of-sample results")
    
    # Mock example of parameter ranges
    example_ranges = {
        'fast_period': [5, 10, 15, 20, 25],
        'slow_period': [20, 30, 40, 50, 60],
        'rsi_period': [10, 14, 18, 22],
        'threshold': [0.1, 0.15, 0.2, 0.25]
    }
    
    print(f"\n🔧 EXAMPLE PARAMETER RANGES")
    print("-" * 30)
    for param, values in example_ranges.items():
        print(f"{param}: {values}")
    
    print(f"\n⚙️ ANALYSIS CONFIGURATION")
    print("-" * 25)
    analyzer = WalkForwardAnalyzer(
        optimization_window=252,  # 1 year optimization
        validation_window=63,     # 3 months validation
        step_size=21             # Monthly step
    )
    
    print(f"Optimization Window: {analyzer.optimization_window} days")
    print(f"Validation Window: {analyzer.validation_window} days")
    print(f"Step Size: {analyzer.step_size} days")
    
    print(f"\n✅ Walk-forward analysis framework ready!")
    print("   Use run_walk_forward_analysis() with your strategies and data")


if __name__ == "__main__":
    demo_walk_forward_analysis()
