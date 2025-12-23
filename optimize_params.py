"""
Parameter Optimization Script

This script provides a framework for optimizing strategy parameters.
YOU MUST IMPLEMENT YOUR OWN OPTIMIZATION METHODS.

Available approaches you could implement:
- Grid Search: Exhaustive search over parameter combinations
- Random Search: Random sampling of parameter space
- Bayesian Optimization: Uses prior evaluations to guide search
- Genetic Algorithms: Evolutionary approach to find optimal parameters
- Simulated Annealing: Probabilistic technique for global optimization
- Gradient-based methods: If parameters are continuous and differentiable

Usage:
    python optimize_params.py

TODO: Implement your optimization method in the optimize_parameters() function.

See Nautilus Backtesting Docs: https://nautilustrader.io/docs/latest/backtesting/
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

# TODO: Import any additional libraries you need for your optimization method
# Examples:
# import itertools  # For grid search
# import random     # For random search
# import numpy as np  # For numerical operations
# from scipy.optimize import minimize  # For gradient-based methods
# from skopt import gp_minimize  # For Bayesian optimization (scikit-optimize)
# import optuna  # For advanced optimization (Optuna library)

# Import backtest runner components
# Note: You may need to adjust imports based on your run_backtest.py implementation
from run_backtest import load_config, setup_backtest_engine


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    rsi_period: int
    long_entry: float
    long_exit: float
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    total_pnl: float
    num_trades: int


def run_with_params(
    config: Dict[str, Any],
    rsi_period: int,
    long_entry: float,
    long_exit: float
) -> OptimizationResult:
    """
    Run a backtest with specific parameters.
    
    Args:
        config: Base configuration dictionary
        rsi_period: RSI calculation period
        long_entry: RSI threshold for long entry
        long_exit: RSI threshold for long exit
    
    Returns:
        OptimizationResult with performance metrics
    """
    # Update strategy parameters in config
    strategy_config = config['engine']['strategies'][0]['config']
    strategy_config['rsi_period'] = rsi_period
    strategy_config['long_entry'] = long_entry
    strategy_config['long_exit'] = long_exit
    
    try:
        # Set up and run backtest
        engine = setup_backtest_engine(config)
        engine.run()
        
        # Extract performance metrics
        stats_returns = engine.portfolio.analyzer.get_performance_stats_returns()
        stats_general = engine.portfolio.analyzer.get_performance_stats_general()
        
        sharpe = stats_returns.get('Sharpe Ratio (252 days)', 0.0)
        total_return = stats_returns.get('Total Return', 0.0)
        max_drawdown = stats_general.get('Max Drawdown', 0.0)
        
        # Get account PnL
        venues = engine.list_venues()
        total_pnl = 0.0
        num_trades = 0
        
        if venues:
            venue = venues[0]
            account_df = engine.trader.generate_account_report(venue)
            if not account_df.empty and 'total' in account_df.columns:
                initial_equity = account_df['total'].iloc[0]
                final_equity = account_df['total'].iloc[-1]
                total_pnl = final_equity - initial_equity
            
            fills_df = engine.trader.generate_fills_report()
            num_trades = len(fills_df) if not fills_df.empty else 0
        
        return OptimizationResult(
            rsi_period=rsi_period,
            long_entry=long_entry,
            long_exit=long_exit,
            sharpe_ratio=float(sharpe) if sharpe else 0.0,
            total_return=float(total_return) if total_return else 0.0,
            max_drawdown=float(max_drawdown) if max_drawdown else 0.0,
            total_pnl=float(total_pnl),
            num_trades=num_trades,
        )
    
    except Exception as e:
        print(f"⚠️  Error running backtest with params (rsi={rsi_period}, entry={long_entry}, exit={long_exit}): {e}")
        # Return a result with zero performance
        return OptimizationResult(
            rsi_period=rsi_period,
            long_entry=long_entry,
            long_exit=long_exit,
            sharpe_ratio=0.0,
            total_return=0.0,
            max_drawdown=0.0,
            total_pnl=0.0,
            num_trades=0,
        )


def get_parameter_ranges() -> Dict[str, List]:
    """
    Define parameter ranges for optimization.
    
    Returns:
        Dictionary with parameter names as keys and lists of possible values as values
    
    TODO: Adjust these ranges based on your optimization needs and strategy requirements.
    """
    return {
        'rsi_period': [10, 12, 14, 16, 18, 20],  # RSI calculation periods
        'long_entry': [25.0, 28.0, 31.0, 34.0, 37.0, 40.0],  # Oversold thresholds
        'long_exit': [70.0, 75.0, 80.0, 83.0, 86.0, 90.0],    # Overbought thresholds
    }


def evaluate_parameter_combination(
    config: Dict[str, Any],
    rsi_period: int,
    long_entry: float,
    long_exit: float
) -> OptimizationResult:
    """
    Evaluate a single parameter combination by running a backtest.
    
    This is a helper function that wraps run_with_params() for clarity.
    
    Args:
        config: Base configuration dictionary
        rsi_period: RSI calculation period
        long_entry: RSI threshold for long entry
        long_exit: RSI threshold for long exit
    
    Returns:
        OptimizationResult with performance metrics
    """
    return run_with_params(config, rsi_period, long_entry, long_exit)


def optimize_parameters(config_path: str) -> List[OptimizationResult]:
    """
    Main optimization function - YOU MUST IMPLEMENT THIS.
    
    This function should implement your chosen optimization algorithm to find
    the best parameter combinations for the strategy.
    
    Args:
        config_path: Path to configuration YAML file
    
    Returns:
        List of OptimizationResult sorted by your chosen metric (e.g., Sharpe ratio)
    
    TODO: Implement your optimization method here. Some options:
    
    1. GRID SEARCH (Simple but exhaustive):
       - Test all combinations of parameters
       - Good for small parameter spaces
       - Use itertools.product() to generate combinations
    
    2. RANDOM SEARCH (Faster than grid search):
       - Randomly sample parameter combinations
       - Good for large parameter spaces
       - Use random.choice() or random.sample()
    
    3. BAYESIAN OPTIMIZATION (Efficient):
       - Uses prior evaluations to guide search
       - Requires library like scikit-optimize or optuna
       - Good for expensive evaluations
    
    4. GENETIC ALGORITHM (Evolutionary):
       - Maintains population of parameter sets
       - Evolves better solutions over generations
       - Good for complex, non-convex spaces
    
    5. SIMULATED ANNEALING (Probabilistic):
       - Starts with random solution
       - Gradually "cools" to find optimum
       - Good for avoiding local minima
    
    Example structure:
        1. Load configuration
        2. Define parameter ranges (use get_parameter_ranges())
        3. Initialize your optimization algorithm
        4. Loop: Generate parameter combination → Evaluate → Update algorithm
        5. Return sorted results
    
    Hints:
        - Use evaluate_parameter_combination() to test each parameter set
        - Track all results in a list
        - Consider early stopping if you find a good enough solution
        - You may want to limit the number of evaluations (e.g., max_iterations)
    """
    # Load base configuration
    config = load_config(config_path)
    
    # Get parameter ranges
    param_ranges = get_parameter_ranges()
    
    print("🔍 Starting Parameter Optimization")
    print(f"📊 Parameter Ranges:")
    for param, values in param_ranges.items():
        print(f"   {param}: {values}")
    print()
    
    # TODO: IMPLEMENT YOUR OPTIMIZATION METHOD HERE
    # 
    # Example skeleton for grid search (remove and implement your own):
    # 
    results: List[OptimizationResult] = []
    
    for rsi_period in param_ranges['rsi_period']:
        for long_entry in param_ranges['long_entry']:
            for long_exit in param_ranges['long_exit']:
                result = evaluate_parameter_combination(
                    config, rsi_period, long_entry, long_exit
                )
                results.append(result)
    
    results.sort(key=lambda x: x.sharpe_ratio, reverse=True)
    return results
    
    raise NotImplementedError(
        "You must implement optimize_parameters() with your chosen optimization method.\n"
        "See the function docstring for guidance on different approaches."
    )


def objective_function(result: OptimizationResult) -> float:
    """
    Convert OptimizationResult to a single objective value for optimization.
    
    This function defines what you're trying to maximize/minimize.
    You can customize this based on your optimization goals.
    
    Args:
        result: OptimizationResult from a backtest
    
    Returns:
        Single objective value (higher is better for maximization)
    
    TODO: Customize this function based on your optimization goals.
    Examples:
        - Maximize Sharpe ratio: return result.sharpe_ratio
        - Maximize return: return result.total_return
        - Minimize drawdown: return -result.max_drawdown
        - Combined metric: return result.sharpe_ratio * 0.7 + result.total_return * 0.3
    """
    # Default: maximize Sharpe ratio
    # TODO: Implement your own objective function
    return result.sharpe_ratio


def print_optimization_summary(results: List[OptimizationResult], top_n: int = 10):
    """
    Print summary of optimization results.
    
    Args:
        results: List of optimization results (sorted by Sharpe ratio)
        top_n: Number of top results to display
    """
    print("\n" + "="*100)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*100)
    
    if not results:
        print("❌ No results to display")
        return
    
    # Best result
    best = results[0]
    print(f"\n🏆 Best Parameters (by Sharpe Ratio):")
    print(f"   RSI Period: {best.rsi_period}")
    print(f"   Long Entry: {best.long_entry}")
    print(f"   Long Exit: {best.long_exit}")
    print(f"\n   Performance Metrics:")
    print(f"   Sharpe Ratio: {best.sharpe_ratio:.2f}")
    print(f"   Total Return: {best.total_return:.2%}")
    print(f"   Max Drawdown: {best.max_drawdown:.2%}")
    print(f"   Total PnL: ${best.total_pnl:,.2f}")
    print(f"   Number of Trades: {best.num_trades}")
    
    # Top N results
    print(f"\n📊 Top {min(top_n, len(results))} Configurations:")
    print(f"{'Rank':<6} {'RSI':<6} {'Entry':<8} {'Exit':<8} {'Sharpe':<10} {'Return':<12} {'PnL':<15} {'Trades':<8}")
    print("-" * 100)
    
    for i, result in enumerate(results[:top_n], 1):
        print(
            f"{i:<6} {result.rsi_period:<6} {result.long_entry:<8.1f} {result.long_exit:<8.1f} "
            f"{result.sharpe_ratio:<10.2f} {result.total_return:<12.2%} "
            f"${result.total_pnl:<14,.2f} {result.num_trades:<8}"
        )
    
    # Statistics
    if len(results) > 0:
        sharpe_ratios = [r.sharpe_ratio for r in results if r.sharpe_ratio > 0]
        returns = [r.total_return for r in results if r.total_return != 0]
        
        if sharpe_ratios:
            print(f"\n📈 Statistics:")
            print(f"   Average Sharpe Ratio: {sum(sharpe_ratios) / len(sharpe_ratios):.2f}")
            print(f"   Max Sharpe Ratio: {max(sharpe_ratios):.2f}")
            print(f"   Min Sharpe Ratio: {min(sharpe_ratios):.2f}")
        
        if returns:
            print(f"   Average Return: {sum(returns) / len(returns):.2%}")
            print(f"   Max Return: {max(returns):.2%}")
            print(f"   Min Return: {min(returns):.2%}")
    
    print("\n" + "="*100)
    
    # Recommendation
    print("\n💡 Recommendation:")
    print(f"   Update config/backtest_gc.yaml with the best parameters:")
    print(f"   rsi_period: {best.rsi_period}")
    print(f"   long_entry: {best.long_entry}")
    print(f"   long_exit: {best.long_exit}")


def main():
    """
    Main entry point for parameter optimization.
    
    TODO: You may want to add command-line arguments for:
    - Optimization method selection
    - Number of iterations/evaluations
    - Parameter range customization
    - Early stopping criteria
    - Output file for results
    """
    script_dir = Path(__file__).parent
    config_path = script_dir / "config" / "backtest_gc.yaml"
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    
    print("🚀 Starting Parameter Optimization")
    print(f"📁 Configuration: {config_path}\n")
    
    # TODO: Add any pre-optimization setup here
    # Examples:
    # - Set random seed for reproducibility
    # - Load previous results to continue optimization
    # - Set up logging to file
    
    # Run optimization
    try:
        results = optimize_parameters(str(config_path))
        
        if not results:
            print("❌ No results returned from optimization!")
            sys.exit(1)
        
        # Print summary
        print_optimization_summary(results, top_n=10)
        
        # TODO: Add post-optimization actions here
        # Examples:
        # - Save results to CSV/JSON file
        # - Generate plots/visualizations
        # - Update config file with best parameters automatically
        
        print("\n✅ Optimization complete!")
        
    except NotImplementedError as e:
        print(f"\n❌ {e}")
        print("\n💡 You need to implement the optimize_parameters() function.")
        print("   See the function docstring for guidance on different optimization approaches.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Optimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
