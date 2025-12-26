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

import copy
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
import random


# Import backtest runner components
# Note: You may need to adjust imports based on your run_backtest.py implementation
from run_backtest import load_config, setup_backtest_engine


@dataclass
class OptimizationResult:
    """Container for optimization results."""

    rsi_period: int
    long_entry: float
    long_exit: float
    atr_period: int
    sensitivity_base: float
    pyramid_multiplier_base: float
    sharpe_ratio: float
    calmar_ratio: float
    total_return: float
    max_drawdown: float
    total_pnl: float
    num_trades: int


def run_with_params(
    config: Dict[str, Any],
    rsi_period: int,
    long_entry: float,
    long_exit: float,
    atr_period: int,
    sensitivity_base: float,
    pyramid_multiplier_base: float,
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

    # I found that when running the optimization, there were issues saving the configs from the yaml
    # file so this is my solution to saving those configs instead of having whatever configuration
    # the optimize_params.py was changing
    config_copy = copy.deepcopy(config)
    strategy_config = config_copy["engine"]["strategies"][0]["config"]
    engine = setup_backtest_engine(config_copy)

    # Update strategy parameters in config
    strategy_config = config["engine"]["strategies"][0]["config"]
    strategy_config["rsi_period"] = rsi_period
    strategy_config["long_entry"] = long_entry
    strategy_config["long_exit"] = long_exit

    strategy_config["atr_period"] = atr_period
    strategy_config["sensitivity_base"] = sensitivity_base
    strategy_config["pyramid_multiplier_base"] = pyramid_multiplier_base

    try:
        # Set up and run backtest
        engine = setup_backtest_engine(config)
        engine.run()

        # Extract performance metrics
        stats_returns = engine.portfolio.analyzer.get_performance_stats_returns()
        stats_general = engine.portfolio.analyzer.get_performance_stats_general()

        sharpe = stats_returns.get("Sharpe Ratio (252 days)", 0.0)
        total_return = stats_returns.get("Total Return", 0.0)
        max_drawdown = stats_general.get("Max Drawdown", 0.0)

        # Get account PnL
        venues = engine.list_venues()
        total_pnl = 0.0
        num_trades = 0

        if venues:
            venue = venues[0]
            account_df = engine.trader.generate_account_report(venue)
            if not account_df.empty and "total" in account_df.columns:

                # I noticed that the params testing had issues dealing with float values for whatever
                # reason, so this chunk is to solve that issue
                raw_initial = account_df["total"].iloc[0]
                raw_final = account_df["total"].iloc[-1]

                def parse_money(val):
                    if isinstance(val, str):
                        clean_val = val.replace(",", "").split(" ")[0]
                        return float(clean_val)
                    return float(val)

                initial_equity = parse_money(raw_initial)
                final_equity = parse_money(raw_final)

                total_pnl = final_equity - initial_equity

                # Total Returns wasn't calculating for whatever
                # reason so I am calculating that manually here
                calculated_return = total_pnl / initial_equity

                # Maxdraw down also wasn't calculating (hello??!!)
                # so I've also done that manually here
                equity_curve = account_df["total"].apply(parse_money)
                running_peak = equity_curve.cummax()
                drawdown_series = (equity_curve - running_peak) / running_peak
                calculated_max_dd = drawdown_series.min()



            fills_df = engine.trader.generate_fills_report()
            num_trades = len(fills_df) if not fills_df.empty else 0

        # My implementation of calmar ratio
        drawdown_penalty = abs(float(calculated_max_dd))
        if drawdown_penalty == 0:
            drawdown_penalty = 0.0001
        calmar = 0.0
        if calculated_return > 0:
             calmar = calculated_return / drawdown_penalty

        return OptimizationResult(
            rsi_period=rsi_period,
            long_entry=long_entry,
            long_exit=long_exit,
            atr_period=atr_period,
            sensitivity_base=sensitivity_base,
            pyramid_multiplier_base=pyramid_multiplier_base,
            sharpe_ratio=float(sharpe) if sharpe else 0.0,
            calmar_ratio=float(calmar) if calmar else 0.0,
            total_return=float(calculated_return),
            max_drawdown=float(max_drawdown) if max_drawdown else 0.0,
            total_pnl=float(total_pnl),
            num_trades=num_trades,
        )

    except Exception as e:
        print(
            f"⚠️  Error running backtest with params (rsi={rsi_period}, entry={long_entry}, exit={long_exit}): {e}"
        )
        # Return a result with zero performance
        return OptimizationResult(
            rsi_period=rsi_period,
            long_entry=long_entry,
            long_exit=long_exit,
            atr_period=atr_period,
            sensitivity_base=sensitivity_base,
            pyramid_multiplier_base=pyramid_multiplier_base,
            sharpe_ratio=0.0,
            calmar_ratio=0.0,
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
    return [
        Integer(6, 14, name="rsi_period"),
        Real(20.0, 45.0, name="long_entry"),
        Real(55.0, 70.0, name="long_exit"),
        Integer(10, 40, name="atr_period"),
        Real(0.2, 0.8, name="sensitivity_base"),
        Real(1.2, 2.0, name="pyramid_multiplier_base"),
    ]


def evaluate_parameter_combination(
    config: Dict[str, Any],
    rsi_period: int,
    long_entry: float,
    long_exit: float,
    atr_period: int,
    sensitivity_base: float,
    pyramid_multiplier_base: float,
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
    return run_with_params(
        config,
        rsi_period,
        long_entry,
        long_exit,
        atr_period,
        sensitivity_base,
        pyramid_multiplier_base,
    )


def optimize_parameters(config_path: str) -> List[OptimizationResult]:
    """
    Main optimization function - YOU MUST IMPLEMENT THIS.

    This function should implement your chosen optimization algorithm to find
    the best parameter combinations for the strategy.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        List of OptimizationResult sorted by your chosen metric (e.g., Sharpe ratio)

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

    # TODO: IMPLEMENT YOUR OPTIMIZATION METHOD HERE
    results: List[OptimizationResult] = []
    # Bayesian Search CV Implementation

    # @used_named_args is actually kinda goated
    @use_named_args(param_ranges)
    # Backtest Function
    def backtest(
        rsi_period,
        long_entry,
        long_exit,
        atr_period,
        sensitivity_base,
        pyramid_multiplier_base,
    ):
        result = evaluate_parameter_combination(
            config,
            int(rsi_period),
            float(long_entry),
            float(long_exit),
            int(atr_period),
            float(sensitivity_base),
            float(pyramid_multiplier_base),
        )

        results.append(result)
        score = result.calmar_ratio
        # For whatever reasongp_minimize minimizes the 
        # return value, so we return negative score

        return -score

    # To track what iteration we are on when optimizing so I can eat
    def tracking_number(res):
        n_calls = len(res.x_iters)
        print(f"Iteration {n_calls:<2}/ 50")

    # Bayesian Optimization with parameters
    res = gp_minimize(
        func=backtest,
        dimensions=param_ranges,
        n_calls=50,
        n_random_starts=10,
        random_state=6767,
        callback=[tracking_number],
    )

    # Sort Results
    results.sort(key=lambda x: x.calmar_ratio, reverse=True)

    return results


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
    
    # I implemented a version of the calmar ratio towards the end of
    # run_with_params()

    return result.calmar_ratio
    return result.sharpe_ratio


def print_optimization_summary(results: List[OptimizationResult], top_n: int = 10):
    """
    Print summary of optimization results.

    Args:
        results: List of optimization results (sorted by Calmar ratio)
        top_n: Number of top results to display
    """
    print("\n" + "=" * 100)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("=" * 100)

    if not results:
        print("❌ No results to display")
        return

    # Best result
    best = results[0]
    print(f"\n🏆 Best Parameters (by Calmar Ratio):")
    print(f"   RSI Period: {best.rsi_period}")
    print(f"   Long Entry: {best.long_entry}")
    print(f"   Long Exit: {best.long_exit}")
    print(f"   ATR Period: {best.atr_period}")
    print(f"   Sensitivity: {best.sensitivity_base}")
    print(f"   Pyramid Multiplier: {best.pyramid_multiplier_base}")
    print(f"\n   Performance Metrics:")
    print(f"   Calmar Ratio: {best.calmar_ratio:.2f}")
    print(f"   Sharpe Ratio: {best.sharpe_ratio:.2f}")
    print(f"   Total Return: {best.total_return:.2%}")
    print(f"   Max Drawdown: {best.max_drawdown:.2%}")
    print(f"   Total PnL: ${best.total_pnl:,.2f}")
    print(f"   Number of Trades: {best.num_trades}")

    # Top N results
    print(f"\n📊 Top {min(top_n, len(results))} Configurations:")
    print(
        f"{'Rank':<6} {'RSI':<6} {'Entry':<8} {'Exit':<8} {'ATR': <6} {'Sens':<7} {'Pyramid':<8} {'Sharpe':<10} {'Calmar':<10} {'Return':<12} {'PnL':<15} {'Trades':<8}"
    )
    print("-" * 150)

    for i, result in enumerate(results[:top_n], 1):
        print(
            f"{i:<6} {result.rsi_period:<6} {result.long_entry:<8.1f} {result.long_exit:<8.1f} {result.atr_period:<6} {result.sensitivity_base:<8.1f} {result.pyramid_multiplier_base:<8.1f}"
            f"{result.sharpe_ratio:<10.2f} {result.calmar_ratio:<10.2f} {result.total_return:<12.2%} "
            f"${result.total_pnl:<14,.2f} {result.num_trades:<8}"
        )

    # Statistics
    if len(results) > 0:
        sharpe_ratios = [r.sharpe_ratio for r in results if r.sharpe_ratio > 0]
        returns = [r.total_return for r in results if r.total_return != 0]

        if sharpe_ratios:
            print(f"\n📈 Statistics:")
            print(
                f"   Average Sharpe Ratio: {sum(sharpe_ratios) / len(sharpe_ratios):.2f}"
            )
            print(f"   Max Sharpe Ratio: {max(sharpe_ratios):.2f}")
            print(f"   Min Sharpe Ratio: {min(sharpe_ratios):.2f}")

        if returns:
            print(f"   Average Return: {sum(returns) / len(returns):.2%}")
            print(f"   Max Return: {max(returns):.2%}")
            print(f"   Min Return: {min(returns):.2%}")

    if len(results) > 0:
        calmars = [r.calmar_ratio for r in results if r.calmar_ratio > 0]
        if calmars:
            print(f"\n📈 Statistics:")
            print(f"   Average Calmar: {sum(calmars) / len(calmars):.2f}")
            print(f"   Max Calmar: {max(calmars):.2f}")


    print("\n" + "=" * 100)

    # Recommendation
    print("\n💡 Recommendation:")
    print(f"   Update config/backtest_gc.yaml with the best parameters:")
    print(f"   rsi_period: {best.rsi_period}")
    print(f"   long_entry: {best.long_entry}")
    print(f"   long_exit: {best.long_exit}")


def plot_stability_heatmap(results):
    """
    Plots a stability heatmap of RSI Period vs Long Entry to
      check for 'Plateaus' as opposed to 'Spikes/Islands'.
    """
    x = [i.rsi_period for i in results]
    y = [i.long_entry for i in results]
    z = [i.total_return for i in results]

    # Create grid
    plt.figure(figsize=(10, 6))
    plt.tricontourf(x, y, z, levels=20, cmap="RdYlGn")
    plt.colorbar(label="Total Return")

    plt.xlabel("RSI Period")
    plt.ylabel("Long Entry Threshold")
    plt.title("Optimization Stability Surface")

    plt.scatter(x, y, c="black", s=5, alpha=0.3)
    plt.show()

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

        # Stability Heatmap
        plot_stability_heatmap(results)

        print("\n✅ Optimization complete!")

    except NotImplementedError as e:
        print(f"\n❌ {e}")
        print("\n💡 You need to implement the optimize_parameters() function.")
        print(
            "   See the function docstring for guidance on different optimization approaches."
        )
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
