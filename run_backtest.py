"""
Backtest Runner Script

This script loads the backtest configuration from YAML and runs a Nautilus Trader backtest.

Usage:
    python run_backtest.py

The script will:
1. Load configuration from config/backtest_gc.yaml
2. Initialize the Nautilus Trader backtest engine
3. Run the strategy on the GC 1-minute data
4. Print performance metrics

See Nautilus Backtesting Docs: https://nautilustrader.io/docs/latest/backtesting/
"""

import yaml
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timezone

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig, BacktestDataConfig, BacktestVenueConfig
from nautilus_trader.config import ImportableStrategyConfig, LoggingConfig
from nautilus_trader.model.data import Bar, BarType, BarSpecification
from nautilus_trader.model.identifiers import InstrumentId, Venue, Symbol, TraderId
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.model.instruments import FuturesContract
from nautilus_trader.model.enums import AccountType, OmsType, BarAggregation, PriceType, AssetClass


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _iso_to_ns(iso_str: str) -> int:
    """Convert ISO 8601 string to nanoseconds."""
    s = iso_str.strip()
    if "T" not in s:
        s = s + "T23:59:59Z"
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def create_instrument_from_config(config: Dict[str, Any]) -> tuple[InstrumentId, FuturesContract]:
    """Create InstrumentId and FuturesContract from configuration."""
    instrument_config = config['data']['instrument']
    instrument_id = InstrumentId.from_str(instrument_config['id'])
    venue = Venue(instrument_config['venue'])
    
    # Create FuturesContract
    expiration_ns = _iso_to_ns("2099-12-31T23:59:59Z")
    
    instrument = FuturesContract(
        instrument_id=instrument_id,
        raw_symbol=Symbol(instrument_config['symbol']),
        asset_class=AssetClass.COMMODITY,
        currency=USD,
        price_precision=instrument_config.get('price_precision', 2),
        price_increment=Price.from_str("0.10"),  # $0.10 minimum price movement for GC
        multiplier=Quantity.from_int(100),  # 100 oz per contract
        lot_size=Quantity.from_int(instrument_config.get('lot_size', 1)),
        underlying=instrument_config['symbol'],
        activation_ns=0,
        expiration_ns=expiration_ns,
        ts_event=0,
        ts_init=0,
    )
    
    return instrument_id, instrument


def load_parquet_to_bars(data_path: Path, instrument_id: InstrumentId, start_time: str = None, end_time: str = None) -> list[Bar]:
    """
    Load OHLCV data from Parquet file and convert to Nautilus Bar objects.
    
    Args:
        data_path: Path to the parquet file
        instrument_id: The instrument ID for the bars
        start_time: Optional start time filter (ISO format)
        end_time: Optional end time filter (ISO format)
    
    Returns:
        List of Bar objects
    """
    # Read parquet file
    df = pd.read_parquet(data_path)
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    # Filter by time range if provided
    if start_time:
        start_ts = pd.to_datetime(start_time, utc=True)
        df = df[df["timestamp"] >= start_ts]
    if end_time:
        end_ts = pd.to_datetime(end_time, utc=True)
        df = df[df["timestamp"] <= end_ts]
    
    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Create BarType (1-minute bars, last price, external)
    bar_spec = BarSpecification(
        step=1,
        aggregation=BarAggregation.MINUTE,
        price_type=PriceType.LAST
    )
    bar_type = BarType(instrument_id, bar_spec)
    
    # Convert to Nautilus Bar objects
    bars = []
    for _, row in df.iterrows():
        # Convert timestamp to nanoseconds
        ts_event_ns = int(pd.Timestamp(row["timestamp"]).timestamp() * 1_000_000_000)
        ts_init_ns = ts_event_ns  # Use same timestamp for init
        
        # Create Price objects
        open_price = Price.from_str(f"{row['open']:.2f}")
        high_price = Price.from_str(f"{row['high']:.2f}")
        low_price = Price.from_str(f"{row['low']:.2f}")
        close_price = Price.from_str(f"{row['close']:.2f}")
        
        # Create Quantity for volume
        volume = Quantity.from_str(f"{row['volume']:.0f}")
        
        # Create Bar object
        bar = Bar(
            bar_type=bar_type,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            ts_event=ts_event_ns,
            ts_init=ts_init_ns,
        )
        
        bars.append(bar)
    
    return bars


def setup_backtest_engine(config: Dict[str, Any]) -> BacktestEngine:
    """
    Set up and configure the Nautilus Trader backtest engine.
    
    See Nautilus Backtesting Docs: https://nautilustrader.io/docs/latest/backtesting/
    """
    # Create engine configuration
    engine_config = BacktestEngineConfig(
        trader_id=TraderId(config['engine'].get('trader_id', 'BACKTEST-001')),
        logging=LoggingConfig(
            log_level=config.get('logging', {}).get('log_level', 'INFO')
        ),
    )
    
    # Create backtest engine
    engine = BacktestEngine(config=engine_config)
    
    # Set up venue
    venue_config = config['venue']
    venue = Venue(venue_config['name'])
    
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money.from_str(bal) for bal in venue_config['starting_balances']],
    )
    
    # Create and add instrument
    instrument_id, instrument = create_instrument_from_config(config)
    engine.add_instrument(instrument)
    
    # Load strategy configuration
    strategy_config_dict = config['engine']['strategies'][0]
    strategy_config = ImportableStrategyConfig(
        strategy_path=f"{strategy_config_dict['module']}:{strategy_config_dict['class']}",
        config_path=f"{strategy_config_dict['module']}:RsiAlgoConfig",
        config=strategy_config_dict['config'],
    )
    
    # Add strategy to engine
    from nautilus_trader.config import StrategyFactory
    strategy = StrategyFactory.create(strategy_config)
    engine.add_strategy(strategy)
    
    # Load data from Parquet file
    data_config = config['data']['bar_data']
    data_path_str = data_config['path']
    
    # Resolve relative paths relative to config file location
    if not Path(data_path_str).is_absolute():
        # Assume relative to project root (where config file is)
        script_dir = Path(__file__).parent
        data_path = script_dir / data_path_str
    else:
        data_path = Path(data_path_str)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"Loading data from: {data_path}")
    bars = load_parquet_to_bars(
        data_path=data_path,
        instrument_id=instrument_id,
        start_time=data_config.get('start_time'),
        end_time=data_config.get('end_time'),
    )
    
    print(f"Loaded {len(bars):,} bars")
    engine.add_data(bars)
    
    return engine


def print_performance_summary(engine: BacktestEngine):
    """Print backtest performance metrics."""
    print("\n" + "="*80)
    print("BACKTEST PERFORMANCE SUMMARY")
    print("="*80)
    
    # Get portfolio statistics
    try:
        stats_returns = engine.portfolio.analyzer.get_performance_stats_returns()
        stats_general = engine.portfolio.analyzer.get_performance_stats_general()
        stats_pnls = engine.portfolio.analyzer.get_performance_stats_pnls()
        
        print("\n📊 Performance Metrics:")
        print(f"  Total Return: {stats_returns.get('Total Return', 'N/A'):.2%}" if 'Total Return' in stats_returns else "  Total Return: N/A")
        print(f"  Sharpe Ratio: {stats_returns.get('Sharpe Ratio (252 days)', 'N/A'):.2f}" if 'Sharpe Ratio (252 days)' in stats_returns else "  Sharpe Ratio: N/A")
        print(f"  Max Drawdown: {stats_general.get('Max Drawdown', 'N/A'):.2%}" if 'Max Drawdown' in stats_general else "  Max Drawdown: N/A")
        print(f"  Win Rate: {stats_pnls.get('Win Rate', 'N/A'):.2%}" if 'Win Rate' in stats_pnls else "  Win Rate: N/A")
        print(f"  Profit Factor: {stats_returns.get('Profit Factor', 'N/A'):.2f}" if 'Profit Factor' in stats_returns else "  Profit Factor: N/A")
        
        # Get account report
        venues = engine.list_venues()
        if venues:
            venue = venues[0]
            account_df = engine.trader.generate_account_report(venue)
            if not account_df.empty and 'total' in account_df.columns:
                initial_equity = account_df['total'].iloc[0]
                final_equity = account_df['total'].iloc[-1]
                
                # Convert to float if Money object
                if isinstance(initial_equity, str):
                    try:
                        initial_equity = float(initial_equity.split()[0])
                    except:
                        pass
                if isinstance(final_equity, str):
                    try:
                        final_equity = float(final_equity.split()[0])
                    except:
                        pass
                
                total_pnl = final_equity - initial_equity if isinstance(final_equity, (int, float)) and isinstance(initial_equity, (int, float)) else 0
                
                print(f"\n💰 Account Summary:")
                if isinstance(initial_equity, (int, float)):
                    print(f"  Initial Equity: ${initial_equity:,.2f}")
                else:
                    print(f"  Initial Equity: {initial_equity}")
                if isinstance(final_equity, (int, float)):
                    print(f"  Final Equity: ${final_equity:,.2f}")
                else:
                    print(f"  Final Equity: {final_equity}")
                if isinstance(total_pnl, (int, float)):
                    print(f"  Total PnL: ${total_pnl:,.2f}")
                else:
                    print(f"  Total PnL: {total_pnl}")
        
        # Get trades summary
        fills_df = engine.trader.generate_fills_report()
        if not fills_df.empty:
            print(f"\n📈 Trading Summary:")
            print(f"  Total Trades: {len(fills_df)}")
            if 'realized_pnl' in fills_df.columns:
                winning_trades = fills_df[fills_df['realized_pnl'] > 0]
                losing_trades = fills_df[fills_df['realized_pnl'] < 0]
                print(f"  Winning Trades: {len(winning_trades)}")
                print(f"  Losing Trades: {len(losing_trades)}")
        
    except Exception as e:
        print(f"\n⚠️  Error generating performance summary: {e}")
        print("This may be due to incomplete backtest execution or data issues.")
    
    print("\n" + "="*80)


def main():
    """Main entry point for running backtest."""
    # Get script directory
    script_dir = Path(__file__).parent
    config_path = script_dir / "config" / "backtest_gc.yaml"
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    
    print("🚀 Starting Nautilus Trader Backtest")
    print(f"📁 Configuration: {config_path}")
    
    # Load configuration
    try:
        config = load_config(str(config_path))
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)
    
    # Set up backtest engine
    try:
        print("\n🔧 Setting up backtest engine...")
        engine = setup_backtest_engine(config)
        print("✅ Backtest engine configured")
    except Exception as e:
        print(f"❌ Error setting up backtest engine: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Run backtest
    try:
        print("\n▶️  Running backtest...")
        engine.run()
        print("✅ Backtest completed")
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print performance summary
    print_performance_summary(engine)
    
    print("\n✅ Backtest run complete!")
    print("\n💡 Tip: Use optimize_params.py to find optimal parameters")


if __name__ == "__main__":
    main()



    # Hours burned on this project since 12/18/25
    # 15
