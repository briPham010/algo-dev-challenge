"""
RSI Algorithm Strategy Template

This strategy template implements the Pine Script RSI Algo V4 logic using Nautilus Trader.
The template is structured with TODO sections for you to complete the implementation.

Strategy Logic (from Pine Script RSI Algo V4):
- Enter long positions when RSI crosses below long_entry threshold (oversold)
- Exit long positions when RSI crosses above long_exit threshold (overbought)
- Support pyramid logic for adding to positions
- Optional: RSI divergence detection

See Nautilus Strategy Docs: https://nautilustrader.io/docs/latest/strategies/
"""

from typing import Optional
import pandas as pd
import numpy as np

from .indicators import rsi

from nautilus_trader.trading.strategy import Strategy, StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, PositionSide, PriceType
from nautilus_trader.model.events import PositionOpened, PositionClosed
from nautilus_trader.model.objects import Quantity
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.core.message import Event

# ====================================================================
# Indicator Imports - Choose ONE approach:
# ====================================================================
#
# OPTION 1: Use Nautilus Trader's built-in indicators (RECOMMENDED)
# Uncomment the following lines:
from nautilus_trader.indicators.momentum import RelativeStrengthIndex
from nautilus_trader.indicators.volatility import AverageTrueRange
from nautilus_trader.indicators import ExponentialMovingAverage
from nautilus_trader.indicators import MovingAverageConvergenceDivergence
from nautilus_trader.model.enums import PriceType

#
# OPTION 2: Use custom pandas-based indicators (ALTERNATIVE)
# Uncomment the following line if you prefer custom implementations:
# from .indicators import rsi, ema, atr, macd

#
# See indicators.py for more information about both approaches.
# ====================================================================


class RsiAlgoConfig(StrategyConfig):
    """Configuration for RSI Algorithm Strategy."""

    instrument_id: str
    bar_type: str
    rsi_period: int = 14
    long_entry: float = 31.0  # RSI threshold for entering long (oversold)
    long_exit: float = 83.0  # RSI threshold for exiting long (overbought)
    base_qty: int = 2  # Base position size in contracts
    enable_pyramid: bool = True
    max_position: int = 6  # Maximum total position size


class RsiAlgoStrategy(Strategy):
    """
    RSI-based trading strategy that enters long positions on oversold conditions
    and exits on overbought conditions.

    This template mirrors the Pine Script RSI Algo V4 structure but leaves
    implementation details for you to complete.

    Execution occurs on bar close — similar to TradingView behavior.
    """

    def __init__(self, config: RsiAlgoConfig):
        super().__init__(config=config)

        # Store configuration values (don't store config object as it's read-only)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)

        # Construct bar_type with instrument prefix if not already included
        if config.bar_type.startswith(str(self.instrument_id)):
            # Already has instrument prefix
            self.bar_type = BarType.from_str(config.bar_type)
        else:
            # Add instrument prefix
            bar_type_str = f"{self.instrument_id}-{config.bar_type}"
            self.bar_type = BarType.from_str(bar_type_str)

        # Strategy parameters
        self.rsi_period = config.rsi_period
        self.long_entry = config.long_entry
        self.long_exit = config.long_exit
        self.base_qty = config.base_qty
        self.enable_pyramid = config.enable_pyramid
        self.max_position = config.max_position
        self.macd_current = 0
        self.macd_position = 0
        self.current_atr = 0
        self.signal_current = 0

        # State variables
        # TODO: Add any additional state variables you need
        self.highs: list[float] = []
        self.lows: list[float] = []
        self.position: Optional = None
        self.prices: list[float] = (
            []
        )  # Store recent prices for indicator calculation (needed for custom indicators)
        self.bars: list[Bar] = []  # Store recent bars

        self.bar_index = 0

        # Pyramid logic here
        self.pyramid_count = 0
        self.last_entry_bar_index = 0
        self.last_entry_rsi_level = 0.0

        # Indicator values (updated on each bar)
        self.rsi_indicator = RelativeStrengthIndex(period=self.rsi_period)
        self.rsi_indicator = RelativeStrengthIndex(period=14)
        self.atr_indicator = AverageTrueRange(period=14)
        self.macd_indicator = MovingAverageConvergenceDivergence(fast_period=12, slow_period=26, price_type=PriceType.LAST)
        self.signal_indicator = ExponentialMovingAverage(period=9)

        self.current_rsi: Optional[float] = None
        self.previous_rsi: Optional[float] = None
        self.current_atr = None
        self.macd_current = None
        self.signal_current = None


        self.vol_short_indicator = ExponentialMovingAverage(period=46)
        self.vol_long_indicator = ExponentialMovingAverage(period=92)
        self.is_volume_decreasing = False

        # RSI Divergence Variables Here

        self.len = 22
        self.src = "close"
        self.lbR = 5
        self.lbL = 1
        self.range_upper = 60
        self.range_lower = 5
        self.plot_bull = True
        self.plot_hidden_bull = True
        self.plot_bear = True
        self.plot_hidden_bear = False

        # Volumes

        self.volumes: list[float] = []

        # ====================================================================
        # Initialize Indicators - Choose ONE approach:
        # ====================================================================
        #
        # OPTION 1: Initialize Nautilus built-in indicators (RECOMMENDED)
        # Uncomment the following lines:
        # from nautilus_trader.indicators.momentum import RelativeStrengthIndex
        # self.rsi_indicator = RelativeStrengthIndex(period=self.rsi_period)
        #
        # OPTION 2: Custom indicators don't need initialization here
        # They are calculated on-the-fly in on_bar() using pandas
        #
        # ====================================================================

        # Track entry/exit conditions
        self.last_entry_signal: Optional[bool] = None
        self.last_exit_signal: Optional[bool] = None

    def on_start(self):
        """
        Called once when the strategy starts.
        Subscribe to market data and initialize indicators.

        See Nautilus Strategy Docs: https://nautilustrader.io/docs/latest/strategies/
        """
        # Subscribe to bars for the instrument
        self.subscribe_bars(bar_type=self.bar_type)

        self._log.info(
            f"RSI Algorithm Strategy started for {self.instrument_id} "
            f"(RSI period: {self.rsi_period}, Entry: {self.long_entry}, Exit: {self.long_exit})"
        )

        # TODO: Initialize any additional indicators or state here
        #
        # If using Nautilus built-in indicators, they should already be initialized in __init__
        # If using custom indicators, no additional initialization needed here
        #
        # Example for additional built-in indicators:
        # from nautilus_trader.indicators.trend import ExponentialMovingAverage
        # self.ema_indicator = ExponentialMovingAverage(period=20)

    def on_stop(self):
        """
        Called when the strategy stops.
        Clean up and close any open positions.
        """
        try:
            # Close all open positions before stopping
            instrument = self.cache.instrument(self.instrument_id)
            if instrument:
                positions_open = self.cache.positions_open(venue=instrument.venue)
                for pos in positions_open:
                    if pos.instrument_id == self.instrument_id and pos.is_open:
                        self.close_position(pos)
        finally:
            self.unsubscribe_bars(bar_type=self.bar_type)

        self._log.info("RSI Algorithm Strategy stopped")

    def on_bar(self, bar: Bar):
        """
        Called on each bar close.
        This is where the main trading logic goes.

        Execution occurs on bar close — similar to TradingView behavior.

        Args:
            bar: The bar that just closed
        """
        # Update bar history
        self.bar_index += 1
        self.bars.append(bar)

        # Extract close price (handle both Price object and direct float)
        try:
            close_price = (
                float(bar.close.as_double())
                if hasattr(bar.close, "as_double")
                else float(bar.close)
            )
        except (AttributeError, TypeError, ValueError):
            # Fallback: try direct conversion
            close_price = float(bar.close)
        self.prices.append(close_price)

        try:
            h = (
                float(bar.high.as_double())
                if hasattr(bar.high, "as_double")
                else float(bar.high)
            )
            l = (
                float(bar.low.as_double())
                if hasattr(bar.low, "as_double")
                else float(bar.low)
            )
        except:
            h, l = float(bar.high), float(bar.low)

        self.highs.append(h)
        self.lows.append(l)

        # Keep lists same size
        max_bars = max(self.rsi_period * 3, 100)
        if len(self.prices) > max_bars:
            self.highs = self.highs[-max_bars:]
            self.lows = self.lows[-max_bars:]

        # Keep only recent bars for indicator calculation (e.g., last 100 bars)
        # This prevents memory issues with very long backtests
        max_bars = max(self.rsi_period * 3, 100)
        if len(self.bars) > max_bars:
            self.bars = self.bars[-max_bars:]
            self.prices = self.prices[-max_bars:]

        # max_len = 100
        # if len(self.prices) > max_len:
        #     self.prices = self.prices[-max_len:]
        #     self.highs = self.highs[-max_len:]
        #     self.lows = self.lows[-max_len:]
        #     self.volumes = self.volumes[-max_len:]

        # Update position reference (find open position for this instrument)
        self.position = None
        instrument = self.cache.instrument(self.instrument_id)
        if instrument:
            positions_open = self.cache.positions_open(venue=instrument.venue)
            for pos in positions_open:
                if pos.instrument_id == self.instrument_id:
                    self.position = pos
                    break

        try:
            v = float(bar.volume.as_double())
        except:
            v = float(bar.volume)

        self.volumes.append(v)

        # ====================================================================
        # === TODO 1: Compute indicators ===
        # ====================================================================
        #
        # Calculate RSI and any other indicators you need.
        #
        # Choose ONE of the following approaches:
        #
        # ====================================================================
        # OPTION 1: Use Nautilus Trader's Built-in Indicators (RECOMMENDED)
        # ====================================================================
        #
        # Built-in indicators are optimized, well-tested, and integrate seamlessly
        # with Nautilus Trader's bar handling system.
        #
        # Example implementation:
        # Update RSI indicator with the new bar
        # self.rsi_indicator.handle_bar(bar)

        # # Check if indicator is ready (has enough data)
        # if self.rsi_indicator.initialized:
        #     # Get current RSI value
        #     self.current_rsi = self.rsi_indicator.value

        #     # Store previous value for crossover detection
        #     if hasattr(self, "_prev_rsi"):
        #         self.previous_rsi = self._prev_rsi
        #     self._prev_rsi = self.current_rsi
        #
        # Available built-in indicators:
        #   - RelativeStrengthIndex (from nautilus_trader.indicators.momentum)
        #   - ExponentialMovingAverage (from nautilus_trader.indicators.trend)
        #   - AverageTrueRange (from nautilus_trader.indicators.volatility)
        #   - And many more: https://nautilustrader.io/docs/latest/api_reference/indicators/
        #
        # ====================================================================
        # OPTION 2: Use Custom Pandas-based Indicators (ALTERNATIVE)
        # ====================================================================
        #
        # Custom indicators from indicators.py provide flexibility if you need
        # custom logic or prefer pandas-based calculations.
        #
        # Example implementation:
        # if len(self.prices) >= self.rsi_period:
        #     prices_series = pd.Series(self.prices)
        #     rsi_values = rsi(prices_series, period=self.rsi_period)
        #     self.current_rsi = rsi_values.iloc[-1]  # Most recent RSI value
        #     self.previous_rsi = rsi_values.iloc[-2] if len(rsi_values) > 1 else None
        #
        # See indicators.py for available custom indicator functions.
        #
        # ====================================================================

        # TODO: Implement indicator calculation here

        # Choose either Option 1 (built-in) or Option 2 (custom)
        # if len(self.prices) < self.rsi_period + 2:
        #     return

        # # Convert Lists to Pandas Series
        # s_close = pd.Series(self.prices)
        # s_high = pd.Series(self.highs)
        # s_low = pd.Series(self.lows)
        # s_volume = pd.Series(self.volumes)

        # # Calculate RSI
        # rsi_series = rsi(s_close, period=self.rsi_period)

        # # Store current and previous values safely
        # if not rsi_series.empty:
        #     self.current_rsi = rsi_series.iloc[-1]
        #     # Check if we have at least 2 values before accessing [-2]
        #     if len(rsi_series) > 1:
        #         self.previous_rsi = rsi_series.iloc[-2]

        # # Calculate MACD
        # macd_line, signal_line, _ = macd(
        #     s_close, fast_period=12, slow_period=26, signal_period=9
        # )

        # # Store for use in Entry Logic
        # self.macd_current = macd_line.iloc[-1]
        # self.signal_current = signal_line.iloc[-1]

        # # Calculate ATR
        # atr_series = atr(s_high, s_low, s_close, period=14)
        # self.current_atr = atr_series.iloc[-1]

        # # Debug logging (Optional: remove later)
        # # self._log.info(f"RSI: {self.current_rsi:.2f} | MACD: {self.macd_current:.2f}")

        self.atr_indicator.handle_bar(bar)
        self.macd_indicator.handle_bar(bar)

        if self.macd_indicator.initialized:
            raw_macd_value = self.macd_indicator.value
            self.signal_indicator.update_raw(raw_macd_value)


        # Check if indicators are ready
        self.rsi_indicator.handle_bar(bar)
        if self.rsi_indicator.initialized:
            self.current_rsi = self.rsi_indicator.value * 100

        # Store previous value for crossover detection
        if hasattr(self, "_prev_rsi"):
            self.previous_rsi = self._prev_rsi
        self._prev_rsi = self.current_rsi

        self.current_atr = self.atr_indicator.value
        self.macd_current = self.macd_indicator.value
        self.signal_current = self.signal_indicator.value

        # RSI Divergence Indicators
        # Calculate short-term volume average

        # self.vol_short_indicator.update_raw(v)
        # self.vol_long_indicator.update_raw(v)
        # vol_short = self.vol_short_indicator.value
        # vol_long = self.vol_long_indicator.value
        # self.is_volume_decreasing = vol_short < vol_long

        # vol_short = ema(s_volume, period=46).iloc[-1]
        # vol_long = ema(s_volume, period=92).iloc[-1]
        # self.is_volume_decreasing = vol_short < vol_long

        # self.osc = rsi(s_close, self.len)

        if self.bar_index % 10 == 0:
            self._log.info(
                f"BAR: {self.bar_index} | "
                f"MACD: {self.macd_current:.5f} | "
                f"SIGNAL: {self.signal_current:.5f} | "
                f"Safe?: {self.macd_current <= self.signal_current}")

        # If indicators aren't ready yet, skip trading logic
        if self.current_rsi is None:
            return

        # ====================================================================
        # === TODO 2: Implement long entry logic ===
        # ====================================================================
        #
        # Enter long positions when RSI crosses below long_entry threshold.
        #
        # Logic:
        # - Check if RSI has crossed below long_entry (oversold condition)
        # - Only enter if we don't already have a long position
        # - Use self.enter_long() helper method to place the order
        #
        # Hints:
        # - Compare current_rsi and previous_rsi to detect crossovers
        # - Check self.position to see if we already have a position
        # - Use self.is_long property to check for long positions
        #
        # Example structure:
        #   if not self.is_long and self.current_rsi < self.long_entry:
        #       # Check for crossover (RSI was above threshold, now below)
        #       if self.previous_rsi is not None and self.previous_rsi >= self.long_entry:
        #           self.enter_long(qty=self.base_qty)
        #
        # See Nautilus Strategy Docs for order submission:
        # https://nautilustrader.io/docs/latest/strategies/
        #
        # ====================================================================

        # TODO: Implement long entry logic here

        is_macd_safe = self.macd_current <= self.signal_current

        # Trigger: RSI is below entry level
        is_oversold = self.current_rsi < self.long_entry
        #
        if is_macd_safe and is_oversold:
            if not self.is_long and is_oversold and is_macd_safe:
                # Check for crossover (RSI was above threshold, now below)
                if (
                    self.previous_rsi is not None
                    and self.previous_rsi >= self.long_entry
                ):
                    self.enter_long(qty=self.base_qty)

                    self.pyramid_count = 0
                    self.last_entry_bar_index = self.bar_index
                    self.last_entry_rsi_level = self.long_entry

        # ====================================================================
        # === TODO 3: Implement pyramid logic ===
        # ====================================================================
        #
        # Add to existing long positions when conditions are met (pyramiding).
        #
        # Logic:
        # - Only pyramid if enable_pyramid is True
        # - Check that current position size is less than max_position
        # - Add to position when RSI is still oversold (e.g., RSI < long_entry)
        # - Use appropriate position sizing (e.g., base_qty)
        #
        # Hints:
        # - Check self.position.quantity to get current position size
        # - Use self.enter_long() to add to position
        # - Consider adding logic to prevent over-trading (e.g., wait N bars between additions)
        #
        # Example structure:
        #   if (self.is_long and
        #       self.enable_pyramid and
        #       self.current_rsi < self.long_entry and
        #       self.position.quantity.as_int() < self.max_position):
        #       # Add to position
        #       self.enter_long(qty=self.base_qty)
        #
        # ====================================================================

        # TODO: Implement pyramid logic here
        if (
            self.is_long
            and self.enable_pyramid
            and self.current_rsi < self.long_entry
            and self.position.quantity < self.max_position
        ):

            sensitivity = [0.3, 0.3, 0.3, 0.4]

            longPyramidStep1 = self.current_atr * sensitivity[0]
            longPyramidStep2 = self.current_atr * sensitivity[1]
            longPyramidStep3 = self.current_atr * sensitivity[2]
            longPyramidStep4 = self.current_atr * sensitivity[3]

            multiplier = 1
            multipliers = [1.0, 1.0, 2.0, 2.0]

            # Calculate minimum bars, rsiSteps, and multiplier needed for pyramiding
            min_bars = 0
            if self.pyramid_count == 0:
                min_bars = 10
                rsi_step_required = longPyramidStep1
                multiplier = multipliers[0]
            elif self.pyramid_count == 1:
                min_bars = 15
                rsi_step_required = longPyramidStep2
                multiplier = multipliers[1]
            elif self.pyramid_count == 2:
                min_bars = 20
                rsi_step_required = longPyramidStep3
                multiplier = multipliers[2]
            else:
                min_bars = 25
                rsi_step_required = longPyramidStep4
                multiplier = multipliers[3]

            if (self.bar_index - self.last_entry_bar_index) >= min_bars:
                if self.current_rsi <= (self.last_entry_rsi_level - rsi_step_required):
                    if self.macd_current <= self.signal_current:
                        self.base_qty = int(self.base_qty * multiplier)

                        self.enter_long(qty=self.base_qty)

                        self.pyramid_count += 1
                        self.last_entry_bar_index = self.bar_index
                        self.last_entry_rsi_level -= rsi_step_required

        # ====================================================================
        # === TODO 4: Implement exit logic ===
        # ====================================================================
        #
        # Exit long positions when RSI crosses above long_exit threshold.
        #
        # Logic:
        # - Exit when RSI crosses above long_exit (overbought condition)
        # - Only exit if we have a long position
        # - Use self.exit_long() helper method to close the position
        #
        # Hints:
        # - Compare current_rsi and previous_rsi to detect crossovers
        # - Check self.is_long to ensure we have a position to exit
        # - Consider exiting the entire position or partial exits
        #
        # Example structure:
        #   if self.is_long and self.current_rsi > self.long_exit:
        #       # Check for crossover (RSI was below threshold, now above)
        #       if self.previous_rsi is not None and self.previous_rsi <= self.long_exit:
        #           self.exit_long()
        #
        # ====================================================================

        # TODO: Implement exit logic here
        if self.is_long and self.current_rsi > self.long_exit:
            # Check for crossover (RSI was below threshold, now above)
            if self.previous_rsi is not None and self.previous_rsi <= self.long_exit:
                self.exit_long()
                self.pyramid_count = 0
                self.last_entry_bar_index = 0
                self.last_entry_rsi_level = 0.0

        # ====================================================================
        # === OPTIONAL BONUS: Implement RSI divergence logic ===
        # ====================================================================
        #
        # Detect bullish/bearish RSI divergences for enhanced entry/exit signals.
        #
        # Bullish Divergence:
        # - Price makes lower low, but RSI makes higher low
        # - Strong buy signal
        #
        # Bearish Divergence:
        # - Price makes higher high, but RSI makes lower high
        # - Strong sell signal
        #
        # Hints:
        # - Track price highs/lows over a lookback period
        # - Compare RSI values at those price extremes
        # - Use divergence signals to enhance entry/exit logic
        #
        # ====================================================================

        # TODO (OPTIONAL): Implement RSI divergence detection here

        # self.bull_divergence = False
        # self.bear_divergence = False

        # # Checking for bullish divergence

        # # Check for 11 bars of history before attempting pivot checks
        # if len(self.lows) > 12:
        #     pass

        #     # Now we figure out how to find a pivot low

        #     pivot_search = -1 - self.lbR

        #     pivot_search_low = self.lows[pivot_search]
        #     pivot_search_high = self.highs[pivot_search]
        #     pivot_rsi = rsi_series.iloc[pivot_search]

        #     # awooga
        #     current_window_low = self.lows[pivot_search:]
        #     is_pivot_low = (pivot_search_low == min(current_window_low))

        #     if is_pivot_low:
        #         # WE FOUND A PIVOT HOLY

        #     self.last_low_pivot = pivot_search_low

    def on_event(self, event: Event):
        """
        Handle position events (opened, closed, etc.).

        See Nautilus Strategy Docs: https://nautilustrader.io/docs/latest/strategies/
        """
        if isinstance(event, PositionOpened):
            # Find the position that was just opened
            instrument = self.cache.instrument(self.instrument_id)
            if instrument:
                positions_open = self.cache.positions_open(venue=instrument.venue)
                for pos in positions_open:
                    if pos.instrument_id == self.instrument_id:
                        self.position = pos
                        break
            if self.position is not None:
                self._log.info(
                    f"Position opened: {self.position.side} "
                    f"qty={self.position.quantity} @ {self.position.avg_px_open}"
                )
        elif isinstance(event, PositionClosed):
            # Get realized PnL (may be a property or method)
            realized_pnl = None
            if self.position:
                try:
                    # Try as property first
                    realized_pnl = self.position.realized_pnl
                except (AttributeError, TypeError):
                    try:
                        # Try as method
                        realized_pnl = self.position.realized_pnl()
                    except:
                        pass
            self._log.info(
                f"Position closed: PnL={realized_pnl if realized_pnl is not None else 'N/A'}"
            )
            self.position = None

    # ========================================================================
    # Helper Methods
    # ========================================================================

    @property
    def is_long(self) -> bool:
        """Check if we have a long position."""
        return self.position is not None and self.position.side == PositionSide.LONG

    @property
    def is_short(self) -> bool:
        """Check if we have a short position."""
        return self.position is not None and self.position.side == PositionSide.SHORT

    def enter_long(self, qty: int):
        """
        Enter a long position.

        Args:
            qty: Quantity to buy (number of contracts)
        """
        if self.instrument_id is None:
            self._log.error("Instrument ID not set")
            return

        instrument = self.cache.instrument(self.instrument_id)
        if instrument is None:
            self._log.error(f"Instrument {self.instrument_id} not found in cache")
            return

        # Get current market price
        last_price = self.cache.price(self.instrument_id, PriceType.LAST)
        if last_price is None:
            self._log.warning(f"No price available for {self.instrument_id}")
            return

        # Create market order to enter long
        quantity = Quantity.from_int(qty)

        # Submit buy order
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=quantity,
        )

        self.submit_order(order)
        self._log.info(f"Submitted BUY order: {qty} contracts @ {last_price}")

    def exit_long(self):
        """Exit the current long position."""
        if not self.is_long:
            self._log.warning("No long position to exit")
            return

        if self.position is None:
            return

        # Close the entire position
        self.close_position(self.position)
        self._log.info(f"Exiting long position: qty={self.position.quantity}")
