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
from collections import deque

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
    rsi_period: int = 11
    long_entry: float = 25  # RSI threshold for entering long (oversold)
    long_exit: float = 67  # RSI threshold for exiting long (overbought)
    base_qty: int = 2  # Base position size in contracts
    enable_pyramid: bool = True
    max_position: int = 6  # Maximum total position size

    # Configurations for Optimization
    atr_period = 14
    sensitivity = 0.3
    pyramid_multiplier: float = 2


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
        self.atr_period = config.atr_period
        self.pyramid_multiplier = config.pyramid_multiplier

        # State variables
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
        self.max_pyramids = 6
        self.last_entry_bar_index = 0
        self.last_entry_rsi_level = 0.0

        # Indicator values (updated on each bar)
        self.rsi_indicator = RelativeStrengthIndex(period=self.rsi_period)
        self.atr_indicator = AverageTrueRange(period=self.atr_period)
        self.macd_indicator = MovingAverageConvergenceDivergence(
            fast_period=12, slow_period=26, price_type=PriceType.LAST
        )
        self.signal_indicator = ExponentialMovingAverage(period=9)

        self.current_rsi: Optional[float] = None
        self.previous_rsi: Optional[float] = None
        self.current_atr = None
        self.macd_current = None
        self.signal_current = None
        self.require_macd_for_long: bool = True

        # Other optimization fields
        self.sensitivity_base = config.sensitivity # 0.3 is default

        # RSI Divergence Detection Variables
        self.vol_short_indicator = ExponentialMovingAverage(period=46)
        self.vol_long_indicator = ExponentialMovingAverage(period=92)
        self.is_volume_decreasing = False
        self.osc_rsi_period = 22
        self.osc_rsi_indicator = RelativeStrengthIndex(period=self.osc_rsi_period)
        self.current_osc: Optional[float] = None
        self.lbR = 5
        self.lbL = 1
        self.range_upper = 60
        self.range_lower = 5
        self.window_size = self.lbL + self.lbR + 1

        self.prev_bull_cond = False
        self.prev_bear_cond = False

        # DOUBLE ENDED QUEUES TO TRACK PIVOTS (and oscillator)
        self.osc_window = deque(maxlen=self.window_size)
        self.high_window = deque(maxlen=self.window_size)
        self.low_window = deque(maxlen=self.window_size)

        # Other oscillator variables
        self.prev_osc_low = None
        self.prev_price_low = None
        self.prev_osc_high = None
        self.prev_price_high = None

        self.bullish_divergence = False
        self.bearish_divergence = False
        self.prev_bullish_divergence = False
        self.prev_bearish_divergence = False

        self.prev_osc_low_index = 0
        self.prev_osc_high_index = 0

        self.range_lower = 5
        self.range_upper = 60

        # Volumes
        self.volumes: list[float] = []

        # Track entry/exit conditions
        self.last_entry_signal: Optional[bool] = None
        self.last_exit_signal: Optional[bool] = None

    def pivothigh(self, window):
        """
        Python implementation of Pine Script's pivothigh.
        Returns value if valid, None otherwise.
        """

        # Check if window size is sufficient to search
        if len(window) != self.window_size:
            return None

        # Go back by x indexes
        current_index = self.lbR

        # Convert double queues to lists and place pointer
        w_list = list(window)
        target = w_list[current_index]

        # Search left of pointer for something
        # smaller, otherwise exit search
        left_side = w_list[:current_index]
        if left_side and max(left_side) > target:
            return None

        # Search right of pointer for something
        # smaller, otherwise exit search
        right_side = w_list[current_index + 1 :]
        if right_side and max(right_side) >= target:
            return None

        # If our pointer is the greatest point in the window,
        # then we've found a high pivot!
        return target

    def pivotlow(self, window):
        """
        Python implementation of Pine Script's pivotlow.
        Returns value if valid, None otherwise.
        """
        if len(window) != self.window_size:
            return None

        # Go back x indexes
        current_index = self.lbR

        # Convert double queues to lists and place pointer
        w_list = list(window)
        target = w_list[current_index]

        # Search left of pointer for something
        # greater, otherwise exit the search
        left_side = w_list[:current_index]
        if left_side and min(left_side) < target:
            return None

        # Search right of pointer for something
        #  greater, otherwise ext the search
        right_side = w_list[current_index + 1 :]
        if right_side and min(right_side) <= target:
            return None

        # If our pointer is the lowest point in the window,
        # then we've found a low pivot!
        return target

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
            f"(RSI period: {self.rsi_period}, Entry: {self.long_entry}, Exit: {self.long_exit})",
        )

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

        if self.bar_index % 50000 == 0:
            self.log.info(f"bar_index={self.bar_index}")

        # ====================================================================
        # === 1: Compute indicators ===
        # ====================================================================
        #

        # Indicator calculations here
        self.atr_indicator.handle_bar(bar)
        self.macd_indicator.handle_bar(bar)
        self.rsi_indicator.handle_bar(bar)
        self.osc_rsi_indicator.handle_bar(bar)

        # Initialize MACD
        if self.macd_indicator.initialized:
            raw_macd_value = self.macd_indicator.value
            self.signal_indicator.update_raw(raw_macd_value)

        # Check if indicators are ready
        if (
            self.rsi_indicator.initialized
            and self.atr_indicator.initialized
            and self.osc_rsi_indicator.initialized
        ):

            self.current_rsi = self.rsi_indicator.value * 100
            self.current_osc = self.osc_rsi_indicator.value * 100
        else:
            return  # Skip if indicators aren't ready

        # Store previous value for crossover detection
        if hasattr(self, "_prev_rsi"):
            self.previous_rsi = self._prev_rsi
        self._prev_rsi = self.current_rsi

        # Compute indicators
        self.current_atr = self.atr_indicator.value
        self.macd_current = self.macd_indicator.value
        self.signal_current = self.signal_indicator.value

        # RSI Divergence Indicators
        # Calculate short-term volume average
        self.vol_long_indicator.update_raw(v)
        self.vol_short_indicator.update_raw(v)
        vol_short = self.vol_short_indicator.value
        vol_long = self.vol_long_indicator.value
        self.is_volume_decreasing = vol_short < vol_long

        # Update Double Queues (Rolling windows for RSI Divergence)
        if self.current_rsi is not None:
            self.osc_window.append(self.current_osc)
            self.high_window.append(h)
            self.low_window.append(l)

        # If indicators aren't ready yet, skip trading logic
        if self.current_rsi is None:
            return

        # ====================================================================
        # === OPTIONAL BONUS: Implement RSI divergence logic ===
        # ====================================================================

        bull_cond = False
        bear_cond = False
        # RSI Divergence Detection Logic
        if len(self.low_window) >= self.window_size:

            # Bullish Divergence Logic
            current_pivot_index = self.bar_index - self.lbR

            # Check for a low pivot
            osc_low = self.pivotlow(self.osc_window)

            # Low pivot found
            if osc_low is not None:
                curr_price_low = self.low_window[self.lbR]

                # Check if there exists a previous low pivot
                if self.prev_osc_low is not None:
                    # Also check if pivot is in range and not 5000 bars off
                    bars_since_last = current_pivot_index - self.prev_osc_low_index
                    if self.range_lower <= bars_since_last <= self.range_upper:

                        # Check if RSI is hitting a higher low AND the price a lower low
                        # Pinescript: oscHL and priceLL
                        # This indicates the regular bullish divergence
                        regular_bull = (
                            osc_low > self.prev_osc_low
                            and curr_price_low < self.prev_price_low
                        )

                        # Check if RSI is hitting a lower low and price a higher low
                        # Pinescript: oscLL and priceHL
                        # This indicates the hidden bullish divergence
                        hidden_bull = (
                            osc_low < self.prev_osc_low
                            and curr_price_low > self.prev_price_low
                        )
                        bull_cond = regular_bull or hidden_bull

                # Update pivot memory
                self.prev_osc_low = osc_low
                self.prev_price_low = curr_price_low
                self.prev_osc_low_index = current_pivot_index

            # Bearish Divergence Logic

            # Check for pivot high
            osc_high = self.pivothigh(self.osc_window)

            # Once pivot high has been found:
            if osc_high is not None:
                current_pivot_index = self.bar_index - self.lbR
                curr_price_high = self.high_window[self.lbR]

                # Check if there exists a high pivot previously
                if self.prev_osc_high is not None:

                    # Check if pivot is in range and not 5000 bars off
                    bars_since_last = current_pivot_index - self.prev_osc_high_index
                    if self.range_lower <= bars_since_last <= self.range_upper:

                        # Check if RSI is hitting a lower high AND the price a higher high
                        # Pinescript: oscLH and PriceHH
                        # This indicates bearish divergence

                        bear_cond = (
                            osc_high < self.prev_osc_high
                            and curr_price_high > self.prev_price_high
                        )
                        self.bearish_divergence = bear_cond and not self.prev_bear_cond

                # Update pivot memory
                self.prev_osc_high = osc_high
                self.prev_price_high = curr_price_high
                self.prev_osc_high_index = current_pivot_index

        self.bullish_divergence = bull_cond and not self.prev_bull_cond
        self.bearish_divergence = bear_cond and not self.prev_bear_cond

        self.prev_bull_cond = bull_cond
        self.prev_bear_cond = bear_cond

        # ====================================================================
        # === 4: Implement exit logic ===
        # ====================================================================

        if self.is_long:

            # Exit #1: if the current RSI value is greater than the long exit level
            if self.current_rsi > self.long_exit:

                # Check for crossover (RSI was below threshold, now above)
                if (
                    self.previous_rsi is not None
                    and self.previous_rsi <= self.long_exit
                ):

                    # Exit and reset values
                    self.exit_long()
                    self.pyramid_count = 0
                    self.last_entry_bar_index = 0
                    self.last_entry_rsi_level = 0.0
                    return

            # Exit #2: If there is bearish divergence and volume is decreasing
            elif self.bearish_divergence and self.is_volume_decreasing:

                # Exit and reset values
                self.exit_long()
                self.pyramid_count = 0
                self.last_entry_bar_index = 0
                self.last_entry_rsi_level = 0.0
                return

        # Long Entry logic
        # Check MACD condition
        is_macd_safe = (
            self.macd_current <= self.signal_current and self.require_macd_for_long
        ) or not self.require_macd_for_long

        # Trigger: RSI is below entry level
        is_oversold = self.current_rsi < self.long_entry

                # Checking for long positions
        if not self.is_long:

            # Entry #1: RSI threshold entry gated by MACD path
            if is_macd_safe and is_oversold:
                # Check for crossover (RSI was above threshold, now below)
                if (
                    self.previous_rsi is not None
                    and self.previous_rsi >= self.long_entry
                ):
                    # Enter position if requirements are met
                    self.enter_long(qty=self.base_qty)

                    # Reset variables once position entered
                    self.pyramid_count = 0
                    self.last_entry_bar_index = self.bar_index
                    self.last_entry_rsi_level = self.long_entry

            # Entry #2: Bullish RSI Divergence path
            elif self.bullish_divergence:
                # Enter position if requirements are met
                self.enter_long(qty=self.base_qty)

                # Reset variables once position entered
                self.pyramid_count = 0
                self.last_entry_bar_index = self.bar_index
                self.last_entry_rsi_level = self.long_entry

        # ====================================================================
        # === Implement pyramid logic ===
        # ====================================================================

        # Pyramid Logic
        if (
            self.is_long
            and self.enable_pyramid
            and self.current_rsi < self.long_entry
            and self.pyramid_count < self.max_pyramids
        ):

            # Sensitivities bases, adjusted for optimization
            s = self.sensitivity_base
            sensitivity = [s, s, s, s + 0.1]

            # Pyramid Step Calculations
            long_pyramid_step_1 = self.current_atr * sensitivity[0]
            long_pyramid_step_2 = self.current_atr * sensitivity[1]
            long_pyramid_step_3 = self.current_atr * sensitivity[2]
            long_pyramid_step_4 = self.current_atr * sensitivity[3]

            # Multiplier values, adjusted for optimization
            m = self.pyramid_multiplier
            multipliers = [1.0, 1.0, m, m]

            multiplier = 1
            # Calculate minimum bars, rsiSteps, and multiplier needed for pyramiding
            # based on current pyramid count
            min_bars = 0
            if self.pyramid_count == 0:
                min_bars = 10
                rsi_step_required = long_pyramid_step_1
                multiplier = multipliers[0]

            elif self.pyramid_count == 1:
                min_bars = 15
                rsi_step_required = long_pyramid_step_2
                multiplier = multipliers[1]

            elif self.pyramid_count == 2:
                min_bars = 20
                rsi_step_required = long_pyramid_step_3
                multiplier = multipliers[2]

            else:
                min_bars = 25
                rsi_step_required = long_pyramid_step_4
                multiplier = multipliers[3]

            # Check minimum bar spacing
            spacing_met = (self.bar_index - self.last_entry_bar_index) >= min_bars

            # Check if rsi condition is met
            rsi_pyramid_condition = self.current_rsi <= (
                self.last_entry_rsi_level - rsi_step_required
            )

            # Attempt pyramid adds
            if spacing_met:
                if (rsi_pyramid_condition and is_macd_safe) or self.bullish_divergence:
                
                    # I realized that there was a problem with the pyramiding logic
                    # Even if the limit was below, the algorithm would buy a qty that
                    # blows pass the limit, here is the fix:

                    # Propose a quantity to buy, and if its below the max, we can buy
                    # it, otherwise just buy up to the max position allowed
                    proposed_qty = int(self.base_qty * multiplier)
                    current_qty = float(self.position.quantity)
                    remaining_qty = self.max_position - current_qty
                    actual_qty_to_buy = min(proposed_qty, remaining_qty)

                    if actual_qty_to_buy > 0:
                        self.enter_long(qty=int(actual_qty_to_buy))

                        # Update pyramid values
                        self.pyramid_count += 1
                        self.last_entry_bar_index = self.bar_index

                        # Note: last_entry_rsi_level is not literal: it
                        # behaves like a threshold reference that moves
                        # downwards after each pyramid
                        self.last_entry_rsi_level -= rsi_step_required
                        return

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
