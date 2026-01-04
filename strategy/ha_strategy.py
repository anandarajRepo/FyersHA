# strategy/ha_strategy.py

"""
Heikin Ashi Trading Strategy Implementation
Complete strategy with WebSocket integration, risk management, and performance tracking
Based on Advanced Heikin Ashi strategy from BackTest project
"""

import asyncio
import logging
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, time

from config.settings import FyersConfig, HeikinAshiStrategyConfig, TradingConfig, SignalType
from config.websocket_config import WebSocketConfig
from config.symbols import symbol_manager, get_ha_symbols, validate_ha_symbol
from models.trading_models import (
    Position, HASignal, LiveQuote, HeikinAshiCandle, TradeResult,
    StrategyMetrics, MarketState, create_position_from_signal,
    create_trade_result_from_position, validate_signal_quality, calculate_portfolio_risk
)
from services.fyers_websocket_service import HybridDataService
from services.analysis_service import HATechnicalAnalysisService
from services.market_timing_service import MarketTimingService
from strategy.order_manager import OrderManager
from strategy.paper_order_manager import PaperOrderManager
from utils.symbol_manager import SymbolManager
from utils.paper_trade_logger import PaperTradeLogger

logger = logging.getLogger(__name__)


class HeikinAshiStrategy:
    """Complete Heikin Ashi Trading Strategy"""

    def __init__(self, fyers_config: FyersConfig, strategy_config: HeikinAshiStrategyConfig,
                 trading_config: TradingConfig, ws_config: WebSocketConfig):

        # Configuration
        self.fyers_config = fyers_config
        self.strategy_config = strategy_config
        self.trading_config = trading_config
        self.ws_config = ws_config

        # Services
        self.data_service = HybridDataService(fyers_config, ws_config)
        self.analysis_service = HATechnicalAnalysisService(self.data_service)
        self.timing_service = MarketTimingService(trading_config)

        # Initialize order manager based on paper trading mode
        if trading_config.enable_paper_trading:
            logger.info("Initializing PAPER TRADING mode - No real orders will be placed")
            self.paper_logger = PaperTradeLogger()
            self.order_manager = PaperOrderManager(fyers_config, self.paper_logger)
            self.is_paper_trading = True
        else:
            logger.info("Initializing LIVE TRADING mode")
            self.order_manager = OrderManager(fyers_config)
            self.paper_logger = None
            self.is_paper_trading = False

        # Symbol manager for ATM option generation (optional)
        self.symbol_manager = SymbolManager()
        self.use_auto_symbol_generation = False  # Can be enabled via config

        # Configure analysis service with strategy parameters
        self.analysis_service.ha_smoothing = strategy_config.ha_smoothing
        self.analysis_service.adx_period = strategy_config.adx_period
        self.analysis_service.atr_period = strategy_config.atr_period
        self.analysis_service.volume_percentile = strategy_config.volume_percentile

        # Strategy state
        self.positions: Dict[str, Position] = {}
        self.completed_trades: List[TradeResult] = []
        self.metrics = StrategyMetrics()
        self.market_state = MarketState(timestamp=datetime.now())

        # Heikin Ashi specific state
        self.signals_generated_today: List[HASignal] = []
        self.daily_pnl = 0.0
        self.max_daily_loss = self.strategy_config.portfolio_value * 0.02  # 2% max daily loss

        # Get trading universe from centralized symbol manager
        self.trading_symbols = get_ha_symbols()
        logger.info(f"Heikin Ashi Strategy initialized with {len(self.trading_symbols)} symbols")

        # Real-time data tracking
        self.live_quotes: Dict[str, LiveQuote] = {}
        self.symbol_dataframes: Dict[str, pd.DataFrame] = {}

        # Square-off time
        self.square_off_time = time(
            trading_config.square_off_hour,
            trading_config.square_off_minute
        )

        # Add data callback
        self.data_service.add_data_callback(self._on_live_data_update)

    async def initialize(self) -> bool:
        """Initialize strategy and data connections"""
        try:
            logger.info("Initializing Heikin Ashi Strategy...")

            # Verify broker connection
            if not self.order_manager.verify_broker_connection():
                logger.error("Failed to verify broker connection")
                return False

            logger.info("Broker connection verified")

            # Connect to data service
            if not self.data_service.connect():
                logger.error("Failed to connect to data service")
                return False

            # Subscribe to all symbols
            symbols = get_ha_symbols()
            if not self.data_service.subscribe_symbols(symbols):
                logger.error("Failed to subscribe to symbols")
                return False

            # Initialize market state
            self._update_market_state()

            # Log strategy configuration
            logger.info(f"Heikin Ashi Strategy initialized successfully:")
            logger.info(f"  Total symbols: {len(symbols)}")
            logger.info(f"  Max positions: {self.strategy_config.max_positions}")
            logger.info(f"  Risk per trade: {self.strategy_config.risk_per_trade_pct}%")
            logger.info(f"  HA Smoothing: {self.strategy_config.ha_smoothing}")
            logger.info(f"  ADX Period: {self.strategy_config.adx_period}, Threshold: {self.strategy_config.adx_threshold}")
            logger.info(f"  ATR Period: {self.strategy_config.atr_period}, Multiplier: {self.strategy_config.atr_multiplier}")

            return True

        except Exception as e:
            logger.error(f"Strategy initialization failed: {e}")
            return False

    def _on_live_data_update(self, symbol: str, live_quote: LiveQuote):
        """Handle real-time data updates"""
        try:
            # Validate symbol
            if not validate_ha_symbol(symbol):
                logger.warning(f"Received data for invalid symbol: {symbol}")
                return

            # Update internal storage
            self.live_quotes[symbol] = live_quote

            # Update symbol dataframe with new data
            self._update_symbol_dataframe(symbol, live_quote)

            # Update position tracking if we have a position
            if symbol in self.positions:
                self._update_position_tracking(symbol, live_quote)

        except Exception as e:
            logger.error(f"Error handling live data update for {symbol}: {e}")

    def _update_symbol_dataframe(self, symbol: str, live_quote: LiveQuote):
        """Update symbol's dataframe with new quote"""
        try:
            # Create new row from live quote
            new_data = {
                'open': live_quote.open_price,
                'high': live_quote.high_price,
                'low': live_quote.low_price,
                'close': live_quote.ltp,
                'volume': live_quote.volume
            }

            # Initialize or update dataframe
            if symbol not in self.symbol_dataframes:
                self.symbol_dataframes[symbol] = pd.DataFrame([new_data],
                    index=[live_quote.timestamp])
            else:
                df = self.symbol_dataframes[symbol]
                # Keep last N candles (e.g., 100 for indicators)
                max_candles = max(self.strategy_config.adx_period,
                                  self.strategy_config.atr_period,
                                  self.strategy_config.ha_smoothing) * 3

                new_row = pd.DataFrame([new_data], index=[live_quote.timestamp])
                df = pd.concat([df, new_row])

                # Keep only recent data
                if len(df) > max_candles:
                    df = df.iloc[-max_candles:]

                self.symbol_dataframes[symbol] = df

        except Exception as e:
            logger.error(f"Error updating dataframe for {symbol}: {e}")

    def _update_position_tracking(self, symbol: str, live_quote: LiveQuote):
        """Update position tracking with current price"""
        try:
            position = self.positions[symbol]
            current_price = live_quote.ltp

            # Update price extremes for trailing stops
            position.update_price_extremes(current_price)

            # Calculate unrealized P&L
            if position.signal_type == SignalType.LONG:
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:
                position.unrealized_pnl = (position.entry_price - current_price) * abs(position.quantity)

            # Update trailing stop if enabled
            if self.strategy_config.enable_trailing_stops:
                new_stop = self.analysis_service.calculate_trailing_stop(
                    position.signal_type,
                    position.entry_price,
                    current_price,
                    position.highest_price,
                    position.lowest_price,
                    self.strategy_config.trailing_stop_pct
                )

                # Check for breakeven trigger
                profit_pct = ((current_price - position.entry_price) / position.entry_price * 100)
                if profit_pct >= self.strategy_config.breakeven_profit_pct and not position.breakeven_triggered:
                    position.breakeven_triggered = True
                    new_stop = max(new_stop, position.entry_price)
                    logger.info(f"Breakeven triggered for {symbol} at Rs.{current_price:.2f}")

                # Only update if trailing stop is better
                should_modify = False
                if position.signal_type == SignalType.LONG and new_stop > position.current_stop_loss:
                    should_modify = True
                elif position.signal_type == SignalType.SHORT and new_stop < position.current_stop_loss:
                    should_modify = True

                if should_modify:
                    asyncio.create_task(
                        self.order_manager.modify_stop_loss(position, new_stop)
                    )
                    logger.info(f"Trailing stop modified for {symbol}: Rs.{new_stop:.2f}")

        except Exception as e:
            logger.error(f"Error updating position tracking for {symbol}: {e}")

    async def run_strategy_cycle(self):
        """Main strategy execution cycle"""
        try:
            # Update market state
            self._update_market_state()

            # Check if we're in trading hours
            if not self.timing_service.is_trading_time():
                return

            # Monitor existing positions
            await self._monitor_positions()

            # Check for square-off time
            if self._is_square_off_time():
                await self._square_off_all_positions()
                return

            # Scan for new signals if we can take more positions
            if len(self.positions) < self.strategy_config.max_positions:
                if self.timing_service.is_signal_generation_time():
                    await self._scan_for_signals()

            # Update strategy metrics
            self._update_strategy_metrics()

        except Exception as e:
            logger.error(f"Error in strategy cycle: {e}")

    async def _monitor_positions(self):
        """Monitor existing positions for exit conditions"""
        positions_to_close = []

        for symbol, position in self.positions.items():
            try:
                # Get current price
                live_quote = self.live_quotes.get(symbol)
                if not live_quote:
                    continue

                current_price = live_quote.ltp

                # Check stop loss hit
                if position.signal_type == SignalType.LONG:
                    if current_price <= position.current_stop_loss:
                        positions_to_close.append((symbol, current_price, "STOP_LOSS"))
                        continue
                else:
                    if current_price >= position.current_stop_loss:
                        positions_to_close.append((symbol, current_price, "STOP_LOSS"))
                        continue

                # Check target hit
                if position.signal_type == SignalType.LONG:
                    if current_price >= position.target_price:
                        positions_to_close.append((symbol, current_price, "TARGET"))
                        continue
                else:
                    if current_price <= position.target_price:
                        positions_to_close.append((symbol, current_price, "TARGET"))
                        continue

                # Check for Heikin Ashi bearish signal
                if symbol in self.symbol_dataframes:
                    df = self.symbol_dataframes[symbol]
                    if len(df) > 0:
                        df_processed = self.analysis_service.process_symbol_data(symbol, df.copy())
                        if len(df_processed) > 0 and df_processed.iloc[-1].get('sell_signal', False):
                            positions_to_close.append((symbol, current_price, "HA_BEARISH"))
                            continue

            except Exception as e:
                logger.error(f"Error monitoring position for {symbol}: {e}")

        # Close positions that hit exit conditions
        for symbol, exit_price, exit_reason in positions_to_close:
            await self._close_position(symbol, exit_price, exit_reason)

    async def _scan_for_signals(self):
        """Scan all symbols for new trading signals"""
        for symbol in self.trading_symbols:
            try:
                # Skip if we already have a position
                if symbol in self.positions:
                    continue

                # Check if we have enough data
                if symbol not in self.symbol_dataframes:
                    continue

                df = self.symbol_dataframes[symbol]
                min_candles = max(self.strategy_config.adx_period,
                                  self.strategy_config.atr_period) * 2

                if len(df) < min_candles:
                    continue

                # Process data with all indicators
                df_processed = self.analysis_service.process_symbol_data(symbol, df.copy())

                if len(df_processed) == 0:
                    continue

                # Check for buy signal
                if df_processed.iloc[-1].get('buy_signal', False):
                    signal = await self._generate_signal(symbol, df_processed)
                    if signal:
                        await self._enter_position(signal)

            except Exception as e:
                logger.error(f"Error scanning {symbol} for signals: {e}")

    async def _generate_signal(self, symbol: str, df: pd.DataFrame) -> Optional[HASignal]:
        """Generate trading signal from processed data"""
        try:
            latest_candle = self.analysis_service.get_latest_ha_candle(symbol, df)
            if not latest_candle:
                return None

            # Get current price
            live_quote = self.live_quotes.get(symbol)
            if not live_quote:
                return None

            entry_price = live_quote.ltp

            # Calculate stop loss (ATR-based)
            atr = latest_candle.atr
            stop_loss = entry_price - (atr * self.strategy_config.atr_multiplier)

            # Calculate target
            risk = entry_price - stop_loss
            target_price = entry_price + (risk * self.strategy_config.target_multiplier)

            # Calculate risk/reward
            risk_amount, reward_amount, rr_ratio = self.analysis_service.calculate_risk_reward(
                entry_price, stop_loss, target_price, SignalType.LONG
            )

            # Calculate volume ratio
            avg_volume = df['volume'].tail(20).mean() if len(df) >= 20 else df['volume'].mean()
            volume_ratio = live_quote.volume / avg_volume if avg_volume > 0 else 1.0

            # Create signal
            signal = HASignal(
                symbol=symbol,
                signal_type=SignalType.LONG,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                confidence=min(latest_candle.adx / 50.0, 1.0),  # Normalize ADX to 0-1
                volume_ratio=volume_ratio,
                momentum_score=df.iloc[-1].get('bullish_streak', 0),
                adx=latest_candle.adx,
                atr=latest_candle.atr,
                consecutive_candles=int(df.iloc[-1].get('bullish_streak', 0)),
                ha_candle=latest_candle,
                timestamp=datetime.now(),
                risk_amount=risk_amount,
                reward_amount=reward_amount
            )

            # Validate signal quality
            if not validate_signal_quality(signal, self.strategy_config.min_confidence):
                logger.debug(f"Signal for {symbol} failed quality validation")
                return None

            logger.info(f"Generated signal for {symbol}: Entry Rs.{entry_price:.2f}, "
                        f"SL Rs.{stop_loss:.2f}, Target Rs.{target_price:.2f}, "
                        f"ADX: {latest_candle.adx:.1f}, Confidence: {signal.confidence:.2f}")

            return signal

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None

    async def _enter_position(self, signal: HASignal):
        """Enter a new position based on signal"""
        try:
            # Calculate position size
            quantity = self.analysis_service.calculate_position_size(
                self.strategy_config.portfolio_value,
                self.strategy_config.risk_per_trade_pct,
                signal.entry_price,
                signal.stop_loss
            )

            if quantity <= 0:
                logger.warning(f"Invalid quantity for {signal.symbol}")
                return

            # Create position
            position = create_position_from_signal(signal, quantity)

            # Place order (paper trading for now)
            order_id = await self.order_manager.place_order(
                symbol=signal.symbol,
                quantity=quantity,
                price=signal.entry_price,
                side="BUY" if signal.signal_type == SignalType.LONG else "SELL"
            )

            if order_id:
                position.order_id = order_id
                self.positions[signal.symbol] = position
                self.signals_generated_today.append(signal)

                logger.info(f"Entered position: {signal.symbol} {signal.signal_type.value} "
                            f"@ Rs.{signal.entry_price:.2f}, Qty: {quantity}, "
                            f"SL: Rs.{signal.stop_loss:.2f}, Target: Rs.{signal.target_price:.2f}")
            else:
                logger.error(f"Failed to place order for {signal.symbol}")

        except Exception as e:
            logger.error(f"Error entering position for {signal.symbol}: {e}")

    async def _close_position(self, symbol: str, exit_price: float, exit_reason: str):
        """Close an existing position"""
        try:
            position = self.positions.get(symbol)
            if not position:
                return

            # Create trade result
            trade_result = create_trade_result_from_position(position, exit_price, exit_reason)

            # Update daily P&L
            self.daily_pnl += trade_result.net_pnl

            # Add to completed trades
            self.completed_trades.append(trade_result)

            # Log to paper trade logger if in paper mode
            if self.is_paper_trading and self.paper_logger:
                self.paper_logger.log_exit(trade_result)

            # Remove from positions
            del self.positions[symbol]

            logger.info(f"Closed position: {symbol} @ Rs.{exit_price:.2f}, "
                        f"Reason: {exit_reason}, P&L: Rs.{trade_result.net_pnl:.2f}")

            # Place exit order
            await self.order_manager.place_exit_order(position, exit_price)

        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")

    async def _square_off_all_positions(self):
        """Square off all positions at end of day"""
        logger.info("Square-off time reached - closing all positions")

        for symbol in list(self.positions.keys()):
            live_quote = self.live_quotes.get(symbol)
            if live_quote:
                await self._close_position(symbol, live_quote.ltp, "SQUARE_OFF")

    def _is_square_off_time(self) -> bool:
        """Check if it's square-off time"""
        try:
            current_time = datetime.now().time()
            return current_time >= self.square_off_time
        except:
            return False

    def _update_market_state(self):
        """Update market state"""
        self.market_state.timestamp = datetime.now()
        self.market_state.max_positions_reached = len(self.positions) >= self.strategy_config.max_positions
        self.market_state.daily_loss_limit_hit = self.daily_pnl < -self.max_daily_loss

    def _update_strategy_metrics(self):
        """Update strategy performance metrics"""
        self.metrics.update_metrics(self.completed_trades)
        self.metrics.daily_pnl = self.daily_pnl

    async def run(self):
        """Main strategy run loop"""
        if not await self.initialize():
            logger.error("Strategy initialization failed")
            return

        logger.info("Heikin Ashi Strategy started - monitoring market...")

        try:
            while True:
                await self.run_strategy_cycle()
                await asyncio.sleep(self.trading_config.monitoring_interval)

        except KeyboardInterrupt:
            logger.info("Strategy stopped by user")
        except Exception as e:
            logger.error(f"Strategy error: {e}")
        finally:
            self.data_service.disconnect()
            logger.info("Strategy stopped")
