# models/trading_models.py

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List
from config.settings import SignalType

logger = logging.getLogger(__name__)


@dataclass
class LiveQuote:
    """Real-time quote data from WebSocket"""
    symbol: str
    ltp: float  # Last traded price
    open_price: float
    high_price: float
    low_price: float
    volume: int
    previous_close: float
    timestamp: datetime
    change: float = 0.0
    change_pct: float = 0.0

    def __post_init__(self):
        if self.previous_close > 0:
            self.change = self.ltp - self.previous_close
            self.change_pct = (self.change / self.previous_close) * 100


@dataclass
class OpenRange:
    """Opening Range data for ORB strategy"""
    symbol: str
    high: float
    low: float
    range_size: float
    range_pct: float
    volume: int
    start_time: datetime
    end_time: datetime


@dataclass
class HeikinAshiCandle:
    """Heikin Ashi candle data"""
    symbol: str
    ha_open: float
    ha_high: float
    ha_low: float
    ha_close: float
    ha_open_smooth: float  # EMA smoothed
    ha_close_smooth: float  # EMA smoothed
    ha_bullish: bool
    ha_bearish: bool
    ha_body: float
    volume: int
    timestamp: datetime
    adx: float = 0.0
    atr: float = 0.0
    good_volume: bool = False


@dataclass
class HASignal:
    """Heikin Ashi Signal"""
    symbol: str
    signal_type: SignalType  # LONG or SHORT
    entry_price: float
    stop_loss: float
    target_price: float

    # Signal quality metrics
    confidence: float
    volume_ratio: float
    momentum_score: float
    adx: float
    atr: float
    consecutive_candles: int

    # Heikin Ashi specific
    ha_candle: HeikinAshiCandle

    # Timing
    timestamp: datetime

    # Risk metrics
    risk_amount: float
    reward_amount: float
    risk_reward_ratio: float = field(init=False)

    def __post_init__(self):
        if self.risk_amount > 0:
            self.risk_reward_ratio = self.reward_amount / self.risk_amount


@dataclass
class ORBSignal:
    """Opening Range Breakout Signal"""
    symbol: str
    signal_type: SignalType  # LONG or SHORT
    entry_price: float
    stop_loss: float
    target_price: float

    # Signal quality metrics
    confidence: float
    volume_ratio: float
    breakout_strength: float

    # ORB specific
    opening_range: OpenRange
    breakout_level: float

    # Timing
    timestamp: datetime

    # Risk metrics
    risk_amount: float
    reward_amount: float
    risk_reward_ratio: float = field(init=False)

    def __post_init__(self):
        if self.risk_amount > 0:
            self.risk_reward_ratio = self.reward_amount / self.risk_amount


@dataclass
class Position:
    """Trading position with Heikin Ashi specific features"""
    symbol: str
    signal_type: SignalType

    # Position details
    entry_price: float
    quantity: int
    stop_loss: float
    target_price: float

    # Timing
    entry_time: datetime

    # Tracking
    highest_price: float = 0.0  # For trailing stops
    lowest_price: float = 0.0  # For trailing stops
    current_stop_loss: float = 0.0  # Dynamic stop loss

    # Orders
    order_id: Optional[str] = None
    sl_order_id: Optional[str] = None
    target_order_id: Optional[str] = None

    # Performance
    unrealized_pnl: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0

    # Heikin Ashi specific
    entry_adx: float = 0.0
    entry_atr: float = 0.0
    breakeven_triggered: bool = False

    def __post_init__(self):
        self.current_stop_loss = self.stop_loss
        self.highest_price = self.entry_price if self.signal_type == SignalType.LONG else 0.0
        self.lowest_price = self.entry_price if self.signal_type == SignalType.SHORT else float('inf')

    def update_price_extremes(self, current_price: float):
        """Update price extremes for trailing stop calculation"""
        if self.signal_type == SignalType.LONG:
            self.highest_price = max(self.highest_price, current_price)

            # Calculate favorable and adverse excursion
            self.max_favorable_excursion = max(
                self.max_favorable_excursion,
                current_price - self.entry_price
            )
            self.max_adverse_excursion = min(
                self.max_adverse_excursion,
                current_price - self.entry_price
            )
        else:  # SHORT
            self.lowest_price = min(self.lowest_price, current_price)

            # Calculate favorable and adverse excursion
            self.max_favorable_excursion = max(
                self.max_favorable_excursion,
                self.entry_price - current_price
            )
            self.max_adverse_excursion = min(
                self.max_adverse_excursion,
                self.entry_price - current_price
            )


@dataclass
class TradeResult:
    """Completed trade result"""
    symbol: str
    signal_type: SignalType

    # Trade details
    entry_price: float
    exit_price: float
    quantity: int

    # Timing
    entry_time: datetime
    exit_time: datetime
    holding_period: float  # in minutes

    # Performance
    gross_pnl: float
    net_pnl: float = field(init=False)

    # Metrics
    exit_reason: str  # "TARGET", "STOP_LOSS", "TRAILING_STOP", "TIME_EXIT", "HA_BEARISH"
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0

    # Heikin Ashi specific
    entry_adx: float = 0.0
    entry_atr: float = 0.0

    # Performance
    commission: float = 0.0

    def __post_init__(self):
        self.net_pnl = self.gross_pnl - self.commission
        self.holding_period = (self.exit_time - self.entry_time).total_seconds() / 60


@dataclass
class StrategyMetrics:
    """Strategy performance metrics"""
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # P&L metrics
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    max_portfolio_risk: float = 0.0

    # Strategy specific
    long_trades: int = 0
    short_trades: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0

    # Timing metrics
    avg_holding_period: float = 0.0  # minutes

    # Recent performance
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0

    def update_metrics(self, trades: List[TradeResult]):
        """Update metrics from trade list"""
        if not trades:
            return

        self.total_trades = len(trades)
        self.winning_trades = sum(1 for t in trades if t.net_pnl > 0)
        self.losing_trades = self.total_trades - self.winning_trades
        self.win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0

        self.total_pnl = sum(t.net_pnl for t in trades)
        self.gross_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
        self.gross_loss = sum(t.net_pnl for t in trades if t.net_pnl < 0)

        # Strategy specific metrics
        long_trades_list = [t for t in trades if t.signal_type == SignalType.LONG]
        short_trades_list = [t for t in trades if t.signal_type == SignalType.SHORT]

        self.long_trades = len(long_trades_list)
        self.short_trades = len(short_trades_list)

        if self.long_trades > 0:
            self.long_win_rate = (sum(1 for t in long_trades_list if t.net_pnl > 0) / self.long_trades) * 100

        if self.short_trades > 0:
            self.short_win_rate = (sum(1 for t in short_trades_list if t.net_pnl > 0) / self.short_trades) * 100

        # Timing metrics
        if trades:
            self.avg_holding_period = sum(t.holding_period for t in trades) / len(trades)


@dataclass
class MarketState:
    """Current market state for strategy decisions"""
    timestamp: datetime

    # Market indicators
    nifty_change_pct: float = 0.0
    market_trend: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    volatility_regime: str = "NORMAL"  # HIGH, NORMAL, LOW

    # Volume indicators
    market_volume_ratio: float = 1.0  # Current vs average

    # Risk indicators
    max_positions_reached: bool = False
    daily_loss_limit_hit: bool = False
    signal_generation_active: bool = True


# Utility functions for working with models
def create_ha_signal_from_candle(candle: HeikinAshiCandle, signal_type: SignalType,
                                  entry_price: float, stop_loss: float, target_price: float,
                                  **kwargs) -> HASignal:
    """Create HA signal from candle"""
    return HASignal(
        symbol=candle.symbol,
        signal_type=signal_type,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_price=target_price,
        ha_candle=candle,
        timestamp=datetime.now(),
        **kwargs
    )


def create_position_from_signal(signal: HASignal, quantity: int, **kwargs) -> Position:
    """Create position from HA signal"""
    return Position(
        symbol=signal.symbol,
        signal_type=signal.signal_type,
        entry_price=signal.entry_price,
        quantity=quantity,
        stop_loss=signal.stop_loss,
        target_price=signal.target_price,
        entry_time=datetime.now(),
        entry_adx=signal.adx,
        entry_atr=signal.atr,
        **kwargs
    )


def create_trade_result_from_position(position: Position, exit_price: float, exit_reason: str) -> TradeResult:
    """Create trade result from closed position"""
    exit_time = datetime.now()

    # Calculate gross P&L
    if position.signal_type == SignalType.LONG:
        gross_pnl = (exit_price - position.entry_price) * position.quantity
    else:
        gross_pnl = (position.entry_price - exit_price) * abs(position.quantity)

    # Calculate holding period in minutes
    holding_period = (exit_time - position.entry_time).total_seconds() / 60

    return TradeResult(
        symbol=position.symbol,
        signal_type=position.signal_type,
        entry_price=position.entry_price,
        exit_price=exit_price,
        quantity=abs(position.quantity),
        entry_time=position.entry_time,
        exit_time=exit_time,
        holding_period=holding_period,
        gross_pnl=gross_pnl,
        exit_reason=exit_reason,
        max_favorable_excursion=position.max_favorable_excursion,
        max_adverse_excursion=position.max_adverse_excursion,
        entry_adx=position.entry_adx,
        entry_atr=position.entry_atr
    )


def validate_signal_quality(signal: HASignal, min_confidence: float = 0.55) -> bool:
    """Validate signal quality based on various criteria"""
    logger.info(f"Validating signal quality for {signal.symbol}, confidence: {signal.confidence}, "
                f"adx: {signal.adx}, volume_ratio: {signal.volume_ratio}")

    if signal.confidence < min_confidence:
        return False

    if signal.risk_amount <= 0:
        return False

    if signal.volume_ratio < 1.0:  # Below average volume
        return False

    return True


def calculate_portfolio_risk(positions: List[Position], portfolio_value: float) -> Dict[str, float]:
    """Calculate portfolio risk metrics"""
    total_risk = sum(abs(pos.entry_price - pos.current_stop_loss) * abs(pos.quantity) for pos in positions)
    total_invested = sum(abs(pos.entry_price * pos.quantity) for pos in positions)

    return {
        'total_risk_amount': total_risk,
        'total_invested': total_invested,
        'risk_percentage': (total_risk / portfolio_value) * 100 if portfolio_value > 0 else 0,
        'invested_percentage': (total_invested / portfolio_value) * 100 if portfolio_value > 0 else 0,
        'average_risk_per_position': total_risk / len(positions) if positions else 0
    }
