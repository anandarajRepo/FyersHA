# config/settings.py

import os
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class FyersConfig:
    client_id: str
    secret_key: str
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    base_url: str = "https://api-t1.fyers.in/api/v3"


@dataclass
class HeikinAshiStrategyConfig:
    """Heikin Ashi Strategy Configuration"""
    # Portfolio settings
    portfolio_value: float = 100000
    risk_per_trade_pct: float = 1.0
    max_positions: int = 3

    # Heikin Ashi specific parameters
    ha_smoothing: int = 3  # EMA period for Heikin Ashi smoothing
    consecutive_candles: int = 2  # Number of consecutive bullish candles required

    # Technical indicators
    adx_period: int = 14  # Period for ADX calculation
    adx_threshold: float = 25.0  # Minimum ADX for trend confirmation
    atr_period: int = 14  # Period for ATR calculation
    atr_multiplier: float = 2.0  # Multiplier for ATR trailing stop
    volume_percentile: float = 60.0  # Volume percentile threshold

    # Risk management
    stop_loss_pct: float = 1.5  # Stop loss as % from entry
    target_multiplier: float = 2.0  # Target as multiple of risk
    trailing_stop_pct: float = 0.5  # Trailing stop adjustment
    breakeven_profit_pct: float = 1.0  # Move to breakeven at 1% profit

    # Signal filtering
    min_confidence: float = 0.65
    min_volume_ratio: float = 1.5  # Current vs average volume

    # Position management
    enable_trailing_stops: bool = True
    enable_partial_exits: bool = True
    partial_exit_pct: float = 50.0  # % to exit at first target

    # Time interval for candle resampling
    tick_interval: str = '1min'  # Options: None (raw ticks), '5s', '10s', '30s', '1min', '5min'


@dataclass
class TradingConfig:
    # Market hours (IST)
    market_start_hour: int = 9
    market_start_minute: int = 15
    market_end_hour: int = 15
    market_end_minute: int = 30

    # Strategy timing
    signal_generation_end_hour: int = 15  # Stop generating signals at 3:00 PM
    signal_generation_end_minute: int = 0
    square_off_hour: int = 15  # Square off time
    square_off_minute: int = 20

    # Monitoring
    monitoring_interval: int = 1  # seconds
    position_update_interval: int = 5  # seconds
