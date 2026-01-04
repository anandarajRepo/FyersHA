# services/analysis_service.py

"""
Heikin Ashi Technical Analysis Service
Provides Heikin Ashi calculations, technical indicators, and signal generation
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime

from config.settings import SignalType
from models.trading_models import LiveQuote, HeikinAshiCandle, HASignal

logger = logging.getLogger(__name__)


class HATechnicalAnalysisService:
    """Technical analysis service for Heikin Ashi strategy"""

    def __init__(self, data_service):
        self.data_service = data_service
        self.ha_smoothing = 3
        self.adx_period = 14
        self.atr_period = 14
        self.volume_percentile = 60.0

    def calculate_heikin_ashi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Heikin Ashi candles with EMA smoothing"""
        if df.empty or len(df) < 2:
            return df

        ha_df = df.copy()

        # Calculate standard Heikin Ashi
        ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        ha_df['ha_open'] = (df['open'] + df['close']) / 2

        # Calculate subsequent HA opens
        for i in range(1, len(ha_df)):
            ha_df.iloc[i, ha_df.columns.get_loc('ha_open')] = (
                ha_df.iloc[i-1]['ha_open'] + ha_df.iloc[i-1]['ha_close']
            ) / 2

        ha_df['ha_high'] = ha_df[['high', 'ha_open', 'ha_close']].max(axis=1)
        ha_df['ha_low'] = ha_df[['low', 'ha_open', 'ha_close']].min(axis=1)

        # Apply EMA smoothing
        if self.ha_smoothing > 1:
            ha_df['ha_open_smooth'] = ha_df['ha_open'].ewm(span=self.ha_smoothing, adjust=False).mean()
            ha_df['ha_close_smooth'] = ha_df['ha_close'].ewm(span=self.ha_smoothing, adjust=False).mean()
            ha_df['ha_high_smooth'] = ha_df['ha_high'].ewm(span=self.ha_smoothing, adjust=False).mean()
            ha_df['ha_low_smooth'] = ha_df['ha_low'].ewm(span=self.ha_smoothing, adjust=False).mean()
        else:
            ha_df['ha_open_smooth'] = ha_df['ha_open']
            ha_df['ha_close_smooth'] = ha_df['ha_close']
            ha_df['ha_high_smooth'] = ha_df['ha_high']
            ha_df['ha_low_smooth'] = ha_df['ha_low']

        # Determine candle color and body size
        ha_df['ha_bullish'] = ha_df['ha_close_smooth'] > ha_df['ha_open_smooth']
        ha_df['ha_bearish'] = ha_df['ha_close_smooth'] < ha_df['ha_open_smooth']
        ha_df['ha_body'] = abs(ha_df['ha_close_smooth'] - ha_df['ha_open_smooth'])

        return ha_df

    def calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ADX for trend strength"""
        if df.empty or len(df) < self.adx_period + 1:
            return df

        # Calculate True Range
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # Calculate Directional Movements
        df['dm_plus'] = np.where(
            (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
            np.maximum(df['high'] - df['high'].shift(1), 0),
            0
        )

        df['dm_minus'] = np.where(
            (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
            np.maximum(df['low'].shift(1) - df['low'], 0),
            0
        )

        # Smooth using Wilder's smoothing
        alpha = 1 / self.adx_period
        df['tr_smooth'] = df['tr'].ewm(alpha=alpha, adjust=False).mean()
        df['dm_plus_smooth'] = df['dm_plus'].ewm(alpha=alpha, adjust=False).mean()
        df['dm_minus_smooth'] = df['dm_minus'].ewm(alpha=alpha, adjust=False).mean()

        # Calculate DI
        df['di_plus'] = 100 * (df['dm_plus_smooth'] / df['tr_smooth'])
        df['di_minus'] = 100 * (df['dm_minus_smooth'] / df['tr_smooth'])

        # Calculate ADX
        df['dx'] = 100 * abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
        df['adx'] = df['dx'].ewm(alpha=alpha, adjust=False).mean()

        return df

    def calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range for dynamic stops"""
        if df.empty or len(df) < self.atr_period + 1:
            return df

        if 'tr' not in df.columns:
            df['prev_close'] = df['close'].shift(1)
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['prev_close'])
            df['tr3'] = abs(df['low'] - df['prev_close'])
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # Calculate ATR using EMA
        df['atr'] = df['tr'].ewm(span=self.atr_period, adjust=False).mean()

        return df

    def identify_good_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify periods with good volume"""
        if df.empty:
            return df

        volume_threshold = df['volume'].quantile(self.volume_percentile / 100)
        df['good_volume'] = df['volume'] > volume_threshold
        return df

    def count_consecutive_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Count consecutive bullish candles"""
        if df.empty:
            return df

        df['bullish_streak'] = 0
        streak = 0
        for i in range(len(df)):
            if df.iloc[i]['ha_bullish']:
                streak += 1
            else:
                streak = 0
            df.iloc[i, df.columns.get_loc('bullish_streak')] = streak

        return df

    def generate_buy_signals(self, df: pd.DataFrame, min_consecutive: int = 2,
                             adx_threshold: float = 25.0) -> pd.DataFrame:
        """Generate buy signals based on Heikin Ashi and indicators"""
        if df.empty:
            return df

        # BUY SIGNAL: All conditions must be true
        df['buy_signal'] = (
            (df['ha_bullish']) &                                    # Bullish HA candle
            (df['bullish_streak'] >= min_consecutive) &            # Momentum (consecutive candles)
            (df['adx'] > adx_threshold) &                          # Trending market
            (df['good_volume']) &                                   # Good volume
            (~df['adx'].isna()) &                                  # Valid ADX
            (~df['atr'].isna())                                    # Valid ATR
        )

        # SELL SIGNAL: Heikin Ashi turns bearish
        df['sell_signal'] = (
            (df['ha_bearish']) &
            (df['ha_bullish'].shift(1))  # Was bullish, now bearish
        )

        return df

    def calculate_trailing_stop(self, signal_type: SignalType, entry_price: float,
                                 current_price: float, highest_price: float,
                                 lowest_price: float, trailing_pct: float) -> float:
        """Calculate trailing stop loss"""
        if signal_type == SignalType.LONG:
            # For long positions, trail below the highest price
            trailing_stop = highest_price * (1 - trailing_pct / 100)
            return max(trailing_stop, entry_price)  # Never below entry
        else:
            # For short positions, trail above the lowest price
            trailing_stop = lowest_price * (1 + trailing_pct / 100)
            return min(trailing_stop, entry_price)  # Never above entry

    def calculate_position_size(self, portfolio_value: float, risk_pct: float,
                                 entry_price: float, stop_loss: float) -> int:
        """Calculate position size based on risk percentage"""
        risk_amount = portfolio_value * (risk_pct / 100)
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share <= 0:
            return 0

        quantity = int(risk_amount / risk_per_share)
        return max(quantity, 0)

    def calculate_risk_reward(self, entry_price: float, stop_loss: float,
                              target_price: float, signal_type: SignalType) -> Tuple[float, float, float]:
        """Calculate risk and reward amounts"""
        if signal_type == SignalType.LONG:
            risk = entry_price - stop_loss
            reward = target_price - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - target_price

        if risk <= 0:
            return 0.0, 0.0, 0.0

        risk_reward_ratio = reward / risk if risk > 0 else 0.0

        return risk, reward, risk_reward_ratio

    def get_latest_ha_candle(self, symbol: str, df: pd.DataFrame) -> Optional[HeikinAshiCandle]:
        """Get the latest Heikin Ashi candle for a symbol"""
        if df.empty or len(df) == 0:
            return None

        try:
            latest = df.iloc[-1]

            return HeikinAshiCandle(
                symbol=symbol,
                ha_open=latest.get('ha_open', 0.0),
                ha_high=latest.get('ha_high', 0.0),
                ha_low=latest.get('ha_low', 0.0),
                ha_close=latest.get('ha_close', 0.0),
                ha_open_smooth=latest.get('ha_open_smooth', 0.0),
                ha_close_smooth=latest.get('ha_close_smooth', 0.0),
                ha_bullish=latest.get('ha_bullish', False),
                ha_bearish=latest.get('ha_bearish', False),
                ha_body=latest.get('ha_body', 0.0),
                volume=int(latest.get('volume', 0)),
                timestamp=latest.name if isinstance(latest.name, datetime) else datetime.now(),
                adx=latest.get('adx', 0.0),
                atr=latest.get('atr', 0.0),
                good_volume=latest.get('good_volume', False)
            )
        except Exception as e:
            logger.error(f"Error getting latest HA candle for {symbol}: {e}")
            return None

    def process_symbol_data(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Process complete data for a symbol - calculate all indicators"""
        if df.empty:
            return df

        try:
            # Calculate all indicators
            df = self.calculate_heikin_ashi(df)
            df = self.calculate_adx(df)
            df = self.calculate_atr(df)
            df = self.identify_good_volume(df)
            df = self.count_consecutive_candles(df)
            df = self.generate_buy_signals(df)

            return df
        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {e}")
            return df
