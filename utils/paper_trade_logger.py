"""
Paper Trading Logger for FyersHA
Logs all paper trading activity including entries, exits, and P&L
"""

import logging
import csv
import os
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
from models.trading_models import Position, TradeResult, SignalType

logger = logging.getLogger(__name__)


class PaperTradeLogger:
    """
    Logs paper trading activity to CSV and text files
    """

    def __init__(self, log_dir: str = "logs"):
        """
        Initialize paper trade logger

        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Setup log files
        self.trade_log_file = self.log_dir / "paper_trades.csv"
        self.detailed_log_file = self.log_dir / "paper_trades_detailed.log"

        # Initialize CSV if it doesn't exist
        if not self.trade_log_file.exists():
            self._initialize_csv()

        # Track daily statistics
        self.daily_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'gross_profit': 0.0,
            'gross_loss': 0.0
        }

        logger.info(f"Paper Trade Logger initialized - Log files in {self.log_dir}")

    def _initialize_csv(self):
        """Initialize CSV file with headers"""
        headers = [
            'timestamp',
            'symbol',
            'signal_type',
            'entry_time',
            'entry_price',
            'exit_time',
            'exit_price',
            'quantity',
            'gross_pnl',
            'net_pnl',
            'exit_reason',
            'holding_period_min',
            'entry_adx',
            'entry_atr',
            'max_favorable_excursion',
            'max_adverse_excursion'
        ]

        with open(self.trade_log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

        logger.info(f"Initialized paper trade CSV: {self.trade_log_file}")

    def log_entry(self, position: Position):
        """
        Log a paper trade entry

        Args:
            position: Position that was entered
        """
        try:
            log_message = (
                f"\n{'='*80}\n"
                f"PAPER TRADE ENTRY\n"
                f"{'='*80}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Symbol: {position.symbol}\n"
                f"Signal Type: {position.signal_type.value}\n"
                f"Entry Price: {position.entry_price:.2f}\n"
                f"Quantity: {position.quantity}\n"
                f"Stop Loss: {position.stop_loss:.2f}\n"
                f"Target Price: {position.target_price:.2f}\n"
                f"ADX: {position.entry_adx:.2f}\n"
                f"ATR: {position.entry_atr:.2f}\n"
                f"{'='*80}\n"
            )

            # Log to detailed file
            with open(self.detailed_log_file, 'a') as f:
                f.write(log_message)

            logger.info(f"Paper trade ENTRY logged: {position.symbol} @ {position.entry_price}")

        except Exception as e:
            logger.error(f"Error logging paper trade entry: {e}")

    def log_exit(self, trade_result: TradeResult):
        """
        Log a paper trade exit

        Args:
            trade_result: Completed trade result
        """
        try:
            # Calculate profit percentage
            profit_pct = (trade_result.net_pnl / (trade_result.entry_price * trade_result.quantity)) * 100

            log_message = (
                f"\n{'='*80}\n"
                f"PAPER TRADE EXIT\n"
                f"{'='*80}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Symbol: {trade_result.symbol}\n"
                f"Signal Type: {trade_result.signal_type.value}\n"
                f"Entry Price: {trade_result.entry_price:.2f}\n"
                f"Exit Price: {trade_result.exit_price:.2f}\n"
                f"Quantity: {trade_result.quantity}\n"
                f"Holding Period: {trade_result.holding_period:.1f} minutes\n"
                f"Exit Reason: {trade_result.exit_reason}\n"
                f"Gross P&L: Rs.{trade_result.gross_pnl:.2f}\n"
                f"Net P&L: Rs.{trade_result.net_pnl:.2f} ({profit_pct:+.2f}%)\n"
                f"Max Favorable Excursion: {trade_result.max_favorable_excursion:.2f}\n"
                f"Max Adverse Excursion: {trade_result.max_adverse_excursion:.2f}\n"
                f"{'='*80}\n"
            )

            # Log to detailed file
            with open(self.detailed_log_file, 'a') as f:
                f.write(log_message)

            # Log to CSV
            self._log_to_csv(trade_result)

            # Update daily stats
            self._update_daily_stats(trade_result)

            logger.info(f"Paper trade EXIT logged: {trade_result.symbol} - P&L: Rs.{trade_result.net_pnl:.2f}")

        except Exception as e:
            logger.error(f"Error logging paper trade exit: {e}")

    def _log_to_csv(self, trade_result: TradeResult):
        """
        Log trade result to CSV file

        Args:
            trade_result: Completed trade result
        """
        try:
            row = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': trade_result.symbol,
                'signal_type': trade_result.signal_type.value,
                'entry_time': trade_result.entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                'entry_price': f"{trade_result.entry_price:.2f}",
                'exit_time': trade_result.exit_time.strftime('%Y-%m-%d %H:%M:%S'),
                'exit_price': f"{trade_result.exit_price:.2f}",
                'quantity': trade_result.quantity,
                'gross_pnl': f"{trade_result.gross_pnl:.2f}",
                'net_pnl': f"{trade_result.net_pnl:.2f}",
                'exit_reason': trade_result.exit_reason,
                'holding_period_min': f"{trade_result.holding_period:.1f}",
                'entry_adx': f"{trade_result.entry_adx:.2f}",
                'entry_atr': f"{trade_result.entry_atr:.2f}",
                'max_favorable_excursion': f"{trade_result.max_favorable_excursion:.2f}",
                'max_adverse_excursion': f"{trade_result.max_adverse_excursion:.2f}"
            }

            with open(self.trade_log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writerow(row)

        except Exception as e:
            logger.error(f"Error writing to CSV: {e}")

    def _update_daily_stats(self, trade_result: TradeResult):
        """
        Update daily statistics

        Args:
            trade_result: Completed trade result
        """
        self.daily_stats['total_trades'] += 1

        if trade_result.net_pnl > 0:
            self.daily_stats['winning_trades'] += 1
            self.daily_stats['gross_profit'] += trade_result.net_pnl
        else:
            self.daily_stats['losing_trades'] += 1
            self.daily_stats['gross_loss'] += abs(trade_result.net_pnl)

        self.daily_stats['total_pnl'] += trade_result.net_pnl

    def get_daily_summary(self) -> Dict[str, Any]:
        """
        Get daily trading summary

        Returns:
            Dictionary with daily statistics
        """
        win_rate = 0.0
        if self.daily_stats['total_trades'] > 0:
            win_rate = (self.daily_stats['winning_trades'] / self.daily_stats['total_trades']) * 100

        profit_factor = 0.0
        if self.daily_stats['gross_loss'] > 0:
            profit_factor = self.daily_stats['gross_profit'] / self.daily_stats['gross_loss']

        return {
            'total_trades': self.daily_stats['total_trades'],
            'winning_trades': self.daily_stats['winning_trades'],
            'losing_trades': self.daily_stats['losing_trades'],
            'win_rate': win_rate,
            'total_pnl': self.daily_stats['total_pnl'],
            'gross_profit': self.daily_stats['gross_profit'],
            'gross_loss': self.daily_stats['gross_loss'],
            'profit_factor': profit_factor
        }

    def print_daily_summary(self):
        """Print formatted daily summary"""
        summary = self.get_daily_summary()

        print("\n" + "=" * 80)
        print("PAPER TRADING DAILY SUMMARY")
        print("=" * 80)
        print(f"Total Trades: {summary['total_trades']}")
        print(f"Winning Trades: {summary['winning_trades']}")
        print(f"Losing Trades: {summary['losing_trades']}")
        print(f"Win Rate: {summary['win_rate']:.2f}%")
        print(f"Total P&L: Rs.{summary['total_pnl']:.2f}")
        print(f"Gross Profit: Rs.{summary['gross_profit']:.2f}")
        print(f"Gross Loss: Rs.{summary['gross_loss']:.2f}")
        print(f"Profit Factor: {summary['profit_factor']:.2f}")
        print("=" * 80)

    def log_position_update(self, position: Position, current_price: float):
        """
        Log position update (for tracking open positions)

        Args:
            position: Current position
            current_price: Current market price
        """
        try:
            unrealized_pnl = 0.0
            if position.signal_type == SignalType.LONG:
                unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:
                unrealized_pnl = (position.entry_price - current_price) * abs(position.quantity)

            log_message = (
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"{position.symbol} - Current: {current_price:.2f}, "
                f"Entry: {position.entry_price:.2f}, "
                f"Unrealized P&L: Rs.{unrealized_pnl:+.2f}\n"
            )

            with open(self.detailed_log_file, 'a') as f:
                f.write(log_message)

        except Exception as e:
            logger.error(f"Error logging position update: {e}")

    def reset_daily_stats(self):
        """Reset daily statistics (call at start of new day)"""
        self.daily_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'gross_profit': 0.0,
            'gross_loss': 0.0
        }
        logger.info("Daily statistics reset")
