"""
Paper Trading Order Manager for FyersHA
Simulates order execution without placing real orders
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

from config.settings import FyersConfig, SignalType
from models.trading_models import Position
from utils.paper_trade_logger import PaperTradeLogger

logger = logging.getLogger(__name__)


class PaperOrderManager:
    """
    Simulates order execution for paper trading
    Does NOT place real orders with broker
    """

    def __init__(self, fyers_config: FyersConfig, paper_logger: PaperTradeLogger = None):
        """
        Initialize paper order manager

        Args:
            fyers_config: Fyers configuration (not used for orders, only for logging)
            paper_logger: Paper trade logger instance
        """
        self.fyers_config = fyers_config
        self.paper_logger = paper_logger or PaperTradeLogger()

        # Simulated order tracking
        self.simulated_orders: Dict[str, Dict[str, Any]] = {}
        self.order_counter = 0

        logger.info("Paper Order Manager initialized - NO REAL ORDERS WILL BE PLACED")

    def _generate_order_id(self) -> str:
        """Generate a simulated order ID"""
        self.order_counter += 1
        return f"PAPER_{datetime.now().strftime('%Y%m%d')}_{self.order_counter:05d}"

    async def place_entry_order(self, position: Position) -> bool:
        """
        Simulate entry order placement

        Args:
            position: Position object with entry details

        Returns:
            bool: Always True (simulated success)
        """
        try:
            order_id = self._generate_order_id()
            position.order_id = order_id

            # Simulate order execution
            self.simulated_orders[order_id] = {
                'symbol': position.symbol,
                'type': 'ENTRY',
                'position': position,
                'status': 'FILLED',
                'timestamp': datetime.now(),
                'side': 'BUY' if position.signal_type == SignalType.LONG else 'SELL',
                'quantity': abs(position.quantity),
                'price': position.entry_price
            }

            logger.info(f"[PAPER] Entry order simulated - {position.symbol}: "
                        f"Side={self.simulated_orders[order_id]['side']}, "
                        f"Qty={abs(position.quantity)}, "
                        f"Price={position.entry_price:.2f}")

            # Log to paper trade logger
            self.paper_logger.log_entry(position)

            return True

        except Exception as e:
            logger.error(f"Error simulating entry order for {position.symbol}: {e}")
            return False

    async def place_stop_loss_order(self, position: Position) -> bool:
        """
        Simulate stop loss order placement

        Args:
            position: Position object with stop loss details

        Returns:
            bool: Always True (simulated success)
        """
        try:
            order_id = self._generate_order_id()
            position.sl_order_id = order_id

            self.simulated_orders[order_id] = {
                'symbol': position.symbol,
                'type': 'STOP_LOSS',
                'position': position,
                'status': 'PENDING',
                'timestamp': datetime.now(),
                'side': 'SELL' if position.signal_type == SignalType.LONG else 'BUY',
                'quantity': abs(position.quantity),
                'stop_price': position.stop_loss
            }

            logger.info(f"[PAPER] Stop loss order simulated - {position.symbol}: "
                        f"Stop={position.stop_loss:.2f}")

            return True

        except Exception as e:
            logger.error(f"Error simulating stop loss order for {position.symbol}: {e}")
            return False

    async def place_target_order(self, position: Position) -> bool:
        """
        Simulate target order placement

        Args:
            position: Position object with target details

        Returns:
            bool: Always True (simulated success)
        """
        try:
            order_id = self._generate_order_id()
            position.target_order_id = order_id

            self.simulated_orders[order_id] = {
                'symbol': position.symbol,
                'type': 'TARGET',
                'position': position,
                'status': 'PENDING',
                'timestamp': datetime.now(),
                'side': 'SELL' if position.signal_type == SignalType.LONG else 'BUY',
                'quantity': abs(position.quantity),
                'limit_price': position.target_price
            }

            logger.info(f"[PAPER] Target order simulated - {position.symbol}: "
                        f"Target={position.target_price:.2f}")

            return True

        except Exception as e:
            logger.error(f"Error simulating target order for {position.symbol}: {e}")
            return False

    async def place_exit_order(self, position: Position, exit_price: float) -> bool:
        """
        Simulate market exit order

        Args:
            position: Position to exit
            exit_price: Exit price (simulated market price)

        Returns:
            bool: Always True (simulated success)
        """
        try:
            order_id = self._generate_order_id()

            self.simulated_orders[order_id] = {
                'symbol': position.symbol,
                'type': 'EXIT',
                'position': position,
                'status': 'FILLED',
                'timestamp': datetime.now(),
                'side': 'SELL' if position.signal_type == SignalType.LONG else 'BUY',
                'quantity': abs(position.quantity),
                'price': exit_price
            }

            logger.info(f"[PAPER] Exit order simulated - {position.symbol}: "
                        f"Price={exit_price:.2f}")

            # Cancel pending SL/Target orders
            await self.cancel_pending_orders(position)

            return True

        except Exception as e:
            logger.error(f"Error simulating exit order for {position.symbol}: {e}")
            return False

    async def modify_stop_loss(self, position: Position, new_stop: float) -> bool:
        """
        Simulate stop loss modification

        Args:
            position: Position with stop loss order
            new_stop: New stop loss price

        Returns:
            bool: Always True (simulated success)
        """
        try:
            if not position.sl_order_id:
                logger.warning(f"No stop loss order to modify for {position.symbol}")
                return False

            # Update the simulated order
            if position.sl_order_id in self.simulated_orders:
                self.simulated_orders[position.sl_order_id]['stop_price'] = new_stop
                position.current_stop_loss = new_stop

                logger.info(f"[PAPER] Stop loss modified - {position.symbol}: "
                            f"New Stop={new_stop:.2f}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error modifying simulated stop loss for {position.symbol}: {e}")
            return False

    async def cancel_pending_orders(self, position: Position) -> bool:
        """
        Simulate cancelling pending orders

        Args:
            position: Position with orders to cancel

        Returns:
            bool: Always True (simulated success)
        """
        try:
            cancelled_count = 0

            # Cancel stop loss order
            if position.sl_order_id and position.sl_order_id in self.simulated_orders:
                self.simulated_orders[position.sl_order_id]['status'] = 'CANCELLED'
                cancelled_count += 1
                logger.info(f"[PAPER] Cancelled SL order: {position.sl_order_id}")

            # Cancel target order
            if position.target_order_id and position.target_order_id in self.simulated_orders:
                self.simulated_orders[position.target_order_id]['status'] = 'CANCELLED'
                cancelled_count += 1
                logger.info(f"[PAPER] Cancelled target order: {position.target_order_id}")

            logger.info(f"[PAPER] Cancelled {cancelled_count} pending orders for {position.symbol}")
            return True

        except Exception as e:
            logger.error(f"Error cancelling simulated orders for {position.symbol}: {e}")
            return False

    async def cancel_order(self, order_id: str) -> bool:
        """
        Simulate cancelling a specific order

        Args:
            order_id: Order ID to cancel

        Returns:
            bool: Always True (simulated success)
        """
        try:
            if order_id in self.simulated_orders:
                self.simulated_orders[order_id]['status'] = 'CANCELLED'
                logger.info(f"[PAPER] Order cancelled: {order_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error cancelling simulated order {order_id}: {e}")
            return False

    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a simulated order

        Args:
            order_id: Order ID

        Returns:
            Order status dictionary or None
        """
        try:
            return self.simulated_orders.get(order_id)

        except Exception as e:
            logger.error(f"Error getting simulated order status for {order_id}: {e}")
            return None

    def get_positions_summary(self) -> Dict[str, Any]:
        """
        Get summary of simulated positions

        Returns:
            Dictionary with position information
        """
        try:
            # Count filled entry orders
            filled_entries = [
                order for order in self.simulated_orders.values()
                if order['type'] == 'ENTRY' and order['status'] == 'FILLED'
            ]

            return {
                'success': True,
                'positions': filled_entries,
                'count': len(filled_entries),
                'is_paper_trading': True
            }

        except Exception as e:
            logger.error(f"Error getting simulated positions summary: {e}")
            return {
                'success': False,
                'error': str(e),
                'positions': [],
                'count': 0,
                'is_paper_trading': True
            }

    def verify_broker_connection(self) -> bool:
        """
        Simulate broker connection verification

        Returns:
            bool: Always True (paper trading doesn't require real connection)
        """
        logger.info("[PAPER] Broker connection simulated - Paper trading mode active")
        return True

    def get_paper_logger(self) -> PaperTradeLogger:
        """Get the paper trade logger instance"""
        return self.paper_logger

    def print_summary(self):
        """Print paper trading summary"""
        print("\n" + "=" * 80)
        print("PAPER TRADING SESSION SUMMARY")
        print("=" * 80)
        print(f"Total Simulated Orders: {len(self.simulated_orders)}")

        # Count by type
        entry_orders = sum(1 for o in self.simulated_orders.values() if o['type'] == 'ENTRY')
        exit_orders = sum(1 for o in self.simulated_orders.values() if o['type'] == 'EXIT')
        sl_orders = sum(1 for o in self.simulated_orders.values() if o['type'] == 'STOP_LOSS')
        target_orders = sum(1 for o in self.simulated_orders.values() if o['type'] == 'TARGET')

        print(f"Entry Orders: {entry_orders}")
        print(f"Exit Orders: {exit_orders}")
        print(f"Stop Loss Orders: {sl_orders}")
        print(f"Target Orders: {target_orders}")

        # Print trade summary
        self.paper_logger.print_daily_summary()

        print("=" * 80)
        print("NOTE: This was PAPER TRADING - No real orders were placed")
        print("=" * 80)
