# main.py - Heikin Ashi Trading Strategy

"""
Heikin Ashi Trading Strategy - Complete Main Entry Point
Full algorithmic trading system with WebSocket data integration
Based on Advanced Heikin Ashi strategy from BackTest project
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from dotenv import load_dotenv

# Core imports
from config.settings import FyersConfig, HeikinAshiStrategyConfig, TradingConfig
from config.websocket_config import WebSocketConfig
from strategy.ha_strategy import HeikinAshiStrategy

# Import the enhanced authentication system
from utils.enhanced_auth_helper import (
    setup_auth_only,
    authenticate_fyers,
    test_authentication,
    update_pin_only
)

# Load environment variables
load_dotenv()


# Configure enhanced logging
def setup_logging():
    """Setup enhanced logging configuration"""
    log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()

    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'ha_strategy.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Set specific log levels for external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def load_configuration():
    """Load all configuration from environment variables"""
    try:
        # Fyers configuration
        fyers_config = FyersConfig(
            client_id=os.environ.get('FYERS_CLIENT_ID'),
            secret_key=os.environ.get('FYERS_SECRET_KEY'),
            access_token=os.environ.get('FYERS_ACCESS_TOKEN'),
            refresh_token=os.environ.get('FYERS_REFRESH_TOKEN')
        )

        # Heikin Ashi Strategy configuration
        strategy_config = HeikinAshiStrategyConfig(
            portfolio_value=float(os.environ.get('PORTFOLIO_VALUE', 100000)),
            risk_per_trade_pct=float(os.environ.get('RISK_PER_TRADE', 1.0)),
            max_positions=int(os.environ.get('MAX_POSITIONS', 3)),

            # Heikin Ashi parameters
            ha_smoothing=int(os.environ.get('HA_SMOOTHING', 3)),
            consecutive_candles=int(os.environ.get('CONSECUTIVE_CANDLES', 2)),

            # Technical indicators
            adx_period=int(os.environ.get('ADX_PERIOD', 14)),
            adx_threshold=float(os.environ.get('ADX_THRESHOLD', 25.0)),
            atr_period=int(os.environ.get('ATR_PERIOD', 14)),
            atr_multiplier=float(os.environ.get('ATR_MULTIPLIER', 2.0)),
            volume_percentile=float(os.environ.get('VOLUME_PERCENTILE', 60.0)),

            # Risk management
            stop_loss_pct=float(os.environ.get('STOP_LOSS_PCT', 1.5)),
            target_multiplier=float(os.environ.get('TARGET_MULTIPLIER', 2.0)),
            trailing_stop_pct=float(os.environ.get('TRAILING_STOP_PCT', 0.5)),
            breakeven_profit_pct=float(os.environ.get('BREAKEVEN_PROFIT_PCT', 1.0)),

            # Signal filtering
            min_confidence=float(os.environ.get('MIN_CONFIDENCE', 0.65)),
            min_volume_ratio=float(os.environ.get('MIN_VOLUME_RATIO', 1.5)),

            # Position management
            enable_trailing_stops=os.environ.get('ENABLE_TRAILING_STOPS', 'true').lower() == 'true',
            enable_partial_exits=os.environ.get('ENABLE_PARTIAL_EXITS', 'true').lower() == 'true',
            partial_exit_pct=float(os.environ.get('PARTIAL_EXIT_PCT', 50.0)),

            # Time interval
            tick_interval=os.environ.get('TICK_INTERVAL', '1min')
        )

        # Trading configuration
        trading_config = TradingConfig(
            market_start_hour=9,
            market_start_minute=15,
            market_end_hour=15,
            market_end_minute=30,
            signal_generation_end_hour=15,
            signal_generation_end_minute=0,
            square_off_hour=15,
            square_off_minute=20,
            monitoring_interval=int(os.environ.get('MONITORING_INTERVAL', 1)),
            position_update_interval=int(os.environ.get('POSITION_UPDATE_INTERVAL', 5)),
            enable_paper_trading=os.environ.get('ENABLE_PAPER_TRADING', 'false').lower() == 'true',
            paper_trade_log_file=os.environ.get('PAPER_TRADE_LOG_FILE', 'logs/paper_trades.log')
        )

        # WebSocket configuration
        ws_config = WebSocketConfig(
            reconnect_interval=5,
            max_reconnect_attempts=int(os.environ.get('WS_MAX_RECONNECT_ATTEMPTS', 10)),
            ping_interval=int(os.environ.get('WS_PING_INTERVAL', 30)),
            connection_timeout=int(os.environ.get('WS_CONNECTION_TIMEOUT', 30))
        )

        return fyers_config, strategy_config, trading_config, ws_config

    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


async def run_ha_strategy():
    """Main function to run the Heikin Ashi strategy with enhanced authentication"""
    try:
        logger.info("=" * 60)
        logger.info("STARTING HEIKIN ASHI TRADING STRATEGY")
        logger.info("=" * 60)

        # Load configuration
        fyers_config, strategy_config, trading_config, ws_config = load_configuration()

        # Validate basic configuration
        if not all([fyers_config.client_id, fyers_config.secret_key]):
            logger.error("Missing required Fyers API credentials")
            logger.error("Please set FYERS_CLIENT_ID and FYERS_SECRET_KEY in .env file")
            logger.error("Run 'python main.py auth' to setup authentication")
            return

        # Enhanced authentication with auto-refresh
        config_dict = {'fyers_config': fyers_config}
        if not authenticate_fyers(config_dict):
            logger.error("Authentication failed. Please run 'python main.py auth' to setup authentication")
            return

        logger.info("Authentication successful - Access token validated")

        # Log strategy configuration
        logger.info(f"Portfolio Value: Rs.{strategy_config.portfolio_value:,}")
        logger.info(f"Risk per Trade: {strategy_config.risk_per_trade_pct}%")
        logger.info(f"Max Positions: {strategy_config.max_positions}")
        logger.info(f"HA Smoothing: {strategy_config.ha_smoothing}")
        logger.info(f"ADX Period: {strategy_config.adx_period}, Threshold: {strategy_config.adx_threshold}")
        logger.info(f"ATR Period: {strategy_config.atr_period}, Multiplier: {strategy_config.atr_multiplier}x")
        logger.info(f"Stop Loss: {strategy_config.stop_loss_pct}%")
        logger.info(f"Target Multiple: {strategy_config.target_multiplier}x")

        # Create and run strategy
        strategy = HeikinAshiStrategy(
            fyers_config=config_dict['fyers_config'],
            strategy_config=strategy_config,
            trading_config=trading_config,
            ws_config=ws_config
        )

        # Run strategy
        logger.info("Initializing Heikin Ashi Strategy...")
        await strategy.run()

    except KeyboardInterrupt:
        logger.info("Strategy stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        logger.exception("Full error details:")


def show_strategy_help():
    """Show comprehensive Heikin Ashi strategy help and configuration"""
    print("\n" + "=" * 80)
    print("HEIKIN ASHI TRADING STRATEGY - CONFIGURATION GUIDE")
    print("=" * 80)

    print("\n STRATEGY OVERVIEW:")
    print("• Uses Heikin Ashi candles for trend identification")
    print("• ADX for trend strength confirmation (>25)")
    print("• ATR for dynamic trailing stops")
    print("• Volume analysis for signal validation")
    print("• Monitors 40+ stocks across multiple sectors")

    print("\n CONFIGURATION PARAMETERS:")
    print("Edit .env file or set environment variables:")

    print("\n Portfolio Settings:")
    print("  PORTFOLIO_VALUE=100000         # Total portfolio value (Rs.1L)")
    print("  RISK_PER_TRADE=1.0            # Risk per trade (1% of portfolio)")
    print("  MAX_POSITIONS=3               # Maximum concurrent positions")

    print("\n Heikin Ashi Strategy Parameters:")
    print("  HA_SMOOTHING=3                # EMA smoothing period for HA candles")
    print("  CONSECUTIVE_CANDLES=2         # Required consecutive bullish candles")
    print("  ADX_PERIOD=14                 # ADX calculation period")
    print("  ADX_THRESHOLD=25.0            # Minimum ADX for trend confirmation")
    print("  ATR_PERIOD=14                 # ATR calculation period")
    print("  ATR_MULTIPLIER=2.0            # ATR multiplier for trailing stops")
    print("  VOLUME_PERCENTILE=60.0        # Volume percentile threshold")

    print("\n Risk Management:")
    print("  STOP_LOSS_PCT=1.5             # Initial stop loss percentage")
    print("  TARGET_MULTIPLIER=2.0         # Target as multiple of risk (1:2 ratio)")
    print("  TRAILING_STOP_PCT=0.5         # Trailing stop adjustment percentage")
    print("  BREAKEVEN_PROFIT_PCT=1.0      # Move to breakeven at 1% profit")
    print("  ENABLE_TRAILING_STOPS=true    # Enable dynamic trailing stops")

    print("\n Signal Filtering:")
    print("  MIN_CONFIDENCE=0.65           # Minimum signal confidence (65%)")
    print("  MIN_VOLUME_RATIO=1.5          # Volume vs average ratio")

    print("\n System Settings:")
    print("  MONITORING_INTERVAL=1         # Strategy monitoring cycle (seconds)")
    print("  TICK_INTERVAL=1min            # Candle timeframe (1min, 5min, etc.)")
    print("  LOG_LEVEL=INFO                # Logging verbosity")

    print("\n EXPECTED PERFORMANCE:")
    print("  Daily Signals: 3-8 high-quality setups")
    print("  Win Rate Target: 60-70%")
    print("  Risk-Reward: 1:2 ratio (1.5% risk, 3% target)")
    print("  Monthly Target: 15-25% portfolio growth")

    print("\n  IMPORTANT NOTES:")
    print("  • Start with paper trading or small amounts")
    print("  • Monitor closely during initial weeks")
    print("  • Adjust parameters based on market conditions")
    print("  • Keep trading PIN secure for token refresh")


def show_authentication_status():
    """Show detailed authentication status"""
    print("\n" + "=" * 60)
    print("FYERS API AUTHENTICATION STATUS")
    print("=" * 60)

    # Check current credentials
    client_id = os.environ.get('FYERS_CLIENT_ID')
    access_token = os.environ.get('FYERS_ACCESS_TOKEN')
    refresh_token = os.environ.get('FYERS_REFRESH_TOKEN')

    print(f" Credential Status:")
    print(f"  Client ID: {' Set' if client_id else ' Missing'}")
    print(f"  Access Token: {' Set' if access_token else ' Missing'}")
    print(f"  Refresh Token: {' Set' if refresh_token else ' Missing'}")

    print(f"\n Available Commands:")
    print(f"  Setup Authentication: python main.py auth")
    print(f"  Test Authentication: python main.py test-auth")
    print(f"  Update Trading PIN: python main.py update-pin")
    print(f"  Run Strategy: python main.py run")


def main():
    """Enhanced main entry point with comprehensive CLI interface"""

    # Display header
    print("=" * 80)
    print("    HEIKIN ASHI TRADING STRATEGY")
    print("    Advanced Algorithmic Trading System v1.0")
    print("=" * 80)

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "run":
            logger.info(" Starting Heikin Ashi Strategy")
            asyncio.run(run_ha_strategy())

        elif command == "auth":
            print(" Setting up Fyers API Authentication")
            setup_auth_only()

        elif command == "test-auth":
            print(" Testing Fyers API Authentication")
            test_authentication()

        elif command == "update-pin":
            print(" Updating Trading PIN")
            update_pin_only()

        elif command == "status":
            show_authentication_status()

        elif command == "help":
            show_strategy_help()

        else:
            print(f" Unknown command: {command}")
            print("\n Available commands:")
            commands = [
                ("run", "Run the Heikin Ashi trading strategy"),
                ("auth", "Setup Fyers API authentication"),
                ("test-auth", "Test authentication status"),
                ("update-pin", "Update trading PIN"),
                ("status", "Show authentication status"),
                ("help", "Show strategy configuration guide"),
            ]

            for cmd, desc in commands:
                print(f"  python main.py {cmd:<12} - {desc}")

    else:
        # Interactive menu
        print(" Advanced Heikin Ashi algorithmic trading with real-time data")
        print(" Comprehensive risk management and position monitoring")
        print("\nSelect an option:")

        menu_options = [
            ("1", " Run Heikin Ashi Trading Strategy"),
            ("2", " Setup Fyers Authentication"),
            ("3", " Test Authentication"),
            ("4", " Update Trading PIN"),
            ("5", " Show Authentication Status"),
            ("6", " Strategy Configuration Guide"),
            ("7", " Exit")
        ]

        for option, description in menu_options:
            print(f"{option:>2}. {description}")

        choice = input(f"\nSelect option (1-{len(menu_options)}): ").strip()

        if choice == "1":
            logger.info(" Starting Heikin Ashi Strategy")
            asyncio.run(run_ha_strategy())

        elif choice == "2":
            print(" Setting up Fyers API Authentication")
            setup_auth_only()

        elif choice == "3":
            print(" Testing Fyers API Authentication")
            test_authentication()

        elif choice == "4":
            print(" Updating Trading PIN")
            update_pin_only()

        elif choice == "5":
            show_authentication_status()

        elif choice == "6":
            show_strategy_help()

        elif choice == "7":
            print("\n Goodbye! Happy Trading! ")
            print("  Remember: Trade responsibly and manage your risk!")

        else:
            print(f" Invalid choice: {choice}")
            print("Please select a number between 1 and 7")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n Interrupted by user - Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
        logger.exception("Full error details:")
        sys.exit(1)
