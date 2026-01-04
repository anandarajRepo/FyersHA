#!/usr/bin/env python3
"""
Standalone script to generate ATM option symbols for trading
Run this before starting the trading session to generate symbols for the day
"""

import os
import sys
import logging
from dotenv import load_dotenv
from symbol_manager import SymbolManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Generate ATM symbols for today's trading"""
    load_dotenv()

    # Get credentials from environment
    client_id = os.getenv('FYERS_CLIENT_ID')
    access_token = os.getenv('FYERS_ACCESS_TOKEN')

    if not client_id or not access_token:
        logger.error("FYERS_CLIENT_ID and FYERS_ACCESS_TOKEN must be set in .env file")
        sys.exit(1)

    # Configuration
    indices = ['NIFTY', 'BANKNIFTY']  # Add more indices as needed
    num_strikes_otm = 1  # Generate ATM + 1 strike on each side

    print("\n" + "=" * 80)
    print("ATM OPTION SYMBOL GENERATOR FOR FYERSHA")
    print("=" * 80)
    print(f"Generating symbols for: {', '.join(indices)}")
    print(f"Strikes: ATM + {num_strikes_otm} OTM on each side")
    print("=" * 80 + "\n")

    # Initialize symbol manager
    symbol_manager = SymbolManager()

    # Initialize generator
    if not symbol_manager.initialize_generator(client_id, access_token):
        logger.error("Failed to initialize symbol generator")
        sys.exit(1)

    # Generate symbols
    symbols = symbol_manager.get_or_generate_symbols(
        indices=indices,
        num_strikes_otm=num_strikes_otm,
        force_regenerate=True  # Always regenerate
    )

    if symbols:
        print(f"\nSuccessfully generated {len(symbols)} symbols!")
        print("\nGenerated symbols:")
        for i, symbol in enumerate(symbols, 1):
            print(f"  {i:2d}. {symbol}")

        # Print cache info
        symbol_manager.print_cache_info()

        print("\n" + "=" * 80)
        print("Symbols saved to: daily_symbols.json")
        print("These symbols will be automatically loaded by the trading strategy")
        print("=" * 80 + "\n")

    else:
        logger.error("Failed to generate symbols")
        sys.exit(1)


if __name__ == "__main__":
    main()
