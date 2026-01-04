# FyersHA - ATM Option Symbol Generation & Paper Trading

This document describes the new features added to FyersHA for automatic ATM (At-The-Money) option symbol generation and paper trading mode.

## Features Added

### 1. ATM Option Symbol Generation

Automatically generates option symbols for NIFTY, BANKNIFTY, FINNIFTY, and MIDCPNIFTY based on current spot prices.

**Features:**
- Fetches real-time spot prices from Fyers API
- Calculates ATM strike prices automatically
- Determines correct expiry dates (weekly/monthly)
- Generates symbols in Fyers format (e.g., `NSE:NIFTY25D0926000CE`)
- Daily symbol caching to avoid repeated API calls
- Supports multiple strikes (ATM + OTM)

**Files:**
- `utils/symbol_generator.py` - Core symbol generation logic
- `utils/symbol_manager.py` - Symbol caching and management
- `utils/generate_symbols.py` - Standalone script to generate symbols

### 2. Paper Trading Mode

Simulates trading without placing real orders, with comprehensive trade logging.

**Features:**
- Simulated order execution (entry, stop loss, target)
- CSV trade log with all trade details
- Detailed text log for debugging
- Daily P&L tracking
- Win rate and profit factor calculation
- No broker API calls for orders

**Files:**
- `strategy/paper_order_manager.py` - Simulated order manager
- `utils/paper_trade_logger.py` - Trade logging system
- `logs/paper_trades.csv` - CSV log of all trades
- `logs/paper_trades_detailed.log` - Detailed trade log

## Usage

### Generating ATM Option Symbols

#### Option 1: Using the Standalone Script

```bash
# Run the symbol generator
python utils/generate_symbols.py
```

This will:
1. Fetch current spot prices for configured indices
2. Calculate ATM strikes
3. Determine expiry dates
4. Generate option symbols
5. Save to `daily_symbols.json`

#### Option 2: Programmatically

```python
from utils.symbol_manager import SymbolManager

# Initialize
symbol_manager = SymbolManager()
symbol_manager.initialize_generator(client_id, access_token)

# Generate symbols for NIFTY and BANKNIFTY
# ATM + 1 OTM strike on each side
symbols = symbol_manager.get_or_generate_symbols(
    indices=['NIFTY', 'BANKNIFTY'],
    num_strikes_otm=1
)

print(f"Generated {len(symbols)} symbols")
```

### Symbol Format Examples

**NIFTY (Weekly Expiry):**
- `NSE:NIFTY25D0926000CE` - NIFTY 26000 Call, Dec 09, 2025
- `NSE:NIFTY25D0926000PE` - NIFTY 26000 Put, Dec 09, 2025

**BANKNIFTY (Monthly Expiry):**
- `NSE:BANKNIFTY25DEC58000CE` - BANKNIFTY 58000 Call, Dec 2025
- `NSE:BANKNIFTY25DEC58000PE` - BANKNIFTY 58000 Put, Dec 2025

### Enabling Paper Trading Mode

Update your `.env` file or set environment variables:

```bash
# Enable paper trading mode (default: false)
ENABLE_PAPER_TRADING=true

# Paper trade log location (optional)
PAPER_TRADE_LOG_FILE=logs/paper_trades.log
```

Or modify `main.py`:

```python
trading_config = TradingConfig(
    enable_paper_trading=True,  # Enable paper mode
    paper_trade_log_file="logs/paper_trades.log"
)
```

### Running Paper Trading

```bash
# Run strategy in paper trading mode
python main.py run
```

**Output:**
- Real-time logs showing simulated trades
- `logs/paper_trades.csv` - CSV with all trade details
- `logs/paper_trades_detailed.log` - Detailed trade information
- Daily summary printed at end of session

### Paper Trade Logs

#### CSV Log Format

The `paper_trades.csv` file contains:
- timestamp
- symbol
- signal_type (LONG/SHORT)
- entry_time, entry_price
- exit_time, exit_price
- quantity
- gross_pnl, net_pnl
- exit_reason (TARGET, STOP_LOSS, SQUARE_OFF, etc.)
- holding_period_min
- entry_adx, entry_atr
- max_favorable_excursion, max_adverse_excursion

#### Example CSV Entry

```csv
timestamp,symbol,signal_type,entry_time,entry_price,exit_time,exit_price,quantity,gross_pnl,net_pnl,exit_reason,holding_period_min,entry_adx,entry_atr,max_favorable_excursion,max_adverse_excursion
2026-01-04 10:30:00,NIFTY25D0926000CE,LONG,2026-01-04 10:15:00,150.50,2026-01-04 14:30:00,165.75,50,762.50,762.50,TARGET,255.0,28.50,12.30,15.25,-2.50
```

## Configuration

### Symbol Generation Settings

Edit `utils/generate_symbols.py` to customize:

```python
# Indices to generate symbols for
indices = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']

# Number of OTM strikes on each side (0 = ATM only)
num_strikes_otm = 1  # Generates ATM, ATM+50, ATM-50 for NIFTY
```

### Index Configuration

Edit `utils/symbol_generator.py` to modify index settings:

```python
INDEX_CONFIG = {
    'NIFTY': {
        'spot_symbol': 'NSE:NIFTY50-INDEX',
        'strike_interval': 50,  # 50 point strikes
        'lot_size': 50,
        'expiry_day': 1,  # Tuesday
        'expiry_type': 'weekly'
    },
    # Add more indices...
}
```

## Integration with Strategy

The strategy automatically supports both modes:

```python
# Strategy initializes based on config
strategy = HeikinAshiStrategy(
    fyers_config=fyers_config,
    strategy_config=strategy_config,
    trading_config=trading_config,  # Paper mode flag here
    ws_config=ws_config
)

# In paper mode:
# - Orders are simulated
# - Trades are logged to CSV/text files
# - Daily summary is printed
# - No real broker API calls for orders
```

## Daily Workflow

### 1. Start of Day

```bash
# Generate today's option symbols
python utils/generate_symbols.py
```

### 2. Run Strategy

```bash
# Paper trading (recommended for testing)
ENABLE_PAPER_TRADING=true python main.py run

# Live trading (use with caution!)
ENABLE_PAPER_TRADING=false python main.py run
```

### 3. End of Day

Review paper trade logs:
- `logs/paper_trades.csv` - Import into Excel for analysis
- `logs/paper_trades_detailed.log` - Review detailed trade flow

## Important Notes

### Paper Trading
- **NO REAL ORDERS** are placed in paper trading mode
- All execution is simulated
- Use this mode to test strategy before going live
- Paper trading does NOT account for:
  - Slippage
  - Order rejection
  - Partial fills
  - Network latency

### Live Trading
- Set `ENABLE_PAPER_TRADING=false` for live trading
- **REAL ORDERS** will be placed with your broker
- **REAL MONEY** will be at risk
- Test thoroughly in paper mode first
- Start with small position sizes

### Symbol Generation
- Run symbol generator daily before market open
- Symbols are cached in `daily_symbols.json`
- Cache is automatically invalidated next day
- Requires valid Fyers access token
- Requires market data permissions

## Troubleshooting

### Symbol Generation Fails

```
Error: Could not fetch spot price for NIFTY
```

**Solutions:**
1. Check Fyers access token is valid
2. Verify market data subscription
3. Check internet connection
4. Run during market hours for live prices

### Paper Trade Logs Not Created

```
Error: Permission denied: logs/paper_trades.csv
```

**Solutions:**
1. Create logs directory: `mkdir -p logs`
2. Check write permissions: `chmod 755 logs`
3. Verify disk space available

### Orders Still Being Placed in Paper Mode

```
Warning: Real orders being placed despite paper mode
```

**Solutions:**
1. Verify `ENABLE_PAPER_TRADING=true` in .env
2. Check trading_config.enable_paper_trading is True
3. Restart the strategy completely

## Example Output

### Symbol Generation

```
================================================================================
ATM OPTION SYMBOL GENERATOR FOR FYERSHA
================================================================================
Generating symbols for: NIFTY, BANKNIFTY
Strikes: ATM + 1 OTM on each side
================================================================================

2026-01-04 09:00:00 - INFO - Generating symbols for NIFTY...
2026-01-04 09:00:01 - INFO - NIFTY spot price: 26050.25
2026-01-04 09:00:01 - INFO - NIFTY ATM Strike: 26050 (Spot: 26050.25)
2026-01-04 09:00:01 - INFO - NIFTY Next Expiry: 2026-01-09 Tuesday
2026-01-04 09:00:01 - INFO - Generated 6 symbols for NIFTY

Successfully generated 12 symbols!

Generated symbols:
   1. NSE:NIFTY25D0926000CE
   2. NSE:NIFTY25D0926000PE
   3. NSE:NIFTY25D0926050CE
   4. NSE:NIFTY25D0926050PE
   5. NSE:NIFTY25D0926100CE
   6. NSE:NIFTY25D0926100PE
   ...
```

### Paper Trading Session

```
================================================================================
PAPER TRADE ENTRY
================================================================================
Time: 2026-01-04 10:15:00
Symbol: NSE:NIFTY25D0926000CE
Signal Type: LONG
Entry Price: 150.50
Quantity: 50
Stop Loss: 145.00
Target Price: 160.00
ADX: 28.50
ATR: 12.30
================================================================================

[PAPER] Entry order simulated - NSE:NIFTY25D0926000CE: Side=BUY, Qty=50, Price=150.50

================================================================================
PAPER TRADE EXIT
================================================================================
Time: 2026-01-04 14:30:00
Symbol: NSE:NIFTY25D0926000CE
Signal Type: LONG
Entry Price: 150.50
Exit Price: 165.75
Quantity: 50
Holding Period: 255.0 minutes
Exit Reason: TARGET
Gross P&L: Rs.762.50
Net P&L: Rs.762.50 (+10.13%)
================================================================================

================================================================================
PAPER TRADING DAILY SUMMARY
================================================================================
Total Trades: 5
Winning Trades: 3
Losing Trades: 2
Win Rate: 60.00%
Total P&L: Rs.1250.50
Gross Profit: Rs.2150.00
Gross Loss: Rs.899.50
Profit Factor: 2.39
================================================================================
NOTE: This was PAPER TRADING - No real orders were placed
================================================================================
```

## Next Steps

1. **Test in Paper Mode**: Run strategy for several days in paper mode
2. **Analyze Results**: Review CSV logs to understand strategy performance
3. **Optimize Parameters**: Adjust strategy config based on paper trading results
4. **Go Live**: Once confident, switch to live trading with small positions
5. **Monitor**: Continuously monitor and adjust

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review this documentation
3. Check main README.md for general strategy information
4. Verify Fyers API credentials and permissions
