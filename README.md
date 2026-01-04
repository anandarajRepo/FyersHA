# Heikin Ashi Trading Strategy

An advanced algorithmic trading system implementing the Heikin Ashi strategy with real-time data integration via Fyers API.

## Overview

This project implements a sophisticated Heikin Ashi trading strategy that combines:
- **Heikin Ashi candles** with EMA smoothing for trend identification
- **ADX (Average Directional Index)** for trend strength confirmation
- **ATR (Average True Range)** for dynamic trailing stops
- **Volume analysis** for signal validation
- **Real-time WebSocket data** from Fyers API
- **Comprehensive risk management** with trailing stops and breakeven protection

## Strategy Logic

### Entry Conditions (ALL must be true):
1. Heikin Ashi turns bullish (close > open)
2. Multiple consecutive bullish HA candles (momentum confirmation)
3. ADX > threshold (trending market, not choppy)
4. Volume > percentile threshold (good liquidity)
5. Before square-off time (3:20 PM IST)

### Exit Conditions (ANY triggers exit):
1. ATR-based dynamic trailing stop (adapts to volatility)
2. Breakeven stop (after 1% profit, stop moves to entry)
3. Heikin Ashi turns bearish
4. Target hit (2x risk-reward ratio)
5. 3:20 PM square-off (mandatory)

## Features

-  Real-time market data via Fyers WebSocket API
-  Advanced Heikin Ashi calculation with EMA smoothing
-  Multi-indicator confirmation (ADX, ATR, Volume)
-  Dynamic position sizing based on risk percentage
-  ATR-based trailing stops
-  Breakeven protection
-  Comprehensive logging and monitoring
-  Enhanced authentication with auto-refresh
-  Paper trading support

## Installation

1. Clone the repository:
```bash
git clone https://github.com/anandarajRepo/FyersHA.git
cd FyersHA
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file from template:
```bash
cp .env.template .env
```

5. Configure your Fyers API credentials in `.env`:
```
FYERS_CLIENT_ID=your_client_id
FYERS_SECRET_KEY=your_secret_key
```

## Configuration

Edit the `.env` file to customize strategy parameters:

### Portfolio Settings
- `PORTFOLIO_VALUE`: Total portfolio value (default: Rs.100,000)
- `RISK_PER_TRADE`: Risk per trade as % of portfolio (default: 1.0%)
- `MAX_POSITIONS`: Maximum concurrent positions (default: 3)

### Heikin Ashi Parameters
- `HA_SMOOTHING`: EMA smoothing period (default: 3)
- `CONSECUTIVE_CANDLES`: Required consecutive bullish candles (default: 2)

### Technical Indicators
- `ADX_PERIOD`: ADX calculation period (default: 14)
- `ADX_THRESHOLD`: Minimum ADX for trend confirmation (default: 25.0)
- `ATR_PERIOD`: ATR calculation period (default: 14)
- `ATR_MULTIPLIER`: ATR multiplier for trailing stops (default: 2.0)
- `VOLUME_PERCENTILE`: Volume percentile threshold (default: 60%)

### Risk Management
- `STOP_LOSS_PCT`: Initial stop loss percentage (default: 1.5%)
- `TARGET_MULTIPLIER`: Target as multiple of risk (default: 2.0)
- `TRAILING_STOP_PCT`: Trailing stop adjustment (default: 0.5%)
- `BREAKEVEN_PROFIT_PCT`: Profit % to trigger breakeven (default: 1.0%)

## Usage

### Setup Authentication
```bash
python main.py auth
```

### Test Authentication
```bash
python main.py test-auth
```

### Run the Strategy
```bash
python main.py run
```

### Update Trading PIN
```bash
python main.py update-pin
```

### Show Help
```bash
python main.py help
```

### Show Authentication Status
```bash
python main.py status
```

## Project Structure

```
FyersHA/
├── config/
│   ├── settings.py           # Strategy configuration
│   ├── websocket_config.py   # WebSocket configuration
│   └── symbols.py            # Trading symbols configuration
├── models/
│   └── trading_models.py     # Data models
├── services/
│   ├── fyers_websocket_service.py  # WebSocket data service
│   ├── market_timing_service.py    # Market timing utilities
│   └── analysis_service.py         # Technical analysis
├── strategy/
│   ├── ha_strategy.py        # Main strategy implementation
│   └── order_manager.py      # Order management
├── utils/
│   └── enhanced_auth_helper.py  # Authentication utilities
├── logs/                     # Log files
├── main.py                   # Main application
├── requirements.txt          # Dependencies
├── .env.template            # Environment template
└── README.md                # This file
```

## Strategy Performance

### Expected Metrics:
- **Daily Signals**: 3-8 high-quality setups
- **Win Rate Target**: 60-70%
- **Risk-Reward**: 1:2 ratio (1.5% risk, 3% target)
- **Monthly Target**: 15-25% portfolio growth
- **Max Drawdown**: <5% with proper risk management

## Symbols Covered

The strategy monitors 40+ stocks across multiple sectors:
- **FMCG**: NESTLEIND, COLPAL, HINDUNILVR, ITC, BRITANNIA, etc.
- **IT**: TCS, INFY, WIPRO, HCLTECH, TECHM, etc.
- **Banking**: HDFCBANK, ICICIBANK, SBIN, AXISBANK, etc.
- **Auto**: MARUTI, TATAMOTORS, BAJAJ-AUTO, M&M, etc.
- **Others**: RELIANCE, SUNPHARMA, TATASTEEL, etc.

## Risk Management

The strategy implements multiple layers of risk management:
1. **Position Sizing**: Risk-based position sizing (1% per trade)
2. **Stop Losses**: ATR-based dynamic stop losses
3. **Trailing Stops**: Automatic profit protection
4. **Breakeven**: Move stops to entry after 1% profit
5. **Daily Loss Limit**: 2% maximum daily loss
6. **Square-Off**: Mandatory position closing at 3:20 PM

## Logging

All trading activity is logged to:
- Console (real-time monitoring)
- `logs/ha_strategy.log` (file logging)

Log levels can be configured via `LOG_LEVEL` environment variable.

## Important Notes

 **Risk Warning**: Trading involves risk. This strategy is provided for educational purposes. Start with paper trading or small amounts.

 **Testing**: Thoroughly backtest and paper trade before live trading.

 **Monitoring**: Closely monitor the strategy during initial weeks.

 **Market Conditions**: Adjust parameters based on market conditions.

## Architecture Reference

This project follows the architecture from the FyersORB project:
- **Configuration Management**: Centralized settings and parameters
- **Data Models**: Type-safe trading models
- **Data Services**: Real-time WebSocket data integration
- **Enhanced Authentication**: Auto-refresh token management
- **Strategy Pattern**: Clean separation of concerns

## Strategy Reference

The Heikin Ashi strategy logic is adapted from the AdvancedHeikinAshi backtester from the BackTest project, implementing:
- Smoothed Heikin Ashi candles
- ADX trend strength filter
- ATR-based dynamic stops
- Volume confirmation
- Multi-candle momentum

## License

MIT License

## Support

For issues and questions:
- GitHub Issues: https://github.com/anandarajRepo/FyersHA/issues

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Happy Trading! **

*Remember to trade responsibly and always manage your risk!*
