# config/symbols.py

"""
Symbol Configuration for Heikin Ashi Trading Strategy
Single source of truth for symbol to Fyers format mapping
"""

from typing import Dict, List, Tuple


class HASymbolManager:
    """Symbol manager for Heikin Ashi strategy"""

    def __init__(self):
        # Simple mapping: symbol -> Fyers WebSocket format
        self._symbol_mappings: Dict[str, str] = {
            # Large Cap Stocks - FMCG
            "NESTLEIND": "NSE:NESTLEIND-EQ",
            "COLPAL": "NSE:COLPAL-EQ",
            "HINDUNILVR": "NSE:HINDUNILVR-EQ",
            "ITC": "NSE:ITC-EQ",
            "BRITANNIA": "NSE:BRITANNIA-EQ",
            "DABUR": "NSE:DABUR-EQ",
            "MARICO": "NSE:MARICO-EQ",
            "TATACONSUM": "NSE:TATACONSUM-EQ",

            # IT Sector
            "TCS": "NSE:TCS-EQ",
            "INFY": "NSE:INFY-EQ",
            "WIPRO": "NSE:WIPRO-EQ",
            "HCLTECH": "NSE:HCLTECH-EQ",
            "TECHM": "NSE:TECHM-EQ",

            # Banking
            "HDFCBANK": "NSE:HDFCBANK-EQ",
            "ICICIBANK": "NSE:ICICIBANK-EQ",
            "SBIN": "NSE:SBIN-EQ",
            "AXISBANK": "NSE:AXISBANK-EQ",
            "KOTAKBANK": "NSE:KOTAKBANK-EQ",
            "INDUSINDBK": "NSE:INDUSINDBK-EQ",

            # Auto
            "MARUTI": "NSE:MARUTI-EQ",
            "BAJAJ-AUTO": "NSE:BAJAJ-AUTO-EQ",
            "M&M": "NSE:M&M-EQ",
            "HEROMOTOCO": "NSE:HEROMOTOCO-EQ",
            "EICHERMOT": "NSE:EICHERMOT-EQ",

            # Pharma
            "SUNPHARMA": "NSE:SUNPHARMA-EQ",
            "DRREDDY": "NSE:DRREDDY-EQ",
            "CIPLA": "NSE:CIPLA-EQ",
            "DIVISLAB": "NSE:DIVISLAB-EQ",

            # Energy & Others
            "RELIANCE": "NSE:RELIANCE-EQ",
            "ONGC": "NSE:ONGC-EQ",
            "IOC": "NSE:IOC-EQ",
            "BPCL": "NSE:BPCL-EQ",
            "NTPC": "NSE:NTPC-EQ",
            "POWERGRID": "NSE:POWERGRID-EQ",

            # Metals
            "TATASTEEL": "NSE:TATASTEEL-EQ",
            "JSWSTEEL": "NSE:JSWSTEEL-EQ",
            "HINDALCO": "NSE:HINDALCO-EQ",
            "VEDL": "NSE:VEDL-EQ",
        }

        # Create reverse mapping for quick lookups
        self._reverse_mappings = {v: k for k, v in self._symbol_mappings.items()}

    def get_fyers_symbol(self, symbol: str) -> str:
        """Get Fyers format symbol"""
        return self._symbol_mappings.get(symbol.upper())

    def get_display_symbol(self, fyers_symbol: str) -> str:
        """Get display symbol from Fyers format"""
        return self._reverse_mappings.get(fyers_symbol)

    def get_all_symbols(self) -> List[str]:
        """Get all available symbols"""
        return list(self._symbol_mappings.keys())

    def get_all_fyers_symbols(self) -> List[str]:
        """Get all symbols in Fyers format"""
        return list(self._symbol_mappings.values())

    def create_symbol_mappings(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Create forward and reverse mapping dictionaries"""
        return self._symbol_mappings.copy(), self._reverse_mappings.copy()

    def validate_symbol(self, symbol: str) -> bool:
        """Check if symbol is supported"""
        return symbol.upper() in self._symbol_mappings

    def get_trading_universe_size(self) -> int:
        """Get total number of tradable symbols"""
        return len(self._symbol_mappings)

    def export_for_websocket(self) -> Dict[str, str]:
        """Export symbols in format suitable for WebSocket subscription"""
        return self._symbol_mappings.copy()

    def get_symbol_summary(self) -> Dict:
        """Get summary of symbol universe"""
        return {
            'total_symbols': len(self._symbol_mappings),
            'symbols': self.get_all_symbols(),
            'fyers_symbols': self.get_all_fyers_symbols()
        }


# Global symbol manager instance
symbol_manager = HASymbolManager()


# Convenience functions for easy access
def get_ha_symbols() -> List[str]:
    """Get all Heikin Ashi trading symbols (display format)"""
    return symbol_manager.get_all_symbols()


def get_ha_fyers_symbols() -> List[str]:
    """Get all symbols in Fyers WebSocket format"""
    return symbol_manager.get_all_fyers_symbols()


def convert_to_fyers_format(symbol: str) -> str:
    """Convert display symbol to Fyers format"""
    return symbol_manager.get_fyers_symbol(symbol)


def convert_from_fyers_format(fyers_symbol: str) -> str:
    """Convert Fyers format to display symbol"""
    return symbol_manager.get_display_symbol(fyers_symbol)


def validate_ha_symbol(symbol: str) -> bool:
    """Validate if symbol is supported for HA trading"""
    return symbol_manager.validate_symbol(symbol)


def get_symbol_mappings() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Get symbol mappings for WebSocket services"""
    return symbol_manager.create_symbol_mappings()


# Example usage and testing
if __name__ == "__main__":
    print("Heikin Ashi Symbol Manager Test")
    print("=" * 50)

    # Test symbol manager
    print(f"Total symbols: {symbol_manager.get_trading_universe_size()}")

    # Display all symbols
    print(f"\nAll symbols:")
    for symbol in get_ha_symbols():
        print(f"  {symbol} -> {convert_to_fyers_format(symbol)}")
