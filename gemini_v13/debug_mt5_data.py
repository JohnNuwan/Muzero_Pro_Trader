import MetaTrader5 as mt5
from datetime import datetime
import pandas as pd

def debug_mt5():
    if not mt5.initialize():
        print(f"Initialize failed, error code = {mt5.last_error()}")
        return

    symbol = "EURUSD"
    print(f"Checking symbol: {symbol}")
    
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"{symbol} not found, can not call order_check()")
        # Try to find similar symbols
        symbols = mt5.symbols_get()
        print(f"Total symbols: {len(symbols)}")
        for s in symbols:
            if "EURUSD" in s.name:
                print(f"Found similar: {s.name}")
    else:
        print(f"{symbol} found!")
        print(f"Spread: {info.spread}")
        print(f"Digits: {info.digits}")
        
        # Try fetching data
        print("Attempting to fetch rates...")
        rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M15, datetime.now(), 10)
        if rates is None:
            print(f"Rates are None. Error: {mt5.last_error()}")
        else:
            print(f"Fetched {len(rates)} rates")
            print(rates[:2])

    mt5.shutdown()

if __name__ == "__main__":
    debug_mt5()
