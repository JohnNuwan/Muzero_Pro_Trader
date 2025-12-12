import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

def check():
    if not mt5.initialize():
        print("MT5 Init Failed")
        return

    print(f"Connected to: {mt5.terminal_info().name}")
    
    # 1. List all symbols matching *USD*
    print("\n--- Searching for USD symbols ---")
    symbols = mt5.symbols_get(group="*USD*")
    if symbols:
        for s in symbols[:5]:
            print(f"Found: {s.name}")
    else:
        print("No *USD* symbols found")

    # 2. List all symbols matching *500*
    print("\n--- Searching for *500* symbols ---")
    symbols = mt5.symbols_get(group="*500*")
    if symbols:
        for s in symbols[:5]:
            print(f"Found: {s.name}")
    else:
        print("No *500* symbols found")

    # 3. Try to fetch data for specific symbols from config
    targets = ["EURUSD", "US500.cash", "BTCUSD", "XAUUSD", "GER40.cash", "US30.cash", "US100.cash"]
    for t in targets:
        rates = mt5.copy_rates_from_pos(t, mt5.TIMEFRAME_M15, 0, 5000)
        if rates is None:
            print(f"❌ {t}: No data (Error: {mt5.last_error()})")
        else:
            print(f"✅ {t}: Got {len(rates)} candles")

    mt5.shutdown()

if __name__ == "__main__":
    check()
