"""
V19 M15 Data Collection Script
Collects M15 data for all 11 V19 symbols from MT5
"""
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from gemini_v19.utils.indicators import Indicators

# V19 Symbols (from main_v19_multi.py)
SYMBOLS = [
    "EURUSD",
    "XAUUSD",
    "BTCUSD",
    "US30.cash",
    "US500.cash",
    "USDJPY",
    "GBPUSD",
    "USDCAD",
    "USDCHF",
    "GER40.cash",
    "US100.cash"
]

def collect_m15_data(symbol, days=730):
    """Collect M15 data for a symbol (2 years default)"""
    print(f"\n📊 Collecting M15 data for {symbol}...")
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Fetch from MT5
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, start_date, end_date)
    
    if rates is None or len(rates) == 0:
        print(f"   ❌ No data available for {symbol}")
        return None
        
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Add indicators
    print(f"   🔧 Adding indicators...")
    df = Indicators.add_all(df)
    
    # Save
    output_path = f"gemini_v19/training/data/{symbol}_M15.csv"
    df.to_csv(output_path, index=False)
    
    print(f"   ✅ Saved {len(df)} bars to {output_path}")
    return len(df)

def main():
    print("🚀 V19 M15 Data Collection")
    print("=" * 50)
    
    # Initialize MT5
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        return
        
    print(f"✅ MT5 Connected")
    print(f"📍 Account: {mt5.account_info().login}")
    
    # Create output directory
    os.makedirs("gemini_v19/training/data", exist_ok=True)
    
    # Collect for all symbols
    results = {}
    for symbol in SYMBOLS:
        bars = collect_m15_data(symbol)
        results[symbol] = bars
        
    # Summary
    print("\n" + "=" * 50)
    print("📊 COLLECTION SUMMARY")
    print("=" * 50)
    
    for symbol, bars in results.items():
        if bars:
            print(f"✅ {symbol:15} : {bars:5} bars")
        else:
            print(f"❌ {symbol:15} : FAILED")
            
    # Count success
    success = sum(1 for b in results.values() if b)
    print(f"\n🎯 Success: {success}/{len(SYMBOLS)} symbols")
    
    mt5.shutdown()
    print("\n✅ Collection Complete!")

if __name__ == "__main__":
    main()
