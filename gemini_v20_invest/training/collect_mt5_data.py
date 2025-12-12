"""
V20 Training Data Collector
Collect historical MT5 index data for AlphaZero training
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
sys.path.append('gemini_v15')

from gemini_v15.utils.indicators import Indicators

# Indices MT5 disponibles
TRAINING_SYMBOLS = {
    'US30.cash': 'DowJones',
    'CAC40.cash': 'CAC40', 
    'GER40.cash': 'DAX',
    'US500.cash': 'SP500',
    'US100.cash': 'NASDAQ',
}

def collect_symbol_data(symbol, years=2, timeframe=mt5.TIMEFRAME_D1):
    """
    Collect historical data from MT5 for one symbol
    """
    print(f"\n{'='*60}")
    print(f"📊 Collecting {symbol} - {years} years D1 data")
    print(f"{'='*60}")
    
    # Initialize MT5
    if not mt5.initialize():
        print(f"❌ MT5 initialization failed for {symbol}")
        return None
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*years + 30)  # +30 for indicators warmup
    
    # Fetch rates
    rates = mt5.copy_rates_range(
        symbol,
        timeframe,
        start_date,
        end_date
    )
    
    if rates is None or len(rates) == 0:
        print(f"❌ No data for {symbol}")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    print(f"✅ Downloaded {len(df)} bars")
    print(f"   Period: {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
    
    # Apply 26 indicators (same as V19)
    print("🔧 Applying 26 indicators...")
    df = Indicators.add_all(df)
    
    # Remove NaN rows (warmup period)
    df = df.dropna()
    
    print(f"✅ Final dataset: {len(df)} bars (after indicator warmup)")
    
    return df

def save_training_data(symbol, df, output_dir='gemini_v20_invest/training/data'):
    """
    Save processed data to CSV
    """
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{TRAINING_SYMBOLS[symbol]}_D1_2y.csv"
    filepath = os.path.join(output_dir, filename)
    
    df.to_csv(filepath, index=False)
    print(f"💾 Saved to: {filepath}")
    
    return filepath

def collect_all_training_data():
    """
    Collect data for all training symbols
    """
    print("\n🏋️ V20 TRAINING DATA COLLECTION")
    print("="*60)
    
    results = {}
    
    for symbol, name in TRAINING_SYMBOLS.items():
        df = collect_symbol_data(symbol, years=2)
        
        if df is not None:
            filepath = save_training_data(symbol, df)
            results[symbol] = {
                'name': name,
                'bars': len(df),
                'file': filepath
            }
        else:
            print(f"⚠️ Skipping {symbol} - no data")
    
    # Summary
    print("\n" + "="*60)
    print("📊 COLLECTION SUMMARY")
    print("="*60)
    
    for symbol, info in results.items():
        print(f"{info['name']:12} : {info['bars']:5} bars → {info['file']}")
    
    print(f"\n✅ Collected {len(results)}/{len(TRAINING_SYMBOLS)} datasets")
    
    mt5.shutdown()
    return results

if __name__ == "__main__":
    results = collect_all_training_data()
    
    if len(results) > 0:
        print("\n🎯 Ready for training!")
        print("   Next: python -m gemini_v20_invest.training.train_v20")
    else:
        print("\n❌ No data collected - check MT5 symbols")
