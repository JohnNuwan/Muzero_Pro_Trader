import MetaTrader5 as mt5

def check_v19_expansion_symbols():
    if not mt5.initialize():
        print("❌ MT5 Init Failed")
        return

    target_symbols = [
        "USDJPY", "GBPUSD", "USDCAD", "USDCHF",  # Forex
        "GER40.cash", "US100.cash"               # CFDs
    ]

    print(f"🔍 Checking availability for {len(target_symbols)} new symbols...")
    
    all_symbols = mt5.symbols_get()
    all_names = [s.name for s in all_symbols] if all_symbols else []
    
    available = []
    missing = []
    
    for symbol in target_symbols:
        # Exact match check first
        if symbol in all_names:
            available.append(symbol)
        else:
            # Try to find close matches (e.g. suffixes)
            match = None
            for s in all_names:
                if s.startswith(symbol):
                    match = s
                    break
            
            if match:
                available.append(f"{symbol} (found as {match})")
            else:
                missing.append(symbol)
            
    print(f"\n✅ Available ({len(available)}):")
    for s in available:
        print(f"  - {s}")
        
    if missing:
        print(f"\n❌ Missing ({len(missing)}):")
        print(", ".join(missing))
    
    mt5.shutdown()

if __name__ == "__main__":
    check_v19_expansion_symbols()
