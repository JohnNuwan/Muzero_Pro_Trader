import MetaTrader5 as mt5
import pandas as pd

def check_dow_symbols():
    if not mt5.initialize():
        print("❌ MT5 Init Failed")
        return

    # Liste officielle Dow Jones 30
    dow_30 = [
        "MMM", "AXP", "AMGN", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DIS", 
        "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "MCD", "MRK", 
        "MSFT", "NKE", "PG", "CRM", "TRV", "UNH", "VZ", "V", "WBA", "WMT"
    ]

    print(f"🔍 Checking {len(dow_30)} Dow Jones symbols in MT5...")
    
    available = []
    missing = []
    
    # Récupérer tous les symboles disponibles
    all_symbols = mt5.symbols_get()
    all_names = [s.name for s in all_symbols] if all_symbols else []
    
    print(f"📊 Total symbols in MT5: {len(all_names)}")
    
    for symbol in dow_30:
        # Recherche exacte ou partielle (ex: "AAPL" ou "AAPL.US" ou "AAPL_US")
        match = None
        for s in all_names:
            if symbol == s or s.startswith(symbol + ".") or s.startswith(symbol + "_"):
                match = s
                break
        
        if match:
            available.append((symbol, match))
        else:
            missing.append(symbol)
            
    print(f"\n✅ Available ({len(available)}):")
    for base, match in available:
        print(f"  - {base} -> {match}")
        
    print(f"\n❌ Missing ({len(missing)}):")
    print(", ".join(missing))
    
    mt5.shutdown()

if __name__ == "__main__":
    check_dow_symbols()
