from gemini_core import GeminiCore
import MetaTrader5 as mt5
import time

def force_sync():
    print("Initializing Gemini Core...")
    gemini = GeminiCore()
    
    print("Connecting to MT5...")
    if not mt5.initialize():
        print("MT5 Initialization failed")
        return

    print("Forcing Trade Sync...")
    gemini.sync_trades()
    
    print("Sync Complete.")
    mt5.shutdown()

if __name__ == "__main__":
    force_sync()
