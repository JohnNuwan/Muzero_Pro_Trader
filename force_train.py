import sys
import os
import json
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from gemini_core import GeminiCore

def force_train():
    print("Initializing Gemini Core...")
    bot = GeminiCore()
    
    # Load config to get symbols
    with open("backend/gemini_config.json", "r") as f:
        config = json.load(f)
        
    symbols = list(config.keys())
    print(f"Found symbols: {symbols}")
    
    if not bot.connect():
        print("Failed to connect to MT5")
        return

    for symbol in symbols:
        print(f"\n--- Training {symbol} ---")
        success = bot.train_model(symbol)
        if success:
            print(f"Successfully trained {symbol}")
        else:
            print(f"Failed to train {symbol}")
            
    print("\nDone. Check gemini_config.json for updates.")

if __name__ == "__main__":
    force_train()
