import MetaTrader5 as mt5

PROJECT_NAME = "Gemini V13 - The Sovereign"
VERSION = "13.0.0"
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "XAUUSD", "US30.cash", "GER40.cash"] # Multi-Symbol Support
TIMEFRAME = 16385 # mt5.TIMEFRAME_H1 (16385 is the int value)
SYMBOL = SYMBOLS[0] # Default for single-agent training
DEPOSIT = 10000.0
LOT_SIZE = 0.10 # Increased for 40k Demo Account
MAGIC_NUMBER = 13000 # Unique ID for Gemini V13
DEVIATION = 20 # Slippage tolerance
