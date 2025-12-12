import MetaTrader5 as mt5
import sys

print(f"Python Executable: {sys.executable}")
print(f"MT5 Version: {mt5.__version__}")
print(f"MT5 Path: {mt5.__file__}")
print(f"Has calendar_events: {hasattr(mt5, 'calendar_events')}")
