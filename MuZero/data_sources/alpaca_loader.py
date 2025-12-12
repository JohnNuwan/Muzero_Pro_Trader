
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
try:
    from MuZero.credentials import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_ENDPOINT
except ImportError:
    print("⚠️ Credentials not found. Please create MuZero/credentials.py")
    ALPACA_API_KEY = ""
    ALPACA_SECRET_KEY = ""
    ALPACA_ENDPOINT = "https://paper-api.alpaca.markets"

class AlpacaDataLoader:
    def __init__(self):
        self.headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
            "accept": "application/json"
        }
        self.base_url = ALPACA_ENDPOINT

    def get_bars(self, symbol, timeframe="1Min", limit=1000):
        """
        Fetch bars from Alpaca API.
        timeframe: '1Min', '5Min', '15Min', '1H', '1D'
        """
        # Map MT5 symbols to Alpaca symbols if needed
        symbol_map = {
            "EURUSD": "EUR/USD", # Alpaca uses slash for forex? Or just EURUSD? Alpaca supports Crypto and Stocks.
            # Forex on Alpaca requires specific subscription usually.
            # Crypto: BTC/USD
            "BTCUSD": "BTC/USD",
            "XAUUSD": "XAU/USD", # Not sure if supported on basic paper
            "US30.cash": "DIA", # ETF proxy? Or Index? Alpaca has stocks.
            "US500.cash": "SPY",
            "US100.cash": "QQQ",
            "GER40.cash": "EWG", # ETF proxy
        }
        
        alpaca_symbol = symbol_map.get(symbol, symbol)
        
        # Construct URL
        # For Crypto: /v1beta3/crypto/us/bars
        # For Stocks: /v2/stocks/bars
        
        if "BTC" in symbol or "ETH" in symbol:
            url = f"https://data.alpaca.markets/v1beta3/crypto/us/bars"
            params = {
                "symbols": alpaca_symbol,
                "timeframe": timeframe,
                "limit": limit
            }
        else:
            # Stocks
            url = f"https://data.alpaca.markets/v2/stocks/bars"
            params = {
                "symbols": alpaca_symbol,
                "timeframe": timeframe,
                "limit": limit
            }

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "bars" in data and alpaca_symbol in data["bars"]:
                bars = data["bars"][alpaca_symbol]
                df = pd.DataFrame(bars)
                # Rename columns to match our env expectations (open, high, low, close, volume)
                df.rename(columns={
                    "t": "time",
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume"
                }, inplace=True)
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                return df
            else:
                print(f"⚠️ No data found for {alpaca_symbol}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching Alpaca data for {symbol}: {e}")
            return None

if __name__ == "__main__":
    loader = AlpacaDataLoader()
    print("Testing Alpaca Loader...")
    df = loader.get_bars("BTCUSD", limit=10)
    if df is not None:
        print(df.head())
    else:
        print("Failed.")
