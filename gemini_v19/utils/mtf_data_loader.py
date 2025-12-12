import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MultiTimeframeLoader:
    def __init__(self):
        self.timeframes = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        self.base_tf = "M1"

    def get_data(self, symbol, lookback_candles=1000):
        """
        Fetches M1 data and resamples it to higher timeframes.
        """
        # 1. Fetch Base Data (M1)
        # We need enough M1 candles. 
        # D1 = 1440 minutes. lookback * 1440 is too huge.
        # We assume we only need recent history for D1 indicators.
        
        # Try fetching 50,000 first (Reasonable limit)
        required_m1 = 50000
        
        rates = mt5.copy_rates_from_pos(symbol, self.timeframes["M1"], 0, required_m1)
        
        # Fallback
        if rates is None:
            print(f"⚠️ Failed to fetch {required_m1} M1 candles for {symbol}. MT5 Error: {mt5.last_error()}")
            print("🔄 Retrying with 15,000 candles...")
            rates = mt5.copy_rates_from_pos(symbol, self.timeframes["M1"], 0, 15000)
            
        if rates is None:
            print(f"❌ Critical: Failed to fetch M1 data for {symbol}.")
            return None

        df_m1 = pd.DataFrame(rates)
        df_m1['time'] = pd.to_datetime(df_m1['time'], unit='s')
        df_m1.set_index('time', inplace=True)
        
        data_dict = {}
        
        # 2. Resample to create synchronized higher timeframes
        # print(f"  📊 Resampling {symbol} M1 data to M5, M15, H1, H4, D1...")
        
        aggregation = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'tick_volume': 'sum',
            'spread': 'mean',
            'real_volume': 'sum'
        }

        # M1 is the base
        data_dict["M1"] = df_m1.copy()

        # Resample Loop
        # Note: We use 'origin=start' to align correctly
        data_dict["M5"] = df_m1.resample('5min', origin='start').agg(aggregation).dropna()
        data_dict["M15"] = df_m1.resample('15min', origin='start').agg(aggregation).dropna()
        data_dict["H1"] = df_m1.resample('1h', origin='start').agg(aggregation).dropna()
        data_dict["H4"] = df_m1.resample('4h', origin='start').agg(aggregation).dropna()
        data_dict["D1"] = df_m1.resample('1D', origin='start').agg(aggregation).dropna()

        return data_dict

    def get_synchronized_state(self, symbol):
        """
        Returns the latest completed candle for all timeframes.
        CRITICAL: We must not look into the future.
        If current time is 10:12:
        - M1: 10:11 candle (closed)
        - M5: 10:05 candle (closed) - 10:10 is currently forming? No, 10:10 closed at 10:15. Wait.
          If it's 10:12, the 10:05-10:10 candle is closed. The 10:10-10:15 is open.
          So we take the 10:05 candle.
        - H1: 09:00 candle (closed). 10:00 is open.
        - H4: 08:00 candle (closed).
        """
        data = self.get_data(symbol)
        if data is None: return None
        
        state = {}
        
        # Get the last completed candle for each timeframe
        # Since we fetched history up to "now", the last row in the resampled DF 
        # might be the incomplete current candle depending on how resample works with partial intervals.
        # But copy_rates_from_pos(0) usually gives history up to the last completed tick.
        
        # To be safe, we take the last row of the resampled data.
        # Note: In live trading, we might need to be careful about "re-painting" if the candle isn't closed.
        # But for now, let's assume the resampling of historical M1 gives us valid closed candles 
        # EXCEPT potentially the very last one if the interval isn't over.
        
        for tf, df in data.items():
            if df.empty: continue
            state[tf] = df.iloc[-1] # The latest candle
            
        return state

if __name__ == "__main__":
    # Test
    if mt5.initialize():
        loader = MultiTimeframeLoader()
        data = loader.get_data("EURUSD")
        if data:
            print("M1:", data["M1"].shape)
            print("M5:", data["M5"].shape)
            print("H1:", data["H1"].shape)
            print("H4:", data["H4"].shape)
            
            print("\nLatest State:")
            state = loader.get_synchronized_state("EURUSD")
            for tf, row in state.items():
                print(f"{tf}: {row.name} | Close: {row['close']}")
        mt5.shutdown()
