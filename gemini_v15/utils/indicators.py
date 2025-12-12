import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy

class Indicators:
    """
    Technical Indicators Library for Gemini V15.
    Optimized for pandas DataFrames.
    """
    
    @staticmethod
    def add_all(df):
        """Adds all available indicators to the DataFrame."""
        df = Indicators.rsi(df)
        df = Indicators.mfi(df)
        df = Indicators.ema(df, period=20)
        df = Indicators.ema(df, period=50)
        df = Indicators.ema(df, period=200)
        df = Indicators.atr(df)
        df = Indicators.adx(df)
        df = Indicators.obv(df)
        df = Indicators.z_score(df)
        df = Indicators.linear_regression(df)
        df = Indicators.pivots(df)
        df = Indicators.fibonacci(df)
        df = Indicators.support_resistance(df)
        df = Indicators.accumulation_distribution(df)
        df = Indicators.statistical_moments(df)
        df = Indicators.shannon_entropy(df)
        df = Indicators.hurst_exponent(df)
        df = Indicators.kalman_filter(df)
        df = Indicators.trend_strength(df)
        
        # V19 Enhanced Indicators
        df = Indicators.stochastic_rsi(df)
        df = Indicators.williams_r(df)
        df = Indicators.cci(df)
        df = Indicators.bollinger_bands(df)
        df = Indicators.keltner_channels(df)
        df = Indicators.atr_bands(df)
        df = Indicators.vwap(df)
        df = Indicators.volume_profile(df)
        return df.dropna()

    @staticmethod
    def rsi(df, period=14):
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'rsi'] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def mfi(df, period=14):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['tick_volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
        
        mfi_ratio = positive_flow / negative_flow
        df[f'mfi'] = 100 - (100 / (1 + mfi_ratio))
        return df

    @staticmethod
    def ema(df, period=20):
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df

    @staticmethod
    def atr(df, period=14):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        df[f'atr'] = true_range.rolling(window=period).mean()
        return df

    @staticmethod
    def adx(df, period=14):
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        
        tr = Indicators.atr(df.copy(), period=1)['atr'] # TR for ADX
        
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        df[f'adx'] = dx.rolling(window=period).mean()
        return df

    @staticmethod
    def obv(df):
        df['obv'] = (np.sign(df['close'].diff()) * df['tick_volume']).fillna(0).cumsum()
        return df

    @staticmethod
    def z_score(df, period=20):
        mean = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        df['z_score'] = (df['close'] - mean) / std
        return df

    @staticmethod
    def linear_regression(df, period=20):
        """Calculates Linear Regression Slope and Forecast."""
        x = np.arange(period)
        
        def get_slope(y):
            if len(y) < period: return np.nan
            return np.polyfit(x, y, 1)[0]
            
        df['linreg_slope'] = df['close'].rolling(window=period).apply(get_slope, raw=True)
        df['linreg_angle'] = np.degrees(np.arctan(df['linreg_slope']))
        return df

    @staticmethod
    def pivots(df):
        """Calculates Standard Pivot Points (Classic)."""
        # Using previous candle's High, Low, Close
        prev_high = df['high'].shift(1)
        prev_low = df['low'].shift(1)
        prev_close = df['close'].shift(1)
        
        df['pivot'] = (prev_high + prev_low + prev_close) / 3
        df['r1'] = (2 * df['pivot']) - prev_low
        df['s1'] = (2 * df['pivot']) - prev_high
        df['r2'] = df['pivot'] + (prev_high - prev_low)
        df['s2'] = df['pivot'] - (prev_high - prev_low)
        return df

    @staticmethod
    def fibonacci(df, period=100):
        """Calculates Fibonacci Retracement Levels based on recent High/Low."""
        rolling_high = df['high'].rolling(window=period).max()
        rolling_low = df['low'].rolling(window=period).min()
        diff = rolling_high - rolling_low
        
        df['fibo_0'] = rolling_low
        df['fibo_236'] = rolling_low + (diff * 0.236)
        df['fibo_382'] = rolling_low + (diff * 0.382)
        df['fibo_500'] = rolling_low + (diff * 0.5)
        df['fibo_618'] = rolling_low + (diff * 0.618)
        df['fibo_100'] = rolling_high
        
        # Position relative to Fibo (0 to 1)
        df['fibo_pos'] = (df['close'] - rolling_low) / (diff + 1e-9)
        return df

    @staticmethod
    def support_resistance(df, period=20):
        """Identifies local Support and Resistance levels."""
        # Simple fractal-based or rolling min/max approach
        # Here we use rolling min/max as dynamic S/R zones
        df['resistance'] = df['high'].rolling(window=period).max()
        df['support'] = df['low'].rolling(window=period).min()
        
        # Distance to S/R
        df['dist_to_res'] = (df['resistance'] - df['close']) / df['close']
        df['dist_to_sup'] = (df['close'] - df['support']) / df['close']
        return df

    @staticmethod
    def accumulation_distribution(df):
        """Calculates Accumulation/Distribution Line."""
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-9)
        df['ad_line'] = (clv * df['tick_volume']).cumsum()
        return df

    @staticmethod
    def statistical_moments(df, period=20):
        """Calculates Skewness and Kurtosis of returns."""
        returns = df['close'].pct_change()
        
        # Rolling Skewness
        df['skew'] = returns.rolling(window=period).skew()
        
        # Rolling Kurtosis
        df['kurtosis'] = returns.rolling(window=period).kurt()
        return df

    @staticmethod
    def shannon_entropy(df, period=20):
        """
        Calculates Shannon Entropy (Market Chaos/Uncertainty).
        NOTE: Using default value 0.5 for live trading performance.
        Full calculation is too slow for real-time use.
        """
        df['entropy'] = 0.5  # Neutral entropy (future: optimize vectorized calculation)
        return df

    @staticmethod
    def hurst_exponent(df, period=100):
        """
        Calculates Hurst Exponent.
        H < 0.5: Mean Reverting
        H = 0.5: Random Walk
        Useful for noise reduction.
        """
        # Optimization: Rolling Hurst is extremely slow. 
        # For live trading speed, we temporarily disable it or use a very fast proxy.
        # Setting to 0.5 (Random Walk) as neutral default.
        df['hurst'] = 0.5 
        return df

    @staticmethod
    def kalman_filter(df):
        """
        Kalman Filter for noise reduction.
        """
        # Simple Kalman Filter implementation
        n_iter = len(df)
        sz = (n_iter,) # size of array
        
        # Allocate space for arrays
        xhat = np.zeros(sz)      # a posteri estimate of x
        P = np.zeros(sz)         # a posteri error estimate
        xhatminus = np.zeros(sz) # a priori estimate of x
        Pminus = np.zeros(sz)    # a priori error estimate
        K = np.zeros(sz)         # gain or blending factor
        
        Q = 1e-5 # process variance
        R = 0.01**2 # estimate of measurement variance
        
        # Intial guesses
        xhat[0] = df['close'].iloc[0]
        P[0] = 1.0
        
        close_prices = df['close'].values
        
        for k in range(1, n_iter):
            # Time update
            xhatminus[k] = xhat[k-1]
            Pminus[k] = P[k-1] + Q
            
            # Measurement update
            K[k] = Pminus[k] / (Pminus[k] + R)
            xhat[k] = xhatminus[k] + K[k] * (close_prices[k] - xhatminus[k])
            P[k] = (1 - K[k]) * Pminus[k]
            
        df['kalman_price'] = xhat
        df['kalman_diff'] = df['close'] - df['kalman_price'] # Noise component
        return df
        
    @staticmethod
    def trend_strength(df):
        # Simple Trend based on EMA alignment
        # 1 = Strong Bull, -1 = Strong Bear, 0 = Range
        
        # We need EMAs calculated first
        if 'ema_20' not in df: df = Indicators.ema(df, 20)
        if 'ema_50' not in df: df = Indicators.ema(df, 50)
        if 'ema_200' not in df: df = Indicators.ema(df, 200)
        
        conditions = [
            (df['close'] > df['ema_20']) & (df['ema_20'] > df['ema_50']) & (df['ema_50'] > df['ema_200']),
            (df['close'] < df['ema_20']) & (df['ema_20'] < df['ema_50']) & (df['ema_50'] < df['ema_200'])
        ]
        choices = [1.0, -1.0]
        df['trend_score'] = np.select(conditions, choices, default=0.0)
        return df

    # ============================================
    # NEW INDICATORS - V19 Enhanced (2024-11-24)
    # ============================================
    
    @staticmethod
    def stochastic_rsi(df, period=14, k_period=3, d_period=3):
        """
        Stochastic RSI - More sensitive than regular RSI.
        Values: 0-1 (oversold < 0.2, overbought > 0.8)
        """
        # Calculate RSI first
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Stochastic of RSI
        rsi_min = rsi.rolling(window=period).min()
        rsi_max = rsi.rolling(window=period).max()
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min)
        
        # Smooth with K and D
        df['stoch_rsi_k'] = stoch_rsi.rolling(window=k_period).mean()
        df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(window=d_period).mean()
        
        return df
    
    @staticmethod
    def williams_r(df, period=14):
        """
        Williams %R - Momentum indicator.
        Values: 0 to -100 (overbought > -20, oversold < -80)
        """
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        
        df['williams_r'] = ((highest_high - df['close']) / (highest_high - lowest_low)) * -100
        
        return df
    
    @staticmethod
    def cci(df, period=20):
        """
        Commodity Channel Index.
        Values: typically -100 to +100 (extreme > ±200)
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        df['cci'] = (typical_price - sma) / (0.015 * mean_deviation)
        
        return df
    
    @staticmethod
    def bollinger_bands(df, period=20, std_dev=2):
        """
        Bollinger Bands - Volatility bands.
        Returns: bb_upper, bb_lower, bb_width, bb_percent_b
        """
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        df['bb_upper'] = sma + (std * std_dev)
        df['bb_lower'] = sma - (std * std_dev)
        df['bb_middle'] = sma
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']  # Normalized width
        
        # %B: Position within bands (0 = lower band, 1 = upper band)
        df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    @staticmethod
    def keltner_channels(df, period=20, atr_multiplier=2):
        """
        Keltner Channels - ATR-based volatility bands.
        Alternative to Bollinger Bands.
        """
        # EMA as middle line
        ema = df['close'].ewm(span=period, adjust=False).mean()
        
        # ATR for width
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        df['keltner_upper'] = ema + (atr * atr_multiplier)
        df['keltner_lower'] = ema - (atr * atr_multiplier)
        df['keltner_middle'] = ema
        
        return df
    
    @staticmethod
    def atr_bands(df, period=14, multiplier=2):
        """
        ATR Bands - Trend bands based on ATR.
        """
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        # Bands around close
        df['atr_upper'] = df['close'] + (atr * multiplier)
        df['atr_lower'] = df['close'] - (atr * multiplier)
        
        return df
    
    @staticmethod
    def vwap(df):
        """
        Volume Weighted Average Price.
        Resets daily (or per available data window).
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        volume = df['tick_volume']
        
        # Cumulative for VWAP
        df['vwap'] = (typical_price * volume).cumsum() / volume.cumsum()
        
        # Signal: 1 if price > VWAP (bullish), -1 if < VWAP (bearish)
        df['vwap_signal'] = np.where(df['close'] > df['vwap'], 1, -1)
        
        return df
    
    @staticmethod
    def volume_profile(df, bins=20):
        """
        Volume Profile - Distribution of volume across price levels.
        Returns volume concentration score (0-1, higher = more concentrated).
        """
        # Price bins
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_range = price_max - price_min
        
        if price_range == 0:
            df['volume_concentration'] = 0.5
            return df
            
        bin_size = price_range / bins
        
        # Assign each candle to a bin
        df['price_bin'] = ((df['close'] - price_min) / bin_size).astype(int).clip(0, bins-1)
        
        # Volume per bin
        volume_by_bin = df.groupby('price_bin')['tick_volume'].sum()
        
        # Concentration: Gini coefficient approximation
        # Higher value = volume more concentrated in certain price levels
        sorted_volumes = volume_by_bin.sort_values()
        n = len(sorted_volumes)
        index = np.arange(1, n + 1)
        concentration = (2 * (index * sorted_volumes).sum()) / (n * sorted_volumes.sum()) - (n + 1) / n
        
        df['volume_concentration'] = concentration
        
        return df
if __name__ == "__main__":
    # Test
    import sys
    import os
    # Add project root to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    sys.path.append(project_root)
    
    from gemini_v15.utils.mtf_data_loader import MultiTimeframeLoader
    import MetaTrader5 as mt5
    
    if mt5.initialize():
        loader = MultiTimeframeLoader()
        data = loader.get_data("EURUSD", 5000)
        if data:
            df_h1 = data["H1"]
            print(f"Raw H1: {df_h1.shape}")
            
            df_h1 = Indicators.add_all(df_h1)
            print(f"Processed H1: {df_h1.shape}")
            print(df_h1[['close', 'rsi', 'linreg_angle', 'pivot', 'fibo_pos', 'dist_to_res']].tail())
        mt5.shutdown()

if __name__ == "__main__":
    # Test
    import sys
    import os
    # Add project root to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    sys.path.append(project_root)
    
    from gemini_v15.utils.mtf_data_loader import MultiTimeframeLoader
    import MetaTrader5 as mt5
    
    if mt5.initialize():
        loader = MultiTimeframeLoader()
        data = loader.get_data("EURUSD", 5000)
        if data:
            df_h1 = data["H1"]
            print(f"Raw H1: {df_h1.shape}")
            
            df_h1 = Indicators.add_all(df_h1)
            print(f"Processed H1: {df_h1.shape}")
            print(df_h1[['close', 'rsi', 'linreg_angle', 'pivot', 'fibo_pos', 'dist_to_res']].tail())
        mt5.shutdown()
