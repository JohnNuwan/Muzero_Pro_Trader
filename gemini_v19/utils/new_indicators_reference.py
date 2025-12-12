# New Enhanced Indicators for V19
# To be added to indicators.py

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
    Alternative to BollingerBands.
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
