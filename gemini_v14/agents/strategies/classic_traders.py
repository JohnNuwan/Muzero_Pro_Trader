import numpy as np
from gemini_v14.agents.base_agent import TraderAgent

class TurtleAgent(TraderAgent):
    """
    Classic Turtle Trading Strategy.
    Buy on 20-day High.
    Sell on 10-day Low.
    (Adapted for intraday using 'lookback' periods)
    """
    def __init__(self, name="Turtle", entry_lookback=20, exit_lookback=10):
        super().__init__(name)
        self.entry_lookback = entry_lookback
        self.exit_lookback = exit_lookback
        self.prices = []
        
    def act(self, observation):
        # Obs: [Price, RSI, Trend, Volatility, Z-Score, Fibo]
        price = observation[0]
        self.prices.append(price)
        
        if len(self.prices) > self.entry_lookback:
            self.prices.pop(0)
            
        if len(self.prices) < self.entry_lookback:
            return 0 # Wait for enough data
            
        # Turtle Logic
        high_channel = max(self.prices[:-1])
        low_channel = min(self.prices[:-1])
        
        if price > high_channel:
            return 1 # Buy (Breakout)
        elif price < low_channel:
            return 2 # Sell (Breakdown)
            
        return 0 # Wait

class MovingAverageAgent(TraderAgent):
    """
    Simple Moving Average Crossover.
    Buy when Fast MA > Slow MA.
    Sell when Fast MA < Slow MA.
    """
    def __init__(self, name="MA_Cross", fast_period=10, slow_period=30):
        super().__init__(name)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.prices = []
        
    def act(self, observation):
        price = observation[0]
        self.prices.append(price)
        
        if len(self.prices) > self.slow_period:
            self.prices.pop(0)
            
        if len(self.prices) < self.slow_period:
            return 0
            
        fast_ma = np.mean(self.prices[-self.fast_period:])
        slow_ma = np.mean(self.prices[-self.slow_period:])
        
        # Simple Logic: If Fast crosses above Slow -> Buy
        # But we need to know if it JUST crossed. 
        # For simplicity, we return signal if condition is met, 
        # the Environment/Manager handles if we are already in position.
        
        if fast_ma > slow_ma:
            return 1 # Bullish
        elif fast_ma < slow_ma:
            return 2 # Bearish
            
        return 0
