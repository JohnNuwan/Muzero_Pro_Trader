
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SyntheticCommissionTrinityEnv(gym.Env):
    """
    Synthetic Environment that mimics CommissionTrinityEnv for testing/training 
    without MetaTrader5 connection.
    Generates realistic-looking market data (Random Walk + Sine Waves).
    """
    def __init__(self, symbol="EURUSD", lookback=1000):
        super().__init__()
        self.symbol = symbol
        self.lookback = lookback
        self.observation_shape = (84,) # Matches V19
        self.action_space = spaces.Discrete(5) # HOLD, BUY, SELL, SPLIT, CLOSE
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32)
        
        self.current_step = 0
        self.max_steps = 1000
        self.balance = 10000.0
        self.position = 0 # 0: None, 1: Long, -1: Short
        self.entry_price = 0.0
        
        # Generate synthetic price data
        t = np.linspace(0, 100, self.max_steps + 100)
        self.prices = 1.1000 + 0.01 * np.sin(t) + 0.005 * np.random.randn(len(t))
        self.prices = np.cumsum(np.random.randn(len(t)) * 0.0001) + 1.1000

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.balance = 10000.0
        self.position = 0
        self.entry_price = 0.0
        return self._get_obs(), {}

    def step(self, action):
        current_price = self.prices[self.current_step]
        reward = 0
        done = False
        
        # Simple Trading Logic
        if action == 1: # BUY
            if self.position == -1: # Close Short
                pnl = (self.entry_price - current_price) * 100000
                self.balance += pnl
                reward += pnl
                self.position = 0
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
                
        elif action == 2: # SELL
            if self.position == 1: # Close Long
                pnl = (current_price - self.entry_price) * 100000
                self.balance += pnl
                reward += pnl
                self.position = 0
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
                
        elif action == 4: # CLOSE ALL
            if self.position == 1:
                pnl = (current_price - self.entry_price) * 100000
                self.balance += pnl
                reward += pnl
            elif self.position == -1:
                pnl = (self.entry_price - current_price) * 100000
                self.balance += pnl
                reward += pnl
            self.position = 0

        # Step
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done, False, {'balance': self.balance}

    def _get_obs(self):
        # Return random observation matching shape
        return np.random.randn(*self.observation_shape).astype(np.float32)
