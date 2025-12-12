import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from MuZero.utils.mtf_data_loader import MultiTimeframeLoader
from MuZero.utils.indicators import Indicators

class DeepTrinityEnv(gym.Env):
    """
    Gymnasium Environment for Gemini V15 (Deep RL).
    Handles Multi-Timeframe Data and Advanced Indicators.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, symbol="EURUSD", lookback=1000):
        super(DeepTrinityEnv, self).__init__()
        
        self.symbol = symbol
        self.lookback = lookback
        self.loader = MultiTimeframeLoader()
        
        # Load Data
        self.raw_data = self.loader.get_data(symbol, lookback)
        if self.raw_data is None:
            raise ValueError(f"Could not load data for {symbol}")
            
        # Process Indicators for all timeframes
        self.data = {}
        for tf, df in self.raw_data.items():
            self.data[tf] = Indicators.add_all(df.copy())
            
        # Align data (intersection of indices)
        # We need to ensure that at step T, we have data for all TFs
        # This is complex because H4 has fewer rows than M1.
        # Strategy: We step through M1 (Base). For other TFs, we take the latest available closed candle.
        self.m1_data = self.data["M1"]
        self.valid_indices = self.m1_data.index
        
        # Define Action Space (Discrete for PPO/DQN)
        # 0: Hold
        # 1: Buy
        # 2: Sell
        self.action_space = spaces.Discrete(3)
        
        # Define Observation Space
        # We need to flatten all indicators from all TFs into a single vector.
        # Let's count features.
        # M1: ~30 features
        # M5: ~30 features
        # H1: ~30 features
        # H4: ~30 features
        # Total ~120 floats.
        
        # We'll dynamically determine the shape based on the first row
        self.current_step = 0
        self.max_steps = len(self.m1_data) - 1
        
        obs = self._get_observation()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )
        
        # Trading State
        self.balance = 10000.0
        self.position = None # None, 'long', 'short'
        self.entry_price = 0.0
        self.equity_curve = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 100 # Start with enough history
        self.balance = 10000.0
        self.position = None
        self.entry_price = 0.0
        self.equity_curve = [self.balance]
        
        return self._get_observation(), {}

    def step(self, action):
        # 1. Get current market data
        current_price = self.m1_data.iloc[self.current_step]['close']
        
        # 2. Execute Action
        reward = 0
        done = False
        
        # Simple Logic for Training (Can be made more complex with spread/commission)
        if action == 1: # BUY
            if self.position == 'short':
                # Close Short
                pnl = (self.entry_price - current_price) * 100000 # Standard Lot approx
                self.balance += pnl
                self.position = None
                reward += pnl
            
            if self.position is None:
                self.position = 'long'
                self.entry_price = current_price
                
        elif action == 2: # SELL
            if self.position == 'long':
                # Close Long
                pnl = (current_price - self.entry_price) * 100000
                self.balance += pnl
                self.position = None
                reward += pnl
                
            if self.position is None:
                self.position = 'short'
                self.entry_price = current_price
        
        # 3. Calculate Reward (PnL + Stability)
        # If holding, unrealized PnL
        if self.position == 'long':
            unrealized_pnl = (current_price - self.entry_price) * 100000
            # Small reward for holding winning trade, penalty for losing
            reward += unrealized_pnl * 0.01 
        elif self.position == 'short':
            unrealized_pnl = (self.entry_price - current_price) * 100000
            reward += unrealized_pnl * 0.01
            
        # 4. Next Step
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
            
        self.equity_curve.append(self.balance)
        
        # Info dict
        info = {'balance': self.balance, 'step': self.current_step}
        
        return self._get_observation(), reward, done, False, info

    def _get_observation(self):
        """
        Constructs the Multi-Timeframe State Vector.
        """
        # Get current M1 time
        current_time = self.m1_data.index[self.current_step]
        
        state_vector = []
        
        # For each timeframe, find the latest closed candle relative to current_time
        # Added M15 and D1
        for tf in ["M1", "M5", "M15", "H1", "H4", "D1"]:
            df = self.data.get(tf) # Use .get() to avoid error if D1 is missing (e.g. not enough history)
            
            if df is None or df.empty:
                 # Fallback if data missing
                state_vector.extend([0.0] * 22)
                continue

            # We use 'asof' to find the closest index in the past
            # Since indices are timestamps, we find the row <= current_time
            try:
                # Use searchsorted for speed if indices are sorted (they are)
                idx = df.index.searchsorted(current_time, side='right') - 1
                if idx < 0: idx = 0
                row = df.iloc[idx]
                
                # Normalize/Scale data (Crucial for Neural Nets)
                # For now, we just append raw values, but in production we MUST normalize (Z-Score or MinMax)
                # We'll assume Indicators.z_score handles some normalization, but we need more.
                # Let's pick specific columns to avoid passing timestamps or non-numeric
                
                features = [
                    row['rsi'], row['mfi'], row['adx'], row['z_score'], 
                    row['trend_score'], row['linreg_angle'], row['fibo_pos'],
                    row['dist_to_res'], row['dist_to_sup'], row['skew'],
                    row['kurtosis'], row['entropy'], row['hurst'],
                    # V19 New Features
                    row['stoch_rsi_k'], row['williams_r'], row['cci'],
                    row['bb_percent_b'], row['bb_width'], row['vwap_signal'],
                    row['volume_concentration'],
                    # Specific Trading
                    row['obv'], row['spread']
                ]
                # Handle NaNs (fill with 0)
                features = [0.0 if np.isnan(x) else x for x in features]
                
                state_vector.extend(features)
                
            except Exception as e:
                # Fallback if data missing
                state_vector.extend([0.0] * 22) # 22 features per TF

        # Add Time Features (Cyclical Encoding)
        # Hour (0-23) -> Sin/Cos
        # Day (0-6) -> Sin/Cos
        hour = current_time.hour
        day = current_time.dayofweek
        
        state_vector.append(np.sin(2 * np.pi * hour / 24.0))
        state_vector.append(np.cos(2 * np.pi * hour / 24.0))
        state_vector.append(np.sin(2 * np.pi * day / 7.0))
        state_vector.append(np.cos(2 * np.pi * day / 7.0))
                
        return np.array(state_vector, dtype=np.float32)

if __name__ == "__main__":
    # Test the Environment
    import sys
    import os
    # Add project root to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    sys.path.append(project_root)

    import MetaTrader5 as mt5
    print("Initializing MT5...")
    if mt5.initialize():
        print("MT5 Initialized. Creating Env...")
        env = DeepTrinityEnv("EURUSD", 50) # Reduced to 50 for quick test
        print("Env Created. Resetting...")
        obs, _ = env.reset()
        print("Reset Done.")
        print(f"Observation Shape: {obs.shape}")
        print(f"Sample Observation (First 10): {obs[:10]}")
        
        # Random Walk Test
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Balance: {info['balance']:.2f}")
            if done: break
        mt5.shutdown()
