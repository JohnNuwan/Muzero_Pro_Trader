import numpy as np
import pandas as pd
import random
import os
from gemini_v19.utils.selfplay_config import SELF_PLAY_CONFIG
from gemini_v19.utils.indicators import Indicators

class SimulatedMarket:
    """
    Simulated Market Environment for Self-Play.
    Loads historical data and simulates trading steps.
    """
    def __init__(self, symbol, timeframe='H1'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = self._load_data()
        self.current_step = 0
        self.balance = SELF_PLAY_CONFIG['initial_balance']
        self.position = 0 # 0: Flat, 1: Long, -1: Short
        self.entry_price = 0.0
        self.equity = self.balance
        
    def _load_data(self):
        """Load historical data from CSV"""
        # Assuming data is stored in gemini_v20_invest/data/training/ or similar
        # For V19, we might need to fetch or use existing data.
        # Let's assume we have some data or we fetch it.
        # For now, we'll try to load from a standard path or create dummy data if missing for testing.
        
        # In a real scenario, we would have a data loader.
        # Let's look for data in 'data/' folder relative to project root.
        data_path = f"gemini_v19/training/data/{self.symbol}_{self.timeframe}.csv"
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df['time'] = pd.to_datetime(df['time'])
            # Ensure indicators are present
            if 'rsi' not in df.columns:
                df = Indicators.add_all(df)
            return df
        else:
            # Fallback: Generate dummy data or error
            # print(f"⚠️ Data not found for {self.symbol}. Using random walk.")
            return self._generate_dummy_data()

    def _generate_dummy_data(self):
        """Generate random walk data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=2000, freq='H')
        price = 1.1000
        data = []
        for d in dates:
            change = np.random.normal(0, 0.001)
            price *= (1 + change)
            data.append({
                'time': d,
                'open': price,
                'high': price * 1.001,
                'low': price * 0.999,
                'close': price,
                'tick_volume': 1000
            })
        df = pd.DataFrame(data)
        df = Indicators.add_all(df)
        return df

    def reset(self):
        """Reset environment to a random point in history"""
        self.balance = SELF_PLAY_CONFIG['initial_balance']
        self.equity = self.balance
        self.position = 0
        self.entry_price = 0.0
        
        # Pick random start index, leaving enough room for max_steps
        max_steps = SELF_PLAY_CONFIG['max_steps']
        if len(self.data) > max_steps + 100:
            self.current_step = random.randint(100, len(self.data) - max_steps - 1)
        else:
            self.current_step = 100
            
        return self._get_state()

    def step(self, action):
        """
        Execute action
        0: HOLD
        1: BUY
        2: SELL
        """
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0
        done = False
        
        # Execute Action
        if action == 1: # BUY
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
            elif self.position == -1: # Close Short, Open Long
                # Close Short
                pnl = (self.entry_price - current_price) * 100000 # Standard lot
                self.balance += pnl
                # Open Long
                self.position = 1
                self.entry_price = current_price
                
        elif action == 2: # SELL
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
            elif self.position == 1: # Close Long, Open Short
                # Close Long
                pnl = (current_price - self.entry_price) * 100000
                self.balance += pnl
                # Open Short
                self.position = -1
                self.entry_price = current_price
        
        elif action == 0: # HOLD
            pass
            
        # Move to next step
        self.current_step += 1
        next_price = self.data.iloc[self.current_step]['close']
        
        # Calculate Unrealized PnL for Reward
        if self.position == 1:
            unrealized_pnl = (next_price - self.entry_price) * 100000
        elif self.position == -1:
            unrealized_pnl = (self.entry_price - next_price) * 100000
        else:
            unrealized_pnl = 0
            
        # Reward is change in equity (simplified)
        new_equity = self.balance + unrealized_pnl
        reward = new_equity - self.equity
        self.equity = new_equity
        
        # Check Done
        if self.current_step >= len(self.data) - 1:
            done = True
            
        info = {'equity': self.equity, 'balance': self.balance}
        truncated = False
            
        return self._get_state(), reward, done, truncated, info

    def _get_state(self):
        """
        Get current state vector (84 features) to match AlphaZeroTradingNet input.
        Structure:
        - 6 Timeframes * 13 Features = 78
        - 4 Time Features (sin/cos hour/day)
        - 2 Position Features (State, PnL)
        Total: 84
        """
        # 1. Extract Core Features from current step
        # We only have one timeframe loaded (self.data), so we will use it to fill
        # the corresponding slot or replicate it.
        # For robustness in self-play with single-source data, we can replicate 
        # the features across timeframes or just fill the primary one.
        # Let's replicate it to avoid sparse inputs which might confuse the net 
        # if it expects dense data. Or better, just fill all slots with the same 
        # trend info (fractal assumption).
        
        row = self.data.iloc[self.current_step]
        
        # List of 13 features used in CommissionTrinityEnv
        feature_names = [
            'rsi', 'mfi', 'adx', 'z_score', 
            'trend_score', 'linreg_angle', 'fibo_pos',
            'dist_to_res', 'dist_to_sup', 'skew',
            'kurtosis', 'entropy', 'hurst'
        ]
        
        # Extract values, handling missing ones with 0.0
        core_features = []
        for f in feature_names:
            val = row.get(f, 0.0)
            if pd.isna(val): val = 0.0
            core_features.append(float(val))
            
        # 2. Build 78-dim Multi-Timeframe Vector
        # We replicate the core features 6 times (M1, M5, M15, H1, H4, D1)
        # This is a simplification for self-play using single-timeframe data.
        state_vector = []
        for _ in range(6):
            state_vector.extend(core_features)
            
        # 3. Time Features (4)
        current_time = row['time']
        hour = current_time.hour
        day = current_time.dayofweek
        state_vector.append(np.sin(2 * np.pi * hour / 24.0))
        state_vector.append(np.cos(2 * np.pi * hour / 24.0))
        state_vector.append(np.sin(2 * np.pi * day / 7.0))
        state_vector.append(np.cos(2 * np.pi * day / 7.0))
        
        # 4. Position Features (2)
        pos_state = float(self.position) # 0, 1, -1
        
        pnl_pct = 0.0
        current_price = row['close']
        if self.position != 0 and self.entry_price > 0:
            if self.position == 1:
                pnl_pct = (current_price - self.entry_price) / self.entry_price
            else:
                pnl_pct = (self.entry_price - current_price) / self.entry_price
                
        state_vector.append(pos_state)
        state_vector.append(pnl_pct)
        
        # Final conversion
        return np.array(state_vector, dtype=np.float32)
