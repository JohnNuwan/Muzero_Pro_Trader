import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from rich.console import Console

class TrinityEnv(gym.Env):
    """
    Gemini V14 - The Trinity Environment
    Separates concerns between Trader (Entry) and Manager (Exit).
    """
    metadata = {'render_modes': ['human'], 'render_fps': 0}

    def __init__(self, data, deposit=10000.0, multiplier=100000):
        super(TrinityEnv, self).__init__()
        self.data = data
        self.deposit = deposit
        self.multiplier = multiplier
        self.balance = deposit
        self.equity = deposit
        self.positions = []
        self.current_step = 0
        self.max_steps = len(self.data) - 1
        self.console = Console()
        
        # --- Action Spaces ---
        # Trader: 0=Wait, 1=Buy, 2=Sell
        self.trader_action_space = spaces.Discrete(3)
        
        # Manager: 0=Hold, 1=Close 50%, 2=Close 100%, 3=Move SL to BE (Not impl yet)
        self.manager_action_space = spaces.Discrete(3) 
        
        # --- Observation Spaces ---
        # Trader sees Market State: [Price, RSI, Trend, Volatility, Z-Score, Fibo]
        self.trader_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        # Manager sees Trade State: [PnL%, Duration, Volatility, Distance_to_Level]
        self.manager_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.deposit
        self.equity = self.deposit
        self.positions = []
        self.current_step = 0
        
        return self._get_state()

    def _get_state(self):
        # Determine who needs to act
        has_position = len(self.positions) > 0
        
        row = self.data.iloc[self.current_step]
        price = row['close']
        
        # Common Market Features
        rsi = row.get('rsi', 50.0)
        trend = row.get('trend', 0.0)
        volatility = row.get('volatility_sma', 0.0)
        z_score = row.get('z_score', 0.0)
        fibo_pos = row.get('fibo_pos', 0.5)
        
        trader_obs = np.array([price, rsi, trend, volatility, z_score, fibo_pos], dtype=np.float32)
        
        manager_obs = np.zeros(4, dtype=np.float32)
        if has_position:
            pos = self.positions[0]
            # Calculate PnL
            if pos['type'] == 1: # Buy
                pnl_pct = (price - pos['price']) / pos['price']
            else: # Sell
                pnl_pct = (pos['price'] - price) / pos['price']
                
            duration = self.current_step - pos['step']
            
            manager_obs = np.array([pnl_pct * 100, duration, volatility, z_score], dtype=np.float32)
            
        return {
            "mode": "MANAGER" if has_position else "TRADER",
            "trader_obs": trader_obs,
            "manager_obs": manager_obs
        }

    def step(self, action):
        # Action interpretation depends on who is acting
        state = self._get_state()
        mode = state["mode"]
        
        reward = 0
        terminated = False
        truncated = False
        info = {}
        
        row = self.data.iloc[self.current_step]
        price = row['close']
        
        if mode == "TRADER":
            # Action: 0=Wait, 1=Buy, 2=Sell
            if action == 1: # Buy
                self.positions.append({'type': 1, 'price': price, 'step': self.current_step, 'vol': 1.0})
                info['log'] = f"[green]TRADER: BUY @ {price:.5f}[/green]"
            elif action == 2: # Sell
                self.positions.append({'type': -1, 'price': price, 'step': self.current_step, 'vol': 1.0})
                info['log'] = f"[red]TRADER: SELL @ {price:.5f}[/red]"
            else:
                reward = 0 # Small penalty for waiting? No, patience is good.
                
        elif mode == "MANAGER":
            # Action: 0=Hold, 1=Close 50%, 2=Close 100%
            pos = self.positions[0]
            
            # Calculate Current PnL
            if pos['type'] == 1:
                current_pnl = (price - pos['price']) * self.multiplier * pos['vol']
            else:
                current_pnl = (pos['price'] - price) * self.multiplier * pos['vol']
            
            self.equity = self.balance + current_pnl
            
            if action == 1: # Close 50%
                if pos['vol'] > 0.5: # Can only partial close once or if enough vol
                    realized = current_pnl * 0.5
                    self.balance += realized
                    self.positions[0]['vol'] *= 0.5
                    info['log'] = f"[blue]MANAGER: SECURE 50% ({realized:.2f})[/blue]"
                    reward = realized # Reward for securing profit
                else:
                    reward = -10 # Penalty for invalid action
                    
            elif action == 2: # Close 100%
                self.balance += current_pnl
                self.positions = []
                info['log'] = f"[bold]MANAGER: CLOSE ALL ({current_pnl:.2f})[/bold]"
                reward = current_pnl
                
            elif action == 0: # Hold
                reward = 0 # Reward comes at close
                
        # Advance Step
        self.current_step += 1
        if self.current_step >= self.max_steps:
            terminated = True
            
        # Return new state
        new_state = self._get_state()
        
        return new_state, reward, terminated, truncated, info
