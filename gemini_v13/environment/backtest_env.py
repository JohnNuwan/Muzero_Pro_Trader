import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from rich.console import Console

class MT5BacktestEnv(gym.Env):
    """
    Environnement de Backtest rapide sur données historiques (CSV/DataFrame).
    Simule le marché sans attendre les ticks réels.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 0}

    def __init__(self, data, deposit=10000.0, multiplier=100000):
        super(MT5BacktestEnv, self).__init__()
        self.data = data # Pandas DataFrame avec colonnes ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
        self.deposit = deposit
        self.multiplier = multiplier
        self.balance = deposit
        self.equity = deposit
        self.positions = []
        self.current_step = 0
        self.max_steps = len(self.data) - 1
        
        # Action Space: 0=Hold, 1=Buy, 2=Sell, 3=Close
        self.action_space = spaces.Discrete(4)
        
        # Observation Space: [Price, RSI, Trend, Volatility, Z-Score, Fibo_Pos, Pos_Type, Pos_Profit]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        self.console = Console()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.deposit
        self.equity = self.deposit
        self.positions = []
        self.current_step = 0
        
        # Daily Tracking
        self.daily_start_equity = self.deposit
        self.current_day = self.data.iloc[0]['time'].day
        self.daily_bonus_given = False
        
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        self._take_action(action)
        
        # Daily Logic
        current_date = self.data.iloc[self.current_step]['time']
        if current_date.day != self.current_day:
            self.current_day = current_date.day
            self.daily_start_equity = self.equity
            self.daily_bonus_given = False
            
        self.current_step += 1
        terminated = self.equity <= 0 or self.current_step >= self.max_steps
        truncated = False
        
        # Calculate Daily Return
        daily_return = (self.equity - self.daily_start_equity) / self.daily_start_equity
        
        # Reward Shaping
        reward = self._calculate_reward(daily_return)
        
        # Hard Stop (Daily Loss Limit -0.2%)
        if daily_return < -0.002:
            reward -= 1000 # Huge Penalty
            terminated = True # Game Over for this episode (Force Risk Management)
            self.last_log = "[bold red]DAILY STOP LOSS HIT (-0.2%)[/bold red]"
            
        observation = self._get_observation()
        
        info = {"log": getattr(self, "last_log", "")}
        
        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, daily_return):
        base_reward = getattr(self, "last_pnl", 0.0)
        
        # Bonus Target (+10% Daily)
        if daily_return > 0.10 and not self.daily_bonus_given:
            base_reward += 1000 # Huge Bonus
            self.daily_bonus_given = True
            self.last_log = "[bold green]DAILY TARGET HIT (+10%) 🚀[/bold green]"
            
        return base_reward

    def _get_observation(self):
        row = self.data.iloc[self.current_step]
        price = row['close']
        
        # Vrais indicateurs calculés dans data_loader
        # Obs: [Price, RSI, Trend, Volatility, Z-Score, Fibo_Pos, Pos_Type, Pos_Profit]
        rsi = row['rsi'] if 'rsi' in row else 50.0
        trend = row['trend'] if 'trend' in row else 0.0
        volatility = row['volatility_sma'] if 'volatility_sma' in row else 0.0
        z_score = row['z_score'] if 'z_score' in row else 0.0
        fibo_pos = row['fibo_pos'] if 'fibo_pos' in row else 0.5
        
        # Position State
        pos_type = 0
        pos_profit = 0.0
        if len(self.positions) > 0:
            entry_price = self.positions[0]['price']
            p_type = self.positions[0]['type']
            if p_type == 1:
                pos_profit = (price - entry_price) * self.multiplier
                pos_type = 1
            elif p_type == -1:
                pos_profit = (entry_price - price) * self.multiplier
                pos_type = -1
        
        return np.array([price, rsi, trend, volatility, z_score, fibo_pos, pos_type, pos_profit], dtype=np.float32)

    def _take_action(self, action):
        row = self.data.iloc[self.current_step]
        price = row['close'] # On assume exécution au Close de la bougie (simplification)
        self.last_log = ""
        self.last_pnl = 0.0
        
        if action == 1: # Buy
            if len(self.positions) == 0:
                self.positions.append({'type': 1, 'price': price})
                # self.last_log = f"[green]BUY @ {price:.5f}[/green]" # Trop verbeux pour backtest rapide
        elif action == 2: # Sell
            if len(self.positions) == 0:
                self.positions.append({'type': -1, 'price': price})
                # self.last_log = f"[red]SELL @ {price:.5f}[/red]"
        elif action == 3: # Close
            if len(self.positions) > 0:
                entry = self.positions[0]
                pnl = 0
                trade_return = 0.0
                
                if entry['type'] == 1:
                    pnl = (price - entry['price']) * self.multiplier
                    trade_return = (price - entry['price']) / entry['price']
                else:
                    pnl = (entry['price'] - price) * self.multiplier
                    trade_return = (entry['price'] - price) / entry['price']
                
                # Minimum Yield Enforcement (+0.10%)
                # Si le trade est positif mais inférieur à 0.10%, on le considère comme un échec (Scalping inefficace)
                if 0 < trade_return < 0.001:
                    self.last_pnl = -500 # Pénalité pour "Micro-Gain" (Commission/Spread simulation)
                    self.last_log = f"[yellow]SCALP PENALTY | Return: {trade_return*100:.3f}% < 0.10%[/yellow]"
                else:
                    self.last_pnl = pnl
                    self.last_log = f"[blue]CLOSE | PnL: {pnl:.2f} | Ret: {trade_return*100:.3f}%[/blue]"
                
                self.balance += pnl # On met à jour la balance réelle avec le vrai PnL (même si petit)
                self.equity = self.balance
                self.positions = []
            
# Alias for compatibility
BacktestEnv = MT5BacktestEnv
