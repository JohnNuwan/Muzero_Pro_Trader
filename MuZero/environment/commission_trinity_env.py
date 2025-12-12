import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from MuZero.environment.deep_trinity_env import DeepTrinityEnv

class CommissionTrinityEnv(DeepTrinityEnv):
    """
    V16 Environment: Commission-Aware, Pyramiding, and Splitting.
    Inherits observation logic from DeepTrinityEnv.
    
    Action Space (Discrete 5):
    0: Hold
    1: Buy (Open Long or Pyramid Long)
    2: Sell (Open Short or Pyramid Short)
    3: Split (Close 50% of position)
    4: Close All
    """
    def __init__(self, symbol="EURUSD", lookback=1000, start_index=0, end_index=None, timeframe="M1",
                 quality_trade_multiplier=2.0, enable_final_growth_bonus=False, 
                 final_growth_threshold=0.10, final_growth_bonus=100.0):
        super().__init__(symbol, lookback)
        
        # V2 Enhanced Rewards
        self.quality_trade_multiplier = quality_trade_multiplier  # Default: 2.0, V2: 5.0
        self.enable_final_growth_bonus = enable_final_growth_bonus
        self.final_growth_threshold = final_growth_threshold  # 10%
        self.final_growth_bonus = final_growth_bonus  # +100 points
        
        # Data Splitting
        self.start_index = start_index
        self.timeframe = timeframe
        
        # Select Primary Data Source
        if timeframe == "M5":
            self.primary_data = self.data["M5"]
        elif timeframe == "H1":
            self.primary_data = self.data["H1"]
        else:
            self.primary_data = self.m1_data # Default M1
            
        self.end_index = end_index if end_index is not None else len(self.primary_data) - 1
        
        # Action Space (Discrete 5)
        self.action_space = spaces.Discrete(5)
        
        # Observation Space: Base + 2 (Position Size, PnL %)
        base_shape = self.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(base_shape + 2,), dtype=np.float32
        )
        
        # Trading State Extensions
        self.position_size = 0.0 
        self.avg_entry_price = 0.0
        
        # Config
        self.commission_rate = 0.00005 # 0.005% (~0.5 pips) - Realistic Raw Spread
        self.sl_pips = 10
        self.tp_pips = 100
        self.pip_size = 0.0001 
        
        # Adjust for JPY and Indices
        if "JPY" in symbol:
            self.pip_size = 0.01
            self.initial_balance = 1000000.0 # ~10k USD in JPY
        elif "US30" in symbol or "GER40" in symbol or "US100" in symbol:
             self.pip_size = 1.0 # Points
             self.initial_balance = 100000.0 # Indices often high value
        elif "BTC" in symbol:
             self.pip_size = 1.0
             self.initial_balance = 100000.0
        elif "XAU" in symbol:
             self.pip_size = 0.01
             self.initial_balance = 10000.0
        else:
             self.initial_balance = 10000.0

        # Growth Tracking (NEW)
        self.last_equity_check = self.initial_balance
        self.steps_since_last_trade = 0
        self.total_profitable_trades = 0
        self.total_trades = 0
             
        self.balance = self.initial_balance
        self.peak_equity = self.initial_balance

    def reset(self, seed=None, options=None):
        # Call parent reset to handle data pointers
        obs, info = super().reset(seed=seed, options=options)
        
        # Override Step for Data Splitting
        self.current_step = max(100, self.start_index)
        # Limit episode length to 1000 steps to prevent infinite holding penalties
        self.max_steps = min(self.current_step + 1000, self.end_index)
        
        self.position_size = 0.0
        self.avg_entry_price = 0.0
        self.balance = self.initial_balance 
        self.peak_equity = self.initial_balance
        self.equity_curve = [self.initial_balance]
        
        # Reset Growth Tracking
        self.last_equity_check = self.initial_balance
        self.steps_since_last_trade = 0
        self.total_profitable_trades = 0
        self.total_trades = 0
        
        return self._get_full_observation(obs), info

    def _get_observation(self):
        """
        Constructs the Multi-Timeframe State Vector (Corrected for Primary Timeframe).
        """
        # Handle Initialization Case (super().__init__ calls this before primary_data is set)
        if not hasattr(self, 'primary_data'):
            # Fallback to M1 data (which is set in DeepTrinityEnv)
            if hasattr(self, 'm1_data'):
                current_time = self.m1_data.index[self.current_step]
            else:
                # Extreme edge case, shouldn't happen if DeepTrinityEnv init order is correct
                return np.zeros(1, dtype=np.float32) 
        else:
            # Get current time from PRIMARY data (M5, H1, etc.)
            current_time = self.primary_data.index[self.current_step]
        
        state_vector = []
        
        # For each timeframe, find the latest closed candle relative to current_time
        for tf in ["M1", "M5", "M15", "H1", "H4", "D1"]:
            df = self.data.get(tf) 
            
            if df is None or df.empty:
                state_vector.extend([0.0] * 13)
                continue

            try:
                # Find row <= current_time
                idx = df.index.searchsorted(current_time, side='right') - 1
                if idx < 0: idx = 0
                row = df.iloc[idx]
                
                features = [
                    row['rsi'], row['mfi'], row['adx'], row['z_score'], 
                    row['trend_score'], row['linreg_angle'], row['fibo_pos'],
                    row['dist_to_res'], row['dist_to_sup'], row['skew'],
                    row['kurtosis'], row['entropy'], row['hurst']
                ]
                features = [0.0 if np.isnan(x) else x for x in features]
                state_vector.extend(features)
                
            except Exception:
                state_vector.extend([0.0] * 13)

        # Time Features
        hour = current_time.hour
        day = current_time.dayofweek
        state_vector.append(np.sin(2 * np.pi * hour / 24.0))
        state_vector.append(np.cos(2 * np.pi * hour / 24.0))
        state_vector.append(np.sin(2 * np.pi * day / 7.0))
        state_vector.append(np.cos(2 * np.pi * day / 7.0))
                
        return np.array(state_vector, dtype=np.float32)

    def _get_full_observation(self, base_obs=None):
        if base_obs is None:
            base_obs = self._get_observation()
            
        # Add Position State
        pos_state = 0.0
        if self.position_size > 0: pos_state = 1.0
        elif self.position_size < 0: pos_state = -1.0
        
        pnl_pct = 0.0
        if self.position_size != 0:
            current_price = self.primary_data.iloc[self.current_step]['close']
            if self.position_size > 0:
                pnl_pct = (current_price - self.avg_entry_price) / self.avg_entry_price
            else:
                pnl_pct = (self.avg_entry_price - current_price) / self.avg_entry_price
                
        extra_features = np.array([pos_state, pnl_pct], dtype=np.float32)
        return np.concatenate([base_obs, extra_features])

    def step(self, action):
        current_price = self.primary_data.iloc[self.current_step]['close']
        ema_200 = self.primary_data.iloc[self.current_step]['ema_200']
        
        reward = 0
        done = False
        realized_pnl = 0.0 # Track for metrics
        
        # --- Dynamic Trade Sizing (Normalization) ---
        # Goal: ~10-20 USD risk per trade or consistent exposure
        trade_size = 10000.0 # Default Forex (0.1 Lot)
        
        if "BTC" in self.symbol: trade_size = 0.1 # ~9k exposure
        elif "ETH" in self.symbol: trade_size = 1.0 # ~3k exposure
        elif "XAU" in self.symbol: trade_size = 1.0 # 1 oz (~2.6k exposure)
        elif "US30" in self.symbol: trade_size = 0.1 # ~4k exposure
        elif "GER40" in self.symbol: trade_size = 0.1 
        elif "US500" in self.symbol: trade_size = 0.1
        elif "US100" in self.symbol: trade_size = 0.1
        elif "JPY" in self.symbol: trade_size = 10000.0
        
        # --- 1. Check SL/TP Hits (Simulated) ---
        if self.position_size != 0:
            pnl_per_unit = (current_price - self.avg_entry_price) if self.position_size > 0 else (self.avg_entry_price - current_price)
            pnl_pips = pnl_per_unit / self.pip_size
            
            if pnl_pips <= -self.sl_pips:
                pnl = -self.sl_pips * self.pip_size * abs(self.position_size)
                commission = abs(self.position_size) * current_price * self.commission_rate
                realized_pnl = pnl - commission
                self.balance += realized_pnl
                self.position_size = 0
                self.avg_entry_price = 0
                reward -= 1.0 
            elif pnl_pips >= self.tp_pips:
                pnl = self.tp_pips * self.pip_size * abs(self.position_size)
                commission = abs(self.position_size) * current_price * self.commission_rate
                realized_pnl = pnl - commission
                self.balance += realized_pnl
                
                # TP Hit is always a quality trade
                reward += 2.0
                reward += 2.0 # Extra bonus for hitting TP (Quality)
                
                self.position_size = 0
                self.avg_entry_price = 0

        # --- 2. Execute Action (With Trend Filter) ---
        if self.position_size == 0 and action in [3, 4]:
            action = 0 
        
        # Trend Filter: Only Buy above EMA200, Sell below EMA200
        # If holding a position, we allow closing (Action 3, 4) or Pyramiding ONLY if trend is still valid
        if action == 1: # BUY
             if current_price < ema_200:
                 action = 0 # Force Hold if counter-trend
        elif action == 2: # SELL
             if current_price > ema_200:
                 action = 0 # Force Hold if counter-trend
            
        # Max Position Size (Risk Management)
        MAX_POS = 5 * trade_size
        
        if action == 1: # BUY / PYRAMID
            if self.position_size <= 0: 
                if self.position_size < 0: pass 
                
                cost = trade_size * current_price
                commission = cost * self.commission_rate
                self.balance -= commission
                
                total_val = (self.position_size * self.avg_entry_price) + (trade_size * current_price)
                self.position_size += trade_size
                
                if abs(self.position_size) < 1e-9: self.avg_entry_price = 0.0
                else: self.avg_entry_price = total_val / self.position_size
                
            else: # Pyramid
                # Check Max Size
                if self.position_size >= MAX_POS:
                    action = 0 # Force Hold
                else:
                    current_pnl = (current_price - self.avg_entry_price) / self.avg_entry_price
                    if current_pnl > 0.001: 
                        cost = trade_size * current_price
                        commission = cost * self.commission_rate
                        self.balance -= commission
                        
                        total_val = (self.position_size * self.avg_entry_price) + (trade_size * current_price)
                        self.position_size += trade_size
                        self.avg_entry_price = total_val / self.position_size
                        reward += 0.1 
                    else:
                        reward -= 0.1 
                    
        elif action == 2: # SELL / PYRAMID
            if self.position_size >= 0: 
                cost = trade_size * current_price
                commission = cost * self.commission_rate
                self.balance -= commission
                
                if self.position_size > 0: self.position_size = 0 
                
                total_val = (abs(self.position_size) * self.avg_entry_price) + (trade_size * current_price)
                self.position_size -= trade_size
                
                if abs(self.position_size) < 1e-9: self.avg_entry_price = 0.0
                else: self.avg_entry_price = total_val / abs(self.position_size)
                
            else: # Pyramid
                # Check Max Size
                if self.position_size <= -MAX_POS:
                    action = 0 # Force Hold
                else:
                    current_pnl = (self.avg_entry_price - current_price) / self.avg_entry_price
                    if current_pnl > 0.001:
                        cost = trade_size * current_price
                        commission = cost * self.commission_rate
                        self.balance -= commission
                        
                        total_val = (abs(self.position_size) * self.avg_entry_price) + (trade_size * current_price)
                        self.position_size -= trade_size
                        self.avg_entry_price = total_val / abs(self.position_size)
                        reward += 0.1
                    else:
                        reward -= 0.1

        elif action == 3: # SPLIT
            if abs(self.position_size) > 0:
                close_amt = abs(self.position_size) * 0.5
                pnl = 0
                trade_ret = 0
                if self.position_size > 0:
                    pnl = (current_price - self.avg_entry_price) * close_amt
                    trade_ret = (current_price - self.avg_entry_price) / self.avg_entry_price
                    self.position_size -= close_amt
                else:
                    pnl = (self.avg_entry_price - current_price) * close_amt
                    trade_ret = (self.avg_entry_price - current_price) / self.avg_entry_price
                    self.position_size += close_amt
                    
                commission = close_amt * current_price * self.commission_rate
                realized_pnl = pnl - commission
                self.balance += realized_pnl
                
                if pnl > 0: reward += 0.5
                
                # Quality Trade Bonus (> 0.10%)
                if trade_ret > 0.001:
                    reward += 2.0
                
        elif action == 4: # CLOSE ALL
            if abs(self.position_size) > 0:
                pnl = 0
                trade_ret = 0
                if self.position_size > 0:
                    pnl = (current_price - self.avg_entry_price) * abs(self.position_size)
                    trade_ret = (current_price - self.avg_entry_price) / self.avg_entry_price
                else:
                    pnl = (self.avg_entry_price - current_price) * abs(self.position_size)
                    trade_ret = (self.avg_entry_price - current_price) / self.avg_entry_price
                    
                commission = abs(self.position_size) * current_price * self.commission_rate
                realized_pnl = pnl - commission
                self.balance += realized_pnl
                self.position_size = 0
                self.avg_entry_price = 0
                
                # Quality Trade Bonus (> 0.10%)
                if trade_ret > 0.001:
                    reward += 2.0

                # Quality Trade Bonus (> 0.10%)
                if trade_ret > 0.001:
                    reward += 2.0

        # --- 3. Calculate Step Reward (GROWTH-ORIENTED) ---
        # Calculate Equity FIRST
        equity = self.balance
        if self.position_size != 0:
            unrealized = 0
            if self.position_size > 0:
                unrealized = (current_price - self.avg_entry_price) * self.position_size
            else:
                unrealized = (self.avg_entry_price - current_price) * abs(self.position_size)
            equity += unrealized
            
        # Track Peak Equity for Drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
            reward += 1.0  # Bonus for new peak!
            
        drawdown = (self.peak_equity - equity) / self.peak_equity
        
        # === NEW: GROWTH REWARDS ===
        # Reward equity growth over time (every 100 steps)
        if self.current_step % 100 == 0:
            growth_pct = (equity - self.last_equity_check) / self.last_equity_check
            if growth_pct > 0:
                reward += growth_pct * 50.0  # Big reward for growth!
            else:
                reward += growth_pct * 30.0  # Moderate penalty for decline
            self.last_equity_check = equity
        
        # === NEW: INACTIVITY PENALTY ===
        self.steps_since_last_trade += 1
        if action == 0:  # HOLD
            # Penalize prolonged inactivity
            if self.steps_since_last_trade > 50:
                reward -= 0.5  # Discourage excessive HOLDing
            elif self.steps_since_last_trade > 100:
                reward -= 2.0  # Strong penalty
        else:
            self.steps_since_last_trade = 0  # Reset on any action
        
        # If holding, small unrealized PnL reward
        if self.position_size != 0:
            reward += (unrealized / self.initial_balance) * 100.0 * 0.02  # Small holding reward
            
        # Drawdown Penalty (Relaxed)
        if drawdown > 0.05:  # > 5% DD (was 3%)
            reward -= 3.0  # Moderate Penalty (was 5.0)
            
        # Asymmetric PnL Reward with QUALITY BONUS (V2)
        if realized_pnl > 0:
            pnl_percent = (realized_pnl / self.initial_balance) * 100.0
            reward += pnl_percent
            
            # Quality Trade Bonus (V2: +5 points per 1% gain instead of +2)
            if pnl_percent >= 1.0:  # If trade is +1% or better
                reward += self.quality_trade_multiplier  # Default: +2, V2: +5
                
            self.total_profitable_trades += 1
        elif realized_pnl < 0:
            reward += (realized_pnl / self.initial_balance) * 150.0  # 1.5x Penalty

        # --- 4. Final Growth Bonus (V2) ---
        if done and self.enable_final_growth_bonus:
            final_growth = (equity - self.initial_balance) / self.initial_balance
            if final_growth >= self.final_growth_threshold:  # +10% or more
                growth_bonus = self.final_growth_bonus  # +100 points
                reward += growth_bonus
                print(f"🎯 FINAL GROWTH BONUS: {final_growth*100:.1f}% growth = +{growth_bonus} points!")
        
        # --- 5. Next Step ---
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
            
        self.equity_curve.append(equity)
        info = {
            'balance': self.balance, 
            'equity': equity, 
            'step': self.current_step,
            'realized_pnl': realized_pnl,
            'drawdown': drawdown
        }
        
        # Safe Observation Retrieval
        try:
            next_obs = self._get_full_observation()
        except IndexError:
            # If we stepped out of bounds, return the last valid observation or zeros
            # This can happen at the very last step
            self.current_step -= 1 # Revert for safety
            next_obs = self._get_full_observation()
            done = True
        
        return next_obs, reward, done, False, info
