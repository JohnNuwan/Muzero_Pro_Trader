"""
MuZero V3 Environment - Pro Trader Edition

New Features:
- SLBE (Stop Loss Break Even) system
- Time-based drawdown penalties
- Enhanced SPLIT/CLOSE rewards
- Dynamic position capacity scaling
- No automatic SL/TP (agent must learn to exit)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from MuZero.environment.deep_trinity_env import DeepTrinityEnv

class CommissionTrinityEnvV3(DeepTrinityEnv):
    """
    V3 Environment: Advanced Risk Management
    
    Action Space (Discrete 5):
    0: Hold
    1: Buy (Open Long or Pyramid Long)
    2: Sell (Open Short or Pyramid Short)
    3: Split (Close 50% of position + Activate SLBE on remaining)
    4: Close All
    """
    def __init__(self, symbol="EURUSD", lookback=1000, start_index=0, end_index=None, timeframe="M1",
                 quality_trade_multiplier=5.0, enable_final_growth_bonus=False, 
                 final_growth_threshold=0.10, final_growth_bonus=0.0,
                 drawdown_penalty_rate=0.05, max_drawdown_penalty=3.0, loss_penalty_multiplier=1.5):
        super().__init__(symbol, lookback)
        
        # V3 Enhanced Rewards
        self.quality_trade_multiplier = quality_trade_multiplier
        self.enable_final_growth_bonus = enable_final_growth_bonus
        self.final_growth_threshold = final_growth_threshold
        self.final_growth_bonus = final_growth_bonus
        
        # V3 Penalties
        self.drawdown_penalty_rate = drawdown_penalty_rate
        self.max_drawdown_penalty = max_drawdown_penalty
        self.loss_penalty_multiplier = loss_penalty_multiplier
        
        # Data Splitting
        self.start_index = start_index
        self.timeframe = timeframe
        
        # Select Primary Data Source
        if timeframe == "M5":
            self.primary_data = self.data["M5"]
        elif timeframe == "H1":
            self.primary_data = self.data["H1"]
        else:
            self.primary_data = self.m1_data
            
        self.end_index = end_index if end_index is not None else len(self.primary_data) - 1
        
        # Action Space (Discrete 5)
        self.action_space = spaces.Discrete(5)
        
        # Observation Space: Base + 3 (Position Size, PnL %, SLBE Active)
        base_shape = self.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(base_shape + 3,), dtype=np.float32
        )
        
        # Trading State Extensions
        self.position_size = 0.0 
        self.avg_entry_price = 0.0
        
        # V3 NEW: SLBE System
        self.slbe_active = False
        self.slbe_price = 0.0
        
        # V3 NEW: Time Tracking
        self.steps_in_drawdown = 0
        self.steps_since_last_trade = 0
        
        # Config
        self.commission_rate = 0.00005
        self.pip_size = 0.0001 
        
        # Adjust for JPY and Indices
        if "JPY" in symbol:
            self.pip_size = 0.01
            self.initial_balance = 1000000.0
        elif "US30" in symbol or "GER40" in symbol or "US100" in symbol:
             self.pip_size = 1.0
             self.initial_balance = 100000.0
        elif "BTC" in symbol:
             self.pip_size = 1.0
             self.initial_balance = 100000.0
        elif "XAU" in symbol:
             self.pip_size = 0.01
             self.initial_balance = 10000.0
        else:
             self.initial_balance = 10000.0

        # Growth Tracking
        self.last_equity_check = self.initial_balance
        self.total_profitable_trades = 0
        self.total_trades = 0
              
        self.balance = self.initial_balance
        self.peak_equity = self.initial_balance
        
        # V3: Dynamic Position Capacity
        self.base_max_positions = 2
        self.secured_positions_count = 0  # Positions with SLBE

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        self.current_step = max(100, self.start_index)
        self.max_steps = min(self.current_step + 1000, self.end_index)
        
        self.position_size = 0.0
        self.avg_entry_price = 0.0
        self.balance = self.initial_balance 
        self.peak_equity = self.initial_balance
        self.equity_curve = [self.initial_balance]
        
        # V3 Resets
        self.slbe_active = False
        self.slbe_price = 0.0
        self.steps_in_drawdown = 0
        self.last_equity_check = self.initial_balance
        self.steps_since_last_trade = 0
        self.total_profitable_trades = 0
        self.total_trades = 0
        self.secured_positions_count = 0
        self.split_count = 0  # V3 Fix: Prevent SPLIT spamming
        
        return self._get_full_observation(), {}
        
    def _get_full_observation(self):
        base_obs = super()._get_observation()
        
        # Position State
        pos_state = 0.0
        if self.position_size > 0: pos_state = 1.0
        elif self.position_size < 0: pos_state = -1.0
        
        # Unrealized PnL %
        pnl_pct = 0.0
        if self.position_size != 0:
            current_price = self.primary_data.iloc[self.current_step]['close']
            if self.position_size > 0:
                pnl_pct = (current_price - self.avg_entry_price) / self.avg_entry_price
            else:
                pnl_pct = (self.avg_entry_price - current_price) / self.avg_entry_price
        
        # V3 NEW: SLBE State
        slbe_state = 1.0 if self.slbe_active else 0.0
        
        # V3 NEW: Time & Volatility Features
        current_row = self.primary_data.iloc[self.current_step]
        
        # 1. Hour of Day (0-1)
        hour_feat = 0.5
        day_feat = 0.5
        if 'time' in current_row:
            try:
                dt = pd.to_datetime(current_row['time'], unit='s')
                hour_feat = dt.hour / 23.0
                day_feat = dt.dayofweek / 6.0
            except:
                pass
                
        # 2. Volatility (High-Low range relative to Close)
        # Normalized: 1% range = 1.0 (capped at 1.0)
        volatility = 0.0
        try:
            high = current_row['high']
            low = current_row['low']
            close = current_row['close']
            if close > 0:
                vol_raw = (high - low) / close
                volatility = min(vol_raw * 100.0, 1.0)
        except:
            pass
            
        extra_features = np.array([pos_state, pnl_pct, slbe_state, hour_feat, day_feat, volatility], dtype=np.float32)
        return np.concatenate([base_obs, extra_features])

    def step(self, action):
        current_price = self.primary_data.iloc[self.current_step]['close']
        ema_200 = self.primary_data.iloc[self.current_step]['ema_200']
        
        reward = 0
        done = False
        realized_pnl = 0.0
        
        # Dynamic Trade Sizing
        trade_size = 10000.0  # Default Forex (0.1 Lot)
        
        if "BTC" in self.symbol: trade_size = 0.1
        elif "ETH" in self.symbol: trade_size = 1.0
        elif "XAU" in self.symbol: trade_size = 1.0
        elif "US30" in self.symbol: trade_size = 0.1
        elif "GER40" in self.symbol: trade_size = 0.1
        elif "US500" in self.symbol: trade_size = 0.1
        elif "US100" in self.symbol: trade_size = 0.1
        elif "JPY" in self.symbol: trade_size = 10000.0
        
        # V3 NEW: Check SLBE Hit
        if self.slbe_active and self.position_size != 0:
            hit_slbe = False
            if self.position_size > 0 and current_price <= self.slbe_price:
                hit_slbe = True
            elif self.position_size < 0 and current_price >= self.slbe_price:
                hit_slbe = True
            
            if hit_slbe:
                # Close position at break-even
                commission = abs(self.position_size) * current_price * self.commission_rate
                self.balance -= commission
                self.position_size = 0
                self.avg_entry_price = 0
                self.slbe_active = False
                self.slbe_price = 0
                reward += 1.0  # Small bonus for protecting capital
                
        # V3 NEW: Activate SLBE if profit > 0.5%
        if not self.slbe_active and self.position_size != 0:
            unrealized_pnl_pct = 0
            if self.position_size > 0:
                unrealized_pnl_pct = (current_price - self.avg_entry_price) / self.avg_entry_price
            else:
                unrealized_pnl_pct = (self.avg_entry_price - current_price) / self.avg_entry_price
            
            if unrealized_pnl_pct >= 0.005:  # +0.5%
                self.slbe_active = True
                self.slbe_price = self.avg_entry_price
                self.secured_positions_count += 1
                reward += 3.0  # BIG BONUS for activating SLBE
                
        # Force HOLD if trying to SPLIT/CLOSE with no position
        if self.position_size == 0 and action in [3, 4]:
            action = 0
        
        # Trend Filter
        if action == 1:  # BUY
             if current_price < ema_200:
                 action = 0
        elif action == 2:  # SELL
             if current_price > ema_200:
                 action = 0
            
        # V3: Dynamic Max Position (increases with secured positions)
        MAX_POS = (self.base_max_positions + self.secured_positions_count) * trade_size
        
        if action == 1:  # BUY / PYRAMID
            if self.position_size <= 0: 
                cost = trade_size * current_price
                commission = cost * self.commission_rate
                self.balance -= commission
                
                total_val = (self.position_size * self.avg_entry_price) + (trade_size * current_price)
                self.position_size += trade_size
                
                if abs(self.position_size) < 1e-9: self.avg_entry_price = 0.0
                else: self.avg_entry_price = total_val / self.position_size
                
                self.split_count = 0  # Reset split count on new position
                
            else:  # Pyramid
                if self.position_size >= MAX_POS:
                    action = 0
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
                    
        elif action == 2:  # SELL / PYRAMID
            if self.position_size >= 0: 
                cost = trade_size * current_price
                commission = cost * self.commission_rate
                self.balance -= commission
                
                if self.position_size > 0: self.position_size = 0
                
                total_val = (abs(self.position_size) * self.avg_entry_price) + (trade_size * current_price)
                self.position_size -= trade_size
                
                if abs(self.position_size) < 1e-9: self.avg_entry_price = 0.0
                else: self.avg_entry_price = total_val / abs(self.position_size)
                
                self.split_count = 0  # Reset split count on new position
                
            else:  # Pyramid
                if self.position_size <= -MAX_POS:
                    action = 0
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

        elif action == 3:  # V3 ENHANCED SPLIT
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
                self.total_trades += 1
                
                # V3 NEW: Enhanced SPLIT Rewards (Limited to 3 times to prevent spamming)
                if self.split_count < 3:
                    if trade_ret > 0.01:  # +1% or better
                        reward += self.quality_trade_multiplier  # +5 points
                        self.total_profitable_trades += 1
                        self.split_count += 1
                    elif pnl > 0:
                        reward += 1.0
                        self.total_profitable_trades += 1
                        self.split_count += 1
                
                # Auto-activate SLBE on remaining 50%
                if not self.slbe_active:
                    self.slbe_active = True
                    self.slbe_price = self.avg_entry_price
                    reward += 2.0  # Bonus for securing
                
        elif action == 4:  # V3 ENHANCED CLOSE ALL
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
                self.slbe_active = False
                self.slbe_price = 0
                self.total_trades += 1
                
                # V3 NEW: Enhanced CLOSE Rewards
                if trade_ret > 0.02:  # +2% or better
                    reward += self.quality_trade_multiplier * 1.5  # +7.5 points
                    self.total_profitable_trades += 1
                elif trade_ret > 0.01:  # +1%
                    reward += self.quality_trade_multiplier  # +5 points
                    self.total_profitable_trades += 1
                elif pnl > 0:
                    reward += 1.0
                    self.total_profitable_trades += 1
                else:
                    # Small penalty for closing in loss (but better than holding forever)
                    reward += (pnl / self.initial_balance) * 100.0  # Asymmetric

        # Calculate Metrics
        unrealized = 0.0
        if self.position_size != 0:
            if self.position_size > 0:
                unrealized = (current_price - self.avg_entry_price) * abs(self.position_size)
            else:
                unrealized = (self.avg_entry_price - current_price) * abs(self.position_size)
        
        equity = self.balance + unrealized
        
        if equity > self.peak_equity:
            self.peak_equity = equity
            
        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        
        # V3 NEW: Time-based Drawdown Penalty
        if unrealized < 0:
            self.steps_in_drawdown += 1
            # Progressive penalty (worse over time)
            time_penalty = -self.drawdown_penalty_rate * (self.steps_in_drawdown / 20)
            reward += time_penalty
        else:
            self.steps_in_drawdown = 0  # Reset if back in profit
        
        # Inactivity Penalty (Force Action)
        self.steps_since_last_trade += 1
        if action in [1, 2, 3, 4]:
            self.steps_since_last_trade = 0
        elif self.steps_since_last_trade > 100:
            reward -= 1.0  # -1.0 penalty for being lazy (every step after 100)
        
        # Holding reward
        if self.position_size != 0:
            reward += (unrealized / self.initial_balance) * 100.0 * 0.02
            
        # Drawdown Penalty
        if drawdown > 0.05:
            reward -= self.max_drawdown_penalty
            
        # Asymmetric PnL Reward with QUALITY BONUS (V3)
        if realized_pnl > 0:
            pnl_percent = (realized_pnl / self.initial_balance) * 100.0
            reward += pnl_percent
            
            # Quality Trade Bonus (already added in SPLIT/CLOSE logic above)
                
        elif realized_pnl < 0:
            reward += (realized_pnl / self.initial_balance) * 100.0 * self.loss_penalty_multiplier

        # V3 NEW: Final Growth Bonus
        if done and self.enable_final_growth_bonus:
            final_growth = (equity - self.initial_balance) / self.initial_balance
            if final_growth >= self.final_growth_threshold:
                growth_bonus = self.final_growth_bonus
                reward += growth_bonus
                print(f"🎯 FINAL GROWTH BONUS: {final_growth*100:.1f}% growth = +{growth_bonus} points!")
        
        # Next Step
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
            
        self.equity_curve.append(equity)
        info = {
            'balance': self.balance, 
            'equity': equity, 
            'step': self.current_step,
            'realized_pnl': realized_pnl,
            'drawdown': drawdown,
            'slbe_active': self.slbe_active,
            'steps_in_drawdown': self.steps_in_drawdown
        }
        
        # Safe Observation Retrieval
        try:
            next_obs = self._get_full_observation()
        except IndexError:
            self.current_step -= 1
            next_obs = self._get_full_observation()
            done = True
        
        return next_obs, reward, done, False, info
