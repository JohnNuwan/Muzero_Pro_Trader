import MetaTrader5 as mt5
import torch
import numpy as np
import time
import os
import sys
import pickle
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from MuZero.models.muzero_network import MuZeroNet
from MuZero.agents.muzero_mcts import MuZeroMCTS
from MuZero.environment.commission_trinity_env_v3 import CommissionTrinityEnvV3
from MuZero.config_v3 import MuZeroConfigV3
from MuZero.training.replay_buffer import GameHistory
from gemini_v19.utils.telegram_notifier import TelegramNotifier
from MuZero.utils.mtf_data_loader import MultiTimeframeLoader
from MuZero.utils.indicators import Indicators
import pandas as pd

# Live Trading Configuration
SYMBOLS = [
    "EURUSD", "XAUUSD", "BTCUSD",
    "US30.cash", "US500.cash", "USDJPY",
    "GBPUSD", "USDCAD", "USDCHF",
    "GER40.cash", "US100.cash"
]

MAGIC_NUMBER = 2024  # MuZero identifier

class MuZeroLiveTrader:
    def __init__(self, symbol="EURUSD", telegram_notifier=None):
        self.symbol = symbol
        self.config = MuZeroConfigV3()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.telegram = telegram_notifier
        
        # Load Best Model
        self.network = MuZeroNet(config=self.config).to(self.device)
        
        self.load_model()
        self.network.eval()
        
        # Environment (for observation building - NOT for state tracking)
        self.env = CommissionTrinityEnvV3(symbol=symbol, lookback=1000)
        
        # Data Loader for Live Updates
        self.loader = MultiTimeframeLoader()
        
        # Anti-Churning Cooldown
        self.last_trade_time = 0
        self.cooldown_seconds = 1800  # 30 minutes cooldown after opening a position
        
        # Anti-Laziness Mechanic
        self.steps_since_trade = 0
        self.inactivity_threshold = 60 # Force consideration after 60 mins
        
        # Live Experience Replay
        self.current_game = GameHistory()
        self.live_buffer_dir = "MuZero/results_v3/live_buffer"
        os.makedirs(self.live_buffer_dir, exist_ok=True)
        
        print(f"✅ MuZero Live Trader initialized for {symbol}")
        
    def load_model(self):
        model_path = "MuZero/weights_v3/best_model_v3.pth"  # Updated to V3 weights
        if os.path.exists(model_path):
            self.network.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
            print(f"✅ Loaded Best Model: {model_path}")
        else:
            print(f"⚠️ Model not found: {model_path}. Using random weights.")
    
    def get_current_observation(self):
        """Build observation from live MT5 data"""
        # 1. Update Environment with latest data
        self.update_environment_data()
        
        # 2. Get observation from updated environment
        # We don't call reset() because it resets balance/positions which we don't want
        # We just want the observation vector for the current step
        obs = self.env._get_full_observation()
        return obs

    def update_environment_data(self):
        """Fetch live data, process indicators, and inject into environment"""
        # Fetch raw data (M1, M5, H1, H4)
        # We need enough history for indicators (e.g. EMA 200)
        raw_data = self.loader.get_data(self.symbol, lookback_candles=2000)
        
        if raw_data is None:
            print(f"⚠️ Warning: Could not fetch live data for {self.symbol}")
            return

        # Process Indicators for all timeframes
        processed_data = {}
        for tf, df in raw_data.items():
            if not df.empty:
                processed_data[tf] = Indicators.add_all(df.copy())
        
        # Inject into Environment
        self.env.data = processed_data
        self.env.m1_data = processed_data["M1"]
        
        # Update valid indices and current step
        self.env.valid_indices = self.env.m1_data.index
        # Point to the last available candle
        self.env.current_step = len(self.env.m1_data) - 1
        
        # 3. Sync Position State from MT5
        positions = mt5.positions_get(symbol=self.symbol, magic=MAGIC_NUMBER)
        if positions and len(positions) > 0:
            pos = positions[0] # Assuming single position per symbol for now
            if pos.type == mt5.ORDER_TYPE_BUY:
                self.env.position_size = pos.volume
            else:
                self.env.position_size = -pos.volume
            self.env.avg_entry_price = pos.price_open
        else:
            self.env.position_size = 0.0
            self.env.avg_entry_price = 0.0
    
    def get_action(self, state):
        """Get action from MuZero MCTS"""
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            hidden_state = self.network.representation(state_tensor)
            
            # Get Root Value for recording
            root_value = self.network.prediction(hidden_state)[1].item()
            
            mcts = MuZeroMCTS(self.config, self.network)
            root = mcts.run(hidden_state, add_exploration_noise=False)
            
            visit_counts = [(action, child.visit_count) for action, child in root.children.items()]
            
            # Build Policy Vector for recording
            policy = np.zeros(self.config.action_space_size)
            total_visits = sum(vc for _, vc in visit_counts)
            if total_visits > 0:
                for action, count in visit_counts:
                    policy[action] = count / total_visits
            
            if not visit_counts:
                return 0, [], policy, root_value
                
            best_action = max(visit_counts, key=lambda x: x[1])[0]
            
            # --- V3.1 "KICK IN THE BUTT" Logic ---
            # If we haven't traded for a long time, we artificially suppress the HOLD action
            # to force the agent to look at the next best thing.
            if self.steps_since_trade > self.inactivity_threshold and best_action == 0:
                print(f"😴 {self.symbol}: Agent is lazy ({self.steps_since_trade} steps). Applying KICK...")
                
                # Find best action that is NOT 0
                other_actions = [x for x in visit_counts if x[0] != 0]
                if other_actions:
                    alt_action, alt_visits = max(other_actions, key=lambda x: x[1])
                    
                    # Only switch if the alternative has at least SOME visits (don't pick random garbage)
                    if alt_visits > total_visits * 0.005: # At least 0.5% consideration (LOWERED from 5% to force action)
                        print(f"🥾 KICK SUCCESS: Switching HOLD -> Action {alt_action} (Visits: {alt_visits})")
                        best_action = alt_action
                    else:
                        print("⚠️ KICK FAILED: No viable alternative (other actions < 0.5% prob). Stying in HOLD.")
            
            return best_action, visit_counts, policy, root_value
    
    def step(self):
        """Execute one trading step"""
        obs = self.get_current_observation()
        action, _, policy, root_value = self.get_action(obs)
        
        # Execute Trade
        if action != 0:
             self.steps_since_trade = 0 # Reset counter
        else:
             self.steps_since_trade += 1
             
        self.execute_trade(action)
        
        # Calculate Reward (Realized PnL approximation or 0 if holding)
        # Note: In live, we can't easily get the exact step reward without tracking equity diff
        # For now, we'll use a simplified reward based on price movement if in position
        reward = 0.0 # Placeholder, improved logic needed for exact reward tracking
        
        # Store in GameHistory
        self.current_game.store(obs, action, reward, policy, root_value, False)
        
        # Debug: Print Policy and Value and LIFE BAR
        life_pct = max(0, (self.inactivity_threshold - self.steps_since_trade) / self.inactivity_threshold)
        life_bars = int(life_pct * 10)
        life_str = "█" * life_bars + "░" * (10 - life_bars)
        
        # Color code: Green > 50%, Yellow > 20%, Red < 20%
        # (Just console text for now)
        
        # if self.symbol == "EURUSD": # COMMENTED OUT to show all symbols
        print(f"🔍 {self.symbol:<10} Val: {root_value:>5.2f} | Life: {life_str} ({int(life_pct*100)}%) | Pol: {['{:.2f}'.format(p) for p in policy]}")

        # Save periodically (e.g., every 100 steps)
        if len(self.current_game) % 100 == 0:
            self.save_game()
            
        return action

    def save_game(self):
        """Save current game history to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.live_buffer_dir}/live_game_{self.symbol}_{timestamp}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(self.current_game, f)
        print(f"💾 Saved Live Game Chunk to {filename}")
        # Reset game history to avoid huge files, or keep appending? 
        # Better to keep appending for context, but maybe chunk it.
        # For simplicity, we keep appending but save snapshots.
        
    def execute_trade(self, action):
        """Execute trade based on MuZero action"""
        if action == 0:  # HOLD
            pass
        elif action == 1:  # BUY
            positions = mt5.positions_get(symbol=self.symbol, magic=MAGIC_NUMBER)
            if positions is None or len(positions) == 0:
                # Check Cooldown
                if time.time() - self.last_trade_time > self.cooldown_seconds:
                    self._open_position("BUY")
                else:
                    remaining = int(self.cooldown_seconds - (time.time() - self.last_trade_time))
                    print(f"⏳ {self.symbol}: Cooldown active ({remaining}s remaining). Skipping BUY.")
            else:
                print(f"{self.symbol}: Already in position, skipping BUY")
        elif action == 2:  # SELL
            positions = mt5.positions_get(symbol=self.symbol, magic=MAGIC_NUMBER)
            if positions is None or len(positions) == 0:
                # Check Cooldown
                if time.time() - self.last_trade_time > self.cooldown_seconds:
                    self._open_position("SELL")
                else:
                    remaining = int(self.cooldown_seconds - (time.time() - self.last_trade_time))
                    print(f"⏳ {self.symbol}: Cooldown active ({remaining}s remaining). Skipping SELL.")
            else:
                print(f"{self.symbol}: Already in position, skipping SELL")
        elif action == 3:  # SPLIT
            self._close_partial(0.5)
        elif action == 4:  # CLOSE ALL
            self._close_all()
    
    def _open_position(self, order_type):
        """Open a new position with SL/TP (adapted from training environment)"""
        account_info = mt5.account_info()
        if account_info is None:
            return
        
        # Symbol-specific lot sizing
        base_lot = 0.10
        if "BTC" in self.symbol or "ETH" in self.symbol:
            base_lot = 0.01
        elif "XAU" in self.symbol:
            base_lot = 0.01
        elif "US30" in self.symbol or "GER40" in self.symbol or "US500" in self.symbol or "US100" in self.symbol:
            base_lot = 0.01
        elif "JPY" in self.symbol:
            base_lot = 0.10
        
        equity_scaling = account_info.equity / 10000.0
        lot = round(base_lot * equity_scaling, 2)
        lot = max(0.01, min(lot, 10.0))
        
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            return
        
        price = symbol_info.ask if order_type == "BUY" else symbol_info.bid
        
        # SL/TP matching training environment BUT adapted per symbol
        pip_size = 0.0001
        sl_pips = 10
        tp_pips = 100
        
        if "JPY" in self.symbol:
            pip_size = 0.01
            sl_pips = 10
            tp_pips = 100
        elif "BTC" in self.symbol or "ETH" in self.symbol:
            pip_size = 1.0
            sl_pips = 200   # 200 USD stop (realistic for crypto volatility)
            tp_pips = 1000  # 1000 USD target
        elif "XAU" in self.symbol:
            pip_size = 0.01
            sl_pips = 50    # $50 stop
            tp_pips = 200   # $200 target
        elif "US30" in self.symbol or "GER40" in self.symbol or "US100" in self.symbol or "US500" in self.symbol:
            pip_size = 1.0
            sl_pips = 100   # 100 points
            tp_pips = 500   # 500 points
        
        if order_type == "BUY":
            sl_price = price - (sl_pips * pip_size)
            tp_price = price + (tp_pips * pip_size)
        else:  # SELL
            sl_price = price + (sl_pips * pip_size)
            tp_price = price - (tp_pips * pip_size)
        
        # Dynamic Filling Mode Selection (Fix for Error 10044/10030)
        # Note: mt5.SYMBOL_FILLING_IOC might be missing in some versions, using int literals
        # 1 = FOK, 2 = IOC
        filling_type = mt5.ORDER_FILLING_FOK  # Default safe backup
        if symbol_info.filling_mode & 2: # 2 is SYMBOL_FILLING_IOC
            filling_type = mt5.ORDER_FILLING_IOC
        elif symbol_info.filling_mode & 1: # 1 is SYMBOL_FILLING_FOK
            filling_type = mt5.ORDER_FILLING_FOK
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": "MuZero",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }
        
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            msg = f"🚀 MuZero {self.symbol}: {order_type} {lot} @ {price:.5f} (SL:{sl_price:.5f}, TP:{tp_price:.5f})"
            print(msg)
            self.last_trade_time = time.time()  # Set Cooldown
            if self.telegram: 
                self.telegram._send_message(msg)
        else:
            error_msg = f"❌ {self.symbol}: Failed to open {order_type}"
            if result:
                error_msg += f" - Code: {result.retcode}"
            print(error_msg)
    
    def _close_partial(self, fraction=0.5):
        """Close a fraction of the position"""
        positions = mt5.positions_get(symbol=self.symbol, magic=MAGIC_NUMBER)
        if positions is None or len(positions) == 0:
            return
        
        for pos in positions:
            close_volume = round(pos.volume * fraction, 2)
            if close_volume < 0.01:
                continue
            
            order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(self.symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": close_volume,
                "type": order_type,
                "position": pos.ticket,
                "price": price,
                "magic": MAGIC_NUMBER,
                "comment": "MuZero SPLIT",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                msg = f"✂️ MuZero {self.symbol}: SPLIT {fraction*100}% @ {price:.5f}"
                print(msg)
                if self.telegram: 
                    self.telegram._send_message(msg)
    
    def _close_all(self):
        """Close all positions"""
        positions = mt5.positions_get(symbol=self.symbol, magic=MAGIC_NUMBER)
        if positions is None or len(positions) == 0:
            return
        
        for pos in positions:
            order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(self.symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": pos.volume,
                "type": order_type,
                "position": pos.ticket,
                "price": price,
                "magic": MAGIC_NUMBER,
                "comment": "MuZero CLOSE",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                msg = f"💰 MuZero {self.symbol}: CLOSED ALL @ {price:.5f}"
                print(msg)
                if self.telegram: 
                    self.telegram._send_message(msg)

def main():
    # Credentials from .env (Security Best Practice)
    load_dotenv() # Loads .env from root
    
    path = "C:\\Program Files\\MetaTrader 5\\terminal64.exe" 
    login = int(os.getenv("MT5_LOGIN"))
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    print(f"🔑 Connecting to {server} as {login}...")

    if not mt5.initialize(login=login, password=password, server=server):
        print(f"❌ MT5 initialization failed - Code: {mt5.last_error()}")
        return
    
    print("✅ MT5 Connected")
    account_info = mt5.account_info()
    if account_info:
        print(f"   Account: {account_info.login}, Balance: {account_info.balance}")
    
    telegram = TelegramNotifier()
    traders = {symbol: MuZeroLiveTrader(symbol, telegram_notifier=telegram) for symbol in SYMBOLS}
    
    print(f"\n🚀 Trading {len(SYMBOLS)} symbols (SL/TP adapted per asset)...")
    if telegram: 
        telegram._send_message(f"🤖 MuZero Live Trading STARTED with adapted SL/TP")
    
    try:
        step_count = 0
        while True:
            step_count += 1
            print(f"\n--- Step {step_count} ({datetime.now().strftime('%H:%M:%S')}) ---")
            
            for symbol, trader in traders.items():
                try:
                    action = trader.step()
                    if action != 0:
                        print(f"{symbol}: Action {action}")
                except Exception as e:
                    print(f"❌ {symbol} error: {e}")
            
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down MuZero Live Trading...")
        if telegram: 
            telegram._send_message("🛑 MuZero Live Trading STOPPED")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
