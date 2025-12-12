import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import os
import sys
from datetime import datetime
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from stable_baselines3 import PPO

# Rich Imports
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.text import Text
from rich import box

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from gemini_v15.utils.mtf_data_loader import MultiTimeframeLoader
from gemini_v15.utils.indicators import Indicators
from gemini_v15.environment.commission_trinity_env import CommissionTrinityEnv
from gemini_v15.agents.mcts_agent import MCTSAgent

# Configuration
SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", 
    "BTCUSD", "ETHUSD", 
    "XAUUSD", 
    "US30.cash", "GER40.cash", "US500.cash", "US100.cash"
]
TIMEFRAMES = ["M1", "M5", "M15", "H1", "H4", "D1"]
FEATURES = [
    'rsi', 'mfi', 'adx', 'z_score', 
    'trend_score', 'linreg_angle', 'fibo_pos',
    'dist_to_res', 'dist_to_sup', 'skew',
    'kurtosis', 'entropy', 'hurst'
]

# Custom SL/TP Settings (in pips)
SYMBOL_SETTINGS = {
    "XAU": (800, 1600),   # Gold: élargi pour plus de respiration (était 500)
    "BTC": (10000, 20000), # Bitcoin: très large (était 5000)
    "ETH": (5000, 10000),  # Ethereum: élargi aussi
    "US30": (1500, 3000), # Dow Jones: élargi
    "US100": (1500, 3000),# Nasdaq: élargi
    "US500": (800, 1600), # S&P500: élargi
    "GER40": (800, 1600), # DAX: élargi
}

DEFAULT_SL_PIPS = 300  # Élargi de 200 à 300
DEFAULT_TP_PIPS = 600  # Élargi de 400 à 600

MAGIC_NUMBER = 161616
MODELS_DIR = os.path.join(current_dir, "models")
COOLDOWN_SECONDS = 300 

# Champion Map (V16 Results)
CHAMPIONS = {
    "BTCUSD": "ppo_v16_BTCUSD",
    "US30.cash": "ppo_v16_US30.cash", # Note: File is ppo_v16_US30.cash (no .zip in map, load handles it)
    "GER40.cash": "ppo_v16_GER40.cash",
    "US100.cash": "ppo_v16_US100.cash",
    "XAUUSD": "ppo_v17_XAUUSD_M5"
}

class GeminiV16Orchestrator:
    def __init__(self):
        if not mt5.initialize():
            print("❌ MT5 Init Failed in Orchestrator Init")
            return

        self.models = {}
        self.mcts_agents = {}
        self.loader = MultiTimeframeLoader()
        self.ignored_tickets = set()
        self.console = Console()
        self.logs = []
        self.market_data = {s: {"price": 0.0, "pos": "FLAT", "pnl": 0.0, "action": "WAIT"} for s in SYMBOLS}
        self.account_info = {"balance": 0.0, "equity": 0.0, "margin": 0.0}
        self.last_trade_time = {} 
        
        # Load Champions
        for symbol in SYMBOLS:
            model_name = CHAMPIONS.get(symbol)
            if model_name:
                # Check for .zip extension in filename or add it
                path = os.path.join(project_root, "gemini_v15", "models", f"{model_name}.zip")
                if not os.path.exists(path):
                     # Try without .zip if it was already included or file has no extension (rare for SB3)
                     path = os.path.join(project_root, "gemini_v15", "models", f"{model_name}")
                
                if os.path.exists(path):
                    try:
                        self.models[symbol] = PPO.load(path)
                        self.console.print(f"[green]Loaded Champion for {symbol}: {model_name}[/green]")
                        
                        # Init MCTS
                        try:
                            # Use M5 for XAU, M1 for others (based on V17 plan)
                            tf = "M5" if "XAU" in symbol else "M1"
                            env = CommissionTrinityEnv(symbol, lookback=2000, timeframe=tf)
                            self.mcts_agents[symbol] = MCTSAgent(self.models[symbol], env, simulations=50)
                            self.console.print(f"[cyan]🧠 MCTS Initialized for {symbol} ({tf})[/cyan]")
                        except Exception as e:
                            self.console.print(f"[red]Failed to init MCTS for {symbol}: {e}[/red]")
                    except Exception as e:
                        self.console.print(f"[red]Failed to load model: {e}[/red]")
                else:
                    self.console.print(f"[red]Model not found for {symbol}: {path}[/red]")
            else:
                 self.console.print(f"[yellow]No Champion defined for {symbol}[/yellow]")

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        if len(self.logs) > 50: self.logs.pop(0)



    def get_symbol_settings(self, symbol):
        # Find matching setting or return default
        for key, (sl, tp) in SYMBOL_SETTINGS.items():
            if key in symbol.upper():
                return sl, tp
        return DEFAULT_SL_PIPS, DEFAULT_TP_PIPS

    def get_v16_observation(self, symbol):
        # 1. Fetch Data
        raw_data = self.loader.get_data(symbol, lookback_candles=1000)
        if raw_data is None: return None
        
        # 2. Process Indicators
        data = {}
        for tf, df in raw_data.items():
            data[tf] = Indicators.add_all(df.copy())
            
        # 3. Construct Base Vector
        m1_data = data["M1"]
        if m1_data.empty: return None
        
        current_time = m1_data.index[-1]
        state_vector = []
        
        for tf in TIMEFRAMES:
            df = data.get(tf)
            if df is None or df.empty:
                state_vector.extend([0.0] * 13)
                continue
                
            try:
                idx = df.index.searchsorted(current_time, side='right') - 1
                if idx < 0: idx = 0
                row = df.iloc[idx]
                
                vals = [row[f] for f in FEATURES]
                vals = [0.0 if np.isnan(x) else x for x in vals]
                state_vector.extend(vals)
            except:
                state_vector.extend([0.0] * 13)
                
        # 4. Time Features
        hour = current_time.hour
        day = current_time.dayofweek
        state_vector.append(np.sin(2 * np.pi * hour / 24.0))
        state_vector.append(np.cos(2 * np.pi * hour / 24.0))
        state_vector.append(np.sin(2 * np.pi * day / 7.0))
        state_vector.append(np.cos(2 * np.pi * day / 7.0))
        
        # 5. V16 Extra Features (Pos State, PnL %)
        pos_state = 0.0
        pnl_pct = 0.0
        
        # Get positions ONLY for this Magic Number
        positions = mt5.positions_get(symbol=symbol)
        active_positions = []
        if positions:
            active_positions = [p for p in positions if p.magic == MAGIC_NUMBER and p.ticket not in self.ignored_tickets]
            
            if active_positions:
                net_vol = sum([p.volume if p.type == mt5.ORDER_TYPE_BUY else -p.volume for p in active_positions])
                
                if net_vol > 0: pos_state = 1.0
                elif net_vol < 0: pos_state = -1.0
                
                tick = mt5.symbol_info_tick(symbol)
                current_price = tick.bid
                
                total_cost = 0
                total_v = 0
                for p in active_positions:
                    total_cost += p.price_open * p.volume
                    total_v += p.volume
                
                if total_v > 0:
                    avg_entry = total_cost / total_v
                    if net_vol > 0:
                        pnl_pct = (current_price - avg_entry) / avg_entry
                    elif net_vol < 0:
                        pnl_pct = (avg_entry - current_price) / avg_entry

        state_vector.append(pos_state)
        state_vector.append(pnl_pct)
        
        # Update Market Data for UI
        tick = mt5.symbol_info_tick(symbol)
        price = tick.bid if tick else 0.0
        
        pos_str = "FLAT"
        if pos_state == 1.0: pos_str = "LONG"
        elif pos_state == -1.0: pos_str = "SHORT"
        
        current_pnl = sum([p.profit for p in active_positions]) if active_positions else 0.0
        
        self.market_data[symbol] = {
            "price": price,
            "pos": pos_str,
            "pnl": current_pnl,
            "action": self.market_data[symbol]["action"] # Keep last action
        }
        
        return np.array(state_vector, dtype=np.float32)

    def close_position(self, position, volume_ratio=1.0):
        tick = mt5.symbol_info_tick(position.symbol)
        if not tick:
            self.log(f"⚠️ Close Failed {position.symbol}: No Tick Data")
            return

        # Determine price (Close Buy at Bid, Close Sell at Ask)
        price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask
        
        # Calculate volume to close
        volume = position.volume * volume_ratio
        
        # Round to 2 decimal places
        volume = round(volume, 2)
        
        # Check minimum volume
        if volume < 0.01:
            self.log(f"⚠️ Cannot split {position.symbol}: Volume {volume} too small (Orig: {position.volume})")
            return 

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": position.ticket,
            "price": price,
            "magic": MAGIC_NUMBER,
            "comment": "Gemini V16 Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            self.log(f"❌ Close Failed {position.symbol}: OrderSend returned None")
            return
            
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.log(f"❌ Close Failed {position.symbol}: {result.comment} ({result.retcode})")
        else:
            self.log(f"✂️ CLOSED {position.symbol} ({volume}) @ {price}")

    def execute_trade(self, symbol, action):
        # Action Map: 0:Hold, 1:Buy, 2:Sell, 3:Split, 4:CloseAll
        if action == 0: 
            self.market_data[symbol]["action"] = "HOLD"
            return
        
        # Cooldown Check
        last_time = self.last_trade_time.get(symbol, 0)
        if time.time() - last_time < COOLDOWN_SECONDS:
            return

        tick = mt5.symbol_info_tick(symbol)
        if tick is None: return
        
        # Filter positions by Magic Number
        positions = mt5.positions_get(symbol=symbol)
        active_positions = [p for p in positions if p.magic == MAGIC_NUMBER and p.ticket not in self.ignored_tickets] if positions else []
        
        # --- CLOSE ALL (désactivé, mapped to SPLIT) ---
        if action == 4:
            # Au lieu de tout fermer, on fait un SPLIT (50%)
            if not active_positions: return
            self.log(f"✂️ {symbol}: SPLIT (Ex-CLOSE ALL mapped to 50% close)")
            self.market_data[symbol]["action"] = "SPLIT"
            for p in active_positions:
                self.close_position(p, volume_ratio=0.5)
            self.last_trade_time[symbol] = time.time()
            return

        # --- SPLIT (Close 50%) ---
        if action == 3:
            if not active_positions: return
            self.log(f"✂️ {symbol}: Splitting Positions (50%)...")
            self.market_data[symbol]["action"] = "SPLIT"
            for p in active_positions:
                self.close_position(p, volume_ratio=0.5)
            self.last_trade_time[symbol] = time.time()
            return

        # --- BUY / SELL ---
        has_long = any(p.type == mt5.ORDER_TYPE_BUY for p in active_positions)
        has_short = any(p.type == mt5.ORDER_TYPE_SELL for p in active_positions)
        
        # Determine Lot Size Dynamically
        lot = self.calculate_lot_size(symbol)

        # BUY
        if action == 1 and not has_long:
            if has_short:
                self.log(f"🔄 {symbol}: Reversing Short -> Long")
                for p in active_positions:
                    if p.type == mt5.ORDER_TYPE_SELL: self.close_position(p)
            
            sl_pips, tp_pips = self.get_symbol_settings(symbol)
            price = tick.ask
            
            # Calculate SL/TP
            point = mt5.symbol_info(symbol).point
            sl = price - sl_pips * point
            tp = price + tp_pips * point
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot,
                "type": mt5.ORDER_TYPE_BUY,
                "price": price,
                "sl": sl,
                "tp": tp,
                "magic": MAGIC_NUMBER,
                "comment": "Gemini V16",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.log(f"❌ Buy Failed {symbol}: {result.comment}")
            else:
                self.log(f"🔵 BUY {symbol} @ {price} | Lot: {lot} | SL: {sl_pips} | TP: {tp_pips}")
                self.market_data[symbol]["action"] = "BUY"
                self.last_trade_time[symbol] = time.time()

        # SELL
        elif action == 2 and not has_short:
            if has_long:
                self.log(f"🔄 {symbol}: Reversing Long -> Short")
                for p in active_positions:
                    if p.type == mt5.ORDER_TYPE_BUY: self.close_position(p)

            sl_pips, tp_pips = self.get_symbol_settings(symbol)
            price = tick.bid
            
            # Calculate SL/TP
            point = mt5.symbol_info(symbol).point
            sl = price + sl_pips * point
            tp = price - tp_pips * point
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot,
                "type": mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": sl,
                "tp": tp,
                "magic": MAGIC_NUMBER,
                "comment": "Gemini V16",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.log(f"❌ Sell Failed {symbol}: {result.comment}")
            else:
                self.log(f"🔴 SELL {symbol} @ {price} | Lot: {lot} | SL: {sl_pips} | TP: {tp_pips}")
                self.market_data[symbol]["action"] = "SELL"
                self.last_trade_time[symbol] = time.time()

    def calculate_lot_size(self, symbol):
        """
        Calculate dynamic lot size based on equity and asset class.
        Conservative scaling (divisé par 2 pour réduire le risque):
        - Forex/Indices: ~0.5 lot per $100k equity (0.05 per $10k)
        - Crypto: ~0.05 lot per $100k equity (0.005 per $10k)
        """
        equity = self.account_info.get("equity", 1000.0)
        
        # Default scaling (Forex/Indices) - divisé par 2
        scaling_factor = 200000.0  # Était 100000, maintenant 200000
        
        if "BTC" in symbol or "ETH" in symbol:
            scaling_factor = 2000000.0 # Était 1M, maintenant 2M
        elif "XAU" in symbol:
            scaling_factor = 2000000.0#200000.0 # Gold: divisé par 2 aussi
            
        raw_lot = equity / scaling_factor
        
        # Round to 2 decimal places
        lot = round(raw_lot, 2)
        
        # Ensure minimum lot size (usually 0.01)
        lot = max(0.01, lot)
        
        return lot
        
    def process_symbol(self, symbol):
        # 0. Ensure Symbol is Selected
        if not mt5.symbol_select(symbol, True):
            # Try lowercase if uppercase fails
            if mt5.symbol_select(symbol.lower(), True):
                symbol = symbol.lower()
            else:
                self.market_data[symbol]["idea"] = "ERR: SYM"
                return

        # 1. Update Price & Global Position Immediately (UI)
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            self.market_data[symbol]["price"] = tick.bid
        else:
            self.market_data[symbol]["idea"] = "ERR: TICK"
            return
            
        # Global Positions (View Only)
        all_positions = mt5.positions_get(symbol=symbol)
        if all_positions:
            net_vol = sum([p.volume if p.type == mt5.ORDER_TYPE_BUY else -p.volume for p in all_positions])
            pos_str = "FLAT"
            if net_vol > 0: pos_str = f"LONG ({net_vol:.2f})"
            elif net_vol < 0: pos_str = f"SHORT ({abs(net_vol):.2f})"
            
            # Calculate Global PnL
            global_pnl = sum([p.profit for p in all_positions])
            
            self.market_data[symbol]["pos"] = pos_str
            self.market_data[symbol]["pnl"] = global_pnl
        else:
            self.market_data[symbol]["pos"] = "FLAT"
            self.market_data[symbol]["pnl"] = 0.0
        
        # 2. Get Observation (V16 Only)
        try:
            obs = self.get_v16_observation(symbol)
        except Exception as e:
            self.market_data[symbol]["idea"] = "ERR: DATA"
            return

        if obs is None: 
            self.market_data[symbol]["idea"] = "NO DATA"
            return
        
        # Predict with Probabilities
        # SB3 PPO Policy
        # Note: Model expects specific input shape. 
        # If symbol case changed, does it affect model? No, model is agnostic to symbol name, just needs data.
        # But we need to make sure `self.models` has the key. 
        # `self.models` uses the original uppercase keys from SYMBOLS list.
        # So we use the original `symbol` (from argument) for model lookup, 
        # but the corrected `symbol` (local var) for MT5 calls.
        
        # Wait, if I change local `symbol` to lowercase, I must use it for MT5, 
        # but keep original for `self.models`.
        # Let's handle this carefully.
        
        # Actually, `process_symbol` receives `symbol` from `SYMBOLS` list (Uppercase).
        # If MT5 needs lowercase, we use a separate variable `mt5_symbol`.
        
        mt5_symbol = symbol
        if not mt5.symbol_select(symbol, True):
            if mt5.symbol_select(symbol.lower(), True):
                mt5_symbol = symbol.lower()
            else:
                self.market_data[symbol]["idea"] = "ERR: SYM"
                return
        
        # Update Price using mt5_symbol
        tick = mt5.symbol_info_tick(mt5_symbol)
        if tick:
            self.market_data[symbol]["price"] = tick.bid
            
        # Global Positions using mt5_symbol
        all_positions = mt5.positions_get(symbol=mt5_symbol)
        if all_positions:
            net_vol = sum([p.volume if p.type == mt5.ORDER_TYPE_BUY else -p.volume for p in all_positions])
            pos_str = "FLAT"
            if net_vol > 0: pos_str = f"LONG ({net_vol:.2f})"
            elif net_vol < 0: pos_str = f"SHORT ({abs(net_vol):.2f})"
            global_pnl = sum([p.profit for p in all_positions])
            self.market_data[symbol]["pos"] = pos_str
            self.market_data[symbol]["pnl"] = global_pnl
        else:
            self.market_data[symbol]["pos"] = "FLAT"
            self.market_data[symbol]["pnl"] = 0.0

        # Get Obs (Pass mt5_symbol to loader?)
        # Loader likely uses `mt5.copy_rates_from_pos(symbol, ...)`
        # So we need to pass `mt5_symbol` to `get_v16_observation`.
        # But `get_v16_observation` calls `self.loader.get_data(symbol)`.
        # I need to update `get_v16_observation` to accept `mt5_symbol` or handle it.
        # For now, let's assume I can pass `mt5_symbol` to `get_v16_observation`.
        
        try:
            obs = self.get_v16_observation(mt5_symbol) 
        except Exception as e:
            self.log(f"⚠️ Failed to get obs for {symbol}: {e}")
            return

        # 4. Predict (MCTS or PPO)
        probs = {}
        action = 0
        
        agent = self.mcts_agents.get(symbol)
        if agent:
            try:
                # Sync Position
                net_vol = 0.0
                avg_entry = 0.0
                if all_positions:
                     net_vol = sum([p.volume if p.type == mt5.ORDER_TYPE_BUY else -p.volume for p in all_positions])
                     total_cost = sum([p.price_open * p.volume for p in all_positions])
                     total_vol = sum([p.volume for p in all_positions])
                     if total_vol > 0: avg_entry = total_cost / total_vol
                
                agent.env.position_size = net_vol
                agent.env.avg_entry_price = avg_entry
                agent.env.balance = self.account_info["balance"]
                
                # 2. Run MCTS
                # self.log(f"🧠 {symbol}: MCTS Thinking...")
                action = agent.search(obs)
                probs[action] = 1.0 # MCTS is deterministic in output
                
            except Exception as e:
                self.log(f"⚠️ MCTS Error {symbol}: {e}")
                # Fallback to PPO
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
        else:
            # Standard PPO
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            probs[action] = 1.0

        # Execute (Pass mt5_symbol)
        self.execute_trade(mt5_symbol, action)

        # 1. Update Price & Global Position Immediately (UI)
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            self.market_data[symbol]["price"] = tick.bid
        else:
            self.market_data[symbol]["idea"] = "ERR: TICK"
            return
            
        # Global Positions (View Only)
        all_positions = mt5.positions_get(symbol=symbol)
        if all_positions:
            net_vol = sum([p.volume if p.type == mt5.ORDER_TYPE_BUY else -p.volume for p in all_positions])
            pos_str = "FLAT"
            if net_vol > 0: pos_str = f"LONG ({net_vol:.2f})"
            elif net_vol < 0: pos_str = f"SHORT ({abs(net_vol):.2f})"
            
            # Calculate Global PnL
            global_pnl = sum([p.profit for p in all_positions])
            
            self.market_data[symbol]["pos"] = pos_str
            self.market_data[symbol]["pnl"] = global_pnl
        else:
            self.market_data[symbol]["pos"] = "FLAT"
            self.market_data[symbol]["pnl"] = 0.0
        
        # 2. Get Observation (V16 Only)
        try:
            obs = self.get_v16_observation(symbol)
        except Exception as e:
            self.market_data[symbol]["idea"] = "ERR: DATA"
            return

        if obs is None: 
            self.market_data[symbol]["idea"] = "NO DATA"
            return
        
        # Predict with Probabilities
        # SB3 PPO Policy
        # Note: Model expects specific input shape. 
        # If symbol case changed, does it affect model? No, model is agnostic to symbol name, just needs data.
        # But we need to make sure `self.models` has the key. 
        # `self.models` uses the original uppercase keys from SYMBOLS list.
        # So we use the original `symbol` (from argument) for model lookup, 
        # but the corrected `symbol` (local var) for MT5 calls.
        
        # Wait, if I change local `symbol` to lowercase, I must use it for MT5, 
        # but keep original for `self.models`.
        # Let's handle this carefully.
        
        # Actually, `process_symbol` receives `symbol` from `SYMBOLS` list (Uppercase).
        # If MT5 needs lowercase, we use a separate variable `mt5_symbol`.
        
        mt5_symbol = symbol
        if not mt5.symbol_select(symbol, True):
            if mt5.symbol_select(symbol.lower(), True):
                mt5_symbol = symbol.lower()
            else:
                self.market_data[symbol]["idea"] = "ERR: SYM"
                return
        
        # Update Price using mt5_symbol
        tick = mt5.symbol_info_tick(mt5_symbol)
        if tick:
            self.market_data[symbol]["price"] = tick.bid
            
        # Global Positions using mt5_symbol
        all_positions = mt5.positions_get(symbol=mt5_symbol)
        if all_positions:
            net_vol = sum([p.volume if p.type == mt5.ORDER_TYPE_BUY else -p.volume for p in all_positions])
            pos_str = "FLAT"
            if net_vol > 0: pos_str = f"LONG ({net_vol:.2f})"
            elif net_vol < 0: pos_str = f"SHORT ({abs(net_vol):.2f})"
            global_pnl = sum([p.profit for p in all_positions])
            self.market_data[symbol]["pos"] = pos_str
            self.market_data[symbol]["pnl"] = global_pnl
        else:
            self.market_data[symbol]["pos"] = "FLAT"
            self.market_data[symbol]["pnl"] = 0.0

        # Get Obs (Pass mt5_symbol to loader?)
        # Loader likely uses `mt5.copy_rates_from_pos(symbol, ...)`
        # So we need to pass `mt5_symbol` to `get_v16_observation`.
        # But `get_v16_observation` calls `self.loader.get_data(symbol)`.
        # I need to update `get_v16_observation` to accept `mt5_symbol` or handle it.
        # For now, let's assume I can pass `mt5_symbol` to `get_v16_observation`.
        
        try:
            obs = self.get_v16_observation(mt5_symbol) 
        except Exception as e:
            self.log(f"⚠️ Failed to get obs for {symbol}: {e}")
            return

        # 4. Predict (MCTS or PPO)
        probs = {}
        action = 0
        
        agent = self.mcts_agents.get(symbol)
        if agent:
            try:
                # Sync Position
                net_vol = 0.0
                avg_entry = 0.0
                if all_positions:
                     net_vol = sum([p.volume if p.type == mt5.ORDER_TYPE_BUY else -p.volume for p in all_positions])
                     total_cost = sum([p.price_open * p.volume for p in all_positions])
                     total_vol = sum([p.volume for p in all_positions])
                     if total_vol > 0: avg_entry = total_cost / total_vol
                
                agent.env.position_size = net_vol
                agent.env.avg_entry_price = avg_entry
                agent.env.balance = self.account_info["balance"]
                
                # 2. Run MCTS
                # self.log(f"🧠 {symbol}: MCTS Thinking...")
                action = agent.search(obs)
                probs[action] = 1.0 # MCTS is deterministic in output
                
            except Exception as e:
                self.log(f"⚠️ MCTS Error {symbol}: {e}")
                # Fallback to PPO
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
        else:
            # Standard PPO
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            probs[action] = 1.0

        # Execute (Pass mt5_symbol)
        self.execute_trade(mt5_symbol, action)

    def generate_layout(self):
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=10)
        )
        
        # Header
        acc = self.account_info
        currency = acc.get("currency", "USD")
        header_text = f"Gemini V16 Orchestrator | Balance: {acc['balance']:.2f} {currency} | Equity: {acc['equity']:.2f} {currency} | Margin: {acc['margin']:.2f} {currency}"
        layout["header"].update(Panel(header_text, style="bold white on blue"))
        
        # Body (Table)
        table = Table(box=box.SIMPLE)
        table.add_column("Symbol", style="cyan")
        table.add_column("Price", justify="right")
        table.add_column("Pos", justify="center")
        table.add_column("PnL", justify="right")
        table.add_column("Model Idea", justify="center", style="magenta")
        table.add_column("Last Action", justify="right")
        
        for symbol in SYMBOLS:
            data = self.market_data.get(symbol, {})
            pnl = data.get("pnl", 0.0)
            pnl_style = "green" if pnl >= 0 else "red"
            
            table.add_row(
                symbol,
                f"{data.get('price', 0.0):.5f}",
                data.get("pos", "FLAT"),
                f"[{pnl_style}]{pnl:.2f}[/{pnl_style}]",
                data.get("idea", "WAIT"),
                data.get("action", "WAIT")
            )
            
        layout["body"].update(Panel(table, title="Market Watch"))
        
        # Footer (Logs)
        log_text = "\n".join(self.logs)
        layout["footer"].update(Panel(log_text, title="Activity Log", style="grey70"))
        
        return layout

    def run(self):
        if not mt5.initialize():
            print("❌ MT5 Init Failed")
            return
            
        self.log("🚀 Gemini V16 Orchestrator Started")
        self.log(f"🎯 Dynamic Rules Active (Gold=500pips, BTC=5000pips)")
        
        # Initial Account Info Fetch
        acc = mt5.account_info()
        if acc:
            self.account_info = {
                "balance": acc.balance,
                "equity": acc.equity,
                "margin": acc.margin,
                "currency": getattr(acc, "currency", "$")
            }
        else:
            self.log("⚠️ Failed to fetch initial account info")

        with Live(self.generate_layout(), refresh_per_second=4) as live:
            while True:
                # Update Account Info
                acc = mt5.account_info()
                if acc:
                    self.account_info = {
                        "balance": acc.balance,
                        "equity": acc.equity,
                        "margin": acc.margin,
                        "currency": getattr(acc, "currency", "$")
                    }
                
                # Scan Markets
                with ThreadPoolExecutor(max_workers=len(SYMBOLS)) as executor:
                    future_to_symbol = {executor.submit(self.process_symbol, symbol): symbol for symbol in SYMBOLS if symbol in self.models}
                    for future in as_completed(future_to_symbol):
                        try:
                            future.result()
                        except Exception as e:
                            self.log(f"⚠️ Error in thread: {e}")
                
                # Update UI
                live.update(self.generate_layout())
                time.sleep(1) # Refresh rate

if __name__ == "__main__":
    bot = GeminiV16Orchestrator()
    bot.run()
