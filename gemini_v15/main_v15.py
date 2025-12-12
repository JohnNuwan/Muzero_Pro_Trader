import os
import sys
import time
import datetime
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from stable_baselines3 import PPO, SAC

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from gemini_v15.utils.mtf_data_loader import MultiTimeframeLoader
from gemini_v15.utils.indicators import Indicators
from gemini_v15.environment.deep_trinity_env import DeepTrinityEnv # For observation construction logic

# Configuration
SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", 
    "BTCUSD", "ETHUSD", 
    "XAUUSD", 
    "US30.cash", "GER40.cash", "US500.cash", "US100.cash"
]
CRYPTO_SYMBOLS = ["BTCUSD", "ETHUSD"]
MODELS_DIR = os.path.join(current_dir, "models")
MAGIC_NUMBER = 151515

class GeminiV15Orchestrator:
    def __init__(self):
        self.models_trader = {}
        self.models_risk = {}
        self.loader = MultiTimeframeLoader()
        self.is_mt5_connected = False
        self.logs = []
        
    def connect_mt5(self):
        if not mt5.initialize():
            print("❌ MT5 Initialization Failed")
            return False
        print("✅ MT5 Initialized")
        self.is_mt5_connected = True
        return True

    def load_models(self):
        print("🧠 Loading V15 Models...")
        for symbol in SYMBOLS:
            # Load Trader (PPO)
            path_ppo = os.path.join(MODELS_DIR, f"ppo_v15_{symbol}.zip")
            if os.path.exists(path_ppo):
                self.models_trader[symbol] = PPO.load(path_ppo)
                print(f"  ✅ Loaded PPO for {symbol}")
            else:
                print(f"  ⚠️ Missing PPO model for {symbol}")

            # Load Risk Manager (SAC)
            path_sac = os.path.join(MODELS_DIR, f"sac_v15_{symbol}.zip")
            if os.path.exists(path_sac):
                self.models_risk[symbol] = SAC.load(path_sac)
                print(f"  ✅ Loaded SAC for {symbol}")
            else:
                print(f"  ⚠️ Missing SAC model for {symbol}")
        print(f"🎉 Loaded {len(self.models_trader)} Traders and {len(self.models_risk)} Risk Managers.")

    def get_observation(self, symbol):
        """
        Constructs the observation vector for a symbol.
        Reuses logic from DeepTrinityEnv but for live inference.
        """
        # Fetch Data (Optimized for Live: fewer candles needed)
        data = self.loader.get_data(symbol, lookback_candles=500) 
        if data is None: return None
        
        # Process Indicators
        processed_data = {}
        for tf, df in data.items():
            processed_data[tf] = Indicators.add_all(df.copy())
            
        # Build Vector
        # We need the LATEST closed candle for each TF
        current_time = pd.Timestamp.now()
        state_vector = []
        
        for tf in ["M1", "M5", "M15", "H1", "H4", "D1"]:
            df = processed_data.get(tf)
            if df is None or df.empty:
                state_vector.extend([0.0] * 13)
                continue
                
            # Get last row (assuming it's the latest closed candle provided by loader)
            row = df.iloc[-1]
            
            features = [
                row['rsi'], row['mfi'], row['adx'], row['z_score'], 
                row['trend_score'], row['linreg_angle'], row['fibo_pos'],
                row['dist_to_res'], row['dist_to_sup'], row['skew'],
                row['kurtosis'], row['entropy'], row['hurst']
            ]
            features = [0.0 if np.isnan(x) else x for x in features]
            state_vector.extend(features)
            
        # Time Features
        hour = current_time.hour
        day = current_time.dayofweek
        state_vector.append(np.sin(2 * np.pi * hour / 24.0))
        state_vector.append(np.cos(2 * np.pi * hour / 24.0))
        state_vector.append(np.sin(2 * np.pi * day / 7.0))
        state_vector.append(np.cos(2 * np.pi * day / 7.0))
        
        return np.array(state_vector, dtype=np.float32)

    def execute_trade(self, symbol, action_trader, action_risk):
        """
        Executes the trade based on AI decisions.
        """
        if action_trader == 0: return # Hold
        
        # Get Symbol Info
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            self.log(f"❌ Symbol Info not found for {symbol}")
            return
            
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                self.log(f"❌ Failed to select {symbol}")
                return
        
        min_lot = symbol_info.volume_min
        max_lot = symbol_info.volume_max
        step_lot = symbol_info.volume_step
        point = symbol_info.point
        stops_level = symbol_info.trade_stops_level
        
        # Decode Risk Params
        # Lot Multiplier: Map [-1, 1] to [1.0, 5.0] multiplier of MIN_LOT
        lot_mult = ((action_risk[0] + 1) / 2) * (5.0 - 1.0) + 1.0
        raw_lot = min_lot * lot_mult
        lot_size = round(raw_lot / step_lot) * step_lot
        lot_size = max(min_lot, min(lot_size, max_lot))
        
        # Decode SL/TP (Pips)
        sl_pips = ((action_risk[1] + 1) / 2) * (100 - 10) + 10
        tp_pips = ((action_risk[2] + 1) / 2) * (200 - 10) + 10
        
        sl_dist = sl_pips * 10 * point 
        tp_dist = tp_pips * 10 * point
        
        # Ensure SL/TP are valid (respect stops_level)
        min_dist = stops_level * point
        if sl_dist < min_dist: sl_dist = min_dist + (10 * point)
        if tp_dist < min_dist: tp_dist = min_dist + (10 * point)

        # Get Tick
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            self.log(f"❌ Tick data not available for {symbol}")
            return
            
        # Execute
        order_type = mt5.ORDER_TYPE_BUY if action_trader == 1 else mt5.ORDER_TYPE_SELL
        price = tick.ask if action_trader == 1 else tick.bid
        
        sl = price - sl_dist if action_trader == 1 else price + sl_dist
        tp = price + tp_dist if action_trader == 1 else price - tp_dist
        
        # Normalize Prices
        price = float(round(price, symbol_info.digits))
        sl = float(round(sl, symbol_info.digits))
        tp = float(round(tp, symbol_info.digits))
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot_size),
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": "Gemini V15 AI",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result is None:
            self.log(f"❌ Order Send Failed (Result is None). Request: {request}")
            self.log(f"   MT5 Last Error: {mt5.last_error()}")
            return
            
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.log(f"❌ Order Failed for {symbol}: {result.comment} (Code: {result.retcode})")
        else:
            self.log(f"✅ Trade Executed: {symbol} | {'BUY' if action_trader==1 else 'SELL'} | Lot: {lot_size} | SL: {sl_pips:.1f} | TP: {tp_pips:.1f}")

    def log(self, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        if len(self.logs) > 20: self.logs.pop(0)

    def run(self):
        if not self.connect_mt5(): return
        self.load_models()
        
        # Setup Rich Dashboard
        from rich.live import Live
        from rich.layout import Layout
        from rich.panel import Panel
        from rich.table import Table
        from rich.console import Console
        from rich import box
        
        self.logs = []
        
        def generate_dashboard(last_signals, account_info, current_status):
            layout = Layout()
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="stats", size=8),
                Layout(name="body", ratio=1),
                Layout(name="logs", size=10),
                Layout(name="status", size=3)
            )
            
            # Header
            layout["header"].update(Panel("🚀 GEMINI V15 - AI TRADING SYSTEM", style="bold white on blue"))
            
            # Stats (Account Info)
            stats_table = Table(box=None, expand=True)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="bold white")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="bold white")
            
            if account_info:
                balance = account_info.balance
                equity = account_info.equity
                profit = account_info.profit
                margin_free = account_info.margin_free
                
                color_profit = "green" if profit >= 0 else "red"
                
                stats_table.add_row("Balance", f"${balance:.2f}", "Equity", f"${equity:.2f}")
                stats_table.add_row("Open PnL", f"[{color_profit}]${profit:.2f}[/{color_profit}]", "Free Margin", f"${margin_free:.2f}")
            else:
                stats_table.add_row("Balance", "Loading...", "Equity", "Loading...")
            
            layout["stats"].update(Panel(stats_table, title="Account Overview", border_style="green"))
            
            # Body: Signals Table
            table = Table(title="Live AI Signals", box=box.ROUNDED, expand=True)
            table.add_column("Symbol", style="cyan")
            table.add_column("Time", style="dim")
            table.add_column("Signal", style="bold")
            table.add_column("Risk (Lot/SL/TP)", style="magenta")
            table.add_column("Status", style="green")
            
            for sig in last_signals:
                color = "green" if "BUY" in sig['signal'] else "red"
                table.add_row(
                    sig['symbol'], 
                    sig['time'], 
                    f"[{color}]{sig['signal']}[/{color}]", 
                    sig['risk'], 
                    sig['status']
                )
            layout["body"].update(table)
            
            # Logs
            log_text = "\n".join(self.logs)
            layout["logs"].update(Panel(log_text, title="System Logs", border_style="yellow"))
            
            # Status Bar
            layout["status"].update(Panel(current_status, style="bold black on white"))
            
            return layout

        print("🚀 Gemini V15 Live Trading Started...")
        last_signals = []
        current_status = "Initializing..."
        
        # Initial Account Info
        account_info = mt5.account_info()
        
        with Live(generate_dashboard(last_signals, account_info, current_status), refresh_per_second=4) as live:
            while True:
                try:
                    now = datetime.datetime.now()
                    is_weekend = now.weekday() >= 5 
                    
                    # Refresh Account Info
                    account_info = mt5.account_info()
                    
                    for symbol in SYMBOLS:
                        current_status = f"🧠 Analyzing {symbol}..."
                        live.update(generate_dashboard(last_signals, account_info, current_status))
                        
                        # Crypto Weekend Logic
                        if is_weekend and symbol not in CRYPTO_SYMBOLS:
                            continue
                            
                        # Check if we have models
                        if symbol not in self.models_trader or symbol not in self.models_risk:
                            continue
                            
                        # Get Observation
                        obs = self.get_observation(symbol)
                        if obs is None: continue
                        
                        # Inference
                        action_trader, _ = self.models_trader[symbol].predict(obs, deterministic=True)
                        action_risk, _ = self.models_risk[symbol].predict(obs, deterministic=True)
                        
                        # Check Positions
                        positions = mt5.positions_get(symbol=symbol)
                        has_position = positions is not None and len(positions) > 0
                        
                        signal_str = "HOLD"
                        if action_trader == 1: signal_str = "BUY"
                        elif action_trader == 2: signal_str = "SELL"
                        
                        status = "Waiting"
                        if has_position:
                            pos_type = positions[0].type
                            type_str = "BUY" if pos_type == mt5.POSITION_TYPE_BUY else "SELL"
                            status = f"In Trade ({type_str})"
                        
                        # Execute
                        if action_trader != 0 and not has_position:
                            self.execute_trade(symbol, action_trader, action_risk)
                            status = "Executed"
                            # Refresh Account Info after trade
                            account_info = mt5.account_info()
                            
                        # Update Signals List (Upsert)
                        # Check if symbol already in list, update it
                        found = False
                        
                        # Decode Risk for Display
                        lot_mult = ((action_risk[0] + 1) / 2) * (5.0 - 1.0) + 1.0
                        sl_pips = ((action_risk[1] + 1) / 2) * (100 - 10) + 10
                        tp_pips = ((action_risk[2] + 1) / 2) * (200 - 10) + 10
                        risk_str = f"L:{lot_mult:.1f}x S:{sl_pips:.0f} T:{tp_pips:.0f}"
                        
                        new_sig = {
                            'symbol': symbol,
                            'time': now.strftime("%H:%M:%S"),
                            'signal': signal_str,
                            'risk': risk_str,
                            'status': status
                        }
                        
                        # Only add if interesting (Trade or Signal)
                        if action_trader != 0 or has_position:
                            # Remove old entry for this symbol
                            last_signals = [s for s in last_signals if s['symbol'] != symbol]
                            # Add new
                            last_signals.insert(0, new_sig) # Add to top
                            last_signals = last_signals[:10] # Keep last 10
                            
                        live.update(generate_dashboard(last_signals, account_info, current_status))
                    
                    current_status = "💤 Sleeping..."
                    live.update(generate_dashboard(last_signals, account_info, current_status))
                    time.sleep(10) 
                    
                except KeyboardInterrupt:
                    print("🛑 Stopping Gemini V15...")
                    break
                except Exception as e:
                    self.log(f"❌ Error in Main Loop: {e}")
                    time.sleep(10)

if __name__ == "__main__":
    bot = GeminiV15Orchestrator()
    bot.run()
