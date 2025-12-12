import MetaTrader5 as mt5
import torch
import numpy as np
import pandas as pd
import time
import os
import sys
from datetime import datetime
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.text import Text
from rich import box

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from gemini_v19.models.alphazero_net import AlphaZeroTradingNet
from gemini_v19.mcts.alphazero_mcts import AlphaZeroMCTS
from gemini_v19.environment.commission_trinity_env import CommissionTrinityEnv
from gemini_v19.live.replay_db import ReplayDatabase
from gemini_v19.utils.config import NETWORK_CONFIG, MCTS_CONFIG, ENV_CONFIG

class AlphaZeroTrader:
    def __init__(self, symbol="EURUSD", magic=1919):
        self.symbol = symbol
        self.magic = magic
        self.console = Console()
        
        # 1. Init MT5
        if not mt5.initialize():
            self.console.print("[bold red]❌ MT5 Init Failed[/bold red]")
            sys.exit(1)
            
        # 2. Load Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = AlphaZeroTradingNet(**NETWORK_CONFIG).to(self.device)
        self.load_model()
        self.network.eval()
        
        # 3. Init Env (for State Construction)
        # We use the Env class mainly to generate the state vector from MT5 data
        self.env = CommissionTrinityEnv(symbol=symbol, lookback=ENV_CONFIG['lookback'])
        
        # 4. Init MCTS
        # We use a fresh MCTS for each decision, or keep it if we want tree reuse (advanced)
        # For now, we init it here but we'll create a new root search each step
        self.mcts_config = MCTS_CONFIG
        
        # 5. Replay DB
        self.db = ReplayDatabase()
        
        # State
        self.last_state = None
        self.last_action = 0
        self.running = True
        
    def load_model(self):
        path = "gemini_v19/models/champions/current_champion.pth"
        if os.path.exists(path):
            self.network.load_state_dict(torch.load(path, map_location=self.device))
            self.console.print(f"[green]✅ Loaded Champion Model: {path}[/green]")
        else:
            self.console.print("[yellow]⚠️ No Champion Found. Using Random Weights.[/yellow]")

    def calculate_lot_size(self):
        """
        Calculate dynamic lot size based on equity.
        Target: ~1:1 Leverage (Conservative) to ~1:2
        Formula: Equity / 100,000
        Example: 40,000 / 100,000 = 0.4 Lots
        """
        account_info = mt5.account_info()
        if account_info is None:
            return 0.01
            
        equity = account_info.equity
        
        # Base scaling
        lot = equity / 100000.0
        
        # Adjust for specific assets if needed (Crypto usually has different contract sizes)
        # For now, we assume standard lots or user adjusts manually if trading exotic pairs
        
        # Round to 2 decimals
        lot = round(lot, 2)
        
        # Min/Max limits
        lot = max(0.01, lot)
        lot = min(100.0, lot)
        
        return lot

    def get_state(self):
        # Update Env Data
        # In a real live loop, we need to fetch latest data from MT5 and update the Env's dataframe
        # CommissionTrinityEnv uses MultiTimeframeLoader. 
        # We need to trigger a data refresh.
        
        # For V19, let's assume the Env has a method to update from MT5
        # If not, we might need to manually fetch and feed.
        # DeepTrinityEnv loads data in __init__.
        # We need a 'update_data()' method. 
        
        # HACK: Re-init env or reload data. Re-init is slow.
        # Better: The Env should be able to update its internal data structures.
        # Let's implement a simple update in the loop:
        # Fetch last 1000 candles for all TFs and update self.env.data
        
        # For now, to keep it simple and functional, we will rely on the Env's internal loader 
        # if we re-create it or update it. 
        # Let's assume we re-create the env for the state generation to ensure freshness (inefficient but safe)
        # Optimization: Implement 'update()' in DeepTrinityEnv later.
        
        self.env = CommissionTrinityEnv(symbol=self.symbol, lookback=ENV_CONFIG['lookback'])
        self.env.reset() # This sets current_step to end
        
        # Get observation
        # We need the observation at the LAST step
        obs = self.env._get_full_observation()
        return obs

    def execute_trade(self, action, policy_dist, value_est):
        # Map Action 0-4 to MT5 Orders
        # 0: Hold, 1: Buy, 2: Sell, 3: Split, 4: Close All
        
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick: return
        
        # Check current positions
        positions = mt5.positions_get(symbol=self.symbol)
        has_pos = len(positions) > 0
        pos_type = positions[0].type if has_pos else None
        volume = positions[0].volume if has_pos else 0.0
        
        trade_executed = False
        lot_size = self.calculate_lot_size()
        
        if action == 1: # BUY
            if not has_pos: # Open Long
                self._send_order(mt5.ORDER_TYPE_BUY, lot_size) 
                trade_executed = True
            elif pos_type == mt5.ORDER_TYPE_SELL: # Close Short & Flip? No, just Close.
                self._close_all()
                self._send_order(mt5.ORDER_TYPE_BUY, lot_size)
                trade_executed = True
                
        elif action == 2: # SELL
            if not has_pos: # Open Short
                self._send_order(mt5.ORDER_TYPE_SELL, lot_size)
                trade_executed = True
            elif pos_type == mt5.ORDER_TYPE_BUY:
                self._close_all()
                self._send_order(mt5.ORDER_TYPE_SELL, lot_size)
                trade_executed = True
                
        elif action == 3: # SPLIT
            if has_pos:
                self._close_partial(0.5) # Split 50%
                trade_executed = True
                
        elif action == 4: # CLOSE ALL
            if has_pos:
                self._close_all()
                trade_executed = True
                
        # Store in DB
        # We store the state that LED to this action
        if self.last_state is not None:
            # Calculate reward (PnL change since last step)
            # This is tricky in live. We use realized PnL + Unrealized change.
            # For simplicity, we store raw reward=0 for now, and update it later or use a proxy.
            # Or we just store the experience and let the Retrainer calculate returns if we store price history.
            # AlphaZero needs (s, pi, z). z is final return.
            # In continuous, we can't know final return yet.
            # We store (s, pi, r).
            
            self.db.store(
                symbol=self.symbol,
                state=self.last_state,
                action=action,
                reward=0.0, # Placeholder
                done=False,
                metadata=str({'policy': policy_dist.tolist(), 'value': value_est})
            )

    def _send_order(self, type, volume):
        tick = mt5.symbol_info_tick(self.symbol)
        price = tick.ask if type == mt5.ORDER_TYPE_BUY else tick.bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": type,
            "price": price,
            "magic": self.magic,
            "comment": "AlphaZero V19",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        mt5.order_send(request)

    def _close_all(self):
        positions = mt5.positions_get(symbol=self.symbol)
        for pos in positions:
            tick = mt5.symbol_info_tick(self.symbol)
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": pos.ticket,
                "price": tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask,
                "magic": self.magic,
            }
            mt5.order_send(request)

    def _close_partial(self, pct):
        positions = mt5.positions_get(symbol=self.symbol)
        for pos in positions:
            vol = round(pos.volume * pct, 2)
            if vol < 0.01: continue
            tick = mt5.symbol_info_tick(self.symbol)
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": vol,
                "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": pos.ticket,
                "price": tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask,
                "magic": self.magic,
            }
            mt5.order_send(request)

    def run(self):
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        with Live(layout, refresh_per_second=1) as live:
            iteration = 0
            while self.running:
                try:
                    iteration += 1
                    self.console.print(f"[yellow]🔄 Iteration {iteration}: Fetching state...[/yellow]")
                    
                    # 1. Get State
                    state = self.get_state()
                    self.last_state = state
                    
                    self.console.print(f"[yellow]🌲 Running MCTS (this may take 10-20 seconds)...[/yellow]")
                    
                    # 2. MCTS Search
                    mcts_init_config = self.mcts_config.copy()
                    if 'temperature' in mcts_init_config:
                        del mcts_init_config['temperature']
                    
                    # Reduce simulations for initial test (50 is heavy)
                    mcts_init_config['n_simulations'] = 10  # Fast test
                        
                    mcts = AlphaZeroMCTS(self.network, self.env, **mcts_init_config)
                    policy, root = mcts.search(state, temperature=self.mcts_config.get('temperature', 0.1))
                    
                    # 3. Select Action
                    action = np.argmax(policy)
                    value = root.Q
                    
                    # 4. Execute
                    self.execute_trade(action, policy, value)
                    
                    # 5. Update UI
                    self.update_ui(layout, action, policy, value)
                    
                    # Sleep (M5 candle = 300s, but we can check every minute or so)
                    # For demo/testing, we sleep 10s
                    time.sleep(10)
                    
                except KeyboardInterrupt:
                    self.running = False
                    
    def update_ui(self, layout, action, policy, value):
        # Header
        layout["header"].update(Panel(Text(f"AlphaZero V19 - {self.symbol}", justify="center", style="bold white"), style="blue"))
        
        # Main
        table = Table(title="MCTS Analysis")
        table.add_column("Action", justify="center")
        table.add_column("Prob", justify="center")
        table.add_column("Q-Value", justify="center")
        
        actions = ["HOLD", "BUY", "SELL", "SPLIT", "CLOSE"]
        for i, a in enumerate(actions):
            style = "green" if i == action else "white"
            table.add_row(a, f"{policy[i]:.2f}", f"{value:.2f}", style=style)
            
        layout["main"].update(Panel(table))
        
        # Footer
        layout["footer"].update(Panel(Text(f"Last Update: {datetime.now().strftime('%H:%M:%S')}", justify="right")))

if __name__ == "__main__":
    trader = AlphaZeroTrader(symbol="EURUSD")
    trader.run()
