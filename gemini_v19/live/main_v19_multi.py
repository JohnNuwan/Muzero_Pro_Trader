import MetaTrader5 as mt5
import torch
import numpy as np
import time
import os
import sys
from datetime import datetime
from threading import Thread
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
from gemini_v19.utils.telegram_notifier import TelegramNotifier
from gemini_v19.utils.logger import V19Logger
from gemini_v19.live.pyramiding import PyramidManager

# Multi-Symbol Configuration (Reduced for speed)
SYMBOLS = [
    "EURUSD",
    "XAUUSD",     # GOLD (essentiel)
    "BTCUSD",
    "US30.cash",
    "US500.cash",
    "USDJPY",     # New V19
    "GBPUSD",     # New V19
    "USDCAD",     # New V19
    "USDCHF",     # New V19
    "GER40.cash", # DAX
    "US100.cash"  # NASDAQ
]

class AlphaZeroTrader:
    def __init__(self, symbol="EURUSD", magic=1919, shared_network=None, telegram=None, logger=None, log_func=None):
        self.symbol = symbol
        self.magic = magic
        self.console = Console()
        self.log_func = log_func
        
        # Use shared network if provided, else create new one
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if shared_network:
            self.network = shared_network
        else:
            self.network = AlphaZeroTradingNet(**NETWORK_CONFIG).to(self.device)
            self.load_model()
        self.network.eval()
        
        # Init Env (for State Construction)
        self.env = CommissionTrinityEnv(symbol=symbol, lookback=ENV_CONFIG['lookback'])
        
        # MCTS Config
        self.mcts_config = MCTS_CONFIG.copy()
        self.mcts_config['n_simulations'] = 50  # High quality decisions (was 5)
        
        # Replay DB
        self.db = ReplayDatabase()
        
        # State
        self.last_state = None
        self.last_action = 0
        self.last_policy = None
        self.last_value = 0.0
        self.last_pnl = 0.0  # Track previous PnL
        self.last_equity = 0.0  # Track previous equity
        self.latest_indicators = {} # Store latest indicators for UI
        self.telegram = telegram
        self.logger = logger
        self.running = True
        
        # Pyramid Manager
        self.pyramid_mgr = PyramidManager(self.symbol, self.magic)
        
    def load_model(self):
        path = "gemini_v19/models/champions/current_champion.pth"
        if os.path.exists(path):
            self.network.load_state_dict(torch.load(path, map_location=self.device))
            self.console.print(f"[green]✅ Loaded Champion Model: {path}[/green]")
        else:
            self.console.print("[yellow]⚠️ No Champion Found. Using Random Weights.[/yellow]")

    def calculate_lot_size(self):
        account_info = mt5.account_info()
        if account_info is None:
            return 0.01
            
        equity = account_info.equity
        lot = equity / 100000.0
        lot = round(lot, 2)
        lot = max(0.01, lot)
        lot = min(100.0, lot)
        
        return lot

    def get_state(self):
        self.env = CommissionTrinityEnv(symbol=self.symbol, lookback=ENV_CONFIG['lookback'])
        self.env.reset()
        obs = self.env._get_full_observation()
        return obs

    def get_latest_indicators(self):
        """Extract key indicators for dashboard display"""
        if not hasattr(self, 'env') or not hasattr(self.env, 'data'):
            return {}
        
        try:
            # Use M1 for most up-to-date info
            if "M1" not in self.env.data: return {}
            df = self.env.data["M1"]
            if df.empty: return {}
            
            row = df.iloc[-1]
            return {
                'rsi': row.get('rsi', 0),
                'stoch_k': row.get('stoch_rsi_k', 0),
                'bb_b': row.get('bb_percent_b', 0),
                'trend': row.get('trend_score', 0),
                'mfi': row.get('mfi', 0),
                'adx': row.get('adx', 0),
                'z_score': row.get('z_score', 0)
            }
        except Exception:
            return {}

    def execute_trade(self, action, policy_dist, value_est):
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick: return
        
        positions = mt5.positions_get(symbol=self.symbol, magic=1919)
        has_pos = len(positions) > 0
        pos_type = positions[0].type if has_pos else None
        
        lot_size = self.calculate_lot_size()
        
        if action == 1: # BUY
            if not has_pos:
                self._send_order(mt5.ORDER_TYPE_BUY, lot_size) 
            elif pos_type == mt5.ORDER_TYPE_SELL:
                self._close_all()
                self._send_order(mt5.ORDER_TYPE_BUY, lot_size)
                
        elif action == 2: # SELL
            if not has_pos:
                self._send_order(mt5.ORDER_TYPE_SELL, lot_size)
            elif pos_type == mt5.ORDER_TYPE_BUY:
                self._close_all()
                self._send_order(mt5.ORDER_TYPE_SELL, lot_size)
                
        elif action == 3: # SPLIT
            if has_pos:
                self._close_partial(0.5)
                
        elif action == 4: # CLOSE ALL
            if has_pos:
                self._close_all()
                
        # Calculate reward based on PnL change
        if self.last_state is not None:
            # Get current PnL for this symbol
            positions = mt5.positions_get(symbol=self.symbol, magic=1919)
            current_pnl = sum([p.profit for p in positions]) if positions else 0.0
            
            # Calculate PnL delta (change since last action)
            pnl_delta = current_pnl - self.last_pnl
            
            # Reward System:
            # - Profit: 1 point per euro (linear)
            # - Loss: -1 point per euro BUT with additional penalty if loss > 10€
            reward = pnl_delta  # Base: 1€ = 1 point
            
            # Additional penalty for significant losses
            if pnl_delta < -10:
                # Extra penalty: -0.5 pts per euro beyond -10€
                penalty = abs(pnl_delta + 10) * 0.5
                reward -= penalty
                
            # Bonus for big wins (encourage profitability)
            if pnl_delta > 50:
                # Bonus: +0.2 pts per euro beyond 50€
                bonus = (pnl_delta - 50) * 0.2
                reward += bonus
            
            # Store in database
            self.db.store(
                symbol=self.symbol,
                state=self.last_state,
                action=action,
                reward=reward,
                done=False,
                metadata=str({'policy': policy_dist.tolist(), 'value': value_est, 'pnl_delta': pnl_delta})
            )
            
        # Update tracking
        positions = mt5.positions_get(symbol=self.symbol, magic=1919)
        self.last_pnl = sum([p.profit for p in positions]) if positions else 0.0

    def _send_order(self, type, volume):
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            if self.logger: self.logger.error(f"Could not get tick for {self.symbol}")
            return
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
        result = mt5.order_send(request)
        
        # Log execution
        if self.logger and result:
            action_name = "BUY" if type == mt5.ORDER_TYPE_BUY else "SELL"
            self.logger.execution(self.symbol, action_name, volume, price, result.retcode == mt5.TRADE_RETCODE_DONE)
        
        # Telegram alert
        if self.telegram and result and result.retcode == mt5.TRADE_RETCODE_DONE:
            action_name = "BUY" if type == mt5.ORDER_TYPE_BUY else "SELL"
            self.telegram.send_trade_alert(
                symbol=self.symbol,
                action=action_name,
                volume=volume,
                price=price,
                confidence=max(self.last_policy) if self.last_policy is not None else 0.2,
                value_estimate=self.last_value,
                context=self.latest_indicators
            )

    def _close_partial(self, percentage):
        """Closes a percentage of all open positions for the symbol."""
        positions = mt5.positions_get(symbol=self.symbol, magic=self.magic)
        if not positions:
            return

        for pos in positions:
            volume_to_close = round(pos.volume * percentage, 2)
            if volume_to_close < 0.01:
                continue

            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                continue
            
            close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
            opposite_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": self.symbol,
                "volume": volume_to_close,
                "type": opposite_type,
                "price": close_price,
                "magic": self.magic,
                "comment": "AlphaZero V19 Partial Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if self.logger and result:
                self.logger.execution(self.symbol, "PARTIAL_CLOSE", volume_to_close, close_price, result.retcode == mt5.TRADE_RETCODE_DONE)

    def _close_all(self):
        """Closes all open positions for the symbol."""
        positions = mt5.positions_get(symbol=self.symbol, magic=self.magic)
        if not positions:
            return

        for pos in positions:
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                continue

            close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
            opposite_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": self.symbol,
                "volume": pos.volume,
                "type": opposite_type,
                "price": close_price,
                "magic": self.magic,
                "comment": "AlphaZero V19 Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if self.logger and result:
                self.logger.execution(self.symbol, "CLOSE_ALL", pos.volume, close_price, result.retcode == mt5.TRADE_RETCODE_DONE)

    def run_iteration(self):
        """Run one trading iteration"""
        try:
            # Get State
            state = self.get_state()
            self.last_state = state
            self.latest_indicators = self.get_latest_indicators()
            
            # MCTS Search
            mcts_init_config = self.mcts_config.copy()
            if 'temperature' in mcts_init_config:
                del mcts_init_config['temperature']
                
            mcts = AlphaZeroMCTS(self.network, self.env, **mcts_init_config)
            policy, root = mcts.search(state, temperature=self.mcts_config.get('temperature', 0.1))
            
            # Select Action
            action = np.argmax(policy)
            value = root.Q
            confidence = policy[action]  # Confidence = probability of selected action
            
            # DEBUG: Print policy distribution
            action_names = ["HOLD", "BUY", "SELL", "SPLIT", "CLOSE"]
            policy_str = ", ".join([f"{action_names[i]}: {policy[i]:.3f}" for i in range(len(policy))])
            self.console.print(f"[dim]{self.symbol} Policy: {policy_str} -> Action: {action_names[action]}[/dim]")
            
            # Store for display
            self.last_policy = policy
            self.last_value = value
            self.last_action = action
            
            # Log decision
            if self.logger:
                self.logger.decision(self.symbol, action, policy, value, self.latest_indicators)
            
            # Confidence filter (Aggressive: 8% threshold)
            MIN_CONFIDENCE = 0.08
            if confidence < MIN_CONFIDENCE and action != 0:  # Allow HOLD regardless
                msg = f"{self.symbol}: Low confidence ({confidence:.1%}) < {MIN_CONFIDENCE:.1%} - Skipping"
                if self.log_func: self.log_func(f"[yellow]{msg}[/yellow]")
                # self.console.print(f"[yellow]{msg}[/yellow]") # Deprecated
                if self.logger:
                    self.logger.info(f"{self.symbol}: SKIPPED (Conf {confidence:.2f} < {MIN_CONFIDENCE}) | Policy: {policy_str}")
                return
            
            if action != 0:
                if self.log_func: self.log_func(f"[green]🚀 {self.symbol}: GO! Action {action_names[action]} (Conf {confidence:.1%})[/green]")
            
            # --- PYRAMIDING LOGIC ---
            # 1. Check for pyramiding opportunities
            main_pos = self.pyramid_mgr.get_main_position()
            action_name = "BUY" if action == 1 else "SELL" if action == 2 else "HOLD"
            
            if main_pos and action != 0: # If we have a position and signal is not HOLD
                # Map action index to string for manager
                # Assuming action 1=BUY, 2=SELL based on previous code context, but need to verify action mapping
                # Actually, looking at execute_trade (not shown but inferred), action 0 is usually HOLD.
                # Let's verify action mapping from AlphaZeroTradingNet or similar. 
                # Standard: 0=HOLD, 1=BUY, 2=SELL usually.
                # Wait, in V20 it was 0-2 BUY, 3-5 SELL, 6 HOLD.
                # In V19 (CommissionTrinityEnv), actions are usually 0=HOLD, 1=BUY, 2=SELL.
                # Let's assume 1=BUY, 2=SELL for now.
                
                signal_dir = "BUY" if action == 1 else "SELL"
                
                if self.pyramid_mgr.can_pyramid(main_pos, signal_dir, confidence):
                    self.pyramid_mgr.add_pyramid(main_pos, signal_dir)
            
            # 2. Monitor existing pyramids (SL -> BE)
            self.pyramid_mgr.monitor_pyramids()
            # ------------------------

            # Execute
            self.execute_trade(action, policy, value)
            
        except Exception as e:
            msg = f"Error in {self.symbol}: {e}"
            if self.log_func: self.log_func(f"[red]{msg}[/red]")
            # self.console.print(f"[red]{msg}[/red]")


class MultiSymbolOrchestrator:
    def __init__(self):
        self.console = Console()
        self.logs = []
        
        # Init MT5
        if not mt5.initialize():
            self.console.print("[bold red]❌ MT5 Init Failed[/bold red]")
            sys.exit(1)
            
        # Load shared model (all symbols use the same network)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = AlphaZeroTradingNet(**NETWORK_CONFIG).to(self.device)
        self.load_model()
        self.network.eval()
        
        # Initialize Logger
        self.logger = V19Logger()
        
        # Initialize Telegram
        self.telegram = TelegramNotifier()
        self.telegram.send_startup_message()
        
        # Timers for periodic notifications
        self.last_recap = time.time()
        self.last_stats = time.time()
        
        # Create traders
        self.traders = {}
        for symbol in SYMBOLS:
            self.traders[symbol] = AlphaZeroTrader(
                symbol=symbol,
                magic=1919,
                shared_network=self.network,
                telegram=self.telegram,
                logger=self.logger,
                log_func=self.log_message
            )
            
        self.running = True

    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        if len(self.logs) > 50: self.logs.pop(0)
            
        self.running = True
        
    def load_model(self):
        path = "gemini_v19/models/champions/current_champion.pth"
        if os.path.exists(path):
            self.network.load_state_dict(torch.load(path, map_location=self.device))
            self.console.print(f"[green]✅ Loaded Champion Model[/green]")
        else:
            self.console.print("[yellow]⚠️ No Champion Found. Using Random Weights.[/yellow]")
            
    def update_ui(self, layout):
        # Get account info
        account_info = mt5.account_info()
        if account_info:
            balance = account_info.balance
            equity = account_info.equity
            margin = account_info.margin
            free_margin = account_info.margin_free
            profit = account_info.profit
        else:
            balance = equity = margin = free_margin = profit = 0.0
            
        # Header - Account Overview
        # Note: We will update the PnL in the header AFTER calculating it from positions
        # For now, we render the header at the end of the function or use a placeholder
        # But to keep it simple, let's calculate total_pnl FIRST.
        
        total_pnl = 0.0
        # First pass to calculate Total PnL for V19
        all_v19_positions = mt5.positions_get(magic=1919)
        if all_v19_positions:
            total_pnl = sum([p.profit for p in all_v19_positions])

        header_text = Text()
        header_text.append("AlphaZero V19 - Multi-Symbol", style="bold white")
        header_text.append(f" | Balance: {balance:.2f}€", style="cyan")
        header_text.append(f" | Equity: {equity:.2f}€", style="green" if equity >= balance else "red")
        header_text.append(f" | V19 PnL: {total_pnl:.2f}€", style="green" if total_pnl >= 0 else "red")
        layout["header"].update(Panel(header_text, style="blue"))
        
        # Main - Split into two sections (Already split in run)
        # layout["main"].split_row(...) -> Removed to avoid reset
        
        # Left: Symbol Status Table
        symbol_table = Table(title="Symbol Analysis", box=box.ROUNDED, expand=True)
        symbol_table.add_column("Symbol", justify="center", style="cyan", width=10)
        symbol_table.add_column("Spread", justify="center", width=6)
        symbol_table.add_column("Action", justify="center", width=8)
        symbol_table.add_column("Prob", justify="center", width=6)
        symbol_table.add_column("Value", justify="center", width=7)
        symbol_table.add_column("Context", justify="center", width=35)
        symbol_table.add_column("Position", justify="center", width=12)
        symbol_table.add_column("PnL", justify="right", width=8)
        
        # total_pnl = 0.0 # Already calculated above
        for symbol, trader in self.traders.items():
            # Get position info (filter by magic number 1919)
            positions = mt5.positions_get(symbol=symbol, magic=1919)
            pos_str = "FLAT"
            pos_pnl = 0.0
            if positions and len(positions) > 0:
                pos = positions[0]
                pos_type = "LONG" if pos.type == mt5.ORDER_TYPE_BUY else "SHORT"
                pos_str = f"{pos_type} {pos.volume:.2f}"
                pos_pnl = pos.profit
                # total_pnl += pos_pnl # Already calculated above
                
            action_names = ["HOLD", "BUY", "SELL", "SPLIT", "CLOSE"]
            action_str = action_names[trader.last_action]
            
            # Get max probability
            max_prob = trader.last_policy[trader.last_action] if trader.last_policy is not None else 0.0
            
            # Color based on action
            color = "yellow"
            if trader.last_action == 1: color = "green"
            elif trader.last_action == 2: color = "red"
            
            pnl_color = "green" if pos_pnl >= 0 else "red"
            
            # Context String
            inds = trader.latest_indicators
            context_str = "-"
            if inds:
                rsi = inds.get('rsi', 0)
                stoch = inds.get('stoch_k', 0)
                bb = inds.get('bb_b', 0)
                mfi = inds.get('mfi', 0)
                adx = inds.get('adx', 0)
                z = inds.get('z_score', 0)
                
                # Colorize RSI
                rsi_c = "red" if rsi > 70 else "green" if rsi < 30 else "white"
                context_str = f"[{rsi_c}]R:{int(rsi)}[/{rsi_c}] M:{int(mfi)} A:{int(adx)} Z:{z:.1f} B:{bb:.1f}"

            # Get live spread
            info = mt5.symbol_info(symbol)
            spread = info.spread if info else 0
            
            symbol_table.add_row(
                symbol,
                f"{spread}",
                f"[{color}]{action_str}[/{color}]",
                f"{max_prob:.2f}",
                f"{trader.last_value:.2f}",
                context_str,
                pos_str,
                f"[{pnl_color}]{pos_pnl:.2f}€[/{pnl_color}]"
            )
            
        layout["main"]["symbols"].update(Panel(symbol_table, title="[bold]Trading Signals[/bold]"))
        
        # Right: Stats Panel
        stats_table = Table(show_header=False, box=box.SIMPLE, expand=True)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", justify="right")
        
        stats_table.add_row("💰 Balance", f"{balance:.2f}€")
        stats_table.add_row("📊 Equity", f"[{'green' if equity >= balance else 'red'}]{equity:.2f}€[/{'green' if equity >= balance else 'red'}]")
        stats_table.add_row("💵 Free Margin", f"{free_margin:.2f}€")
        stats_table.add_row("📈 Total PnL", f"[{'green' if total_pnl >= 0 else 'red'}]{total_pnl:.2f}€[/{'green' if total_pnl >= 0 else 'red'}]")
        stats_table.add_row("", "")
        
        # Count positions (filter by magic 1919 only)
        all_positions = mt5.positions_get(magic=1919)
        total_positions = len(all_positions) if all_positions else 0
        stats_table.add_row("🎯 Open Positions", f"{total_positions}")
        stats_table.add_row("🤖 Active Symbols", f"{len(SYMBOLS)}")
        stats_table.add_row("🌲 MCTS Sims", f"{self.traders[SYMBOLS[0]].mcts_config['n_simulations']}")
        
        layout["main"]["stats"].update(Panel(stats_table, title="[bold]Account Stats[/bold]"))
        
        # Footer (Logs)
        log_text = "\n".join(self.logs)
        layout["footer"].update(Panel(log_text, title="Activity Log", style="grey70"))
        
    def run(self):
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=12)
        )
        
        # Initialize Main Split ONCE
        layout["main"].split_row(
            Layout(name="symbols", ratio=2),
            Layout(name="stats", ratio=1)
        )
        
        # Initial Render (Show empty UI while loading)
        self.update_ui(layout)
        
        with Live(layout, refresh_per_second=1) as live:
            iteration = 0
            while self.running:
                try:
                    iteration += 1
                    iter_start = time.time()
                    
                    # Periodic notifications
                    now = time.time()
                    if now - self.last_recap >= 3600:  # Every hour
                        self.telegram.send_recap(self.traders)
                        self.last_recap = now
                    
                    if now - self.last_stats >= 21600:  # Every 6 hours
                        self.telegram.send_stats(hours=24)
                        self.last_stats = now
                    
                    # Run all traders in parallel using threads
                    threads = []
                    for trader in self.traders.values():
                        t = Thread(target=trader.run_iteration)
                        t.start()
                        threads.append(t)
                        
                    # Wait for all to complete
                    for t in threads:
                        t.join()
                        
                    # Update UI
                    self.update_ui(layout)
                    
                    # Log iteration
                    if self.logger:
                        self.logger.iteration(iteration, time.time() - iter_start)
                    
                    # Sleep
                    time.sleep(30)  # Check every 30s
                    
                except KeyboardInterrupt:
                    if self.logger:
                        self.logger.info("⚠️ User stopped execution")
                    self.running = False
                except Exception as e:
                    if self.logger:
                        self.logger.info(f"❌ ERROR: {e}")
                    time.sleep(5)

if __name__ == "__main__":
    orchestrator = MultiSymbolOrchestrator()
    orchestrator.run()
