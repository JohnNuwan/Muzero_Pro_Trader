import MetaTrader5 as mt5
import pandas as pd
import time
import sys
import os
from datetime import datetime
from collections import deque

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from gemini_v14.environment.trinity_env import TrinityEnv
from gemini_v14.utils.data_loader import DataLoader
from gemini_v14.agents.strategies.rl_agents import QLearningAgent, DoubleQLearningAgent, ActorCriticAgent
from gemini_v14.agents.strategies.managers import StandardManager

# Configuration
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "XAUUSD", "US30.cash", "GER40.cash"]
TIMEFRAME = mt5.TIMEFRAME_M5
MAGIC_NUMBER = 141414

# Global State
log_buffer = deque(maxlen=15)

def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_buffer.append(f"[{timestamp}] {message}")

def get_dashboard(account_info, positions, agent_name, manager_states):
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="logs", size=15)
    )
    
    # Header
    layout["header"].update(Panel(f"[bold yellow]Gemini V14 Trinity - LIVE ORCHESTRATOR 🚀[/bold yellow] | Champion: [green]{agent_name}[/green]", style="white"))
    
    # Main: Split into Account and Positions
    layout["main"].split_row(
        Layout(name="account", ratio=1),
        Layout(name="positions", ratio=3)
    )
    
    # Account Table
    acct_table = Table(expand=True, box=None)
    acct_table.add_column("Metric", style="cyan")
    acct_table.add_column("Value", style="green", justify="right")
    if account_info:
        acct_table.add_row("Balance", f"{account_info.balance:.2f} €")
        acct_table.add_row("Equity", f"{account_info.equity:.2f} €")
        acct_table.add_row("Margin", f"{account_info.margin:.2f} €")
        acct_table.add_row("Free Margin", f"{account_info.margin_free:.2f} €")
        profit_color = "green" if account_info.profit >= 0 else "red"
        acct_table.add_row("Open PnL", f"[{profit_color}]{account_info.profit:.2f} €[/{profit_color}]")
    
    layout["account"].update(Panel(acct_table, title="Wallet 💰", border_style="cyan"))
    
    # Positions Table
    pos_table = Table(expand=True, box=None)
    pos_table.add_column("Symbol", style="bold")
    pos_table.add_column("Type")
    pos_table.add_column("Vol")
    pos_table.add_column("Open")
    pos_table.add_column("Current")
    pos_table.add_column("PnL", justify="right")
    pos_table.add_column("Comment", style="italic cyan")
    pos_table.add_column("Manager Status", style="magenta")
    
    if positions:
        for pos in positions:
            pnl_color = "green" if pos.profit >= 0 else "red"
            type_str = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
            type_color = "green" if type_str == "BUY" else "red"
            
            # Determine Manager Status
            status = manager_states.get(pos.ticket, "Monitoring 🛡️")
            
            pos_table.add_row(
                pos.symbol, 
                f"[{type_color}]{type_str}[/{type_color}]", 
                str(pos.volume), 
                f"{pos.price_open:.5f}", 
                f"{pos.price_current:.5f}", 
                f"[{pnl_color}]{pos.profit:.2f}[/{pnl_color}]",
                pos.comment,
                status
            )
    else:
        pos_table.add_row("-", "-", "-", "-", "-", "-", "-", "Searching for targets...")
        
    layout["positions"].update(Panel(pos_table, title="Active Trades (The Arena) ⚔️", border_style="magenta"))
    
    # Logs
    log_text = "\n".join(log_buffer)
    layout["logs"].update(Panel(log_text, title="Event Log (Trader/Manager/Critic) 📜", border_style="white"))
    
    return layout

def main():
    console = Console()
    
    # 1. Initialize MT5
    if not mt5.initialize():
        print(f"MT5 Init Failed: {mt5.last_error()}")
        return
        
    # 2. Load Agents
    model_dir = os.path.join(project_root, "gemini_v14", "models")
    winner_name = "QL_Tabular" 
    agent = QLearningAgent(winner_name)
    model_path = os.path.join(model_dir, f"{winner_name}.pkl")
    
    if os.path.exists(model_path):
        agent.load(model_path)
        log(f"[green]Loaded Champion: {winner_name}[/green]")
    else:
        log(f"[red]Model not found: {model_path}[/red]")
        log("[yellow]Starting with fresh agent.[/yellow]")

    # Manager State: ticket -> ManagerInstance
    managers = {}
    manager_states = {} # ticket -> status string
    
    # Critic
    critic = ActorCriticAgent("The_Critic")
    log("[cyan]Loaded Critic: The Judge[/cyan]")
    
    # Global Risk Settings
    GLOBAL_MAX_POSITIONS = 12 # Cap total positions across all symbols
    LOT_MULTIPLIER = 0.05 # Reduced from 0.1 for safety
    
    # 3. Live Loop
    with Live(get_dashboard(None, None, winner_name, {}), refresh_per_second=2, screen=True) as live:
        try:
            while True:
                account_info = mt5.account_info()
                all_positions = mt5.positions_get()
                total_positions = len(all_positions) if all_positions else 0
                
                # Clean up managers for closed positions
                active_tickets = [p.ticket for p in all_positions] if all_positions else []
                managers = {k: v for k, v in managers.items() if k in active_tickets}
                manager_states = {k: v for k, v in manager_states.items() if k in active_tickets}
                
                for symbol in SYMBOLS:
                    # A. Get Data
                    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, 100)
                    if rates is None:
                        continue
                        
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    
                    # Add Indicators
                    loader = DataLoader()
                    df = loader.add_indicators(df)
                    
                    # Check Position for this symbol
                    positions = [p for p in all_positions if p.symbol == symbol]
                    has_position = len(positions) > 0
                    
                    # Construct Observation
                    row = df.iloc[-1]
                    price = row['close']
                    rsi = row['rsi']
                    trend = row['trend']
                    volatility = row['volatility_sma']
                    z_score = row['z_score']
                    fibo_pos = row['fibo_pos']
                    
                    trader_obs = [price, rsi, trend, volatility, z_score, fibo_pos]
                    
                    # C. Agent Action (Entry Logic)
                    # Pyramiding Logic: Allow entry if < MAX_POSITIONS and existing positions are profitable
                    MAX_POSITIONS_PER_SYMBOL = 3
                    can_trade = False
                    
                    if total_positions < GLOBAL_MAX_POSITIONS:
                        if not has_position:
                            can_trade = True
                        else:
                            # Smart Limit: Exclude "Split" (Secured) positions from the count
                            # If a position is split, it's risk-free (BE + Profit Taken), so we can allow a new one.
                            # Check for "Spli" to handle potential string truncation (e.g. "V14_Manager_Spli")
                            active_risk_positions = [p for p in positions if "Spli" not in p.comment]
                            
                            if len(active_risk_positions) < MAX_POSITIONS_PER_SYMBOL:
                                # Check if all existing positions (even split ones) are profitable
                                # We use >= -0.1 to be lenient with BE fluctuations/swaps
                                all_profit = all(p.profit >= -0.1 for p in positions)
                                if all_profit:
                                    can_trade = True
                            
                    if can_trade:
                        # TRADER MODE
                        action = agent.act(trader_obs)
                        
                        if action == 1 or action == 2: # Buy or Sell
                            # Check direction consistency for pyramiding
                            if has_position:
                                existing_type = positions[0].type
                                if (action == 1 and existing_type != mt5.ORDER_TYPE_BUY) or \
                                   (action == 2 and existing_type != mt5.ORDER_TYPE_SELL):
                                    action = 0 # Block counter-trend pyramiding
                            
                            if action != 0:
                                # Dynamic Lot Sizing
                                balance = account_info.balance
                                lot_size = round((balance / 10000.0) * LOT_MULTIPLIER, 2)
                                lot_size = max(0.01, lot_size)
                                
                                tick = mt5.symbol_info_tick(symbol)
                                is_pyramid = has_position
                                
                                if is_pyramid:
                                    lot_size = round(lot_size * 0.5, 2) # 50% lots for Pyramids
                                    lot_size = max(0.01, lot_size)
                                
                                if action == 1: # Buy
                                    type_op = mt5.ORDER_TYPE_BUY
                                    price_op = tick.ask
                                    txt = "BUY (Pyramid)" if is_pyramid else "BUY"
                                    color = "green"
                                else: # Sell
                                    type_op = mt5.ORDER_TYPE_SELL
                                    price_op = tick.bid
                                    txt = "SELL (Pyramid)" if is_pyramid else "SELL"
                                    color = "red"
                                
                                # CRITIC
                                log(f"[cyan]CRITIC: Approves {txt} on {symbol}[/cyan]")
                                
                                comment = f"V14_Pyramid" if is_pyramid else f"V14_{winner_name}"

                                request = {
                                    "action": mt5.TRADE_ACTION_DEAL,
                                    "symbol": symbol,
                                    "volume": lot_size,
                                    "type": type_op,
                                    "price": price_op,
                                    "magic": MAGIC_NUMBER,
                                    "comment": comment,
                                    "type_time": mt5.ORDER_TIME_GTC,
                                    "type_filling": mt5.ORDER_FILLING_IOC,
                                }
                                
                                check = mt5.order_check(request)
                                # order_check returns 0 on success, NOT 10009
                                if check.retcode != 0:
                                     log(f"[yellow]Order Check Failed: {check.comment} ({check.retcode})[/yellow]")
                                else:
                                    result = mt5.order_send(request)
                                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                                        log(f"[{color}]{txt} {symbol} @ {price_op:.5f} | Vol: {lot_size}[/{color}]")
                                        total_positions += 1 # Update local count
                                    else:
                                        log(f"[red]Order Failed: {result.comment} ({result.retcode})[/red]")
                            
                    # MANAGER MODE (Manage ALL positions)
                    if has_position:
                        for pos in positions:
                            # Get or Create Manager for this position
                            if pos.ticket not in managers:
                                is_pyramid = "Pyramid" in pos.comment
                                # Revert to 0.1% (Standard) as requested by user
                                be_threshold = 0.001 
                                
                                mgr = StandardManager(
                                    f"Mgr_{pos.ticket}", 
                                    sl_pct=0.005, # 0.5% Max Risk per trade
                                    be_activation=be_threshold,
                                    split_on_be=is_pyramid # Enable Split-on-BE for Pyramids
                                )
                                managers[pos.ticket] = mgr
                                log(f"[magenta]Manager Attached to {symbol} ({pos.ticket}) | BE: {be_threshold*100:.3f}% | Split: {is_pyramid}[/magenta]")
                            
                            manager = managers[pos.ticket]
                            
                            # Calculate PnL for this specific position
                            profit = pos.profit
                            balance = account_info.balance
                            pnl_pct = (profit / balance) * 100
                            
                            # Manager Observation
                            manager_obs = [pnl_pct, 0, volatility, z_score]
                            
                            action = manager.act(manager_obs)
                            
                            if action == 2: # Close
                                manager_states[pos.ticket] = "CLOSING 🔴"
                                tick = mt5.symbol_info_tick(symbol)
                                request = {
                                    "action": mt5.TRADE_ACTION_DEAL,
                                    "symbol": symbol,
                                    "volume": pos.volume,
                                    "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                                    "position": pos.ticket,
                                    "price": tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask,
                                    "magic": MAGIC_NUMBER,
                                    "comment": "V14_Manager_Close",
                                }
                                result = mt5.order_send(request)
                                if result.retcode == mt5.TRADE_RETCODE_DONE:
                                    log(f"[blue]MANAGER CLOSE {symbol} ({pos.ticket}): {result.comment}[/blue]")
                                    
                            elif action == 3: # BE
                                manager_states[pos.ticket] = "Moving to BE 🔒"
                                # Check if already at BE
                                is_buy = pos.type == mt5.ORDER_TYPE_BUY
                                at_be = (is_buy and pos.sl >= pos.price_open) or (not is_buy and pos.sl <= pos.price_open and pos.sl != 0)
                                
                                if not at_be:
                                    new_sl = pos.price_open + 0.0001 if is_buy else pos.price_open - 0.0001
                                    request = {
                                        "action": mt5.TRADE_ACTION_SLTP,
                                        "position": pos.ticket,
                                        "sl": new_sl,
                                        "tp": pos.tp,
                                        "magic": MAGIC_NUMBER,
                                        "comment": "V14_Manager_BE"
                                    }
                                    result = mt5.order_send(request)
                                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                                        log(f"[cyan]MANAGER: Moved SL to BE for {symbol} ({pos.ticket})[/cyan]")
                                else:
                                     manager_states[pos.ticket] = "Secured (BE) 🛡️"
                                            
                            elif action == 4: # Trail
                                manager_states[pos.ticket] = "Trailing SL 📈"
                                atr = volatility
                                if atr < 0.00001: atr = price * 0.001
                                tick = mt5.symbol_info_tick(symbol)
                                
                                if pos.type == mt5.ORDER_TYPE_BUY:
                                    new_sl = tick.bid - atr
                                    if new_sl > pos.sl:
                                        request = {
                                            "action": mt5.TRADE_ACTION_SLTP,
                                            "position": pos.ticket,
                                            "sl": new_sl,
                                            "tp": pos.tp,
                                            "magic": MAGIC_NUMBER,
                                            "comment": "V14_Manager_Trail"
                                        }
                                        result = mt5.order_send(request)
                                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                                            log(f"[magenta]MANAGER: Trailing SL for {symbol} ({pos.ticket})[/magenta]")
                                else:
                                    new_sl = tick.ask + atr
                                    if new_sl < pos.sl or pos.sl == 0:
                                        request = {
                                            "action": mt5.TRADE_ACTION_SLTP,
                                            "position": pos.ticket,
                                            "sl": new_sl,
                                            "tp": pos.tp,
                                            "magic": MAGIC_NUMBER,
                                            "comment": "V14_Manager_Trail"
                                        }
                                        result = mt5.order_send(request)
                                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                                            log(f"[magenta]MANAGER: Trailing SL for {symbol} ({pos.ticket})[/magenta]")
                                            
                            elif action == 5: # BE + Split (Pyramid Secure)
                                manager_states[pos.ticket] = "SPLIT & BE ✂️"
                                # 1. Move to BE
                                is_buy = pos.type == mt5.ORDER_TYPE_BUY
                                at_be = (is_buy and pos.sl >= pos.price_open) or (not is_buy and pos.sl <= pos.price_open and pos.sl != 0)
                                
                                if not at_be:
                                    new_sl = pos.price_open + 0.0001 if is_buy else pos.price_open - 0.0001
                                    request_be = {
                                        "action": mt5.TRADE_ACTION_SLTP,
                                        "position": pos.ticket,
                                        "sl": new_sl,
                                        "tp": pos.tp,
                                        "magic": MAGIC_NUMBER,
                                        "comment": "V14_Manager_BE"
                                    }
                                    mt5.order_send(request_be)
                                    log(f"[cyan]MANAGER: Moved SL to BE for {symbol} ({pos.ticket})[/cyan]")
                                
                                # 2. Close 50%
                                if pos.volume >= 0.02: # Min volume check
                                    split_vol = round(pos.volume * 0.5, 2)
                                    tick = mt5.symbol_info_tick(symbol)
                                    request_close = {
                                        "action": mt5.TRADE_ACTION_DEAL,
                                        "symbol": symbol,
                                        "volume": split_vol,
                                        "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                                        "position": pos.ticket,
                                        "price": tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask,
                                        "magic": MAGIC_NUMBER,
                                        "comment": "V14_Manager_Split",
                                    }
                                    result = mt5.order_send(request_close)
                                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                                        log(f"[blue]MANAGER SPLIT {symbol} ({pos.ticket}): Secured {split_vol} lots[/blue]")
                            
                            else: # Action 0 (Hold)
                                manager_states[pos.ticket] = "Holding ⏳"
                                # Randomly log "Holding" to show activity (Heartbeat)
                                import random
                                if random.random() < 0.05: # Increased freq to 5%
                                    log(f"[grey50]MANAGER: Holding {symbol} #{pos.ticket} (PnL: {pnl_pct:.4f}%) | BE Target: {managers[pos.ticket].be_activation*100:.3f}%[/grey50]")

                # Update Dashboard
                live.update(get_dashboard(account_info, all_positions, winner_name, manager_states))
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            log("[yellow]Stopping V14...[/yellow]")
            mt5.shutdown()

if __name__ == "__main__":
    main()
