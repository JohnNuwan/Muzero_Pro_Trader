import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import json
import os
import sys
import argparse
from datetime import datetime, timedelta
from scipy.optimize import differential_evolution
from rich.console import Console

console = Console()

# CONFIG
LOGIN = 1512112659
PASSWORD = "8Ee7B$z54"
SERVER = "FTMO-Demo"
TIMEFRAME = mt5.TIMEFRAME_M15
CONFIG_FILE = "gemini_config.json"

# PARAMETER BOUNDS (Genes)
# [RSI_Period, MFI_Period, ADX_Period, Z_Score_Thresh, SL_Mult, TP_Mult]
BOUNDS = [
    (7, 25),    # RSI Period
    (7, 25),    # MFI Period
    (7, 25),    # ADX Period
    (1.5, 4.0), # Z-Score Threshold
    (1.0, 3.5), # SL Multiplier
    (1.5, 6.0)  # TP Multiplier
]

def connect_mt5():
    if not mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER):
        console.print("[red]MT5 Init Failed[/red]")
        return False
    return True

def get_data(symbol, days=30):
    """Fetch M15 data for the last N days"""
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, days * 96) # 96 candles per day (M15)
    if rates is None or len(rates) == 0: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def calculate_indicators(df, rsi_p, mfi_p, adx_p):
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=int(rsi_p)).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=int(rsi_p)).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MFI
    typ_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typ_price * df['tick_volume']
    pos_flow = money_flow.where(typ_price > typ_price.shift(), 0).rolling(window=int(mfi_p)).sum()
    neg_flow = money_flow.where(typ_price < typ_price.shift(), 0).rolling(window=int(mfi_p)).sum()
    df['MFI'] = 100 - (100 / (1 + (pos_flow / neg_flow.replace(0, 1))))
    
    # ADX
    high_diff = df['high'].diff()
    low_diff = df['low'].diff()
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=int(adx_p)).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=int(adx_p)).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=int(adx_p)).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['ADX'] = dx.rolling(window=int(adx_p)).mean()
    
    # Z-Score (Fixed 20 for base)
    df['Z_Score'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
    
    # ATR (for SL/TP)
    df['ATR'] = atr
    
    return df.dropna()

def backtest(params, df):
    """Fast Vectorized Backtest / Event Loop"""
    rsi_p, mfi_p, adx_p, z_thresh, sl_mult, tp_mult = params
    
    sim_df = df.copy()
    sim_df = calculate_indicators(sim_df, rsi_p, mfi_p, adx_p)
    
    balance = 10000
    equity = 10000
    position = None # {'type': 0/1, 'price': float, 'sl': float, 'tp': float}
    
    wins = 0
    losses = 0
    
    # Simple Strategy Simulation (Proxy for ML Model)
    # We assume a trade is taken if indicators align (Trend Following + Mean Reversion)
    
    for i in range(len(sim_df)):
        row = sim_df.iloc[i]
        
        # Check Exit
        if position:
            if position['type'] == 0: # BUY
                if row['low'] <= position['sl']:
                    balance -= (position['price'] - position['sl']) # Loss
                    losses += 1
                    position = None
                elif row['high'] >= position['tp']:
                    balance += (position['tp'] - position['price']) # Win
                    wins += 1
                    position = None
            else: # SELL
                if row['high'] >= position['sl']:
                    balance -= (position['sl'] - position['price']) # Loss
                    losses += 1
                    position = None
                elif row['low'] <= position['tp']:
                    balance += (position['price'] - position['tp']) # Win
                    wins += 1
                    position = None
                    
        # Check Entry (if no position)
        if not position:
            # Trend Logic (ADX > 25)
            if row['ADX'] > 25:
                if row['RSI'] > 50 and row['MFI'] > 50: # Bullish
                    sl = row['close'] - (row['ATR'] * sl_mult)
                    tp = row['close'] + (row['ATR'] * tp_mult)
                    position = {'type': 0, 'price': row['close'], 'sl': sl, 'tp': tp}
                elif row['RSI'] < 50 and row['MFI'] < 50: # Bearish
                    sl = row['close'] + (row['ATR'] * sl_mult)
                    tp = row['close'] - (row['ATR'] * tp_mult)
                    position = {'type': 1, 'price': row['close'], 'sl': sl, 'tp': tp}
            
            # Mean Reversion Logic (Z-Score)
            elif abs(row['Z_Score']) > z_thresh:
                if row['Z_Score'] < -z_thresh: # Oversold -> Buy
                    sl = row['close'] - (row['ATR'] * sl_mult)
                    tp = row['close'] + (row['ATR'] * tp_mult)
                    position = {'type': 0, 'price': row['close'], 'sl': sl, 'tp': tp}
                elif row['Z_Score'] > z_thresh: # Overbought -> Sell
                    sl = row['close'] + (row['ATR'] * sl_mult)
                    tp = row['close'] - (row['ATR'] * tp_mult)
                    position = {'type': 1, 'price': row['close'], 'sl': sl, 'tp': tp}

    return -balance # Minimize negative balance (Maximize balance)

def evolve(symbol):
    console.print(f"[bold cyan]🧬 EVOLVING {symbol}...[/bold cyan]")
    
    if not connect_mt5(): return
    
    df = get_data(symbol)
    if df is None:
        console.print("[red]No Data[/red]")
        return

    # Run Differential Evolution
    result = differential_evolution(backtest, BOUNDS, args=(df,), strategy='best1bin', maxiter=5, popsize=10)
    
    best_params = result.x
    console.print(f"[green]✅ EVOLUTION COMPLETE![/green]")
    console.print(f"Best Params: {best_params}")
    
    # Update Config
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            
        if symbol not in config: config[symbol] = {}
        
        config[symbol]["rsi_period"] = int(best_params[0])
        config[symbol]["mfi_period"] = int(best_params[1])
        config[symbol]["adx_period"] = int(best_params[2])
        config[symbol]["z_score_threshold"] = float(best_params[3])
        config[symbol]["sl_multiplier"] = float(best_params[4])
        config[symbol]["tp_multiplier"] = float(best_params[5])
        
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
            
        console.print(f"[blue]💾 Config Updated for {symbol}[/blue]")
        
    except Exception as e:
        console.print(f"[red]Config Update Failed: {e}[/red]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol", type=str, help="Symbol to evolve")
    args = parser.parse_args()
    
    evolve(args.symbol)
    equity = 10000
    peak_equity = 10000
    max_dd = 0
    trades = []
    
    # Simple Loop Backtest
    position = None # (type, price, sl, tp)
    
    for i in range(len(sim_df)):
        row = sim_df.iloc[i]
        price = row['close']
        
        # Check Exit
        if position:
            p_type, p_price, p_sl, p_tp = position
            
            if p_type == 1: # BUY
                if row['low'] <= p_sl: # SL Hit
                    loss = (p_sl - p_price)
                    balance += loss
                    position = None
                elif row['high'] >= p_tp: # TP Hit
                    profit = (p_tp - p_price)
                    balance += profit
                    position = None
            else: # SELL
                if row['high'] >= p_sl: # SL Hit
                    loss = (p_price - p_sl)
                    balance += loss
                    position = None
                elif row['low'] <= p_tp: # TP Hit
                    profit = (p_price - p_tp)
                    balance += profit
                    position = None
                    
        # Update Equity & DD
        # (Simplified: Equity = Balance if no pos, else approx)
        if balance > peak_equity: peak_equity = balance
        dd = (peak_equity - balance) / peak_equity
        if dd > max_dd: max_dd = dd
        
        # FTMO Hard Fail
        if max_dd > 0.04: # 4% Safety Limit
            return -1000 # Heavy Penalty
            
        # Check Entry (if no pos)
        if position is None:
            atr = row['ATR']
            if atr == 0 or np.isnan(atr): continue
            
            # Logic V10: Z-Score Reversion + RSI Trend
            # Buy: Z-Score < -Thresh OR (RSI < 30 and Trend Up)
            # Sell: Z-Score > Thresh OR (RSI > 70 and Trend Down)
            
            # Simplified for Evolution:
            # Mean Reversion Focus
            if row['Z_Score'] < -z_thresh:
                sl = price - (atr * sl_mult)
                tp = price + (atr * tp_mult)
                position = (1, price, sl, tp) # 1 = Buy
                
            elif row['Z_Score'] > z_thresh:
                sl = price + (atr * sl_mult)
                tp = price - (atr * tp_mult)
                position = (-1, price, sl, tp) # -1 = Sell
                
    profit = balance - 10000
    
    # Fitness Function
    # We want High Profit, Low DD.
    # Score = Profit / (MaxDD * 100 + 1)
    
    if max_dd == 0: max_dd = 0.0001
    score = profit / (max_dd * 100)
    
    return score

def objective(params, df):
    """Objective function for Scipy (Minimize negative score)"""
    try:
        score = backtest(params, df)
        return -score # Minimize negative = Maximize positive
    except Exception:
        return 0

def evolve(symbol):
    console.print(f"🧬 [bold magenta]DARWIN EVOLUTION STARTING: {symbol}[/bold magenta]")
    
    if not connect_mt5(): return
    
    df = get_data(symbol, days=10)
    if df is None:
        console.print(f"[red]No data for {symbol}[/red]")
        return
        
    console.print(f"   Running Genetic Algorithm on {len(df)} candles...")
    
    # Run Differential Evolution
    result = differential_evolution(
        objective, 
        BOUNDS, 
        args=(df,), 
        strategy='best1bin', 
        maxiter=10, 
        popsize=10, 
        tol=0.01, 
        mutation=(0.5, 1), 
        recombination=0.7,
        disp=False
    )
    
    best_params = result.x
    best_score = -result.fun
    
    console.print(f"   🏆 [green]Evolution Complete![/green] Score: {best_score:.2f}")
    console.print(f"   🧬 Genes: RSI={int(best_params[0])}, SL={best_params[2]:.2f}, TP={best_params[3]:.2f}, Z={best_params[4]:.2f}")
    
    # Update Config
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            
        if symbol not in config: config[symbol] = {}
        
        config[symbol]['rsi_period'] = int(best_params[0])
        config[symbol]['mfi_period'] = int(best_params[1])
        config[symbol]['sl_mult'] = round(best_params[2], 2)
        config[symbol]['tp_mult'] = round(best_params[3], 2)
        config[symbol]['z_score_threshold'] = round(best_params[4], 2)
        
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
            
        console.print(f"   💾 [cyan]Config Updated for {symbol}[/cyan]")
        
    except Exception as e:
        console.print(f"[red]Config Update Error: {e}[/red]")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        evolve(sys.argv[1])
    else:
        console.print("Usage: python train_evolution.py SYMBOL")
