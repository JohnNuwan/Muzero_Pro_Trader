import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os
import time
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
import csv

console = Console()

# --- ⚙️ CONFIGURATION DU QUARTIER GÉNÉRAL (V7) ---
LOGIN = 1512112659 
PASSWORD = "8Ee7B$z54"
SERVER = "FTMO-Demo"

# Liste des combattants
SYMBOLS = ["XAUUSD", "US30.cash", "US100.cash", "GER40.cash", "EURUSD", "US500.cash"] 

# Lot de base (Ajustement Dynamique V2)
BASE_LOTS = {
    "XAUUSD": 0.1,
    "EURUSD": 0.1,
    "US30.cash": 0.2,
    "US100.cash": 0.2,
    "GER40.cash": 0.2,
    "US500.cash": 0.5
}

# Paramètres FTMO & IA
MAX_SPREAD_POINTS = 300 
TIMEFRAME = mt5.TIMEFRAME_M15
LOOKBACK = 10 
MEMORY_FOLDER = "gemini_memory"
HISTORY_FILE = f"{MEMORY_FOLDER}/trade_history.csv"
CONSECUTIVE_LOSS_LIMIT = 3 

# --- VARIABLES GLOBALES ---
if not os.path.exists(MEMORY_FOLDER): os.makedirs(MEMORY_FOLDER)
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "symbol", "features", "prediction", "confidence", "outcome", "profit"])

loss_streak = {sym: 0 for sym in SYMBOLS}
ban_until = {sym: 0 for sym in SYMBOLS}
active_trades = {} # Pour suivre les trades ouverts et gérer le trailing stop

def connect_mt5():
    if not mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER):
        console.print(Panel.fit("❌ ÉCHEC CRITIQUE : Connexion MT5 impossible", style="bold red"))
        quit()
    account_info = mt5.account_info()
    console.print(Panel.fit(
        f"🤖 [bold cyan]GEMINI V7 ULTIMATE CONNECTÉ[/bold cyan]\n"
        f"Compte: [yellow]{account_info.login}[/yellow] | "
        f"Balance: [green]{account_info.balance:.2f} USD[/green]\n"
        f"Mode: [magenta]HYBRID ENSEMBLE (GBM + MLP)[/magenta]",
        title="[bold green]Système Actif[/bold green]",
        border_style="green"
    ))

# Variables pour limiter l'affichage des messages
_last_spread_warning = {}

def check_environment(symbol):
    """Vérifie si le terrain est sûr (Spread, Heure, Tilt)"""
    if time.time() < ban_until[symbol]:
        return False 

    tick = mt5.symbol_info_tick(symbol)
    if tick is None: return False
        
    point = mt5.symbol_info(symbol).point
    if point is None or point == 0: return False
        
    spread_points = (tick.ask - tick.bid) / point
    limit = MAX_SPREAD_POINTS if "US" in symbol else 50 
    
    if spread_points > limit:
        current_time = time.time()
        if symbol not in _last_spread_warning or (current_time - _last_spread_warning[symbol]) > 60:
            console.print(f"⚠️ [yellow]{symbol}[/yellow]: Spread trop élevé ([red]{spread_points:.0f} pts[/red]). News possible. Pas de trade.")
            _last_spread_warning[symbol] = current_time
        return False
        
    return True

# --- FEATURE ENGINEERING AVANCÉ ---
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_macd(close, fast=12, slow=26, signal=9):
    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line # Histogramme

def calculate_bollinger(close, period=20, std_dev=2):
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    # Position relative dans les bandes (0 = lower, 1 = upper)
    return (close - lower) / (upper - lower)

def get_data(symbol, n_candles=5000, verbose=False):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, n_candles)
    if rates is None or len(rates) == 0: return pd.DataFrame()
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Indicateurs de base
    df['RSI'] = calculate_rsi(df['close'])
    df['ATR'] = calculate_atr(df['high'], df['low'], df['close'])
    
    # Indicateurs Avancés (V7)
    df['MACD_Hist'] = calculate_macd(df['close'])
    df['BB_Pos'] = calculate_bollinger(df['close'])
    
    # Lagged Returns (Momentum à court terme)
    df['Ret_1'] = df['close'].pct_change(1)
    df['Ret_3'] = df['close'].pct_change(3)
    
    # Time Features
    df['Hour'] = df['time'].dt.hour
    
    # Structure (Dow)
    df['last_high'] = df['high'].rolling(window=20, min_periods=1).max()
    df['last_low'] = df['low'].rolling(window=20, min_periods=1).min()
    
    # Pattern Géométrique
    diff = df['last_high'] - df['last_low']
    atr_safe = df['ATR'].replace(0, np.nan).fillna(df['close'] * 0.001)
    df['Dist_Fib'] = (df['close'] - (df['last_low'] + diff * 0.618)) / atr_safe
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Colonnes requises pour le modèle
    cols_required = ['RSI', 'ATR', 'MACD_Hist', 'BB_Pos', 'Ret_1', 'Ret_3', 'Dist_Fib', 'Hour']
    df.dropna(subset=cols_required, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def prepare_features(df):
    features = []
    labels = []
    cols = ['RSI', 'ATR', 'MACD_Hist', 'BB_Pos', 'Ret_1', 'Ret_3', 'Dist_Fib', 'Hour']
    
    if len(df) < LOOKBACK + 1: return np.array([]), np.array([])
    
    for i in range(LOOKBACK, len(df)):
        # On prend juste la dernière ligne de features pour l'instant (Flattening window is heavy for GBM)
        # Pour V7 on simplifie : on prend les indicateurs à l'instant T
        # Mais pour garder la logique temporelle, on peut prendre T, T-1, T-2
        
        # Simplification V7 : Features actuelles + Features lissées
        current_feats = df[cols].iloc[i].values
        prev_feats = df[cols].iloc[i-1].values
        
        seq = np.concatenate([current_feats, prev_feats]) # Context window = 2
        
        if not np.isnan(seq).any():
            features.append(seq)
            # Target: 1 si le prix monte dans les 3 prochaines bougies (plus robuste)
            future_return = (df['close'].iloc[min(i+3, len(df)-1)] - df['close'].iloc[i])
            labels.append(1 if future_return > 0 else 0)
        
    return np.array(features), np.array(labels)

# --- CERVEAU HYBRIDE & MÉMOIRE ---
def train_hybrid_model(X, y):
    # 1. Gradient Boosting (Rapide, précis sur tabulaire)
    gbm = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    
    # 2. Neural Network (Capture le non-linéaire complexe)
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    
    # 3. Ensemble (Vote pondéré)
    # On donne un peu plus de poids au GBM car souvent plus robuste sur la finance bruitée
    ensemble = VotingClassifier(
        estimators=[('gbm', gbm), ('mlp', mlp)],
        voting='soft',
        weights=[1.5, 1.0] 
    )
    
    ensemble.fit(X, y)
    return ensemble

def manage_memory(symbol):
    m_path = f"{MEMORY_FOLDER}/{symbol}_v7_model.pkl"
    s_path = f"{MEMORY_FOLDER}/{symbol}_v7_scaler.pkl"
    
    # Charger l'historique des trades pour le "Self-Learning"
    # (Dans une version future, on pourrait ré-entraîner avec ces données spécifiques)
    
    if os.path.exists(m_path):
        return joblib.load(m_path), joblib.load(s_path)
    else:
        console.print(f"🧠 [bold cyan][{symbol}][/bold cyan] Entraînement V7 (Hybrid)...")
        
        # Activation symbole
        if not mt5.symbol_select(symbol, True):
            console.print(f"❌ Impossible d'activer {symbol}")
            return None, None
            
        df = get_data(symbol, verbose=True)
        if len(df) == 0: return None, None
        
        X, y = prepare_features(df)
        if len(X) == 0: return None, None
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = train_hybrid_model(X_scaled, y)
        
        joblib.dump(model, m_path)
        joblib.dump(scaler, s_path)
        console.print(f"   💾 Cerveau Hybride sauvegardé pour [cyan]{symbol}[/cyan]")
        return model, scaler

# --- EXÉCUTION & GESTION ---
def smart_lot_size(symbol, confidence):
    base = BASE_LOTS.get(symbol, 0.1)
    # V7: Si confiance extrême (>90%), on double
    if confidence > 0.90: return base * 2.0
    if confidence > 0.75: return base * 1.2
    if confidence > 0.60: return base
    return base * 0.5

def record_trade(symbol, features, prediction, confidence, ticket):
    # On enregistre le trade pour l'analyser plus tard (Self-Learning)
    # On ne connait pas encore le profit, on mettra à jour plus tard ou on log juste l'entrée
    pass # Implémentation simplifiée pour l'instant

def execute_trade(symbol, prediction, df, confidence, scaler, model):
    if len(mt5.positions_get(symbol=symbol)) > 0: return
    
    lot = smart_lot_size(symbol, confidence)
    action = mt5.ORDER_TYPE_BUY if prediction == 1 else mt5.ORDER_TYPE_SELL
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if prediction == 1 else tick.bid
    
    atr = df['ATR'].iloc[-1]
    sl_dist = atr * 1.5 # SL un peu plus large pour laisser respirer
    tp_dist = atr * 2.0 # RR 1:1.33
    
    sl = price - sl_dist if prediction == 1 else price + sl_dist
    tp = price + tp_dist if prediction == 1 else price - tp_dist
    
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(f"{lot:.2f}"),
        "type": action,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": 777777, # Magic V7
        "comment": f"Gemini V7 ({confidence:.2f})",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    res = mt5.order_send(req)
    if res.retcode == mt5.TRADE_RETCODE_DONE:
        action_text = "🟢 BUY" if action == mt5.ORDER_TYPE_BUY else "🔴 SELL"
        console.print(Panel.fit(
            f"{action_text} [bold]{symbol}[/bold]\n"
            f"Confiance: [yellow]{confidence:.2%}[/yellow] | "
            f"Lots: [cyan]{lot:.2f}[/cyan]\n"
            f"SL: {sl:.2f} | TP: {tp:.2f}",
            title="[bold green]🚀 TRADE V7 EXÉCUTÉ[/bold green]",
            border_style="green"
        ))
        # Sauvegarde pour Self-Learning
        # features_str = str(list(features))
        # with open(HISTORY_FILE, 'a', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([datetime.now(), symbol, "features_saved", prediction, confidence, "OPEN", 0])
    else:
        console.print(f"❌ Erreur ordre {symbol}: {res.comment}")

def manage_positions():
    """Trailing Stop & Breakeven"""
    positions = mt5.positions_get()
    if positions is None: return
    
    for pos in positions:
        if pos.magic != 777777: continue # On ne touche qu'aux trades V7
        
        symbol = pos.symbol
        point = mt5.symbol_info(symbol).point
        current_price = mt5.symbol_info_tick(symbol).bid if pos.type == 0 else mt5.symbol_info_tick(symbol).ask
        
        # Calcul profit en points
        if pos.type == 0: # BUY
            profit_points = (current_price - pos.price_open) / point
            new_sl = current_price - (100 * point) # Trailing à 100 points
            if new_sl > pos.sl and profit_points > 150: # Sécurisation après 150 points
                req = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "sl": new_sl,
                    "tp": pos.tp,
                    "position": pos.ticket
                }
                mt5.order_send(req)
        else: # SELL
            profit_points = (pos.price_open - current_price) / point
            new_sl = current_price + (100 * point)
            if (new_sl < pos.sl or pos.sl == 0) and profit_points > 150:
                req = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "sl": new_sl,
                    "tp": pos.tp,
                    "position": pos.ticket
                }
                mt5.order_send(req)

# --- MAIN ---
if __name__ == "__main__":
    connect_mt5()
    models = {}
    scalers = {}
    
    # Chargement / Entraînement
    for sym in SYMBOLS:
        m, s = manage_memory(sym)
        if m:
            models[sym] = m
            scalers[sym] = s
        
    console.print("\n")
    console.print(Panel.fit(
        "🥊 [bold cyan]GEMINI V7 - THE CHALLENGER[/bold cyan]\n"
        "[bold green]PRÊT À DÉTRÔNER LA CONCURRENCE[/bold green]",
        title="[bold yellow]ULTIMATE MODE[/bold yellow]",
        border_style="yellow"
    ))
    
    iteration = 0
    while True:
        iteration += 1
        print(f"\rCycle #{iteration} | {datetime.now().strftime('%H:%M:%S')} | Surveillance...", end="")
        
        # Gestion des positions existantes (Trailing Stop)
        manage_positions()
        
        # Affichage "Live Feed"
        table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
        table.add_column("Symbole", style="cyan")
        table.add_column("Prix", justify="right")
        table.add_column("RSI / MACD", justify="center")
        table.add_column("IA Confiance", justify="right")
        table.add_column("Décision", style="bold")

        for sym in SYMBOLS:
            if sym not in models: continue
            if not check_environment(sym): 
                table.add_row(sym, "-", "-", "-", "[yellow]Sleep (Spread/Time)[/yellow]")
                continue
            
            try:
                # Check position
                positions = mt5.positions_get(symbol=sym)
                has_position = positions is not None and len(positions) > 0
                
                # Analyse
                df = get_data(sym, n_candles=200) 
                if len(df) == 0: 
                    table.add_row(sym, "-", "-", "-", "[dim]No Data[/dim]")
                    continue
                
                X_new, _ = prepare_features(df)
                if len(X_new) == 0: 
                    table.add_row(sym, "-", "-", "-", "[dim]No Features[/dim]")
                    continue
                
                # Prédiction
                feat = scalers[sym].transform(X_new[-1].reshape(1, -1))
                pred = models[sym].predict(feat)[0]
                conf = models[sym].predict_proba(feat)[0][pred]
                
                # Données pour l'affichage
                last_close = df['close'].iloc[-1]
                rsi = df['RSI'].iloc[-1]
                macd = df['MACD_Hist'].iloc[-1]
                macd_color = "[green]" if macd > 0 else "[red]"
                rsi_color = "[red]" if rsi > 70 else ("[green]" if rsi < 30 else "[white]")
                
                direction = "🟢 BUY" if pred == 1 else "🔴 SELL"
                conf_str = f"{conf:.1%}"
                conf_style = "[green]" if conf > 0.60 else ("[yellow]" if conf > 0.55 else "[dim]")
                
                status = ""
                if has_position:
                    pos = positions[0]
                    p_profit = pos.profit
                    p_color = "[green]" if p_profit >= 0 else "[red]"
                    status = f"In Trade ({p_color}{p_profit:.2f}$[/{p_color}])"
                elif conf > 0.60:
                    status = f"[bold green]🚀 EXECUTE {direction}[/bold green]"
                    execute_trade(sym, pred, df, conf, scalers[sym], models[sym])
                else:
                    status = f"[dim]Wait ({direction})[/dim]"

                table.add_row(
                    sym, 
                    f"{last_close:.2f}", 
                    f"{rsi_color}{rsi:.1f}[/{rsi_color}] / {macd_color}{macd:.5f}[/{macd_color}]",
                    f"{conf_style}{conf_str}[/{conf_style}]",
                    status
                )
                    
            except Exception as e:
                table.add_row(sym, "ERROR", str(e), "-", "-")
        
        console.print(table)
        console.print(f"[dim]Prochain scan dans 10s...[/dim]")
        time.sleep(10)
