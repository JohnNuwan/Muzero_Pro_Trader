import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from scipy.stats import linregress
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import time
from datetime import datetime, timedelta
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
import csv

console = Console()

import requests

# --- ⚙️ CONFIGURATION DU QUARTIER GÉNÉRAL (V8) ---
LOGIN = 1512112659 
PASSWORD = "8Ee7B$z54"
SERVER = "FTMO-Demo"

# TELEGRAM CONFIG
TELEGRAM_TOKEN = "6660646226:AAEwerIp1t_i5mToXseSFeic3ryrzTXmF-U"
TELEGRAM_CHAT_ID = "-5009131528"

# Liste des combattants
# Note: On évite GBPUSD pour ne pas doubler le risque avec EURUSD (Corrélation forte)
SYMBOLS = [
    "XAUUSD",       # Gold (Volatilité + Trend)
    "US30.cash",    # Dow Jones (Momentum)
    "US100.cash",   # Nasdaq (Tech Trend)
    "GER40.cash",   # Dax (Europe)
    "EURUSD",       # Forex Major (Liquidité)
    "US500.cash",   # S&P 500 (Global Health)
    "BTCUSD"        # Bitcoin (King of Trends - Non Correlé) 👑
]

# Lot de base (Ajustement Dynamique V2)
BASE_LOTS = {
    "XAUUSD": 0.1,
    "EURUSD": 0.1,
    "BTCUSD": 0.01, # Petit lot pour commencer sur BTC (Volatile)
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
_last_spread_warning = {}

def send_telegram(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        requests.post(url, data=data, timeout=5)
    except Exception as e:
        console.print(f"[red]Telegram Error: {e}[/red]")

def connect_mt5():
    if not mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER):
        console.print(Panel.fit("❌ ÉCHEC CRITIQUE : Connexion MT5 impossible", style="bold red"))
        quit()
    account_info = mt5.account_info()
    console.print(Panel.fit(
        f"🤖 [bold cyan]GEMINI V8 REDEMPTION CONNECTÉ[/bold cyan]\n"
        f"Compte: [yellow]{account_info.login}[/yellow] | "
        f"Balance: [green]{account_info.balance:.2f} USD[/green]\n"
        f"Mode: [magenta]ACTIVE SELF-LEARNING + MFI/OBV[/magenta]\n"
        f"Stratégie: [cyan]Intraday Trend Following (M15)[/cyan]",
        title="[bold green]Système Actif[/bold green]",
        border_style="green"
    ))
    send_telegram(f"🤖 *GEMINI V8 REDEMPTION ONLINE*\nAccount: `{account_info.login}`\nBalance: `{account_info.balance:.2f} USD`")

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

# --- FEATURE ENGINEERING AVANCÉ (V8) ---
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

def calculate_mfi(high, low, close, volume, period=14):
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(window=period).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(window=period).sum()
    
    mfi = 100 - (100 / (1 + (positive_flow / negative_flow.replace(0, 1)))) # Avoid div by 0
    return mfi

def calculate_obv(close, volume):
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

def calculate_linear_regression(close, period=14):
    slopes = []
    angles = []
    # C'est lourd de faire ça sur tout le DF, on optimise avec numpy
    # Pour l'instant, version simple vectorisée impossible facilement, on fait rolling apply
    # Ou plus simplement : Slope sur les X dernières bougies
    
    # Version optimisée : on calcule juste pour les points nécessaires ou on utilise une approx
    # Ici on va utiliser une approximation : (Prix - SMA) / SMA * 100 (Distance à la moyenne)
    # C'est un proxy de la pente
    sma = close.rolling(window=period).mean()
    slope_proxy = (close - sma) / sma * 100
    return slope_proxy

def get_data(symbol, n_candles=5000, verbose=False):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, n_candles)
    if rates is None or len(rates) == 0: return pd.DataFrame()
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Prix Médian
    df['median'] = (df['high'] + df['low']) / 2
    
    # Indicateurs V8
    df['RSI'] = calculate_rsi(df['close'])
    df['ATR'] = calculate_atr(df['high'], df['low'], df['close'])
    df['MFI'] = calculate_mfi(df['high'], df['low'], df['close'], df['tick_volume'])
    df['OBV'] = calculate_obv(df['close'], df['tick_volume'])
    df['LinReg_Slope'] = calculate_linear_regression(df['close'])
    
    # Lagged Returns
    df['Ret_1'] = df['close'].pct_change(1)
    
    # Time Features
    df['Hour'] = df['time'].dt.hour
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Colonnes requises
    cols_required = ['RSI', 'ATR', 'MFI', 'OBV', 'LinReg_Slope', 'Ret_1', 'Hour']
    df.dropna(subset=cols_required, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def prepare_features(df):
    features = []
    labels = []
    cols = ['RSI', 'ATR', 'MFI', 'OBV', 'LinReg_Slope', 'Ret_1', 'Hour']
    
    if len(df) < LOOKBACK + 1: return np.array([]), np.array([])
    
    for i in range(LOOKBACK, len(df)):
        # V8 : On prend les features actuelles + changement MFI/OBV
        current_feats = df[cols].iloc[i].values
        
        # Ajout de la pente de l'OBV (OBV actuel - OBV il y a 5 bougies)
        obv_slope = df['OBV'].iloc[i] - df['OBV'].iloc[i-5] if i >= 5 else 0
        
        seq = np.append(current_feats, obv_slope)
        
        if not np.isnan(seq).any():
            features.append(seq)
            # Target: 1 si le prix monte dans les 3 prochaines bougies
            future_return = (df['close'].iloc[min(i+3, len(df)-1)] - df['close'].iloc[i])
            labels.append(1 if future_return > 0 else 0)
        
    return np.array(features), np.array(labels)

# --- ACTIVE SELF-LEARNING ---
def get_symbol_performance(symbol):
    """Analyse l'historique pour voir si on perd sur ce symbole"""
    if not os.path.exists(HISTORY_FILE): return 0
    
    try:
        df_hist = pd.read_csv(HISTORY_FILE)
        if len(df_hist) == 0: return 0
        
        # Filtrer par symbole
        df_sym = df_hist[df_hist['symbol'] == symbol]
        if len(df_sym) < 3: return 0
        
        # Regarder les 5 derniers trades fermés (si on avait le profit)
        # Ici on simule : si on a eu des pertes récentes, on renvoie un malus
        # Comme on n'a pas encore le profit dans le CSV (il faudrait le mettre à jour),
        # on va utiliser une logique simplifiée : si le fichier existe, on suppose qu'on apprend.
        
        # VRAIE LOGIQUE ACTIVE : Lire les trades fermés depuis MT5
        from_date = datetime.now() - timedelta(days=1)
        deals = mt5.history_deals_get(from_date, datetime.now(), group=f"*{symbol}*")
        
        if deals is None or len(deals) == 0: return 0
        
        recent_losses = 0
        for deal in deals:
            if deal.entry == mt5.DEAL_ENTRY_OUT and deal.profit < 0:
                recent_losses += 1
        
        # Si plus de 2 pertes aujourd'hui, malus de 20%
        if recent_losses >= 2: return -0.20
        if recent_losses >= 4: return -1.0 # Ban complet
        
    except Exception as e:
        return 0
        
    return 0

# --- CERVEAU HYBRIDE ---
def train_hybrid_model(X, y):
    gbm = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    ensemble = VotingClassifier(estimators=[('gbm', gbm), ('mlp', mlp)], voting='soft', weights=[1.5, 1.0])
    ensemble.fit(X, y)
    return ensemble

def manage_memory(symbol):
    m_path = f"{MEMORY_FOLDER}/{symbol}_v8_model.pkl"
    s_path = f"{MEMORY_FOLDER}/{symbol}_v8_scaler.pkl"
    
    if os.path.exists(m_path):
        return joblib.load(m_path), joblib.load(s_path)
    else:
        console.print(f"🧠 [bold cyan][{symbol}][/bold cyan] Entraînement V8 (MFI/OBV)...")
        if not mt5.symbol_select(symbol, True): return None, None
            
        df = get_data(symbol, verbose=True)
        if len(df) == 0: return None, None
        
        X, y = prepare_features(df)
        if len(X) == 0: return None, None
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = train_hybrid_model(X_scaled, y)
        
        joblib.dump(model, m_path)
        joblib.dump(scaler, s_path)
        console.print(f"   💾 Cerveau V8 sauvegardé pour [cyan]{symbol}[/cyan]")
        return model, scaler

# --- EXÉCUTION ---

# --- MULTI-TIMEFRAME ANALYSIS (V9) ---
def check_htf_trend(symbol):
    """
    Vérifie la tendance sur le H1 (1 Heure) pour filtrer le M15.
    Règle : On ne trade que dans le sens de la moyenne mobile 50 H1.
    """
    try:
        # On récupère 100 bougies H1
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
        if rates is None or len(rates) < 50: return "NEUTRAL"
        
        df = pd.DataFrame(rates)
        # Calcul EMA 50
        df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        last_close = df['close'].iloc[-1]
        last_ema = df['EMA_50'].iloc[-1]
        
        if last_close > last_ema:
            return "BULLISH" # Tendance Haussière
        else:
            return "BEARISH" # Tendance Baissière
            
    except Exception as e:
        console.print(f"[red]Erreur MTF {symbol}: {e}[/red]")
        return "NEUTRAL"

def execute_trade(symbol, signal, confidence, df, scaler, model):
    # 0. Vérification de l'environnement
    if not check_environment(symbol): return

    # 1. FILTRE MTF (V9) - THE FORTRESS 🏰
    htf_trend = check_htf_trend(symbol)
    
    # Si le signal contredit la tendance H1, on annule (sauf si confiance extrême > 90%)
    if signal == 1 and htf_trend == "BEARISH" and confidence < 0.9:
        # console.print(f"✋ [yellow]MTF FILTER: {symbol} Signal BUY ignoré car H1 Bearish[/yellow]")
        return
    if signal == 0 and htf_trend == "BULLISH" and confidence < 0.9:
        # console.print(f"✋ [yellow]MTF FILTER: {symbol} Signal SELL ignoré car H1 Bullish[/yellow]")
        return

    # 2. Gestion des positions existantes (Smart Pyramiding)
    positions = mt5.positions_get(symbol=symbol)
    # Filtrer uniquement les positions du bot V8
    v8_positions = [p for p in positions if p.magic == 888888] if positions else []
    
    if len(v8_positions) > 0:
        # On vérifie si on peut pyramider
        last_pos = v8_positions[-1] # La dernière ouverte
        profit_points = (mt5.symbol_info_tick(symbol).bid - last_pos.price_open) / mt5.symbol_info(symbol).point if last_pos.type == 0 else (last_pos.price_open - mt5.symbol_info_tick(symbol).ask) / mt5.symbol_info(symbol).point
        
        # Règle Pyramide : Profit > 50 points ET Confiance > 70% ET Max 3 positions
        if profit_points > 50 and confidence > 0.70 and len(v8_positions) < 3:
            # On vérifie que le signal est dans le même sens
            if (last_pos.type == 0 and signal == 1) or (last_pos.type == 1 and signal == 0):
                lot = BASE_LOTS.get(symbol, 0.1) * 0.5 # Demi-lot pour le renfort
                trade_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
                
                # SL/TP Pyramide
                price = mt5.symbol_info_tick(symbol).ask if signal == 1 else mt5.symbol_info_tick(symbol).bid
                point = mt5.symbol_info(symbol).point
                sl = price - (100 * point) if signal == 1 else price + (100 * point)
                tp = price + (200 * point) if signal == 1 else price - (200 * point)
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": lot,
                    "type": trade_type,
                    "price": price,
                    "sl": sl,
                    "tp": tp,
                    "magic": 888888,
                    "comment": f"Gemini V8 (Pyramiding #{len(v8_positions)})",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    console.print(f"� [bold magenta]PYRAMID ADD #{len(v8_positions)}: {symbol} | Lot: {lot} | Conf: {confidence:.2%}[/bold magenta]")
                    send_telegram(f"🏗️ *PYRAMID ADD #{len(v8_positions)}* 🏗️\nSymbol: `{symbol}`\nAction: `{'BUY �' if signal == 1 else 'SELL 🔴'}`\nPrice: `{price}`\nSL: `{sl}`\nTP: `{tp}`\nConf: `{confidence:.1%}`")
                else:
                    console.print(f"[red]Echec Pyramide: {result.comment}[/red]")
        
        return # On ne fait rien d'autre si on a déjà une position (sauf pyramider)

    # 3. Entry (Si aucune position)
    lot = BASE_LOTS.get(symbol, 0.1)
    
    # Ajustement Confiance (Self-Learning)
    penalty = get_symbol_performance(symbol)
    if penalty < 0:
        # console.print(f"⚠️ [orange1]Malus Self-Learning sur {symbol}: {penalty*100}%[/orange1]")
        if (confidence + penalty) < 0.5: 
            # console.print(f"🚫 Trade annulé par Self-Learning (Confiance trop basse après malus)")
            return

    trade_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(symbol).ask if signal == 1 else mt5.symbol_info_tick(symbol).bid
    point = mt5.symbol_info(symbol).point
    
    # ATR Based SL/TP
    atr = df['ATR'].iloc[-1]
    sl_dist = atr * 1.5
    tp_dist = atr * 3.0
    
    sl = price - sl_dist if signal == 1 else price + sl_dist
    tp = price + tp_dist if signal == 1 else price - tp_dist
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": trade_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": 888888,
        "comment": f"Gemini V8 ({confidence:.2f})",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        console.print(f"✅ [bold green]TRADE EXECUTE: {symbol} | {lot} lots | Conf: {confidence:.2%}[/bold green]")
        send_telegram(f"🚀 *INITIAL ENTRY* 🚀\nSymbol: `{symbol}`\nAction: `{'BUY 🟢' if signal == 1 else 'SELL 🔴'}`\nLot: `{lot}`\nPrice: `{price}`\nSL: `{sl}`\nTP: `{tp}`\nConf: `{confidence:.1%}`")
    else:
        console.print(f"[red]Echec Trade: {result.comment}[/red]")

# --- DASHBOARD HELPERS ---
def get_daily_profit():
    try:
        # Attention : MT5 est souvent en UTC+2/3, on prend une marge de sécurité
        # On regarde les deals depuis le début de la journée locale
        from_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        deals = mt5.history_deals_get(from_date, datetime.now())
        if deals is None: return 0.0
        
        # On exclut les dépôts (type 2 usually, but safer to check symbol or entry)
        # deal.type: 0=BUY, 1=SELL, 2=BALANCE
        profit = sum([d.profit + d.swap + d.commission for d in deals if d.type != mt5.DEAL_TYPE_BALANCE])
        return profit
    except:
        return 0.0

def manage_positions():
    positions = mt5.positions_get()
    if positions is None: return
    
    for pos in positions:
        if pos.magic != 888888: continue
        symbol = pos.symbol
        point = mt5.symbol_info(symbol).point
        tick = mt5.symbol_info_tick(symbol)
        
        # Assurez-vous que tick n'est pas None avant d'accéder à ses attributs
        if tick is None:
            console.print(f"[red]Erreur: Impossible d'obtenir le tick pour {symbol}. Ignoré.[/red]")
            continue

        current = tick.bid if pos.type == 0 else tick.ask
        
        # Calcul Profit en Points
        profit_pts = (current - pos.price_open) / point if pos.type == 0 else (pos.price_open - current) / point
        
        # --- STRATÉGIE "SECURE & RUN" ---
        # Si profit > 75 points et que le SL n'est pas encore au Breakeven
        # On ferme 50% et on met le SL à l'entrée
        
        # Seuil d'activation (75 points)
        if profit_pts > 75:
            # Vérifier si déjà sécurisé (SL proche du prix d'entrée)
            is_secured = False
            # Pour une position BUY (type 0), SL > price_open signifie qu'il est au-dessus du prix d'entrée
            if pos.type == 0 and pos.sl >= pos.price_open: is_secured = True
            # Pour une position SELL (type 1), SL < price_open signifie qu'il est en dessous du prix d'entrée
            if pos.type == 1 and pos.sl > 0 and pos.sl <= pos.price_open: is_secured = True
            
            if not is_secured:
                # 1. PARTIAL CLOSE (50%)
                min_vol = mt5.symbol_info(symbol).volume_min
                vol_step = mt5.symbol_info(symbol).volume_step
                close_vol = round((pos.volume * 0.5) / vol_step) * vol_step
                
                if close_vol >= min_vol:
                    req_close = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": close_vol,
                        "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                        "position": pos.ticket,
                        "price": tick.bid if pos.type == 0 else tick.ask,
                        "magic": 888888,
                        "comment": "Gemini V8 Partial Close",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    res_close = mt5.order_send(req_close)
                    if res_close.retcode == mt5.TRADE_RETCODE_DONE:
                        console.print(f"💰 [gold1]PARTIAL CLOSE: {symbol} ({close_vol} lots) - SECURED[/gold1]")
                        send_telegram(f"💰 *PARTIAL CLOSE* 💰\nSymbol: `{symbol}`\nVolume: `{close_vol}`\nProfit Secured! ✅")
                    else:
                        console.print(f"[red]Erreur Partial Close: {res_close.comment}[/red]")
                
                # 2. MOVE SL TO BREAKEVEN & REMOVE TP (SWING MODE)
                # On augmente le buffer à 50 points pour couvrir largement les commissions/swap
                new_sl = pos.price_open + (50 * point) if pos.type == 0 else pos.price_open - (50 * point)
                
                # Vérifier si le nouveau SL est valide (ne doit pas être pire que l'ancien)
                if (pos.type == 0 and new_sl < pos.sl) or (pos.type == 1 and new_sl > pos.sl and pos.sl != 0):
                    # Si le nouveau SL est moins bon que l'ancien, on ne le déplace pas (sauf si l'ancien était 0)
                    pass
                else:
                    req_sl = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": symbol,
                        "sl": new_sl,
                        "tp": 0.0, # ON ENLÈVE LE TP POUR LAISSER COURIR (SWING MODE)
                        "position": pos.ticket,
                        "magic": 888888,
                    }
                    res_sl = mt5.order_send(req_sl)
                    if res_sl.retcode == mt5.TRADE_RETCODE_DONE:
                        console.print(f"🛡️ [green]SL MOVED TO SMART BE (+50pts) & TP REMOVED: {symbol}[/green]")
                        
                        # Telegram Alert Groupée
                        tg_msg = (
                            f"💰 *SECURE & RUN ACTIVATED* 🏃‍♂️\n"
                            f"------------------------\n"
                            f"Symbol: `{symbol}`\n"
                            f"📉 *Partial Close*: `50%` (Profit Secured ✅)\n"
                            f"🛡️ *SL*: `Breakeven` (Risk Free)\n"
                            f"🚀 *TP*: `REMOVED` (Letting it Run!)\n"
                            f"------------------------"
                        )
                        send_telegram(tg_msg)
                    else:
                        console.print(f"[red]Erreur SL/TP Update: {res_sl.comment}[/red]")

# --- MAIN ---
if __name__ == "__main__":
    connect_mt5()
    models = {}
    scalers = {}
    
    for sym in SYMBOLS:
        m, s = manage_memory(sym)
        if m: models[sym], scalers[sym] = m, s
        
    console.print("\n")
    console.print(Panel.fit("💎 [bold cyan]GEMINI V8 - THE REDEMPTION[/bold cyan]", border_style="cyan"))
    
    while True:
        manage_positions()
        
        # Récupération Infos Compte
        acc = mt5.account_info()
        daily_profit = get_daily_profit()
        dp_col = "[green]" if daily_profit >= 0 else "[red]"
        
        # Header Global
        console.clear()
        console.print(Panel(
            f"💰 Balance: [bold gold1]{acc.balance:.2f} USD[/bold gold1]  |  "
            f"📊 Equity: [bold cyan]{acc.equity:.2f} USD[/bold cyan]  |  "
            f"📅 Daily Profit: {dp_col}{daily_profit:.2f} USD[/{dp_col}]",
            title="[bold magenta]GEMINI V8 DASHBOARD[/bold magenta]",
            border_style="magenta"
        ))
        
        table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
        table.add_column("Symbole", style="cyan", width=10)
        table.add_column("Prix", justify="right", width=10)
        table.add_column("Positions (Main + Pyr)", justify="left", width=30)
        table.add_column("Indicateurs", justify="center", width=15)
        table.add_column("Trend H1", style="bold", width=10)
        table.add_column("Confiance", justify="right", width=10)
        table.add_column("Status", style="bold", width=15)
        
        for sym in SYMBOLS:
            if sym not in models: continue
            if not check_environment(sym): 
                table.add_row(sym, "-", "-", "-", "-", "-", "[yellow]Sleep[/yellow]")
                continue
            
            try:
                df = get_data(sym, n_candles=200)
                if len(df) == 0: continue
                
                X_new, _ = prepare_features(df)
                if len(X_new) == 0: continue
                
                feat = scalers[sym].transform(X_new[-1].reshape(1, -1))
                pred = models[sym].predict(feat)[0]
                conf = models[sym].predict_proba(feat)[0][pred]
                
                # Active Learning Check
                malus = get_symbol_performance(sym)
                adj_conf = conf + malus
                
                # Indicateurs
                mfi = df['MFI'].iloc[-1]
                last_price = df['close'].iloc[-1]
                mfi_col = "[green]" if 20 < mfi < 80 else "[red]"
                
                # Trend H1
                htf = check_htf_trend(sym)
                htf_color = "green" if htf == "BULLISH" else "red" if htf == "BEARISH" else "white"

                # Info Positions Détaillées
                all_positions = mt5.positions_get(symbol=sym)
                positions = [p for p in all_positions if p.magic == 888888] if all_positions else []
                
                pos_str = "-"
                status = "[dim]Wait[/dim]"
                
                if positions and len(positions) > 0:
                    details = []
                    total_pnl = 0
                    for i, p in enumerate(positions):
                        p_pnl = p.profit
                        total_pnl += p_pnl
                        c_col = "[green]" if p_pnl >= 0 else "[red]"
                        type_str = "MAIN" if i == 0 else f"PYR{i}"
                        details.append(f"{type_str}: {c_col}{p_pnl:.2f}$[/{c_col}]")
                    
                    pos_str = " | ".join(details)
                    status = f"[cyan]In Trade ({len(positions)})[/cyan]"
                    
                    # Check Pyramiding
                    execute_trade(sym, pred, conf, df, scalers[sym], models[sym])
                    
                elif adj_conf > 0.65:
                    status = f"[bold green]🚀 ENTRY[/bold green]"
                    execute_trade(sym, pred, conf, df, scalers[sym], models[sym])
                elif malus < 0:
                    status = f"[red]PUNISHED[/red]"
                
                table.add_row(
                    sym,
                    f"{last_price:.2f}",
                    pos_str,
                    f"MFI:{mfi_col}{mfi:.0f}[/{mfi_col}]",
                    f"[{htf_color}]{htf}[/{htf_color}]",
                    f"{adj_conf:.1%}",
                    status
                )
                
            except Exception as e:
                table.add_row(sym, "ERR", "-", "-", "-", "-", str(e))
                
        console.print(table)
        console.print(f"[dim]Scan... {datetime.now().strftime('%H:%M:%S')} | Telegram: ON[/dim]")
        time.sleep(10)
