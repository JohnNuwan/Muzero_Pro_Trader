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
import json
import subprocess
import sys
from datetime import datetime, timedelta
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
import csv
import requests

console = Console()

# --- ⚙️ CONFIGURATION DU QUARTIER GÉNÉRAL (V10 - THE EVOLUTION) ---
LOGIN = 1512112659 
PASSWORD = "8Ee7B$z54"
SERVER = "FTMO-Demo"

# TELEGRAM CONFIG
TELEGRAM_TOKEN = "6660646226:AAEwerIp1t_i5mToXseSFeic3ryrzTXmF-U"
TELEGRAM_CHAT_ID = "-5009131528"

# CONFIG DYNAMIQUE (V10)
CONFIG_FILE = "gemini_config.json"
SYMBOL_PARAMS = {} # Sera chargé depuis le JSON

# Paramètres Globaux
MAX_SPREAD_POINTS = 300 
TIMEFRAME = mt5.TIMEFRAME_M15
LOOKBACK = 10 
MEMORY_FOLDER = "gemini_memory"
HISTORY_FILE = f"{MEMORY_FOLDER}/trade_history.csv"
DAILY_LOSS_LIMIT = -300.0 # 🛑 Hard Equity Stop (USD)
REPORT_INTERVAL = 1 * 3600 # 📢 Rapport Telegram toutes les 4 heures

# --- VARIABLES GLOBALES ---
if not os.path.exists(MEMORY_FOLDER): os.makedirs(MEMORY_FOLDER)
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "symbol", "features", "prediction", "confidence", "outcome", "profit"])

loss_streak = {} # Initialisé après chargement des symboles
ban_until = {}
_last_spread_warning = {}
_last_report_time = time.time()
_evolving_symbols = {} # {symbol: process}

def load_config():
    global SYMBOL_PARAMS, loss_streak, ban_until
    try:
        with open(CONFIG_FILE, 'r') as f:
            SYMBOL_PARAMS = json.load(f)
            
        # Init globals for new symbols
        for sym in SYMBOL_PARAMS:
            if sym not in loss_streak: loss_streak[sym] = 0
            if sym not in ban_until: ban_until[sym] = 0
            
        return True
    except Exception as e:
        console.print(f"[red]Erreur chargement config: {e}[/red]")
        return False

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
        f"🧬 [bold magenta]GEMINI V10 - THE EVOLUTION CONNECTÉ[/bold magenta]\n"
        f"Compte: [yellow]{account_info.login}[/yellow] | "
        f"Balance: [green]{account_info.balance:.2f} USD[/green]\n"
        f"Mode: [cyan]GENETIC OPTIMIZATION + HARD STOP[/cyan]\n"
        f"Stratégie: [magenta]Darwin Engine (Adaptive)[/magenta]",
        title="[bold green]Système Actif[/bold green]",
        border_style="green"
    ))
    send_telegram(f"🧬 *GEMINI V10 - THE EVOLUTION ONLINE*\nAccount: `{account_info.login}`\nBalance: `{account_info.balance:.2f} USD`\nEngine: `Darwin`")

def check_environment(symbol):
    """Vérifie si le terrain est sûr (Spread, Heure, Tilt)"""
    if time.time() < ban_until[symbol]:
        return False 
    
    # Check Evolution Status
    if symbol in _evolving_symbols:
        proc = _evolving_symbols[symbol]
        if proc.poll() is None: # Still running
            return False
        else:
            # Evolution finished
            del _evolving_symbols[symbol]
            load_config() # Reload new genes
            send_telegram(f"🧬 *EVOLUTION COMPLETE: {symbol}* 🧬\nNew parameters loaded. Re-entering market.")
            # Reset ban
            ban_until[symbol] = 0
            return True

    tick = mt5.symbol_info_tick(symbol)
    if tick is None: return False
        
    point = mt5.symbol_info(symbol).point
    if point is None or point == 0: return False
        
    spread_points = (tick.ask - tick.bid) / point
    limit = MAX_SPREAD_POINTS if "US" in symbol else 50 
    
    # Tolérance spread plus large pour BTC
    if "BTC" in symbol: limit = 5000

    if spread_points > limit:
        current_time = time.time()
        if symbol not in _last_spread_warning or (current_time - _last_spread_warning[symbol]) > 60:
            console.print(f"⚠️ [yellow]{symbol}[/yellow]: Spread trop élevé ([red]{spread_points:.0f} pts[/red]). News possible. Pas de trade.")
            _last_spread_warning[symbol] = current_time
        return False
        
    return True

# --- FEATURE ENGINEERING AVANCÉ (V8/V9/V10) ---
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
    sma = close.rolling(window=period).mean()
    slope_proxy = (close - sma) / sma * 100
    return slope_proxy

def calculate_adx(high, low, close, period=14):
    """Calcule l'ADX pour déterminer la force de la tendance"""
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
    minus_di = 100 * (abs(minus_dm).ewm(alpha=1/period).mean() / atr)
    
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()
    return adx

def get_data(symbol, n_candles=5000, verbose=False):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, n_candles)
    if rates is None or len(rates) == 0: return pd.DataFrame()
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Prix Médian
    df['median'] = (df['high'] + df['low']) / 2
    
    # Dynamic Params for Indicators (V10)
    params = SYMBOL_PARAMS.get(symbol, {})
    rsi_p = params.get("rsi_period", 14)
    mfi_p = params.get("mfi_period", 14)
    
    # Indicateurs V8/V10/V11
    df['RSI'] = calculate_rsi(df['close'], period=rsi_p)
    df['ATR'] = calculate_atr(df['high'], df['low'], df['close'])
    df['MFI'] = calculate_mfi(df['high'], df['low'], df['close'], df['tick_volume'], period=mfi_p)
    df['OBV'] = calculate_obv(df['close'], df['tick_volume'])
    df['LinReg_Slope'] = calculate_linear_regression(df['close'])
    df['ADX'] = calculate_adx(df['high'], df['low'], df['close']) # V11
    
    # Lagged Returns
    df['Ret_1'] = df['close'].pct_change(1)
    
    # Time Features - Robustesse
    try:
        df['Hour'] = df['time'].dt.hour
    except Exception:
        df['Hour'] = 0
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Colonnes requises
    cols_required = ['RSI', 'ATR', 'MFI', 'OBV', 'LinReg_Slope', 'ADX', 'Ret_1', 'Hour']
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

# --- ACTIVE SELF-LEARNING & EVOLUTION (V10) ---
def get_symbol_performance(symbol):
    """Analyse l'historique pour voir si on perd sur ce symbole"""
    if not os.path.exists(HISTORY_FILE): return 0
    
    try:
        # VRAIE LOGIQUE ACTIVE : Lire les trades fermés depuis MT5
        from_date = datetime.now() - timedelta(days=1)
        deals = mt5.history_deals_get(from_date, datetime.now(), group=f"*{symbol}*")
        
        if deals is None or len(deals) == 0: return 0
        
        recent_losses = 0
        for deal in deals:
            if deal.entry == mt5.DEAL_ENTRY_OUT and deal.profit < 0:
                recent_losses += 1
        
        # V10 TRIGGER: Si 2 pertes -> Lancer Evolution
        if recent_losses >= 2:
            if symbol not in _evolving_symbols and ban_until[symbol] == 0:
                console.print(f"🧬 [bold magenta]TRIGGERING EVOLUTION FOR {symbol}[/bold magenta]")
                send_telegram(f"🧬 *DARWIN TRIGGERED: {symbol}* 🧬\nReason: 2 Recent Losses.\nAction: Evolving Parameters...")
                
                # Launch background process
                # FIX: Use sys.executable AND pass env to ensure we use the same venv
                env = os.environ.copy()
                proc = subprocess.Popen([sys.executable, "train_evolution.py", symbol], env=env)
                _evolving_symbols[symbol] = proc
                
                # Ban temporaire le temps de l'evolution
                ban_until[symbol] = time.time() + 3600 # Fallback 1h si process plante
                return -1.0 # Ban immédiat

        if recent_losses >= 2: return -0.20
        
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
        console.print(f"🧠 [bold cyan][{symbol}][/bold cyan] Entraînement V9 (MFI/OBV)...")
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
        console.print(f"   💾 Cerveau V9 sauvegardé pour [cyan]{symbol}[/cyan]")
        return model, scaler

# --- 📰 FILTRE NEWS ÉCONOMIQUES 2.0 (ForexFactory Feed) ---
_news_cache = []
_last_news_update = 0
NEWS_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

def update_news_cache():
    """Récupère le calendrier économique depuis ForexFactory (JSON)"""
    global _news_cache, _last_news_update
    
    # Mise à jour toutes les 4 heures
    if time.time() - _last_news_update < 14400 and len(_news_cache) > 0:
        return

    try:
        console.print("[dim]📰 Downloading News Calendar...[/dim]")
        response = requests.get(NEWS_URL, timeout=10)
        if response.status_code == 200:
            data = response.json()
            _news_cache = []
            
            # Filtrage : High Impact (Red) & Currencies USD/EUR/GBP
            for event in data:
                impact = event.get('impact', 'Low')
                country = event.get('country', '')
                
                if impact == 'High' and country in ['USD', 'EUR', 'GBP']:
                    # Format date: "2024-11-19T14:30:00-05:00" -> On doit parser
                    # Astuce: On convertit tout en UTC ou on garde le timestamp
                    try:
                        date_str = event['date']
                        # Python < 3.11 ne gère pas bien le timezone 'Z' ou offset parfois, 
                        # mais FF envoie souvent avec offset. On utilise dateutil si dispo, sinon strptime simple
                        # FF format: 2011-01-05T14:30:00+00:00
                        # On simplifie : on stocke l'objet datetime
                        # Hack rapide pour parser l'offset si datetime.fromisoformat ne marche pas (Py < 3.7)
                        # Mais on assume Python 3.9+
                        dt = datetime.fromisoformat(date_str)
                        # Convert to local timestamp for comparison with time.time()
                        ts = dt.timestamp()
                        
                        _news_cache.append({
                            "title": event['title'],
                            "country": country,
                            "timestamp": ts
                        })
                    except Exception:
                        continue
            
            _last_news_update = time.time()
            console.print(f"[green]📰 News Updated: {len(_news_cache)} High Impact Events found[/green]")
        else:
            console.print(f"[red]News Download Failed: {response.status_code}[/red]")
            
    except Exception as e:
        console.print(f"[red]News Update Error: {e}[/red]")

def check_news(symbol):
    """
    Vérifie si on est dans une zone de danger (45min avant/après news).
    Retourne: (Bool, Reason)
    """
    update_news_cache()
    
    if len(_news_cache) == 0: return False, ""
    
    # Déterminer les devises du symbole
    currencies = ["USD"] # Par défaut
    if "EUR" in symbol: currencies.append("EUR")
    if "GBP" in symbol: currencies.append("GBP")
    if "JPY" in symbol: currencies.append("JPY")
    
    now = time.time()
    
    for news in _news_cache:
        if news['country'] in currencies:
            diff = news['timestamp'] - now
            
            # Zone Danger : 45 min avant (2700s) à 45 min après (2700s)
            if -2700 <= diff <= 2700:
                time_str = "NOW"
                if diff > 0: time_str = f"in {int(diff/60)}min"
                elif diff < 0: time_str = f"{int(abs(diff)/60)}min ago"
                
                return True, f"NEWS: {news['title']} ({time_str})"
                
    return False, ""

# --- 📊 INDICATEURS MATHÉMATIQUES (V9 - THE SNIPER) ---
def calculate_z_score(close, period=20):
    """Calcule le Z-Score (Distance à la moyenne en écart-types)"""
    mean = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    z_score = (close - mean) / std
    return z_score

# --- EXÉCUTION ---

# --- MULTI-TIMEFRAME ANALYSIS (V9) ---
def check_htf_trend(symbol):
    """
    Vérifie la tendance sur H1 ET H4 pour filtrer le M15.
    Règle : On ne trade que si H1 et H4 sont alignés (The Fortress).
    """
    try:
        # H1 Analysis
        rates_h1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
        if rates_h1 is None or len(rates_h1) < 50: return "NEUTRAL"
        df_h1 = pd.DataFrame(rates_h1)
        df_h1['EMA_50'] = df_h1['close'].ewm(span=50, adjust=False).mean()
        h1_trend = "BULLISH" if df_h1['close'].iloc[-1] > df_h1['EMA_50'].iloc[-1] else "BEARISH"
        
        # H4 Analysis
        rates_h4 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 100)
        if rates_h4 is None or len(rates_h4) < 50: return "NEUTRAL"
        df_h4 = pd.DataFrame(rates_h4)
        df_h4['EMA_50'] = df_h4['close'].ewm(span=50, adjust=False).mean()
        h4_trend = "BULLISH" if df_h4['close'].iloc[-1] > df_h4['EMA_50'].iloc[-1] else "BEARISH"
        
        if h1_trend == "BULLISH" and h4_trend == "BULLISH":
            return "STRONG BUY"
        elif h1_trend == "BEARISH" and h4_trend == "BEARISH":
            return "STRONG SELL"
        else:
            return "MIXED" # Correction ou Indécision
            
    except Exception as e:
        console.print(f"[red]Erreur MTF {symbol}: {e}[/red]")
        return "NEUTRAL"

def execute_trade(symbol, signal, confidence, df, scaler, model):
    # 0. Vérification de l'environnement
    if not check_environment(symbol): return

    # 0.5 FILTRE NEWS (V9) 📰
    is_news, news_msg = check_news(symbol)
    if is_news:
        return

    # 1. ANALYSE TENDANCE & MATHS (V9) 🏰 + 🎯
    htf_trend = check_htf_trend(symbol)
    
    # Calcul Z-Score (Mean Reversion)
    df['Z_Score'] = calculate_z_score(df['close'])
    z_score = df['Z_Score'].iloc[-1]
    
    # ADX Regime (V11)
    adx = df['ADX'].iloc[-1]
    regime = "RANGE"
    if adx > 25: regime = "TREND"
    
    # Dynamic Z-Score Threshold (V10)
    params = SYMBOL_PARAMS.get(symbol, {})
    z_thresh = params.get("z_score_threshold", 2.5)
    
    # LOGIQUE DE DÉCISION HYBRIDE (V11 ADAPTIVE)
    trend_confirm = False
    mean_reversion = False
    
    # MODE TREND (FORTRESS) - Prioritaire si ADX > 25
    if regime == "TREND":
        if signal == 1 and htf_trend == "STRONG BUY": trend_confirm = True
        if signal == 0 and htf_trend == "STRONG SELL": trend_confirm = True
        
    # MODE RANGE (SNIPER) - Prioritaire si ADX < 25
    else:
        # En Range, on cherche les excès (Z-Score)
        if z_score < -z_thresh and signal == 1: mean_reversion = True # Oversold -> Buy
        if z_score > z_thresh and signal == 0: mean_reversion = True  # Overbought -> Sell
        
        # Mais on peut aussi prendre du trend si c'est très fort
        if signal == 1 and htf_trend == "STRONG BUY": trend_confirm = True
        if signal == 0 and htf_trend == "STRONG SELL": trend_confirm = True

    # Décision Finale
    if not trend_confirm and not mean_reversion:
        return

    # 2. Gestion des positions existantes (Smart Pyramiding)
    positions = mt5.positions_get(symbol=symbol)
    v8_positions = [p for p in positions if p.magic == 888888] if positions else []
    
    params = SYMBOL_PARAMS.get(symbol, {})
    pyr_threshold = params.get("pyr", 100)
    
    if len(v8_positions) > 0:
        last_pos = v8_positions[-1]
        profit_points = (mt5.symbol_info_tick(symbol).bid - last_pos.price_open) / mt5.symbol_info(symbol).point if last_pos.type == 0 else (last_pos.price_open - mt5.symbol_info_tick(symbol).ask) / mt5.symbol_info(symbol).point
        
        penalty = get_symbol_performance(symbol)
        
        # Pyramiding seulement en TREND fort (ADX > 25) ou profit très élevé
        if profit_points > pyr_threshold and (confidence + penalty) > 0.70 and len(v8_positions) < 3:
            if regime == "RANGE" and profit_points < pyr_threshold * 1.5: return # Plus dur de pyramider en range

            if is_news: return 

            if (last_pos.type == 0 and signal == 1) or (last_pos.type == 1 and signal == 0):
                lot = params.get("lot", 0.01) * 0.5
                trade_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
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
                    "comment": f"Gemini V11 (Pyr #{len(v8_positions)})",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    console.print(f"🚀 [bold magenta]PYRAMID ADD #{len(v8_positions)}: {symbol} | Lot: {lot} | Conf: {confidence:.2%}[/bold magenta]")
                    
                    # Telegram Enhanced
                    strat_type = "TREND (Fortress)" if trend_confirm else "SNIPER (Range)"
                    tg_msg = (
                        f"🏗️ *PYRAMID ADD #{len(v8_positions)}* 🏗️\n"
                        f"Symbol: `{symbol}`\n"
                        f"Action: `{'BUY 🟢' if signal == 1 else 'SELL 🔴'}`\n"
                        f"Regime: `{regime} (ADX {adx:.0f})`\n"
                        f"Strat: `{strat_type}`\n"
                        f"Z-Score: `{z_score:.2f}`\n"
                        f"Conf: `{confidence:.1%}`"
                    )
                    send_telegram(tg_msg)
        
        return

    # 3. Entry (Si aucune position)
    lot = params.get("lot", 0.01)
    penalty = get_symbol_performance(symbol)
    if penalty < 0:
        if (confidence + penalty) < 0.5: return

    trade_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(symbol).ask if signal == 1 else mt5.symbol_info_tick(symbol).bid
    point = mt5.symbol_info(symbol).point
    
    atr = df['ATR'].iloc[-1]
    sl_mult = params.get("sl_mult", 1.5)
    tp_mult = params.get("tp_mult", 2.0)
    
    # Ajustement V11 : En Range, TP plus court
    if regime == "RANGE":
        tp_mult = tp_mult * 0.8
    
    sl_dist = atr * sl_mult
    tp_dist = atr * tp_mult
    
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
        "comment": f"Gemini V11 ({confidence:.2f})",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        console.print(f"✅ [bold green]TRADE EXECUTE: {symbol} | {lot} lots | Conf: {confidence:.2%}[/bold green]")
        
        # Telegram Enhanced
        strat_type = "TREND (Fortress)" if trend_confirm else "SNIPER (Range)"
        tg_msg = (
            f"🚀 *INITIAL ENTRY* 🚀\n"
            f"Symbol: `{symbol}`\n"
            f"Action: `{'BUY 🟢' if signal == 1 else 'SELL 🔴'}`\n"
            f"Regime: `{regime} (ADX {adx:.0f})`\n"
            f"Strat: `{strat_type}`\n"
            f"Z-Score: `{z_score:.2f}`\n"
            f"Lot: `{lot}`\n"
            f"Conf: `{confidence:.1%}`"
        )
        send_telegram(tg_msg)
    else:
        console.print(f"[red]Echec Trade: {result.comment}[/red]")

# --- DASHBOARD HELPERS ---
def get_daily_stats():
    """Récupère les stats détaillées de la journée (Wins, Losses, Pips, Profit)"""
    try:
        from_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        deals = mt5.history_deals_get(from_date, datetime.now())
        
        if deals is None: 
            return {"profit": 0.0, "wins": 0, "losses": 0, "pips": 0.0}
            
        profit = 0.0
        wins = 0
        losses = 0
        total_pips = 0.0
        
        for deal in deals:
            if deal.magic != 888888: continue # On ne compte que nos trades
            if deal.entry == mt5.DEAL_ENTRY_OUT: # C'est une sortie (fermeture)
                # Profit USD
                p = deal.profit + deal.swap + deal.commission
                profit += p
                
                # Win/Loss Count
                if p > 0: wins += 1
                else: losses += 1
                
        return {"profit": profit, "wins": wins, "losses": losses}
        
    except Exception as e:
        console.print(f"[red]Error Stats: {e}[/red]")
        return {"profit": 0.0, "wins": 0, "losses": 0}

def send_periodic_report():
    """Envoie un rapport Telegram périodique complet"""
    global _last_report_time
    if time.time() - _last_report_time < REPORT_INTERVAL:
        return
    
    _last_report_time = time.time()
    
    # 1. Positions en cours
    positions = mt5.positions_get()
    pos_msg = "🛑 *NO OPEN POSITIONS*"
    
    if positions and len(positions) > 0:
        pos_list = []
        for pos in positions:
            if pos.magic != 888888: continue
            symbol = pos.symbol
            profit = pos.profit
            type_str = "BUY" if pos.type == 0 else "SELL"
            pos_list.append(f"• {symbol} ({type_str}): `{profit:.2f}$`")
            
        if len(pos_list) > 0:
            pos_msg = "📊 *OPEN POSITIONS* 📊\n" + "\n".join(pos_list)
            
    # 2. Stats Historique (Aujourd'hui)
    stats = get_daily_stats()
    win_rate = 0
    if (stats['wins'] + stats['losses']) > 0:
        win_rate = (stats['wins'] / (stats['wins'] + stats['losses'])) * 100
    
    report = (
        f"{pos_msg}\n"
        f"------------------------\n"
        f"📅 *DAILY REPORT*\n"
        f"💰 Profit: `{stats['profit']:.2f} USD`\n"
        f"🏆 Wins: `{stats['wins']}` | ❌ Losses: `{stats['losses']}`\n"
        f"📈 Win Rate: `{win_rate:.1f}%`\n"
        f"🤖 Status: `RUNNING`"
    )
    send_telegram(report)

def manage_positions():
    positions = mt5.positions_get()
    if positions is None: return
    
    for pos in positions:
        if pos.magic != 888888: continue
        symbol = pos.symbol
        point = mt5.symbol_info(symbol).point
        tick = mt5.symbol_info_tick(symbol)
        
        if tick is None: continue

        current = tick.bid if pos.type == 0 else tick.ask
        profit_pts = (current - pos.price_open) / point if pos.type == 0 else (pos.price_open - current) / point
        
        # --- STRATÉGIE "SECURE & RUN" (V9 OPTIMIZED) ---
        params = SYMBOL_PARAMS.get(symbol, {})
        be_trigger = params.get("be", 50) 
        
        if profit_pts > be_trigger:
            is_secured = False
            if pos.type == 0 and pos.sl >= pos.price_open: is_secured = True
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
                        "comment": "Gemini V10 Partial Close",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    mt5.order_send(req_close)
                    send_telegram(f"💰 *PARTIAL CLOSE* 💰\nSymbol: `{symbol}`\nVolume: `{close_vol}`\nProfit Secured! ✅")
                
                # 2. MOVE SL TO BREAKEVEN & REMOVE TP
                be_buffer = 50 * point 
                if "BTC" in symbol: be_buffer = 500 * point 
                
                new_sl = pos.price_open + be_buffer if pos.type == 0 else pos.price_open - be_buffer
                
                req_sl = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "sl": new_sl,
                    "tp": 0.0,
                    "position": pos.ticket,
                    "magic": 888888,
                }
                mt5.order_send(req_sl)
                send_telegram(f"🛡️ *SECURE & RUN* 🛡️\nSymbol: `{symbol}`\nSL: `Breakeven`\nTP: `REMOVED`")

# --- MAIN ---
if __name__ == "__main__":
    load_config()
    connect_mt5()
    
    SYMBOLS = list(SYMBOL_PARAMS.keys())
    models = {}
    scalers = {}
    
    for sym in SYMBOLS:
        m, s = manage_memory(sym)
        if m: models[sym], scalers[sym] = m, s
        
    console.print("\n")
    console.print(Panel.fit("🧬 [bold magenta]GEMINI V10 - THE EVOLUTION[/bold magenta]", border_style="magenta"))
    
    while True:
        # 1. Check Hard Equity Stop
        daily_profit = get_daily_stats()['profit']
        if daily_profit < DAILY_LOSS_LIMIT:
            console.print(Panel.fit(f"🛑 [bold red]HARD STOP ACTIVATED: Daily Loss {daily_profit:.2f} < {DAILY_LOSS_LIMIT}[/bold red]"))
            send_telegram(f"🛑 *HARD EQUITY STOP TRIGGERED* 🛑\nDaily Loss: `{daily_profit:.2f} USD`\nTrading Halted for the day.")
            time.sleep(3600) 
            continue

        # 2. Manage Positions & Reports
        manage_positions()
        send_periodic_report()
        
        # 3. Dashboard & Trading Logic
        acc = mt5.account_info()
        dp_col = "[green]" if daily_profit >= 0 else "[red]"
        
        console.clear()
        console.print(Panel(
            f"💰 Balance: [bold gold1]{acc.balance:.2f} USD[/bold gold1]  |  "
            f"📊 Equity: [bold cyan]{acc.equity:.2f} USD[/bold cyan]  |  "
            f"📅 Daily Profit: {dp_col}{daily_profit:.2f} USD[/{dp_col}]",
            title="[bold magenta]GEMINI V10 DASHBOARD[/bold magenta]",
            border_style="magenta"
        ))
        
        table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
        table.add_column("Symbole", style="cyan", width=10)
        table.add_column("Prix", justify="right", width=10)
        table.add_column("Positions (Main + Pyr)", justify="left", width=30)
        table.add_column("Indicateurs", justify="center", width=15)
        table.add_column("Trend H1/H4", style="bold", width=10)
        table.add_column("Z-Score", justify="right", width=10)
        table.add_column("Confiance", justify="right", width=10)
        table.add_column("Status", style="bold", width=15)
        
        for sym in SYMBOLS:
            if sym not in models: continue
            # Fetch positions regardless of status
            all_positions = mt5.positions_get(symbol=sym)
            positions = [p for p in all_positions if p.magic == 888888] if all_positions else []
            pos_str = "-"
            if positions:
                 details = []
                 for i, p in enumerate(positions):
                     p_pnl = p.profit
                     c_col = "[green]" if p_pnl >= 0 else "[red]"
                     type_str = "MAIN" if i == 0 else f"PYR{i}"
                     details.append(f"{type_str}: {c_col}{p_pnl:.2f}$[/{c_col}]")
                 pos_str = " | ".join(details)

            if not check_environment(sym): 
                # Check if evolving
                if sym in _evolving_symbols:
                     table.add_row(sym, "-", pos_str, "-", "-", "-", "-", "[magenta]🧬 EVOLVING...[/magenta]")
                else:
                     table.add_row(sym, "-", pos_str, "-", "-", "-", "-", "[yellow]Sleep/Ban[/yellow]")
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
                
                # Z-Score Calculation for Display
                df['Z_Score'] = calculate_z_score(df['close'])
                z_score = df['Z_Score'].iloc[-1]
                z_col = "[red]" if abs(z_score) > 2.5 else "[white]"
                
                last_price = df['close'].iloc[-1]
                mfi_col = "[green]" if 20 < mfi < 80 else "[red]"
                
                # Trend H1/H4
                htf = check_htf_trend(sym)
                htf_color = "green" if "BUY" in htf else "red" if "SELL" in htf else "white"
                
                # Info Positions Détaillées (Déjà calculé plus haut)
                # On utilise pos_str calculé au début de la boucle
                
                if positions and len(positions) > 0:
                    status = f"[cyan]In Trade ({len(positions)})[/cyan]"
                    
                    # Check Pyramiding
                    execute_trade(sym, pred, conf, df, scalers[sym], models[sym])
                    
                elif adj_conf > 0.65 and not is_news:
                    status = f"[bold green]🚀 SCAN[/bold green]"
                    execute_trade(sym, pred, conf, df, scalers[sym], models[sym])
                elif malus < 0:
                    status = f"[red]PUNISHED[/red]"
                elif is_news:
                    status = f"[bold red]NEWS BLOCK[/bold red]"
                
                table.add_row(
                    sym,
                    f"{last_price:.2f}",
                    pos_str,
                    f"MFI:{mfi_col}{mfi:.0f}[/{mfi_col}]",
                    f"[{htf_color}]{htf}[/{htf_color}]",
                    f"{z_col}{z_score:.2f}[/{z_col}]",
                    f"{adj_conf:.1%}",
                    status
                )
                
            except Exception as e:
                table.add_row(sym, "ERR", "-", "-", "-", "-", "-", str(e))
                
        console.print(table)
        console.print(f"[dim]Scan... {datetime.now().strftime('%H:%M:%S')} | Telegram: ON[/dim]")
        time.sleep(10)
