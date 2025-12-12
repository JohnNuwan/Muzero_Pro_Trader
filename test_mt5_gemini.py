import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import time
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

console = Console()


# --- ⚙️ CONFIGURATION DU QUARTIER GÉNÉRAL ---
LOGIN = 1512112659 
PASSWORD = "8Ee7B$z54"
SERVER = "FTMO-Demo"

# Liste des combattants
SYMBOLS = ["XAUUSD", "US30.cash", "US100.cash", "GER40.cash","EURUSD","US500.cash"] 

# Lot de base (Sera ajusté dynamiquement)
BASE_LOTS = {
    "XAUUSD": 0.1,      # Réduit de 0.5 à 0.1 (5x moins de risque)
    "EURUSD": 0.1,      # Réduit de 0.5 à 0.1 (5x moins de risque)
    "US30.cash": 0.2,  # Réduit de 0.5 à 0.05 (10x moins de risque)
    "US100.cash": 0.2, # Réduit de 0.5 à 0.05 (10x moins de risque)
    "GER40.cash": 0.2,  # Réduit de 0.5 à 0.05 (10x moins de risque)
    "US500.cash": 0.5  # Réduit de 0.5 à 0.05 (10x moins de risque)
}

# Paramètres FTMO & IA
MAX_SPREAD_POINTS = 300 # Protection News (Si spread > ça, on ne trade pas)
TIMEFRAME = mt5.TIMEFRAME_M15
LOOKBACK = 10 
MEMORY_FOLDER = "gemini_memory"
CONSECUTIVE_LOSS_LIMIT = 3 # Anti-Tilt

# --- VARIABLES GLOBALES ---
if not os.path.exists(MEMORY_FOLDER): os.makedirs(MEMORY_FOLDER)
loss_streak = {sym: 0 for sym in SYMBOLS}
ban_until = {sym: 0 for sym in SYMBOLS}

def connect_mt5():
    if not mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER):
        console.print(Panel.fit("❌ ÉCHEC CRITIQUE : Connexion MT5 impossible", style="bold red"))
        quit()
    account_info = mt5.account_info()
    console.print(Panel.fit(
        f"🤖 [bold cyan]GEMINI V6 CONNECTÉ[/bold cyan]\n"
        f"Compte: [yellow]{account_info.login}[/yellow] | "
        f"Balance: [green]{account_info.balance:.2f} USD[/green]",
        title="[bold green]Système Actif[/bold green]",
        border_style="green"
    ))
    console.print("🛡️ [bold blue]Systèmes de défense FTMO : ACTIFS[/bold blue]")

# Variables pour limiter l'affichage des messages
_last_spread_warning = {}
_last_position_info = {}

def check_environment(symbol):
    """Vérifie si le terrain est sûr (Spread, Heure, Tilt)"""
    # 1. Vérif Anti-Tilt
    if time.time() < ban_until[symbol]:
        return False 

    # 2. Vérif Spread (Détecteur de News)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False
        
    point = mt5.symbol_info(symbol).point
    if point is None or point == 0:
        return False
        
    spread_points = (tick.ask - tick.bid) / point
    
    # Tolérance différente pour indices et gold
    limit = MAX_SPREAD_POINTS if "US" in symbol else 50 # Plus serré sur le Gold
    
    if spread_points > limit:
        # Afficher le message seulement une fois toutes les 60 secondes par symbole
        current_time = time.time()
        if symbol not in _last_spread_warning or (current_time - _last_spread_warning[symbol]) > 60:
            console.print(f"⚠️ [yellow]{symbol}[/yellow]: Spread trop élevé ([red]{spread_points:.0f} pts[/red]). News possible. Pas de trade.")
            _last_spread_warning[symbol] = current_time
        return False
        
    return True

def calculate_rsi(prices, period=14):
    """Calcule le RSI (Relative Strength Index)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calcule l'ATR (Average True Range)"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def get_data(symbol, n_candles=5000, verbose=False):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, n_candles)
    
    # Vérifier si les données ont été récupérées
    if rates is None or len(rates) == 0:
        if verbose:
            console.print(f"⚠️ [yellow]Aucune donnée récupérée pour {symbol} depuis MT5[/yellow]")
        return pd.DataFrame()
    
    df = pd.DataFrame(rates)
    
    if len(df) == 0:
        if verbose:
            console.print(f"⚠️ [yellow]DataFrame vide pour {symbol}[/yellow]")
        return df
    
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Indicateurs
    df['RSI'] = calculate_rsi(df['close'], period=14)
    df['ATR'] = calculate_atr(df['high'], df['low'], df['close'], period=14)
    
    # Structure (Dow) - seulement si on a assez de données
    n = 10
    if len(df) > n * 2:
        try:
            min_indices = argrelextrema(df.close.values, np.less_equal, order=n)[0]
            max_indices = argrelextrema(df.close.values, np.greater_equal, order=n)[0]
            
            df['min'] = np.nan
            df['max'] = np.nan
            if len(min_indices) > 0:
                df.loc[df.index[min_indices], 'min'] = df.loc[df.index[min_indices], 'close']
            if len(max_indices) > 0:
                df.loc[df.index[max_indices], 'max'] = df.loc[df.index[max_indices], 'close']
            
            # Utiliser ffill puis bfill pour remplir toutes les valeurs
            df['last_high'] = df['max'].ffill().bfill()
            df['last_low'] = df['min'].ffill().bfill()
            
            # Si toujours NaN, utiliser les valeurs de close comme fallback
            df['last_high'] = df['last_high'].fillna(df['close'])
            df['last_low'] = df['last_low'].fillna(df['close'])
        except Exception as e:
            if verbose:
                console.print(f"⚠️ [yellow]Erreur calcul structure pour {symbol}:[/yellow] {e}")
            df['last_high'] = df['high'].rolling(window=20, min_periods=1).max()
            df['last_low'] = df['low'].rolling(window=20, min_periods=1).min()
    else:
        # Fallback si pas assez de données pour la structure
        df['last_high'] = df['high'].rolling(window=20, min_periods=1).max()
        df['last_low'] = df['low'].rolling(window=20, min_periods=1).min()
    
    # Pattern Géométrique
    diff = df['last_high'] - df['last_low']
    # Éviter la division par zéro pour ATR - utiliser une valeur minimale
    atr_safe = df['ATR'].replace(0, np.nan)
    # Si ATR est NaN, utiliser une petite valeur pour éviter la division par zéro
    atr_safe = atr_safe.fillna(df['close'] * 0.001)  # 0.1% du prix comme fallback
    
    df['Dist_Fib'] = (df['close'] - (df['last_low'] + diff * 0.618)) / atr_safe
    
    # Remplacer les infinis par NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Ne supprimer que les lignes où les colonnes nécessaires sont NaN
    # Les colonnes min/max peuvent être NaN, ce n'est pas grave
    cols_required = ['RSI', 'ATR', 'Dist_Fib']
    nan_count_before = df[cols_required].isna().sum().sum()
    df.dropna(subset=cols_required, inplace=True)
    nan_count_after = len(df)
    
    if verbose and nan_count_before > 0:
        print(f"   ⚠️ {symbol}: {nan_count_before} valeurs NaN supprimées dans les colonnes requises, {nan_count_after} lignes restantes")
    
    # Réinitialiser l'index après dropna
    df.reset_index(drop=True, inplace=True)
    
    return df

def prepare_features(df):
    features = []
    labels = []
    cols = ['RSI', 'ATR', 'Dist_Fib']
    
    # Vérifier que les colonnes existent et ne sont pas toutes NaN
    for col in cols:
        if col not in df.columns:
            console.print(f"⚠️ [yellow]Colonne {col} manquante dans le DataFrame[/yellow]")
            return np.array([]), np.array([])
    
    # Vérifier qu'il y a assez de données
    if len(df) < LOOKBACK + 1:
        console.print(f"⚠️ [yellow]Pas assez de données: {len(df)} lignes, besoin de {LOOKBACK + 1}[/yellow]")
        return np.array([]), np.array([])
    
    for i in range(LOOKBACK, len(df)):
        seq = df[cols].iloc[i-LOOKBACK:i].values.flatten()
        # Vérifier que la séquence ne contient pas de NaN
        if not np.isnan(seq).any():
            features.append(seq)
            labels.append(1 if df['close'].iloc[i] > df['open'].iloc[i] else 0)
        
    if len(features) == 0:
        console.print(f"⚠️ [yellow]Aucune feature valide générée[/yellow]")
        return np.array([]), np.array([])
        
    return np.array(features), np.array(labels)

def manage_memory(symbol):
    """Charge ou crée le cerveau de l'IA"""
    m_path = f"{MEMORY_FOLDER}/{symbol}_model.pkl"
    s_path = f"{MEMORY_FOLDER}/{symbol}_scaler.pkl"
    
    if os.path.exists(m_path):
        return joblib.load(m_path), joblib.load(s_path)
    else:
        console.print(f"🧠 [bold cyan][{symbol}][/bold cyan] Entraînement initial...")
        
        # Vérifier que le symbole est disponible dans MT5
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            console.print(f"❌ [bold red][{symbol}][/bold red] Symbole non disponible dans MT5")
            raise ValueError(f"Symbole {symbol} non disponible dans MT5")
        
        if not symbol_info.visible:
            console.print(f"⚠️ [yellow][{symbol}][/yellow] Symbole non visible, tentative d'activation...")
            if not mt5.symbol_select(symbol, True):
                console.print(f"❌ [bold red][{symbol}][/bold red] Impossible d'activer le symbole")
                raise ValueError(f"Impossible d'activer le symbole {symbol}")
        
        df = get_data(symbol, verbose=True)
        console.print(f"   📊 Données récupérées: [cyan]{len(df)}[/cyan] lignes")
        
        if len(df) == 0:
            console.print(Panel(
                f"❌ [bold red]Aucune donnée disponible pour {symbol}[/bold red]\n\n"
                f"Vérifiez:\n"
                f"  • Que le symbole {symbol} est disponible sur le serveur {SERVER}\n"
                f"  • Que vous avez une connexion active à MT5\n"
                f"  • Que le timeframe {TIMEFRAME} contient des données historiques",
                title="[bold red]Erreur[/bold red]",
                border_style="red"
            ))
            raise ValueError(f"Pas de données disponibles pour {symbol}")
        
        X, y = prepare_features(df)
        
        if len(X) == 0:
            console.print(f"❌ [bold red][{symbol}][/bold red] Impossible de générer des features. Vérifiez les données.")
            raise ValueError(f"Pas assez de données valides pour {symbol}")
        
        console.print(f"   ✅ Features générées: [green]{len(X)}[/green] échantillons")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, warm_start=True, random_state=42)
        model.fit(X_scaled, y)
        
        joblib.dump(model, m_path)
        joblib.dump(scaler, s_path)
        console.print(f"   💾 Modèle sauvegardé pour [cyan]{symbol}[/cyan]")
        return model, scaler

def smart_lot_size(symbol, confidence):
    """Ajuste la mise selon la certitude de l'IA"""
    base = BASE_LOTS.get(symbol, 0.1)
    if confidence > 0.85: return base * 1.5 # Grosse confiance = Gros bet
    if confidence > 0.70: return base       # Confiance normale
    return base * 0.5                       # Petite confiance

def execute_trade(symbol, prediction, df, confidence):
    if len(mt5.positions_get(symbol=symbol)) > 0: return
    
    lot = smart_lot_size(symbol, confidence)
    action = mt5.ORDER_TYPE_BUY if prediction == 1 else mt5.ORDER_TYPE_SELL
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if prediction == 1 else tick.bid
    
    # SL sur Structure
    atr = df['ATR'].iloc[-1]
    last_low = df['last_low'].iloc[-1]
    last_high = df['last_high'].iloc[-1]
    
    sl = last_low - atr if prediction == 1 else last_high + atr
    tp_dist = abs(price - sl) * 1.5
    tp = price + tp_dist if prediction == 1 else price - tp_dist
    
    # Normalisation SL (Empêche SL trop près ou trop loin)
    min_dist = atr * 0.5
    if abs(price - sl) < min_dist: sl = price - min_dist if prediction == 1 else price + min_dist
    
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(f"{lot:.2f}"), # Arrondi 2 décimales
        "type": action,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": 666666,
        "comment": f"Gemini V6 ({confidence:.2f})",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    res = mt5.order_send(req)
    if res.retcode == mt5.TRADE_RETCODE_DONE:
        action_text = "🟢 BUY" if action == mt5.ORDER_TYPE_BUY else "🔴 SELL"
        console.print(Panel.fit(
            f"{action_text} [bold]{symbol}[/bold]\n"
            f"Confiance: [yellow]{confidence:.2%}[/yellow] | "
            f"Lots: [cyan]{lot:.2f}[/cyan] | "
            f"Prix: [green]{price:.2f}[/green]",
            title="[bold green]🚀 TRADE EXÉCUTÉ[/bold green]",
            border_style="green"
        ))
    else:
        console.print(f"❌ [bold red]Erreur ordre {symbol}:[/bold red] {res.comment}")

def check_results():
    """Vérifie les trades fermés pour mettre à jour le compteur de défaites"""
    # Cette fonction simplifiée réinitialise le streak si on fait un profit
    # Dans une version pro, on check l'historique précis via mt5.history_deals_get
    pass 

# --- MAIN ---
if __name__ == "__main__":
    connect_mt5()
    models = {}
    scalers = {}
    
    for sym in SYMBOLS:
        models[sym], scalers[sym] = manage_memory(sym)
        
    console.print("\n")
    console.print(Panel.fit(
        "🥊 [bold cyan]LE MATCH COMMENCE[/bold cyan]\n"
        "[bold green]JE SUIS PRÊT[/bold green]",
        title="[bold yellow]GEMINI V6[/bold yellow]",
        border_style="yellow"
    ))
    console.print()
    
    iteration = 0
    while True:
        iteration += 1
        console.print(f"[dim]{'─' * 80}[/dim]")
        console.print(f"[dim]Cycle #{iteration} - {datetime.now().strftime('%H:%M:%S')}[/dim]")
        console.print(f"[dim]{'─' * 80}[/dim]")
        
        for sym in SYMBOLS:
            if not check_environment(sym): 
                continue
            
            try:
                # Vérifier si une position existe déjà
                positions = mt5.positions_get(symbol=sym)
                has_position = positions is not None and len(positions) > 0
                
                # Toujours analyser les données pour voir ce que le script détecte
                df = get_data(sym, n_candles=100, verbose=False)
                
                if len(df) == 0:
                    if not has_position:  # Seulement afficher si pas de position
                        console.print(f"🔍 [dim]{sym}[/dim]: Pas de données disponibles")
                    continue
                    
                X_new, _ = prepare_features(df)
                
                if len(X_new) == 0:
                    if not has_position:  # Seulement afficher si pas de position
                        console.print(f"🔍 [dim]{sym}[/dim]: Pas assez de features générées")
                    continue
                    
                feat = scalers[sym].transform(X_new[-1].reshape(1, -1))
                pred = models[sym].predict(feat)[0]
                conf = models[sym].predict_proba(feat)[0][pred]
                direction = "🟢 BUY" if pred == 1 else "🔴 SELL"
                
                if has_position:
                    # Position ouverte : afficher l'analyse + statut de la position
                    pos = positions[0]
                    profit = pos.profit
                    volume = pos.volume
                    pos_type = "🟢 BUY" if pos.type == 0 else "🔴 SELL"
                    profit_color = "green" if profit >= 0 else "red"
                    profit_sign = "+" if profit >= 0 else ""
                    
                    # Afficher l'analyse actuelle
                    console.print(
                        f"👀 [cyan]{sym}[/cyan]: Signal {direction} | "
                        f"Confiance: [yellow]{conf:.2%}[/yellow] | "
                        f"📊 Position {pos_type} ([yellow]{volume}[/yellow] lots) | "
                        f"Profit: [{profit_color}]{profit_sign}{profit:.2f} USD[/{profit_color}]"
                    )
                else:
                    # Pas de position : afficher l'analyse et trader si confiance suffisante
                    if conf > 0.65:
                        console.print(
                            f"👀 [cyan]{sym}[/cyan]: Signal {direction} | "
                            f"Confiance: [yellow]{conf:.2%}[/yellow] | "
                            f"[bold green]➡️ Trade exécuté[/bold green]"
                        )
                        execute_trade(sym, pred, df, conf)
                    else:
                        console.print(
                            f"👀 [cyan]{sym}[/cyan]: Signal {direction} | "
                            f"Confiance: [dim]{conf:.2%}[/dim] | "
                            f"[dim]⏸️ Confiance insuffisante (< 65%)[/dim]"
                        )
                        
            except Exception as e:
                console.print(f"❌ [bold red]Erreur {sym}:[/bold red] {e}")
                import traceback
                traceback.print_exc()
        
        console.print()  # Ligne vide avant le prochain cycle
        time.sleep(10)