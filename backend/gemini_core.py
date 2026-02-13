import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import json
import os
import time
import subprocess
import sys
import joblib
import requests
import threading
from dotenv import load_dotenv
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from database import SessionLocal, Trade, Event, Snapshot

# Load environment variables
load_dotenv()

# --- CONFIG ---
LOGIN = os.getenv("MT5_LOGIN")
if LOGIN:
    LOGIN = int(LOGIN)
PASSWORD = os.getenv("MT5_PASSWORD")
SERVER = os.getenv("MT5_SERVER")
CONFIG_FILE = "gemini_config.json"
MEMORY_FOLDER = "gemini_memory"
HISTORY_FILE = f"{MEMORY_FOLDER}/trade_history.csv"
DAILY_LOSS_LIMIT = -500.0
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
NEWS_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

SESSIONS = {
    "ASIAN": (0, 8),
    "LONDON": (7, 16),
    "NY": (13, 22)
}

class GeminiCore:
    def __init__(self):
        self.running = False
        self.symbols = []
        self.params = {}
        self.models = {}
        self.scalers = {}
        self.loss_streak = {}
        self.ban_until = {}
        self.evolving_symbols = {}
        self.news_cache = []
        self.last_news_update = 0
        self.logs = [] # Store logs for API
        self.market_state = {} # Real-time market data for API
        self.open_positions = [] # Cache for open positions
        self.paper_mode = False
        self.paper_mode = False
        self.cool_down_until = 0
        self.last_perf_check = 0
        
        self.db = SessionLocal()
        
        # Ensure directories
        if not os.path.exists(MEMORY_FOLDER): os.makedirs(MEMORY_FOLDER)
        
        # Load Config
        self.load_config()

    def log(self, message, type="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}") # Keep console output
        self.logs.insert(0, {"time": timestamp, "type": type, "message": message})
        if len(self.logs) > 100: self.logs.pop() # Keep last 100 logs

    def get_logs(self):
        return self.logs

    def get_recent_trades(self, limit=10):
        try:
            from_date = datetime.now() - timedelta(days=7) # Last 7 days
            deals = mt5.history_deals_get(from_date, datetime.now())
            if deals is None: return []
            
            trades = []
            for deal in deals:
                if deal.magic != 121212: continue
                if deal.entry == mt5.DEAL_ENTRY_OUT:
                    trades.append({
                        "ticket": deal.ticket,
                        "symbol": deal.symbol,
                        "type": "BUY" if deal.type == 0 else "SELL",
                        "volume": deal.volume,
                        "profit": deal.profit + deal.swap + deal.commission,
                        "time": datetime.fromtimestamp(deal.time).strftime("%Y-%m-%d %H:%M")
                    })
            return sorted(trades, key=lambda x: x['time'], reverse=True)[:limit]
        except Exception as e:
            self.log(f"Error fetching history: {e}", "ERROR")
            return []
        
    def load_config(self):
        try:
            with open(CONFIG_FILE, 'r') as f:
                self.params = json.load(f)
            self.symbols = list(self.params.keys())
            
            # Init State
            for sym in self.symbols:
                if sym not in self.loss_streak: self.loss_streak[sym] = 0
                if sym not in self.ban_until: self.ban_until[sym] = 0
                
            self.log("Configuration loaded successfully.")
                
        except Exception as e:
            print(f"Config Error: {e}")

    def load_models(self):
        """Load or train models for all symbols"""
        self.log("Loading models...")
        for sym in self.symbols:
            self.load_model(sym)
        self.log("All models loaded.")

    def train_hybrid_model(self, X, y):
        gbm = HistGradientBoostingClassifier(max_iter=100, random_state=42)
        mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        ensemble = VotingClassifier(estimators=[('gbm', gbm), ('mlp', mlp)], voting='soft', weights=[1.5, 1.0])
        ensemble.fit(X, y)
        return ensemble

    def train_model(self, symbol):
        self.log(f"Training new model for {symbol}...", "INFO")
        try:
            df = self.get_data(symbol, n_candles=5000)
            if len(df) > 0:
                X, y = self.prepare_features(df, mode="train")
                if len(X) > 0:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    model = self.train_hybrid_model(X_scaled, y)
                    
                    m_path = f"{MEMORY_FOLDER}/{symbol}_v8_model.pkl"
                    s_path = f"{MEMORY_FOLDER}/{symbol}_v8_scaler.pkl"
                    
                    joblib.dump(model, m_path)
                    joblib.dump(scaler, s_path)
                    
                    self.models[symbol] = model
                    self.scalers[symbol] = scaler
                    
                    # Update Config with Last Trained Date
                    # Load current config and update only last_trained field
                    with open(CONFIG_FILE, 'r') as f:
                        full_config = json.load(f)
                    
                    if symbol not in full_config:
                        full_config[symbol] = {}
                    
                    full_config[symbol]["last_trained"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                    
                    with open(CONFIG_FILE, 'w') as f:
                        json.dump(full_config, f, indent=4)
                    
                    # Also update in-memory params
                    self.params[symbol]["last_trained"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                    
                    self.log(f"Model trained and saved for {symbol}", "SUCCESS")
                    return True
                else:
                    self.log(f"Not enough features to train {symbol}", "WARNING")
            else:
                self.log(f"No data to train {symbol}", "WARNING")
        except Exception as e:
            self.log(f"Training failed for {symbol}: {e}", "ERROR")
        return False


    def load_model(self, symbol):
        m_path = f"{MEMORY_FOLDER}/{symbol}_v8_model.pkl"
        s_path = f"{MEMORY_FOLDER}/{symbol}_v8_scaler.pkl"
        
        if os.path.exists(m_path) and os.path.exists(s_path):
            try:
                self.models[symbol] = joblib.load(m_path)
                self.scalers[symbol] = joblib.load(s_path)
                self.log(f"Model loaded for {symbol}", "SUCCESS")
            except:
                self.train_model(symbol)
        else:
            self.train_model(symbol)

    def connect(self):
        if not mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER):
            self.log("MT5 Init Failed", "ERROR")
            return False
        self.log(f"Connected to MT5 ({SERVER})")
        return True

    def send_telegram(self, message):
        """
        Send a message to Telegram with retry logic
        """
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        
        for attempt in range(3):
            try:
                response = requests.post(url, data=data, timeout=5)
                if response.status_code == 200:
                    return
                else:
                    self.log(f"Telegram Error (Attempt {attempt+1}): {response.text}", "WARN")
            except Exception as e:
                self.log(f"Telegram Connection Error (Attempt {attempt+1}): {e}", "WARN")
            
            time.sleep(1) # Wait before retry
        
        self.log("Telegram failed after 3 attempts", "ERROR")

    def get_account_info(self):
        try:
            acc = mt5.account_info()
            if acc is None: return {}
            return {
                "balance": acc.balance,
                "equity": acc.equity,
                "leverage": acc.leverage,
                "margin_free": acc.margin_free,
                "margin_level": acc.margin_level,
                "currency": acc.currency
            }
        except: return {}

    def get_model_details(self, symbol):
        if symbol not in self.models: return None
        try:
            df = self.get_data(symbol, n_candles=200)
            if len(df) == 0: return None
            
            feat = self.prepare_features(df)
            if len(feat) == 0: return None
            
            feat_scaled = self.scalers[symbol].transform(feat)
            pred = self.models[symbol].predict(feat_scaled)[0]
            conf = self.models[symbol].predict_proba(feat_scaled)[0][pred]
            
            # Feature Importance (if available, or raw features)
            # For now, return raw feature values
            features_dict = {
                "RSI": df['RSI'].iloc[-1],
                "ATR": df['ATR'].iloc[-1],
                "MFI": df['MFI'].iloc[-1],
                "OBV": df['OBV'].iloc[-1],
                "ADX": df['ADX'].iloc[-1],
                "Z_Score": self.calculate_z_score(df['close']).iloc[-1]
            }
            
            return {
                "prediction": int(pred),
                "confidence": float(conf),
                "features": features_dict,
                "last_update": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Model Details Error: {e}")
            return None

    def get_config(self):
        return self.params

    def update_config(self, new_params):
        try:
            self.params.update(new_params)
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.params, f, indent=4)
            self.load_config() # Reload to apply changes
            self.log("Configuration updated via API", "SUCCESS")
            return True
        except Exception as e:
            self.log(f"Config Update Error: {e}", "ERROR")
            return False

    # --- INDICATORS ---
    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_atr(self, high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def calculate_mfi(self, high, low, close, volume, period=14):
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(window=period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(window=period).sum()
        mfi = 100 - (100 / (1 + (positive_flow / negative_flow.replace(0, 1))))
        return mfi

    def calculate_obv(self, close, volume):
        return (np.sign(close.diff()) * volume).fillna(0).cumsum()

    def calculate_linear_regression(self, close, period=14):
        sma = close.rolling(window=period).mean()
        return (close - sma) / sma * 100

    def calculate_adx(self, high, low, close, period=14):
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
        return dx.rolling(period).mean()

    def calculate_z_score(self, close, period=20):
        mean = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        return (close - mean) / std

    def calculate_pivots(self, high, low, close):
        """Calculate Standard Pivot Points"""
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        return pd.DataFrame({'Pivot': pivot, 'R1': r1, 'S1': s1, 'R2': r2, 'S2': s2}, index=[0])

    def calculate_fibonacci(self, high, low):
        """Calculate Fibonacci Retracement Levels from recent High/Low"""
        # Assuming high and low are single values (e.g., max/min of last 100 candles)
        diff = high - low
        return {
            "0.0": low,
            "0.236": low + diff * 0.236,
            "0.382": low + diff * 0.382,
            "0.5": low + diff * 0.5,
            "0.618": low + diff * 0.618,
            "1.0": high
        }

    def get_current_session(self):
        hour = datetime.now().hour
        active_sessions = []
        for name, (start, end) in SESSIONS.items():
            if start <= hour < end:
                active_sessions.append(name)
        return "/".join(active_sessions) if active_sessions else "OVERNIGHT"

    def detect_candle_patterns(self, df):
        """Detect basic candle patterns on the last closed candle"""
        if len(df) < 2: return []
        
        patterns = []
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        body = abs(curr['close'] - curr['open'])
        upper_wick = curr['high'] - max(curr['close'], curr['open'])
        lower_wick = min(curr['close'], curr['open']) - curr['low']
        total_range = curr['high'] - curr['low']
        
        # Doji
        if body <= total_range * 0.1:
            patterns.append("DOJI")
            
        # Hammer (Bullish)
        if lower_wick > body * 2 and upper_wick < body * 0.5:
            patterns.append("HAMMER")
            
        # Shooting Star (Bearish)
        if upper_wick > body * 2 and lower_wick < body * 0.5:
            patterns.append("SHOOTING_STAR")
            
        # Bullish Engulfing
        if curr['close'] > curr['open'] and prev['close'] < prev['open'] and \
           curr['close'] > prev['open'] and curr['open'] < prev['close']:
            patterns.append("BULL_ENGULFING")
            
        # Bearish Engulfing
        if curr['close'] < curr['open'] and prev['close'] > prev['open'] and \
           curr['close'] < prev['close'] and curr['open'] > prev['open']:
            patterns.append("BEAR_ENGULFING")
            
        return patterns

    def detect_chart_patterns(self, df):
        """Detect complex chart patterns (Head & Shoulders, Double Top/Bottom)"""
        if len(df) < 50: return []
        
        patterns = []
        close = df['close'].values
        
        # Find local peaks and valleys (simple window method)
        def find_extrema(data, order=5):
            peaks = []
            valleys = []
            for i in range(order, len(data) - order):
                if all(data[i] > data[i-j] for j in range(1, order+1)) and \
                   all(data[i] > data[i+j] for j in range(1, order+1)):
                    peaks.append((i, data[i]))
                if all(data[i] < data[i-j] for j in range(1, order+1)) and \
                   all(data[i] < data[i+j] for j in range(1, order+1)):
                    valleys.append((i, data[i]))
            return peaks, valleys

        peaks, valleys = find_extrema(close, order=5)
        
        # Double Top Detection
        if len(peaks) >= 2:
            p1 = peaks[-2]
            p2 = peaks[-1]
            # Check if peaks are similar height (within 0.5%) and recent
            if abs(p1[1] - p2[1]) / p1[1] < 0.005 and (len(close) - p2[0]) < 20:
                patterns.append("DOUBLE_TOP")

        # Double Bottom Detection
        if len(valleys) >= 2:
            v1 = valleys[-2]
            v2 = valleys[-1]
            if abs(v1[1] - v2[1]) / abs(v1[1]) < 0.005 and (len(close) - v2[0]) < 20:
                patterns.append("DOUBLE_BOTTOM")

        # Head and Shoulders (Left Shoulder, Head, Right Shoulder)
        if len(peaks) >= 3:
            ls = peaks[-3]
            head = peaks[-2]
            rs = peaks[-1]
            # Head higher than shoulders, shoulders similar height
            if head[1] > ls[1] and head[1] > rs[1] and \
               abs(ls[1] - rs[1]) / ls[1] < 0.01 and \
               (len(close) - rs[0]) < 20:
                patterns.append("HEAD_SHOULDERS")

        # Inverse Head and Shoulders
        if len(valleys) >= 3:
            ls = valleys[-3]
            head = valleys[-2]
            rs = valleys[-1]
            # Head lower than shoulders, shoulders similar height
            if head[1] < ls[1] and head[1] < rs[1] and \
               abs(ls[1] - rs[1]) / abs(ls[1]) < 0.01 and \
               (len(close) - rs[0]) < 20:
                patterns.append("INV_HEAD_SHOULDERS")

        # CONTRADICTION FILTER
        if "DOUBLE_TOP" in patterns and "DOUBLE_BOTTOM" in patterns:
            patterns.remove("DOUBLE_TOP")
            patterns.remove("DOUBLE_BOTTOM")
            # Optionally add "RANGE" pattern but frontend might not expect it
            
        return patterns

    def get_data(self, symbol, n_candles=5000):
        if not mt5.symbol_select(symbol, True):
            print(f"DEBUG: Failed to select {symbol}")
            return pd.DataFrame()
            
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, n_candles)
        if rates is None or len(rates) == 0: 
            print(f"DEBUG: No rates for {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        params = self.params.get(symbol, {})
        rsi_p = params.get("rsi_period", 14)
        mfi_p = params.get("mfi_period", 14)
        
        try:
            df['RSI'] = self.calculate_rsi(df['close'], period=rsi_p)
            df['ATR'] = self.calculate_atr(df['high'], df['low'], df['close'])
            df['MFI'] = self.calculate_mfi(df['high'], df['low'], df['close'], df['tick_volume'], period=mfi_p)
            df['OBV'] = self.calculate_obv(df['close'], df['tick_volume'])
            # df['LinReg_Slope'] = self.calculate_linreg_slope(df['close']) # FIX: Missing method
            df['LinReg_Slope'] = self.calculate_linear_regression(df['close']) # Temporary fix: Use existing method
            df['ADX'] = self.calculate_adx(df['high'], df['low'], df['close'])
            df['Ret_1'] = df['close'].pct_change(1)
            df['Hour'] = df['time'].dt.hour
        except Exception as e:
            print(f"DEBUG: Indicator Error {symbol}: {e}")
            return pd.DataFrame()
            
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        cols_required = ['RSI', 'ATR', 'MFI', 'OBV', 'LinReg_Slope', 'ADX', 'Ret_1', 'Hour']
        
        before_drop = len(df)
        df.dropna(subset=cols_required, inplace=True)
        after_drop = len(df)
        
        if after_drop == 0:
            print(f"DEBUG: All rows dropped for {symbol}. Before: {before_drop}, After: 0")
        
        df.reset_index(drop=True, inplace=True)
        return df

    def prepare_features(self, df, mode="inference"):
        # 1. Calculate Indicators (Base)
        # (Assumes RSI, ATR, MFI, OBV, ADX are already in df from get_data)
        
        # 2. Calculate Price Action Features
        # Pivots
        last_high = df['high'].shift(1)
        last_low = df['low'].shift(1)
        last_close = df['close'].shift(1)
        pivot = (last_high + last_low + last_close) / 3
        r1 = (2 * pivot) - last_low
        s1 = (2 * pivot) - last_high
        
        # Fibonacci (Rolling 100)
        roll_high = df['high'].rolling(100).max()
        roll_low = df['low'].rolling(100).min()
        diff = roll_high - roll_low
        fibo_618 = roll_high - (diff * 0.618)
        
        # Distances (Normalized by ATR)
        # Avoid division by zero
        atr = df['ATR'].replace(0, 0.0001)
        
        df['Dist_Pivot'] = (df['close'] - pivot) / atr
        df['Dist_R1'] = (df['close'] - r1) / atr
        df['Dist_S1'] = (df['close'] - s1) / atr
        df['Dist_Fibo'] = (df['close'] - fibo_618) / atr
        
        # Session Encoding
        # Asian=1, London=2, NY=3, Overnight=0
        # We need a vectorized way to do this
        hours = df['Hour'].values
        sessions = np.zeros(len(df))
        
        # Asian (0-8)
        sessions[(hours >= 0) & (hours < 8)] = 1
        # London (8-16)
        sessions[(hours >= 8) & (hours < 16)] = 2
        # NY (13-21) - Overlap with London handled by simple priority or just distinct code
        # Let's keep it simple: 8-13=London(2), 13-16=Overlap(4), 16-21=NY(3)
        sessions[(hours >= 13) & (hours < 16)] = 4 # Overlap
        sessions[(hours >= 16) & (hours < 22)] = 3 # NY
        
        df['Session_Code'] = sessions
        
        # Feature Selection
        cols = [
            'RSI', 'ATR', 'MFI', 'OBV', 'LinReg_Slope', 'Ret_1', 'Hour',
            'Dist_Pivot', 'Dist_R1', 'Dist_S1', 'Dist_Fibo', 'Session_Code'
        ]
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=cols, inplace=True)
        
        if len(df) < 100: 
            return (np.array([]), np.array([])) if mode == "train" else np.array([])

        if mode == "train":
            # Create Targets (Next candle return > 0?)
            # We want to predict if price will go UP in next 1-3 candles
            # Let's try predicting next candle direction for now
            future_ret = df['close'].shift(-1) - df['close']
            df['Target'] = (future_ret > 0).astype(int)
            
            df.dropna(subset=['Target'], inplace=True)
            
            X = df[cols].values
            y = df['Target'].values
            return X, y
            
        else: # Inference
            # Just return the last row
            return df[cols].iloc[-1].values.reshape(1, -1)

    # --- NEWS FILTER ---
    def update_news_cache(self):
        if time.time() - self.last_news_update < 14400 and len(self.news_cache) > 0: return
        try:
            response = requests.get(NEWS_URL, timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.news_cache = []
                for event in data:
                    if event.get('impact') == 'High' and event.get('country') in ['USD', 'EUR', 'GBP']:
                        try:
                            dt = datetime.fromisoformat(event['date'])
                            self.news_cache.append({
                                "title": event['title'],
                                "country": event['country'],
                                "timestamp": dt.timestamp()
                            })
                        except: continue
                self.last_news_update = time.time()
        except Exception as e:
            print(f"News Error: {e}")
            # Fallback: If news API fails, we don't block trading but log warning
            # In a real production env, you might want to block if critical data is missing
            self.last_news_update = time.time() # Prevent retry spam

    def check_news(self, symbol):
        self.update_news_cache()
        if not self.news_cache: return False, ""
        
        currencies = ["USD"]
        if "EUR" in symbol: currencies.append("EUR")
        if "GBP" in symbol: currencies.append("GBP")
        
        now = time.time()
        for news in self.news_cache:
            if news['country'] in currencies:
                diff = news['timestamp'] - now
                if -2700 <= diff <= 2700:
                    return True, f"NEWS: {news['title']}"
        return False, ""

    def check_session(self, symbol):
        params = self.params.get(symbol, {})
        allowed_sessions = params.get("sessions", ["ASIAN", "LONDON", "NY"])
        
        current_hour = datetime.utcnow().hour
        
        for session in allowed_sessions:
            start, end = SESSIONS.get(session, (0, 24))
            if start <= current_hour < end:
                return True
        return False

    # --- RISK MANAGEMENT ---
    def check_spread(self, symbol):
        spread = mt5.symbol_info(symbol).spread
        max_spread = self.params.get(symbol, {}).get("max_spread", 20)
        if spread > max_spread:
            return False, f"Spread too high ({spread} > {max_spread})"
        return True, ""

    def check_volatility(self, symbol, atr):
        min_atr = self.params.get(symbol, {}).get("min_atr", 0.0001)
        max_atr = self.params.get(symbol, {}).get("max_atr", 1.0)
        if atr < min_atr: return False, f"Low Volatility (ATR {atr:.5f})"
        if atr > max_atr: return False, f"High Volatility (ATR {atr:.5f})"
        return True, ""

    def check_correlation(self, symbol):
        positions = mt5.positions_get()
        if not positions: return True, ""
        
        base_currency = symbol[:3]
        quote_currency = symbol[3:]
        
        for pos in positions:
            if pos.magic != 121212: continue
            pos_symbol = pos.symbol
            if base_currency in pos_symbol or quote_currency in pos_symbol:
                return False, f"Correlation Risk: Already exposed to {base_currency}/{quote_currency}"
        return True, ""

    def check_time_stops(self):
        positions = mt5.positions_get()
        if not positions: return
        
        now = time.time()
        for pos in positions:
            if pos.magic != 121212: continue
            duration = now - pos.time
            max_duration = 4 * 3600 # 4 hours
            
            if duration > max_duration and pos.profit < 0:
                self.close_position(pos.ticket, symbol=pos.symbol, comment="Time Stop")

    def get_latest_atr(self, symbol):
        """Get latest ATR value for a symbol"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 50)
            if rates is None:
                return 0.001  # Fallback
            df = pd.DataFrame(rates)
            atr = self.calculate_atr(df['high'], df['low'], df['close'], period=14)
            return atr.iloc[-1] if len(atr) > 0 else 0.001
        except:
            return 0.001

    def place_order(self, symbol, signal, lot, comment, **kwargs):
        """
        Place an order with enhanced Telegram notifications
        
        Args:
            symbol: Trading symbol
            signal: 1 for BUY, 0 for SELL
            lot: Position size
            comment: Order comment (Entry/Pyramid)
            **kwargs: regime, confidence, z_score, adx, strategy
        """
        # Get prices
        tick = mt5.symbol_info_tick(symbol)
        if signal == 1:  # BUY
            price = tick.ask
            order_type = mt5.ORDER_TYPE_BUY
            action = "BUY 🟢"
        else:  # SELL
            price = tick.bid
            order_type = mt5.ORDER_TYPE_SELL
            action = "SELL 🔴"
        
        # Calculate SL and TP
        params = self.params.get(symbol, {})
        atr = self.get_latest_atr(symbol)
        
        sl_mult = params.get("sl_mult", 2.0)
        tp_mult = params.get("tp_mult", 3.0)
        
        point = mt5.symbol_info(symbol).point
        
        if signal == 1:  # BUY
            sl = price - (atr * sl_mult)
            tp = price + (atr * tp_mult)
        else:  # SELL
            sl = price + (atr * sl_mult)
            tp = price - (atr * tp_mult)

        # Place order
        strategy_name = kwargs.get('strategy', 'Unknown')
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": round(sl, mt5.symbol_info(symbol).digits),
            "tp": round(tp, mt5.symbol_info(symbol).digits),
            "magic": 121212,
            "comment": f"Gemini V12 {comment} {strategy_name}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            regime = kwargs.get('regime', '')
            confidence = kwargs.get('confidence', 0)
            z_score = kwargs.get('z_score', 0)
            adx = kwargs.get('adx', 0)
            strategy_name = kwargs.get('strategy', 'Unknown')
            
            trade = Trade(
                ticket=result.order,
                symbol=symbol,
                type="BUY" if signal == 1 else "SELL",
                volume=lot,
                price_open=price,
                sl=sl,
                tp=tp,
                open_time=datetime.now(),
                strategy=strategy_name,
                status="OPEN"
            )
            self.db.add(trade)
            self.db.commit()
            
            # News status
            is_news, _ = self.check_news(symbol)
            news_status = "⚠️ HIGH IMPACT" if is_news else "✅ SAFE"
            
            # HTF Trend
            htf_trend = self.check_htf_trend(symbol)
            
            # Build beautiful Telegram message
            if comment == "Entry":
                emoji = "🚀"
                title = "INITIAL ENTRY"
            else:  # Pyramid
                emoji = "📈"
                title = "PYRAMID"
            
            message = f"""
{emoji} *{title}* {emoji}
━━━━━━━━━━━━━━━━━
📊 Symbol: `{symbol}`
📍 Action: {action}
💰 Strategy: `{strategy_name}`

📈 *Trade Details:*
├ Entry: `{price:.5f}`
├ SL: `{sl:.5f}` ({abs(price-sl)/point:.0f} pts)
├ TP: `{tp:.5f}` ({abs(tp-price)/point:.0f} pts)
└ Lot: `{lot}`

🎯 *AI Analysis:*
├ Z-Score: `{z_score:.2f}`
├ Trend: `{htf_trend}`
├ Regime: `{regime}`
├ ADX: `{adx:.0f}`
└ Conf: `{confidence*100:.1f}%`

🌍 News: {news_status}
━━━━━━━━━━━━━━━━━
🎲 Ticket: `#{result.order}`
"""
            
            self.send_telegram(message.strip())
            self.log(f"✅ Order Placed: {symbol} {action} @ {price}", "SUCCESS")
            return True
        else:
            error_msg = f"❌ Order Failed: {symbol} - {result.comment if result else 'No response'}"
            self.log(error_msg, "ERROR")
            self.send_telegram(error_msg)
            return False

    def close_position(self, ticket, symbol, comment=""):
        pos = mt5.positions_get(ticket=ticket)
        if not pos: return
        pos = pos[0]
        
        trade_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).bid if trade_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(symbol).ask
        
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": trade_type,
            "position": ticket,
            "price": price,
            "magic": 121212,
            "comment": f"Gemini V12 Close {comment}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(req)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            profit = pos.profit + pos.swap + pos.commission
            pips = abs(price - pos.price_open) / mt5.symbol_info(symbol).point
            
            # Emoji based on profit
            if profit > 0:
                emoji = "💰✅"
                result_text = "PROFIT"
            else:
                emoji = "📉"
                result_text = "LOSS"
            
            message = f"""
🛑 *POSITION CLOSED* 🛑
━━━━━━━━━━━━━━━━━
{emoji} Result: *{result_text}*

📊 Symbol: `{symbol}`
📍 Type: `{'BUY' if pos.type == 0 else 'SELL'}`

💵 *P&L:*
├ Profit: `{profit:+.2f} USD`
├ Pips: `{pips:.1f}`
└ Volume: `{pos.volume}`

📌 *Prices:*
├ Entry: `{pos.price_open:.5f}`
└ Exit: `{price:.5f}`

🔖 Reason: `{comment}`
🎲 Ticket: `#{ticket}`
━━━━━━━━━━━━━━━━━
"""
            
            self.send_telegram(message.strip())
            self.log(f"Position Closed: {symbol} ({comment})", "SUCCESS")

    # --- EVOLUTION ---
    def get_symbol_performance(self, symbol):
        if not os.path.exists(HISTORY_FILE): return 0
        try:
            from_date = datetime.now() - timedelta(days=1)
            deals = mt5.history_deals_get(from_date, datetime.now(), group=f"*{symbol}*")
            if not deals: return 0
            
            recent_losses = 0
            for deal in deals:
                if deal.entry == mt5.DEAL_ENTRY_OUT and deal.profit < 0:
                    recent_losses += 1
            
            # V9 Logic: -0.20 for 2 losses, -1.0 (Ban) for 4 losses
            if recent_losses >= 4:
                if symbol not in self.evolving_symbols and self.ban_until[symbol] == 0:
                    self.trigger_evolution(symbol)
                    return -1.0
            
            if recent_losses >= 2: return -0.20
        except: pass
        return 0

    def trigger_evolution(self, symbol):
        if symbol not in self.evolving_symbols:
            self.log(f"🧬 TRIGGERING EVOLUTION FOR {symbol}", "WARN")
            self.send_telegram(f"🧬 *DARWIN TRIGGERED: {symbol}* 🧬\nReason: 4 Recent Losses.")
            env = os.environ.copy()
            proc = subprocess.Popen([sys.executable, "train_evolution.py", symbol], env=env)
            self.evolving_symbols[symbol] = proc
            self.ban_until[symbol] = time.time() + 3600
            return True
        return False

    # --- TRADING LOGIC ---
    def check_htf_trend(self, symbol):
        try:
            rates_h1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
            rates_h4 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 100)
            if rates_h1 is None or rates_h4 is None: return "NEUTRAL"
            
            df_h1 = pd.DataFrame(rates_h1)
            df_h4 = pd.DataFrame(rates_h4)
            
            df_h1['EMA_50'] = df_h1['close'].ewm(span=50).mean()
            df_h4['EMA_50'] = df_h4['close'].ewm(span=50).mean()
            
            h1_bull = df_h1['close'].iloc[-1] > df_h1['EMA_50'].iloc[-1]
            h4_bull = df_h4['close'].iloc[-1] > df_h4['EMA_50'].iloc[-1]
            
            if h1_bull and h4_bull: return "STRONG BUY"
            if not h1_bull and not h4_bull: return "STRONG SELL"
            return "MIXED"
        except: return "NEUTRAL"

    def calculate_dynamic_lot(self, symbol, risk_percent=0.01, sl_pips=50, multiplier=1.0):
        try:
            # 1. Check for fixed lot in config
            params = self.params.get(symbol, {})
            if "lot" in params:
                return float(params["lot"]) * multiplier

            # 2. Dynamic Calculation
            acc = self.get_account_info()
            equity = acc.get('equity', 10000)
            
            risk_amount = equity * risk_percent * multiplier
            
            tick_value = mt5.symbol_info(symbol).trade_tick_value
            point = mt5.symbol_info(symbol).point
            
            if tick_value == 0: return 0.01
            
            lot = risk_amount / (sl_pips * tick_value)
            
            # Normalize lot
            step = mt5.symbol_info(symbol).volume_step
            lot = round(lot / step) * step
            
            min_lot = mt5.symbol_info(symbol).volume_min
            max_lot = mt5.symbol_info(symbol).volume_max
            
            return max(min_lot, min(max_lot, lot))
        except:
            return 0.01

    def execute_trade(self, symbol, signal, confidence, df, scaler, model):
        # Always update market state (moved to tick)
        
        if not self.running: return # Only trade if running
        if time.time() < self.ban_until[symbol]: return
        
        # News Filter
        is_news, news_msg = self.check_news(symbol)
        if is_news: return

        # Session Filter
        if not self.check_session(symbol): return

        # Spread Filter
        ok, msg = self.check_spread(symbol)
        if not ok: return

        # Volatility Filter
        ok, msg = self.check_volatility(symbol, df['ATR'].iloc[-1])
        if not ok: return
        
        # Correlation Filter
        ok, msg = self.check_correlation(symbol)
        if not ok: return

        # Analysis
        htf_trend = self.check_htf_trend(symbol)
        df['Z_Score'] = self.calculate_z_score(df['close'])
        z_score = df['Z_Score'].iloc[-1]
        adx = df['ADX'].iloc[-1]
        regime = "TREND" if adx > 25 else "RANGE"
        
        params = self.params.get(symbol, {})
        z_thresh = params.get("z_score_threshold", 2.5)
        
        trend_confirm = False
        mean_reversion = False
        high_confidence = confidence > 0.80  # SAFETY: Only trade pure AI if VERY confident (>80%)
        
        # V9 Logic: Allow Mean Reversion regardless of ADX if Z-Score is extreme
        if z_score < -z_thresh and signal == 1: mean_reversion = True
        if z_score > z_thresh and signal == 0: mean_reversion = True
        
        # Trend Logic
        if signal == 1 and htf_trend == "STRONG BUY": trend_confirm = True
        if signal == 0 and htf_trend == "STRONG SELL": trend_confirm = True

        # NEW: Allow trading if AI has high confidence (>70%), even without extreme conditions
        # This prioritizes AI signal quality over traditional technical conditions
        if not trend_confirm and not mean_reversion and not high_confidence: 
            return

        # Determine Strategy Name
        strategy_name = "Unknown"
        if high_confidence and not (trend_confirm or mean_reversion):
            strategy_name = "AI_SIGNAL"  # New strategy type for pure AI plays
        elif regime == "TREND":
            strategy_name = "TREND"
        elif mean_reversion:
            strategy_name = "REVERSION"

        # Positions
        positions = mt5.positions_get(symbol=symbol)
        my_positions = [p for p in positions if p.magic == 121212] if positions else []
        
        # Pyramiding
        if len(my_positions) > 0:
            last_pos = my_positions[-1]
            profit_pts = (mt5.symbol_info_tick(symbol).bid - last_pos.price_open) / mt5.symbol_info(symbol).point if last_pos.type == 0 else (last_pos.price_open - mt5.symbol_info_tick(symbol).ask) / mt5.symbol_info(symbol).point
            pyr_thresh = params.get("pyr", 100)
            
            # V9 Logic: Check penalty for pyramiding too
            penalty = self.get_symbol_performance(symbol)
            
            if profit_pts > pyr_thresh and (confidence + penalty) > 0.7 and len(my_positions) < 3:
                if regime == "RANGE" and profit_pts < pyr_thresh * 1.5: return
                
                if (last_pos.type == 0 and signal == 1) or (last_pos.type == 1 and signal == 0):
                    lot = self.calculate_dynamic_lot(symbol, risk_percent=0.005, multiplier=0.5) 
                    self.place_order(symbol, signal, lot, "Pyramid", regime=regime, confidence=confidence, z_score=z_score, adx=adx, strategy=strategy_name)
            return

        # Entry
        penalty = self.get_symbol_performance(symbol)
        if penalty < 0 and (confidence + penalty) < 0.5: return
        
        # Multi-Level Lot Sizing
        lot_multiplier = 1.0 # Default (TREND)
        
        if strategy_name == "AI_SIGNAL":
            lot_multiplier = 0.5 # Half lot for pure AI
        elif strategy_name == "REVERSION":
            lot_multiplier = 0.5 # Half lot for reversion
            
        lot = self.calculate_dynamic_lot(symbol, risk_percent=0.01, multiplier=lot_multiplier)
        self.place_order(symbol, signal, lot, "Entry", regime=regime, confidence=confidence, z_score=z_score, adx=adx, strategy=strategy_name)

    def tick(self):
        # Hard Stop Check (Always Active)
        stats = self.get_daily_stats()
        if stats['profit'] < DAILY_LOSS_LIMIT and self.running:
            self.log("HARD STOP ACTIVATED", "ERROR")
            self.running = False
            self.send_telegram("🛑 HARD STOP ACTIVATED")
            return

        # Equity Protection (Cool Down)
        if stats['profit'] < DAILY_LOSS_LIMIT * 0.5 and time.time() > self.cool_down_until:
             self.paper_mode = True
             self.cool_down_until = time.time() + 14400 # 4 hours
             self.log("❄️ COOL DOWN ACTIVATED: Paper Mode for 4h", "WARN")
             self.send_telegram("❄️ *COOL DOWN ACTIVATED* ❄️\nSwitched to Paper Mode due to drawdown.")
        
        if time.time() > self.cool_down_until and self.paper_mode:
             self.paper_mode = False
             self.log("🔥 COOL DOWN OVER: Back to Real Trading", "SUCCESS")
             self.send_telegram("🔥 *COOL DOWN OVER* 🔥\nBack to Real Trading.")

        # Time Stops
        self.check_time_stops()

        # Secure & Run (V10 Logic)
        self.manage_positions()
        
        # Sync Closed Trades (V12 Logic)
        self.sync_trades()

        # Snapshot (Only if running)
        if self.running:
            acc = mt5.account_info()
            if acc:
                snap = Snapshot(balance=acc.balance, equity=acc.equity, daily_profit=stats['profit'])
                self.db.add(snap)
                self.db.commit()

        # Symbols Loop (Always Run for Matrix)
        for sym in self.symbols:
            # Check Evolution
            if sym in self.evolving_symbols:
                proc = self.evolving_symbols[sym]
                if proc.poll() is not None:
                    del self.evolving_symbols[sym]
                    self.load_config()
                    self.ban_until[sym] = 0
                    self.log(f"🧬 Evolution Complete: {sym}", "SUCCESS")
                    self.send_telegram(f"🧬 Evolution Complete: {sym}")
                continue
            
            if sym not in self.models: continue
            
            try:
                df = self.get_data(sym, n_candles=1000)
                if len(df) == 0: continue
                
                # Update Market State for API (ALWAYS)
                current_price = df['close'].iloc[-1]
                z_score = self.calculate_z_score(df['close']).iloc[-1]
                trend_h1 = self.check_htf_trend(sym)
                adx = df['ADX'].iloc[-1]
                rsi = df['RSI'].iloc[-1]
                mfi = df['MFI'].iloc[-1]
                regime = "TREND" if adx > 25 else "RANGE"
                
                # Price Action Analysis
                pivots = self.calculate_pivots(df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1])
                # Use max/min of last 100 candles for Fibo
                recent_high = df['high'].tail(100).max()
                recent_low = df['low'].tail(100).min()
                fibs = self.calculate_fibonacci(recent_high, recent_low)
                patterns = self.detect_candle_patterns(df)
                chart_patterns = self.detect_chart_patterns(df)
                
                # Deduplicate and classify patterns
                all_patterns = list(set(patterns + chart_patterns))  # Remove duplicates
                
                # Classify patterns by direction
                bullish_patterns = ['HAMMER', 'BULL_ENGULFING', 'MORNING_STAR', 'SMALL_UP']
                bearish_patterns = ['SHOOTING_STAR', 'BEAR_ENGULFING', 'EVENING_STAR', 'SMALL_DOWN']
                
                formatted_patterns = []
                for p in all_patterns[:3]:  # Limit to 3 patterns
                    if any(bull in p for bull in bullish_patterns):
                        formatted_patterns.append({"name": p, "type": "bullish", "tf": "M15"})
                    elif any(bear in p for bear in bearish_patterns):
                        formatted_patterns.append({"name": p, "type": "bearish", "tf": "M15"})
                    else:
                        formatted_patterns.append({"name": p, "type": "neutral", "tf": "M15"})
                
                # Pyramid Count
                pyramid_count = len([p for p in self.open_positions if p.symbol == sym])
                
                # Current Session
                current_session = self.get_current_session()

                # CREATE MARKET STATE (This was missing!)
                self.market_state[sym] = {
                    "symbol": sym,
                    "price": current_price,
                    "trend": trend_h1,
                    "z_score": round(z_score, 2),
                    "adx": int(adx),
                    "rsi": int(rsi),
                    "mfi": int(mfi),
                    "regime": regime,
                    "status": "SLEEP" if sym in self.ban_until and time.time() < self.ban_until[sym] else "ACTIVE",
                    "evolving": sym in self.evolving_symbols,
                    "pivots": pivots.iloc[0].to_dict(),
                    "fibonacci": fibs,
                    "patterns": formatted_patterns, # Use the formatted list with colors/tf
                    "pyramid_count": pyramid_count,
                    "session": current_session,
                    "last_trained": self.params.get(sym, {}).get("last_trained", "Never")
                }

                # AI Prediction (Always run for visualization)
                try:
                    feat = self.prepare_features(df)
                    if len(feat) > 0:
                        feat_scaled = self.scalers[sym].transform(feat)
                        pred = self.models[sym].predict(feat_scaled)[0]
                        conf = self.models[sym].predict_proba(feat_scaled)[0][pred]
                        
                        # Add confidence to market state for visualization
                        self.market_state[sym]["confidence"] = round(conf * 100, 1)
                        self.market_state[sym]["prediction"] = "BUY" if pred == 1 else "SELL"
                        
                        # Trading Logic (Only if Running)
                        if self.running:
                            self.execute_trade(sym, pred, conf, df, self.scalers[sym], self.models[sym])
                except Exception as e:
                    if self.running: print(f"Prediction Error {sym}: {e}")
                
            except Exception as e:
                # Only print errors if running to avoid spamming console when idle
                if self.running: print(f"Error {sym}: {e}")

    def manage_positions(self):
        # V10 Secure & Run Logic:
        # - Monitor open positions
        # - If profit > BE_Trigger:
        #     1. Partial Close (50%)
        #     2. Move SL to Breakeven
        #     3. Remove TP (Let it run)
        try:
            positions = mt5.positions_get()
            if positions is None: 
                self.open_positions = []
                return
            
            self.open_positions = positions # Update cache
            
            for pos in positions:
                if pos.magic != 121212: continue
                symbol = pos.symbol
                
                # Skip if symbol not in our config (safety)
                if symbol not in self.params: continue

                params = self.params.get(symbol, {})
                be_trigger = params.get("be", 50)

                # Dynamic Secure Trigger ("Petit Poucet")
                # If strategy is AI_SIGNAL or REVERSION, secure faster (75% of normal BE)
                strategy_name = "TREND" # Default
                if pos.comment:
                    if "AI_SIGNAL" in pos.comment: strategy_name = "AI_SIGNAL"
                    elif "REVERSION" in pos.comment: strategy_name = "REVERSION"
                
                if strategy_name in ["AI_SIGNAL", "REVERSION"]:
                    be_trigger = be_trigger * 0.75

                point = mt5.symbol_info(symbol).point
                tick = mt5.symbol_info_tick(symbol)
                if tick is None: continue
                
                current_price = tick.bid if pos.type == 0 else tick.ask
                profit_pts = (current_price - pos.price_open) / point if pos.type == 0 else (pos.price_open - current_price) / point
                
                # Check if already secured (SL is at or better than Open Price)
                is_secured = False
                if pos.type == 0 and pos.sl >= pos.price_open: is_secured = True
                if pos.type == 1 and pos.sl > 0 and pos.sl <= pos.price_open: is_secured = True

                if profit_pts > be_trigger and not is_secured:
                    # 1. Partial Close (50%)
                    min_vol = mt5.symbol_info(symbol).volume_min
                    vol_step = mt5.symbol_info(symbol).volume_step
                    close_vol = round((pos.volume * 0.5) / vol_step) * vol_step
                    
                    partial_close_status = "SKIPPED_VOLUME" # Default
                    
                    # Skip if close volume is too small or = total volume
                    if close_vol >= min_vol and close_vol < pos.volume:
                        req_close = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": symbol,
                            "volume": close_vol,
                            "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                            "position": pos.ticket,  # Links to the specific position
                            "price": tick.bid if pos.type == 0 else tick.ask,
                            "deviation": 20,  # Max price slippage
                            "magic": 121212,
                            "comment": "Gemini V12 Partial",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,  # IOC mode for partial close
                        }
                        res = mt5.order_send(req_close)
                        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                            partial_close_status = "SUCCESS"
                            remaining_vol = pos.volume - close_vol
                            secured_profit = pos.profit * (close_vol / pos.volume)
                            
                            message = f"""
💰 *PARTIAL CLOSE* 💰
━━━━━━━━━━━━━━━━━
✅ Profit Secured!

📊 Symbol: `{symbol}`
📍 Type: `{'BUY' if pos.type == 0 else 'SELL'}`

💵 *Details:*
├ Closed: `{close_vol} lots`
├ Remaining: `{remaining_vol} lots`
└ Secured: `~{secured_profit:+.2f} USD`

🎯 Position: `#{pos.ticket}`
━━━━━━━━━━━━━━━━━
"""
                            self.send_telegram(message.strip())
                            self.log(f"💰 PARTIAL CLOSE: {symbol} ({close_vol} lots)", "SUCCESS")
                        else:
                            partial_close_status = "FAILED"
                            # If partial close failed, log it
                            error_msg = f"Partial close failed for {symbol}: {res.comment if res else 'No response'}"
                            self.log(error_msg, "WARN")
                    
                    # 2. Move SL to Breakeven + Buffer
                    # 2. Move SL to Breakeven + Buffer
                    # User Request: BE = Entry + Spread + Safety Margin (1 pip)
                    # This ensures we cover the spread cost if stopped out
                    info = mt5.symbol_info(symbol)
                    spread_points = info.spread * point
                    be_buffer = spread_points + (10 * point) 
                    
                    if "BTC" in symbol: be_buffer = 100 * point # Exception for BTC
                    new_sl = pos.price_open + be_buffer if pos.type == 0 else pos.price_open - be_buffer
                    
                    req_sl = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": symbol,
                        "sl": new_sl,
                        "tp": 0.0, # Remove TP to let it run
                        "position": pos.ticket,
                        "magic": 121212,
                    }
                    res = mt5.order_send(req_sl)
                    if res.retcode == mt5.TRADE_RETCODE_DONE:
                        
                        # Append partial close note if not successful
                        pc_note = ""
                        if partial_close_status == "SKIPPED_VOLUME":
                            pc_note = "\n⚠️ *Note:* Split skipped (Volume too small)"
                        elif partial_close_status == "FAILED":
                            pc_note = "\n⚠️ *Note:* Split failed (Check logs)"

                        message = f"""
🛡️ *SECURE & RUN* 🛡️
━━━━━━━━━━━━━━━━━
🔒 Position Protected!

📊 Symbol: `{symbol}`
📍 Type: `{'BUY' if pos.type == 0 else 'SELL'}`

🎯 *Updated:*
├ SL: `Breakeven ({pos.price_open:.5f})`
├ TP: `REMOVED (Letting run!)`
└ Volume: `{pos.volume} lots`{pc_note}

💰 Profit Float: `{pos.profit:+.2f} USD`
🎲 Ticket: `#{pos.ticket}`
━━━━━━━━━━━━━━━━━
"""
                        self.send_telegram(message.strip())
                        self.log(f"🛡️ SECURE & RUN: {symbol} SL->BE", "SUCCESS")

        except Exception as e:
            self.log(f"Manage Positions Error: {e}", "ERROR")

    def sync_trades(self):
        """Sync closed trades from MT5 to Database"""
        try:
            # 1. Get all OPEN trades from DB
            db = SessionLocal()
            open_trades = db.query(Trade).filter(Trade.status == "OPEN").all()
            
            if not open_trades:
                db.close()
                return

            # 2. Get currently OPEN positions from MT5
            mt5_positions = mt5.positions_get()
            open_tickets = [p.ticket for p in mt5_positions] if mt5_positions else []
            
            for trade in open_trades:
                # If trade is in DB as OPEN but not in MT5 open positions -> It's CLOSED
                if trade.ticket not in open_tickets:
                    # Get history deals for this position
                    deals = mt5.history_deals_get(position=trade.ticket)
                    
                    if deals and len(deals) > 0:
                        # Calculate total profit (Entry + Exit + Swaps + Commissions)
                        total_profit = sum(d.profit + d.swap + d.commission for d in deals)
                        last_deal = deals[-1]
                        
                        trade.status = "CLOSED"
                        trade.profit = total_profit
                        trade.price_close = last_deal.price
                        trade.close_time = datetime.fromtimestamp(last_deal.time)
                        
                        self.log(f"🔄 Synced Closed Trade: {trade.symbol} ({trade.profit})", "INFO")
                    else:
                        # Fallback if no deals found (e.g. cancelled pending order turned position?)
                        pass
                        
            db.commit()
            db.close()
            
        except Exception as e:
            self.log(f"Sync Trades Error: {e}", "ERROR")

    def import_history(self, days=30):
        """Import historical trades from MT5 to Database"""
        try:
            self.log(f"Importing history for last {days} days...", "INFO")
            from_date = datetime.now() - timedelta(days=days)
            deals = mt5.history_deals_get(from_date, datetime.now())
            
            if deals is None or len(deals) == 0:
                self.log("No history found in MT5.", "WARNING")
                return

            db = SessionLocal()
            existing_tickets = {t.ticket for t in db.query(Trade.ticket).all()}
            
            count = 0
            # Group deals by position_id (ticket)
            position_deals = {}
            for deal in deals:
                if deal.position_id not in position_deals:
                    position_deals[deal.position_id] = []
                position_deals[deal.position_id].append(deal)
                
            for ticket, deal_list in position_deals.items():
                # Check if it's a trade (Entry=0, Exit=1)
                entry_deal = next((d for d in deal_list if d.entry == mt5.DEAL_ENTRY_IN), None)
                exit_deal = next((d for d in deal_list if d.entry == mt5.DEAL_ENTRY_OUT), None)
                
                if not entry_deal: continue # Not a valid trade start
                
                # Extract Strategy from Comment
                # Formats: "Gemini V12 AI_SIGNAL", "Gemini V12 REVERSION", "Gemini V12 TREND"
                strategy = "Unknown"
                comment = entry_deal.comment
                
                if comment:
                    if "AI_SIGNAL" in comment:
                        strategy = "AI_SIGNAL"
                    elif "REVERSION" in comment:
                        strategy = "REVERSION"
                    elif "TREND" in comment:
                        strategy = "TREND"
                    elif "Gemini V12" in comment:
                        strategy = "Gemini V12"
                    elif "Gemini V9" in comment:
                        strategy = "Gemini V9"
                    else:
                        strategy = comment # Use full comment if unknown format
                
                if ticket in existing_tickets:
                    # Update strategy if it differs (refining Unknown or grouping specific comments)
                    existing_trade = db.query(Trade).filter(Trade.ticket == ticket).first()
                    if existing_trade and existing_trade.strategy != strategy and strategy != "Unknown":
                        existing_trade.strategy = strategy
                        count += 1
                    continue
                
                symbol = entry_deal.symbol
                type_str = "BUY" if entry_deal.type == mt5.ORDER_TYPE_BUY else "SELL"
                
                # Calculate Profit
                profit = sum(d.profit + d.swap + d.commission for d in deal_list)
                
                status = "CLOSED" if exit_deal else "OPEN"
                close_time = datetime.fromtimestamp(exit_deal.time) if exit_deal else None
                price_close = exit_deal.price if exit_deal else 0.0
                
                new_trade = Trade(
                    ticket=ticket,
                    symbol=symbol,
                    type=type_str,
                    lot=entry_deal.volume,
                    price_open=entry_deal.price,
                    price_close=price_close,
                    sl=0.0, # History doesn't always store SL/TP in deals
                    tp=0.0,
                    status=status,
                    profit=profit,
                    open_time=datetime.fromtimestamp(entry_deal.time),
                    close_time=close_time,
                    strategy=strategy
                )
                db.add(new_trade)
                count += 1
                
            db.commit()
            db.close()
            self.log(f"Imported/Updated {count} historical trades.", "SUCCESS")
            
        except Exception as e:
            self.log(f"Import History Error: {e}", "ERROR")

    def get_daily_stats(self):
        # Récupère les stats détaillées de la journée (Wins, Losses, Pips, Profit)
        try:
            from_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            deals = mt5.history_deals_get(from_date, datetime.now())
            
            if deals is None: 
                return {"profit": 0.0, "wins": 0, "losses": 0, "pips": 0.0}
                
            profit = 0.0
            wins = 0
            losses = 0
            
            for deal in deals:
                # if deal.magic != 121212: continue # REMOVED: Count ALL trades to match Terminal PnL
                if deal.entry == mt5.DEAL_ENTRY_OUT: # C'est une sortie (fermeture)
                    p = deal.profit + deal.swap + deal.commission
                    profit += p
                    if p > 0: wins += 1
                    else: losses += 1
                    
            return {"profit": profit, "wins": wins, "losses": losses}
            
        except Exception as e:
            print(f"Error Stats: {e}")
            return {"profit": 0.0, "wins": 0, "losses": 0}

    def get_advanced_stats(self):
        # Calcule Sharpe, Drawdown, Profit Factor, Win Rate sur l'historique complet
        try:
            # 1. Récupérer tout l'historique
            from_date = datetime.now() - timedelta(days=30) # 30 derniers jours
            deals = mt5.history_deals_get(from_date, datetime.now())
            
            if deals is None or len(deals) == 0:
                return {"sharpe": 0, "drawdown": 0, "profit_factor": 0, "win_rate": 0, "total_trades": 0, "equity_curve": []}

            profits = []
            equity_curve = []
            running_equity = 0
            gross_profit = 0
            gross_loss = 0
            wins = 0
            total_trades = 0
            
            for deal in deals:
                if deal.magic != 121212: continue
                if deal.entry == mt5.DEAL_ENTRY_OUT:
                    p = deal.profit + deal.swap + deal.commission
                    profits.append(p)
                    running_equity += p
                    equity_curve.append(running_equity)
                    
                    if p > 0: 
                        gross_profit += p
                        wins += 1
                    else: 
                        gross_loss += abs(p)
                    
                    total_trades += 1
            
            if total_trades == 0:
                return {"sharpe": 0, "drawdown": 0, "profit_factor": 0, "win_rate": 0, "total_trades": 0, "equity_curve": []}

            # Win Rate
            win_rate = (wins / total_trades) * 100
            
            # Profit Factor
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 99.99
            
            # Max Drawdown
            max_dd = 0
            peak = -999999
            for eq in equity_curve:
                if eq > peak: peak = eq
                dd = peak - eq
                if dd > max_dd: max_dd = dd
                
            # Sharpe Ratio (Simplified: Mean / StdDev of returns)
            if len(profits) > 1:
                avg_ret = np.mean(profits)
                std_ret = np.std(profits)
                sharpe = (avg_ret / std_ret) * np.sqrt(total_trades) if std_ret > 0 else 0
            else:
                sharpe = 0
                
            return {
                "sharpe": round(sharpe, 2),
                "drawdown": round(max_dd, 2),
                "profit_factor": round(profit_factor, 2),
                "win_rate": round(win_rate, 1),
                "total_trades": total_trades,
                "equity_curve": equity_curve[-50:] # Last 50 points for chart
            }
            
        except Exception as e:
            print(f"Adv Stats Error: {e}")
            return {"sharpe": 0, "drawdown": 0, "profit_factor": 0, "win_rate": 0, "total_trades": 0, "equity_curve": []}

    def trigger_evolution(self, symbol):
        try:
            # Check if already evolving
            if symbol in self.evolving_symbols and self.evolving_symbols[symbol]:
                return False
                
            self.log(f"🧬 Triggering Evolution for {symbol}...", "INFO")
            self.evolving_symbols[symbol] = True
            
            # Run as subprocess
            subprocess.Popen([sys.executable, "train_evolution.py", symbol])
            
            # Reset flag after some time (or rely on script to finish)
            # For now, we just set it to False after a delay in a thread or assume the user will see the logs
            # Better: The script could call an API to say it's done, but for now simple is better.
            # We'll just clear the flag after 1 minute to allow re-trigger if needed
            threading.Timer(60.0, lambda: self.evolving_symbols.update({symbol: False})).start()
            
            return True
        except Exception as e:
            self.log(f"Evolution Trigger Error: {e}", "ERROR")
            return False

            by_symbol[symbol]["total"] += 1
            if profit >= 0: by_symbol[symbol]["wins"] += 1
            
            trades.append({
                "time": datetime.fromtimestamp(d.time).strftime('%Y-%m-%d %H:%M'),
                "ticket": d.ticket,
                "symbol": d.symbol,
                "type": "BUY" if d.type == 0 else "SELL",
                "volume": d.volume,
                "profit": round(profit, 2),
                "comment": d.comment
            })
            
        # Final Calculations
        if stats["total_trades"] > 0:
            stats["win_rate"] = round((stats["wins"] / stats["total_trades"]) * 100, 1)
            
        # Sort Symbol Performance
        sorted_symbols = sorted(by_symbol.values(), key=lambda x: x["profit"], reverse=True)
        
        return {
            "stats": stats,
            "trades": sorted(trades, key=lambda x: x["time"], reverse=True),
            "by_symbol": sorted_symbols
        }

    def get_market_state(self):
        return list(self.market_state.values())
    
    def get_positions(self):
        """Return formatted positions for API"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            result = []
            for pos in positions:
                # Filter only V12 positions (magic 121212)
                if pos.magic != 121212:
                    continue
                    
                result.append({
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "price_current": pos.price_current,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "profit": pos.profit + pos.swap,  # Commission not available on open positions
                    "comment": pos.comment
                })
            return result
        except Exception as e:
            self.log(f"Error getting positions: {e}", "ERROR")
            return []


    def get_history_analysis(self, period="all"):
        """Analyze trade history for the frontend"""
        try:
            db = SessionLocal()
            query = db.query(Trade).filter(Trade.status == "CLOSED")
            
            # Date Filtering
            now = datetime.utcnow()
            if period == "day":
                start_time = now - timedelta(days=1)
                query = query.filter(Trade.close_time >= start_time)
            elif period == "week":
                start_time = now - timedelta(weeks=1)
                query = query.filter(Trade.close_time >= start_time)
            elif period == "month":
                start_time = now - timedelta(days=30)
                query = query.filter(Trade.close_time >= start_time)
                
            trades = query.all()
            db.close()
            
            if not trades:
                return {"stats": {}, "trades": [], "by_symbol": []}
                
            # Calculate Stats
            total_profit = sum(t.profit for t in trades)
            total_trades = len(trades)
            wins = len([t for t in trades if t.profit > 0])
            win_rate = round((wins / total_trades) * 100, 1) if total_trades > 0 else 0
            
            # By Symbol
            by_symbol = {}
            for t in trades:
                if t.symbol not in by_symbol:
                    by_symbol[t.symbol] = {"symbol": t.symbol, "profit": 0.0, "wins": 0, "total": 0}
                by_symbol[t.symbol]["profit"] += t.profit
                by_symbol[t.symbol]["total"] += 1
                if t.profit > 0:
                    by_symbol[t.symbol]["wins"] += 1
            
            # Format Trades
            trade_list = []
            for t in trades:
                trade_list.append({
                    "ticket": t.ticket,
                    "symbol": t.symbol,
                    "type": t.type,
                    "volume": t.lot,
                    "profit": t.profit,
                    "time": t.close_time.strftime("%Y-%m-%d %H:%M") if t.close_time else ""
                })
                
            return {
                "stats": {
                    "total_profit": total_profit,
                    "total_trades": total_trades,
                    "win_rate": win_rate
                },
                "trades": sorted(trade_list, key=lambda x: x['time'], reverse=True),
                "by_symbol": list(by_symbol.values())
            }
        except Exception as e:
            self.log(f"Error analyzing history: {e}", "ERROR")
            return {"stats": {}, "trades": [], "by_symbol": []}

    def get_config(self):
        """Return current configuration params"""
        return {
            "global": {
                "daily_loss_limit": DAILY_LOSS_LIMIT,
                "risk_per_trade": 0.01,
                "max_pyramid_layers": 3,
                "pyramid_risk": 0.005
            },
            "symbols": self.params
        }

    def update_config(self, new_config):
        """Update configuration dynamically"""
        if "global" in new_config:
            # Update global vars (in a real app, these should be class attributes)
            pass 
        
        if "symbols" in new_config:
            self.params.update(new_config["symbols"])
            # Save to file
            try:
                with open("gemini_config.json", "w") as f:
                    json.dump(self.params, f, indent=4)
            except Exception as e:
                print(f"Error saving config: {e}")
        
        return True

    def get_strategy_performance(self):
        """Aggregate trade statistics by strategy"""
        try:
            db = SessionLocal()
            trades = db.query(Trade).all()
            db.close()
            
            stats = {}
            for t in trades:
                strat = t.strategy or "Unknown"
                if strat not in stats:
                    stats[strat] = {"name": strat, "trades": 0, "wins": 0, "profit": 0.0}
                
                stats[strat]["trades"] += 1
                stats[strat]["profit"] += t.profit
                if t.profit > 0:
                    stats[strat]["wins"] += 1
            
            result = []
            for s in stats.values():
                s["win_rate"] = round((s["wins"] / s["trades"]) * 100, 1) if s["trades"] > 0 else 0
                result.append(s)
                
            return result
        except Exception as e:
            self.log(f"Error getting strategy stats: {e}", "ERROR")
            return []

    def start(self):
        self.connect()
        self.load_models()
        self.import_history() # Import last 30 days
        self.running = True
        
    def stop(self):
        self.running = False

    def trigger_evolution(self, symbol): # Public wrapper
        self.log(f"Evolution triggered for {symbol}", "EVOLUTION")
        # In a real scenario, this would trigger the genetic algorithm
        # For now, we'll just retrain the model
        threading.Thread(target=self.train_model, args=(symbol,)).start()
        return True

gemini = GeminiCore()
