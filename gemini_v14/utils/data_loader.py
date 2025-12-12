import MetaTrader5 as mt5
import pandas as pd
import os
from datetime import datetime, timedelta
from rich.console import Console

console = Console()

class DataLoader:
    def __init__(self, data_dir="gemini_v13/data"):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def fetch_data(self, symbol, timeframe, days=30):
        """Télécharge les données historiques depuis MT5"""
        if not mt5.initialize():
            console.print(f"[red]Erreur init MT5: {mt5.last_error()}[/red]")
            return None
            
        utc_from = datetime.now() - timedelta(days=days)
        # Map string timeframe to MT5 constant
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "D1": mt5.TIMEFRAME_D1
        }
        mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_M5)
        
        # Utilisation de copy_rates_from_pos (plus fiable) : start_pos=0 (actuel), count=10000
        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, 10000) 
        
        if rates is None:
            console.print(f"[red]Aucune donnée pour {symbol} (Erreur: {mt5.last_error()})[/red]")
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Sauvegarde CSV
        filename = f"{self.data_dir}/{symbol}_{timeframe}_data.csv"
        df.to_csv(filename, index=False)
        console.print(f"[green]Données sauvegardées : {filename} ({len(df)} bougies)[/green]")
        
        return df

    def load_data(self, symbol, timeframe):
        """Charge les données depuis le CSV local"""
        filename = f"{self.data_dir}/{symbol}_{timeframe}_data.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df['time'] = pd.to_datetime(df['time'])
            return df
        else:
            console.print(f"[yellow]Fichier non trouvé, téléchargement...[/yellow]")
            return self.fetch_data(symbol, timeframe)

    def add_indicators(self, df):
        """Ajoute les indicateurs techniques (Les 'Yeux' de l'Agent)"""
        # 1. Returns & Delta
        df['returns'] = df['close'].pct_change()
        df['delta'] = df['close'] - df['open'] # Simple candle delta
        
        # 2. SMA (Trend)
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['trend'] = 0
        df.loc[df['sma_20'] > df['sma_50'], 'trend'] = 1 # Uptrend
        df.loc[df['sma_20'] < df['sma_50'], 'trend'] = -1 # Downtrend
        
        # 3. RSI (Momentum)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
        
        # 4. Volatility (ATR approx & StdDev)
        df['volatility'] = df['high'] - df['low']
        df['volatility_sma'] = df['volatility'].rolling(window=14).mean()
        df['std_dev'] = df['close'].rolling(window=20).std()
        
        # 5. Z-Score (Mean Reversion)
        # Z = (Price - SMA) / StdDev
        df['z_score'] = (df['close'] - df['sma_20']) / df['std_dev']
        df['z_score'] = df['z_score'].fillna(0)
        
        # 6. Fibonacci Retracement (Dynamic)
        # On prend le High/Low des 100 dernières bougies
        df['roll_high'] = df['high'].rolling(window=100).max()
        df['roll_low'] = df['low'].rolling(window=100).min()
        df['fibo_pos'] = (df['close'] - df['roll_low']) / (df['roll_high'] - df['roll_low']) # 0 to 1 position
        df['fibo_pos'] = df['fibo_pos'].fillna(0.5)
        
        # Nettoyage NaN
        df = df.dropna()
        return df
