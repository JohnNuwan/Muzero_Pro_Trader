import gymnasium as gym
from gymnasium import spaces
import numpy as np
from rich.console import Console
import MetaTrader5 as mt5
import pandas as pd
import time

class MT5TradingEnv(gym.Env):
    """
    Environnement de Trading MT5 compatible Gymnasium (Live Mode).
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, symbol, timeframe, deposit=10000.0):
        super(MT5TradingEnv, self).__init__()
        self.symbol = symbol
        self.timeframe = timeframe
        self.deposit = deposit
        self.balance = deposit
        self.equity = deposit
        self.positions = []
        
        # Action Space: 0=Hold, 1=Buy, 2=Sell, 3=Close
        self.action_space = spaces.Discrete(4)
        
        # Observation Space: [Price, RSI, Trend, Volatility, Z-Score, Fibo_Pos, Pos_Type, Pos_Profit]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        self.console = Console()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.deposit
        self.equity = self.deposit
        self.positions = []
        
        # Get initial state from MT5
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        self._take_action(action)
        observation = self._get_observation()
        reward = self._calculate_reward()
        terminated = False # En live, on ne termine jamais vraiment sauf stop manuel
        truncated = False
        info = {"log": getattr(self, "last_log", "")}
        
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        # 1. Get Market Data
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 100) # Need history for indicators
        if rates is None or len(rates) < 100:
            return np.zeros(8, dtype=np.float32)
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # 2. Calculate Indicators (Same as data_loader.py)
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Trend (SMA)
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Volatility
        df['volatility'] = df['high'] - df['low']
        df['volatility_sma'] = df['volatility'].rolling(window=14).mean()
        
        # Z-Score
        df['std_dev'] = df['close'].rolling(window=20).std()
        df['z_score'] = (df['close'] - df['sma_20']) / df['std_dev']
        
        # Fibo
        roll_high = df['high'].rolling(window=100).max()
        roll_low = df['low'].rolling(window=100).min()
        df['fibo_pos'] = (df['close'] - roll_low) / (roll_high - roll_low)
        
        # Current Values
        current = df.iloc[-1]
        price = current['close']
        rsi = current['rsi'] if not np.isnan(current['rsi']) else 50.0
        
        trend = 0
        if current['sma_20'] > current['sma_50']: trend = 1
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from rich.console import Console
import MetaTrader5 as mt5
import pandas as pd
import time

class MT5TradingEnv(gym.Env):
    """
    Environnement de Trading MT5 compatible Gymnasium (Live Mode).
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, symbol, timeframe, deposit=10000.0):
        super(MT5TradingEnv, self).__init__()
        self.symbol = symbol
        self.timeframe = timeframe
        self.deposit = deposit
        self.balance = deposit
        self.equity = deposit
        self.positions = []
        
        # Action Space: 0=Hold, 1=Buy, 2=Sell, 3=Close
        self.action_space = spaces.Discrete(4)
        
        # Observation Space: [Price, RSI, Trend, Volatility, Z-Score, Fibo_Pos, Pos_Type, Pos_Profit]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        self.console = Console()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.deposit
        self.equity = self.deposit
        self.positions = []
        
        # Get initial state from MT5
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        self._take_action(action)
        observation = self._get_observation()
        reward = self._calculate_reward()
        terminated = False # En live, on ne termine jamais vraiment sauf stop manuel
        truncated = False
        info = {"log": getattr(self, "last_log", "")}
        
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        # 1. Get Market Data
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 100) # Need history for indicators
        if rates is None or len(rates) < 100:
            return np.zeros(8, dtype=np.float32)
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # 2. Calculate Indicators (Same as data_loader.py)
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Trend (SMA)
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Volatility
        df['volatility'] = df['high'] - df['low']
        df['volatility_sma'] = df['volatility'].rolling(window=14).mean()
        
        # Z-Score
        df['std_dev'] = df['close'].rolling(window=20).std()
        df['z_score'] = (df['close'] - df['sma_20']) / df['std_dev']
        
        # Fibo
        roll_high = df['high'].rolling(window=100).max()
        roll_low = df['low'].rolling(window=100).min()
        df['fibo_pos'] = (df['close'] - roll_low) / (roll_high - roll_low)
        
        # Current Values
        current = df.iloc[-1]
        price = current['close']
        rsi = current['rsi'] if not np.isnan(current['rsi']) else 50.0
        
        trend = 0
        if current['sma_20'] > current['sma_50']: trend = 1
        elif current['sma_20'] < current['sma_50']: trend = -1
        
        volatility = current['volatility_sma'] if not np.isnan(current['volatility_sma']) else 0.0
        z_score = current['z_score'] if not np.isnan(current['z_score']) else 0.0
        fibo_pos = current['fibo_pos'] if not np.isnan(current['fibo_pos']) else 0.5
        
        # 3. Position State (REAL MT5 POSITIONS)
        pos_type = 0
        pos_profit = 0.0
        
        # Filter by Symbol AND Magic Number
        positions = mt5.positions_get(symbol=self.symbol)
        my_position = None
        
        if positions:
            for pos in positions:
                if pos.magic == 13000: # MAGIC_NUMBER hardcoded or imported
                    my_position = pos
                    break
        
        if my_position:
            if my_position.type == mt5.POSITION_TYPE_BUY:
                pos_type = 1
            elif my_position.type == mt5.POSITION_TYPE_SELL:
                pos_type = -1
            pos_profit = my_position.profit
        
        # Obs: [Price, RSI, Trend, Volatility, Z-Score, Fibo_Pos, Pos_Type, Pos_Profit]
        return np.array([price, rsi, trend, volatility, z_score, fibo_pos, pos_type, pos_profit], dtype=np.float32)

    def _take_action(self, action):
        # 1. Exécution de l'ordre (RÉEL)
        from config import LOT_SIZE, MAGIC_NUMBER, DEVIATION
        
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick: return
        
        self.last_log = ""
        
        # Check existing position
        positions = mt5.positions_get(symbol=self.symbol)
        my_position = None
        if positions:
            for pos in positions:
                if pos.magic == MAGIC_NUMBER:
                    my_position = pos
                    break
        
        if action == 1: # Buy
            if not my_position:
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": LOT_SIZE,
                    "type": mt5.ORDER_TYPE_BUY,
                    "price": tick.ask,
                    "sl": 0.0, # Pas de SL pour l'instant (géré par l'agent)
                    "tp": 0.0,
                    "deviation": DEVIATION,
                    "magic": MAGIC_NUMBER,
                    "comment": "Gemini V13 Buy",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    self.last_log = f"[red]BUY FAIL: {result.comment}[/red]"
                else:
                    self.last_log = f"[green]BUY EXEC @ {tick.ask:.5f}[/green]"
                    
        elif action == 2: # Sell
            if not my_position:
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": LOT_SIZE,
                    "type": mt5.ORDER_TYPE_SELL,
                    "price": tick.bid,
                    "sl": 0.0,
                    "tp": 0.0,
                    "deviation": DEVIATION,
                    "magic": MAGIC_NUMBER,
                    "comment": "Gemini V13 Sell",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    self.last_log = f"[red]SELL FAIL: {result.comment}[/red]"
                else:
                    self.last_log = f"[red]SELL EXEC @ {tick.bid:.5f}[/red]"
                    
        elif action == 3: # Close
            if my_position:
                # FTMO Rule: Minimum Hold Time (1 minute)
                # Note: mt5.positions_get returns timestamp in seconds
                duration = time.time() - my_position.time
                if duration < 60:
                    self.last_log = f"[yellow]HOLDING (FTMO Rule: {int(duration)}s < 60s)[/yellow]"
                    return

                type_close = mt5.ORDER_TYPE_SELL if my_position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                price_close = tick.bid if my_position.type == mt5.POSITION_TYPE_BUY else tick.ask
                
                current_profit = my_position.profit
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": my_position.volume,
                    "type": type_close,
                    "position": my_position.ticket,
                    "price": price_close,
                    "deviation": DEVIATION,
                    "magic": MAGIC_NUMBER,
                    "comment": "Gemini V13 Close",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    self.last_log = f"[red]CLOSE FAIL: {result.comment}[/red]"
                else:
                    # On utilise le profit capturé juste avant la fermeture pour l'affichage
                    color = "green" if current_profit >= 0 else "red"
                    self.last_log = f"[{color}]CLOSE EXEC ({int(duration)}s) PnL: {current_profit:.2f}[/{color}]"

    def _calculate_reward(self):
        # Reward is the realized PnL of the last action
        # En mode réel, c'est plus dur à tracker instantanément sans lire l'historique
        # On retourne 0 pour l'instant car on ne fait pas d'apprentissage en ligne (Online Learning)
        return 0.0
