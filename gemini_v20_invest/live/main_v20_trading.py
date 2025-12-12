"""
V20 Live Trader
Trade CFD indices on MT5 using champion_v20.pth
"""

import MetaTrader5 as mt5
import numpy as np
import time
from datetime import datetime
import sys
sys.path.append('.')
sys.path.append('gemini_v15')

from gemini_v20_invest.models.alphazero_net import AlphaZeroTradingNet
from gemini_v20_invest.data.v20_database import V20Database
from gemini_v20_invest.utils.telegram_notifier import TelegramNotifier
from gemini_v15.utils.indicators import Indicators
from gemini_v20_invest.mcts.config_v20 import *
import torch
import pandas as pd

# Trading symbols
SYMBOLS = {
    'US30.cash': 'DowJones',
    'GER40.cash': 'DAX', 
    'US500.cash': 'S&P500',
}

MAGIC = 2020
TIMEFRAME = mt5.TIMEFRAME_H1  # Start with H1

class V20Trader:
    """
    V20 Trader using champion_v20.pth
    """
    def __init__(self, symbol, model_path):
        self.symbol = symbol
        self.magic = MAGIC
        
        # Load model
        self.model = AlphaZeroTradingNet(input_dim=INPUT_SIZE, action_dim=OUTPUT_POLICY_SIZE)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # DB for predictions
        self.db = V20Database()
        
        print(f"[OK] V20 Trader initialized: {symbol}")
    
    def get_mt5_data(self, bars=200):
        """Get data from MT5"""
        rates = mt5.copy_rates_from_pos(self.symbol, TIMEFRAME, 0, bars)
        if rates is None or len(rates) == 0:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Apply indicators
        df = Indicators.add_all(df)
        
        return df
    
    def get_state(self):
        """Get current state vector"""
        df = self.get_mt5_data(bars=200)
        if df is None:
            return None
        
        # Get latest values of 46 features
        exclude = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
        features = [c for c in df.columns if c not in exclude and not c.startswith('fibo')][:INPUT_SIZE]
        
        state = df.iloc[-1][features].values.astype(np.float32)
        
        if np.isnan(state).any():
            print(f"[WARNING] NaN in state for {self.symbol}")
            return None
        
        return state
    
    def decide(self):
        """Get MCTS decision from model"""
        state = self.get_state()
        if state is None:
            return 6, 0.0  # HOLD, 0 confidence
        
        # Get prediction from Advisory DB
        prediction = self.db.get_latest_prediction(self.symbol)
        
        # Model forward
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            policy, value = self.model(state_tensor)
            policy = policy.squeeze().numpy()
            value = value.item()
        
        # Apply bias from Advisory if available
        if prediction and prediction['confidence'] > 0.6:
            if prediction['signal'] == 'HAUSSIER':
                # Boost BUY actions
                policy[0] *= 1.2  # BUY_25%
                policy[1] *= 1.5  # BUY_50%
                policy[2] *= 1.8  # BUY_100%
            elif prediction['signal'] == 'BAISSIER':
                # Boost SELL actions
                policy[3] *= 1.2  # SELL_25%
                policy[4] *= 1.5  # SELL_50%
                policy[5] *= 1.8  # SELL_100%
            
            # Renormalize
            policy = policy / policy.sum()
        
        # Select action
        action = np.argmax(policy)
        confidence = policy[action]
        
        return action, confidence, value
    
    def execute_trade(self, action):
        """Execute trade on MT5"""
        if action == 6:  # HOLD
            return
        
        # Get current position
        positions = mt5.positions_get(symbol=self.symbol)
        
        # Get account info
        account = mt5.account_info()
        if account is None:
            print(f"[ERROR] Cannot get account info")
            return
        
        balance = account.balance
        
        # Calculate volume based on action
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(f"[ERROR] Symbol {self.symbol} not found")
            return
        
        min_volume = symbol_info.volume_min
        max_volume = symbol_info.volume_max
        
        # Risk mapping
        if action in [0, 3]:   # 25% -> 1% Risk
            risk_pct = 0.01
        elif action in [1, 4]: # 50% -> 2% Risk
            risk_pct = 0.02
        elif action in [2, 5]: # 100% -> 4% Risk
            risk_pct = 0.04
        else:
            return

        risk_amount = balance * risk_pct

        # Calculate SL distance using ATR (Volatility based)
        # We need to fetch data to get ATR
        df = self.get_mt5_data(bars=50)
        if df is None or 'atr' not in df.columns:
            print(f"[ERROR] Could not calculate ATR for {self.symbol}")
            return
        
        atr = df.iloc[-1]['atr']
        sl_distance = atr * 2.0  # SL at 2x ATR
        
        # Calculate Volume based on Risk
        # Risk = Volume * (SL_Distance / Tick_Size) * Tick_Value
        tick_size = symbol_info.trade_tick_size
        tick_value = symbol_info.trade_tick_value
        
        if tick_size == 0 or tick_value == 0:
            print(f"[ERROR] Invalid tick info for {self.symbol}")
            return
            
        # Volume formula
        raw_volume = risk_amount / ((sl_distance / tick_size) * tick_value)
        
        # Normalize volume
        step = symbol_info.volume_step
        volume = round(raw_volume / step) * step
        volume = max(min_volume, min(volume, max_volume))
        
        # Calculate SL/TP Prices
        if action <= 2:  # BUY
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(self.symbol).ask
            sl_price = price - sl_distance
            tp_price = price + (sl_distance * 2)  # R:R 1:2
        else:  # SELL
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(self.symbol).bid
            sl_price = price + sl_distance
            tp_price = price - (sl_distance * 2)  # R:R 1:2
        
        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "magic": self.magic,
            "comment": f"V20_{ACTIONS[action]}_R{risk_pct*100:.0f}%",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ {self.symbol}: {ACTIONS[action]} @ {price} (Vol: {volume})")
            return True
        else:
            print(f"❌ {self.symbol}: Order failed - {result.comment}")
            return False

class V20Orchestrator:
    """
    Orchestrator for multi-symbol V20 trading
    """
    def __init__(self, symbols, model_path):
        self.symbols = symbols
        self.model_path = model_path
        self.traders = {}
        self.telegram = TelegramNotifier()
        
        # Initialize MT5
        if not mt5.initialize():
            print("❌ MT5 initialization failed")
            return
        
        # Create traders
        for symbol in symbols:
            self.traders[symbol] = V20Trader(symbol, model_path)
        
        print(f"\n🚀 V20 Orchestrator ready - {len(symbols)} symbols")
    
    def run(self):
        """Main trading loop"""
        print("\n" + "="*60)
        print("🤖 V20 TRADING SYSTEM STARTED")
        print("="*60)
        
        iteration = 0
        
        while True:
            iteration += 1
            print(f"\n[Iteration {iteration}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            for symbol, trader in self.traders.items():
                try:
                    action, confidence, value = trader.decide()
                    
                    print(f"{symbol}: {ACTIONS[action]} (Conf: {confidence:.2f}, Value: {value:.2f})")
                    
                    if action != 6 and confidence > 0.7:  # Execute if not HOLD and high confidence
                        trader.execute_trade(action)
                    
                except Exception as e:
                    print(f"[ERROR] {symbol}: {e}")
            
            # Sleep 5 minutes
            print(f"\n⏳ Next analysis in 5 minutes...")
            time.sleep(300)

if __name__ == "__main__":
    MODEL_PATH = "gemini_v20_invest/models/champions/champion_v20.pth"
    
    orchestrator = V20Orchestrator(
        symbols=list(SYMBOLS.keys()),
        model_path=MODEL_PATH
    )
    
    orchestrator.run()
