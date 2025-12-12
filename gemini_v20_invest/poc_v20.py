# V20-Invest: POC with V19 Indicators

import sys
import warnings
# Suppress asyncio warnings from yfinance/telegram interaction
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*Event loop is closed.*')

sys.path.append('gemini_v15')  # Access V19's indicators

from gemini_v20_invest.data.stock_loader import StockDataLoader
from gemini_v20_invest.utils.telegram_notifier import TelegramNotifier
from gemini_v20_invest.data.v20_database import V20Database
from gemini_v15.utils.indicators import Indicators
import numpy as np

def analyze_stock(symbol, db):
    """
    Analyze a stock using V19 indicators and generate a signal
    """
    loader = StockDataLoader()
    
    # 1. Fetch Data
    print(f"\n{'='*50}")
    print(f"📊 Analyse de {symbol}")
    print(f"{'='*50}\n")
    
    df = loader.get_data(symbol, period="1y", interval="1d", force_refresh=True)
    if df is None:
        return None
    
    # Normalize column names for V19 indicators (lowercase)
    df.columns = df.columns.str.lower()
    
    # yfinance uses 'volume', MT5 uses 'tick_volume' - create alias
    if 'volume' in df.columns and 'tick_volume' not in df.columns:
        df['tick_volume'] = df['volume']
    
    # Add 'spread' column (stocks don't have spread like Forex, use 0)
    if 'spread' not in df.columns:
        df['spread'] = 0
    
    # 2. Apply V19 Indicators
    print("🔧 Application des 26 indicateurs V19...")
    df = Indicators.add_all(df)
    
    # 3. Get latest values
    latest = df.iloc[-1]
    last_price = latest['close']  # lowercase after normalization
    
    # Key indicators
    rsi = latest.get('rsi', 50)
    stoch = latest.get('stoch_k', 0.5)
    bb_b = latest.get('bb_b', 0.5)
    macd_signal = latest.get('macd_signal', 0)
    trend = latest.get('adx', 0)
    
    # 4. Get Fundamentals
    fund = loader.get_fundamentals(symbol)
    pe_ratio = fund.get('pe_ratio', 'N/A')
    sector = fund.get('sector', 'Inconnu')
    
    # 5. Simple Signal Logic (V19-style)
    confidence = 0
    action = "HOLD"
    
    # Bullish conditions
    if rsi < 45 and stoch < 0.3 and bb_b < 0.3:
        action = "BUY_50%"
        confidence = 0.75
    elif rsi < 35:
        action = "BUY_100%"
        confidence = 0.90
    # Bearish conditions
    elif rsi > 70 and stoch > 0.8:
        action = "SELL_50%"
        confidence = 0.80
    elif rsi > 80:
        action = "SELL_100%"
        confidence = 0.95
    
    if action == "HOLD":
        print(f"⚪ Signal: HOLD (RSI: {rsi:.0f})")
        return None
    
    # 6. Build Signal
    buy_or_sell = 'survente' if 'BUY' in action else 'surachat'
    opportunity = 'Forte opportunité d\'achat' if confidence > 0.85 else 'Bon point d\'entrée' if 'BUY' in action else 'Sortie recommandée'
    
    signal = {
        'symbol': symbol,
        'action': action,
        'confidence': confidence,
        'price': last_price,
        'target': last_price * 1.15 if 'BUY' in action else last_price * 0.90,
        'stop_loss': last_price * 0.92 if 'BUY' in action else last_price * 1.08,
        'analysis': {
            'RSI': f"{rsi:.0f} ({'Survente' if rsi < 30 else 'Surachat' if rsi > 70 else 'Neutre'})",
            'Stochastic': f"{stoch*100:.0f}%",
            'Bollinger': f"{bb_b:.2f}",
            'MACD': f"{'Haussier' if macd_signal > 0 else 'Baissier'}",
            'Trend (ADX)': f"{trend:.0f}",
            'P/E Ratio': pe_ratio,
            'Sector': sector
        },
        'reason': f"Conditions de {buy_or_sell} détectées. RSI à {rsi:.0f}, "
                  f"Stochastique à {stoch*100:.0f}%. {opportunity}."
    }
    
    print(f"🚀 Signal: {action} | Confiance: {confidence*100:.0f}%")
    
    # Save to database
    signal_id = db.save_signal(signal)
    print(f"💾 Signal sauvegardé (ID: {signal_id})")
    
    return signal

def run_v20_poc():
    """Run V20 POC analysis on select stocks"""
    notifier = TelegramNotifier()
    db = V20Database()
    
    # Test universe
    symbols = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]
    
    print("🧪 V20-INVEST POC")
    print(f"Analyse de {len(symbols)} actions...\n")
    
    signals = []
    for symbol in symbols:
        signal = analyze_stock(symbol, db)
        if signal:
            signals.append(signal)
            # Send notification (without portfolio data for now)
            notifier.send_investment_advice(signal)
            print(f"📤 Notification Telegram envoyée pour {symbol}\n")
    
    print(f"\n{'='*50}")
    print(f"✅ Analyse terminée: {len(signals)} signal(aux) généré(s)")
    print(f"{'='*50}")

if __name__ == "__main__":
    run_v20_poc()

if __name__ == "__main__":
    run_v20_poc()
