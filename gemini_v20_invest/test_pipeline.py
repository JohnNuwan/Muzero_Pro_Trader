from gemini_v20_invest.data.stock_loader import StockDataLoader
from gemini_v20_invest.utils.telegram_notifier import TelegramNotifier
import time

def test_pipeline():
    print("🧪 Starting V20 Pipeline Test...")
    
    # 1. Initialize Components
    loader = StockDataLoader()
    notifier = TelegramNotifier()
    
    # 2. Fetch Data (Real yfinance)
    symbol = "AAPL"
    print(f"\n📥 Fetching data for {symbol}...")
    df = loader.get_data(symbol)
    
    if df is not None:
        last_price = df['Close'].iloc[-1]
        print(f"✅ Data received. Last Price: {last_price:.2f}")
        
        # 3. Fetch Fundamentals
        print(f"📥 Fetching fundamentals...")
        fund = loader.get_fundamentals(symbol)
        pe_ratio = fund.get('pe_ratio', 'N/A')
        print(f"✅ Fundamentals received. P/E: {pe_ratio}")
        
        # 4. Generate Dummy Signal
        print(f"\n🤖 Generating Test Signal...")
        signal = {
            'symbol': symbol,
            'action': 'BUY_25% (TEST)',
            'confidence': 0.92,
            'price': last_price,
            'target': last_price * 1.15,
            'stop_loss': last_price * 0.90,
            'analysis': {
                'P/E Ratio': pe_ratio,
                'Sector': fund.get('sector', 'Tech'),
                'RSI': '42 (Simulated)',
                'Trend': 'Bullish (Simulated)'
            },
            'reason': 'This is a TEST signal to validate V20 pipeline integration.'
        }
        
        # 5. Send Notification
        print(f"📤 Sending Telegram Notification...")
        notifier.send_investment_advice(signal)
        print("✅ Notification sent!")
        
    else:
        print("❌ Failed to fetch data")

if __name__ == "__main__":
    test_pipeline()
