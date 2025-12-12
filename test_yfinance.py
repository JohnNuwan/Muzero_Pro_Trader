import yfinance as yf
print("Start")
try:
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="1d")
    print(df.head())
    print("Success")
except Exception as e:
    print(f"Error: {e}")
