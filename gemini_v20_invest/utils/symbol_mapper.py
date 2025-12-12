"""
Symbol mapper between yfinance tickers and Trade Republic ISINs
"""

# Major US Stocks
US_STOCKS = {
    'AAPL': 'US0378331005',   # Apple
    'MSFT': 'US5949181045',   # Microsoft
    'GOOGL': 'US02079K3059',  # Alphabet (Google)
    'AMZN': 'US0231351067',   # Amazon
    'TSLA': 'US88160R1014',   # Tesla
    'NVDA': 'US67066G1040',   # NVIDIA
    'META': 'US30303M1027',   # Meta (Facebook)
    'NFLX': 'US64110L1061',   # Netflix
    'AMD': 'US0079031078',    # AMD
}

# French CAC 40
FR_STOCKS = {
    'MC.PA': 'FR0000121014',    # LVMH
    'OR.PA': 'FR0000120321',    # L'Oréal
    'SAN.PA': 'FR0000120578',   # Sanofi
    'TTE.PA': 'FR0000120271',   # TotalEnergies
    'AI.PA': 'FR0000120073',    # Air Liquide
}

# ETFs
ETFS = {
    'CW8.PA': 'IE00B4L5Y983',   # iShares MSCI World
    'PE500.PA': 'FR0011550177', # Amundi S&P 500
}

# Combine all
SYMBOL_TO_ISIN = {**US_STOCKS, **FR_STOCKS, **ETFS}
ISIN_TO_SYMBOL = {v: k for k, v in SYMBOL_TO_ISIN.items()}

def get_isin(symbol):
    """Convert yfinance symbol to ISIN"""
    return SYMBOL_TO_ISIN.get(symbol)

def get_symbol(isin):
    """Convert ISIN to yfinance symbol"""
    return ISIN_TO_SYMBOL.get(isin)

if __name__ == "__main__":
    # Test
    print(f"AAPL ISIN: {get_isin('AAPL')}")
    print(f"US0378331005 Symbol: {get_symbol('US0378331005')}")
