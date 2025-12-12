"""
Trade Republic API Connector for V20-Invest
Read-only: portfolio, cash balance, positions
"""
import asyncio
import os
from dotenv import load_dotenv

class TradeRepublicConnector:
    """
    Connector to Trade Republic for portfolio data (read-only)
    """
    def __init__(self):
        load_dotenv()
        self.phone = os.getenv('TR_PHONE_NUMBER')
        self.pin = os.getenv('TR_PIN')
        
        if not self.phone or not self.pin:
            print("WARNING: Trade Republic credentials not found in .env")
            self.enabled = False
        else:
            self.enabled = True
            # Import here to avoid error if not installed
            try:
                from trapi import TRApi  # Module name is trapi
                self.api = TRApi(self.phone, self.pin)
                self.api.login()
                print(f"[OK] Trade Republic connected: {self.phone[-4:]}")
            except Exception as e:
                print(f"ERROR: Trade Republic connection failed: {e}")
                self.enabled = False
    
    async def get_cash_balance(self):
        """Get available cash in EUR"""
        if not self.enabled:
            return None
        try:
            cash_data = await self.api.cash()
            # Extract cash amount from response
            return cash_data.get('availableCash', 0.0)
        except Exception as e:
            print(f"ERROR getting cash: {e}")
            return None
    
    async def get_portfolio(self):
        """Get all positions"""
        if not self.enabled:
            return []
        try:
            portfolio = await self.api.portfolio()
            return portfolio
        except Exception as e:
            print(f"ERROR getting portfolio: {e}")
            return []
    
    async def get_position(self, isin):
        """Get position for specific ISIN"""
        portfolio = await self.get_portfolio()
        for position in portfolio:
            if position.get('isin') == isin:
                return position
        return None
    
    async def get_current_price(self, isin):
        """Get real-time price for ISIN"""
        if not self.enabled:
            return None
        try:
            ticker_data = await self.api.ticker(isin)
            return ticker_data.get('last', {}).get('price')
        except Exception as e:
            print(f"ERROR getting price for {isin}: {e}")
            return None

# Synchronous wrapper for easy use
def get_portfolio_sync():
    """Synchronous wrapper to get portfolio"""
    connector = TradeRepublicConnector()
    if not connector.enabled:
        return None, []
    
    async def run():
        cash = await connector.get_cash_balance()
        portfolio = await connector.get_portfolio()
        return cash, portfolio
    
    return asyncio.run(run())

def get_cash_sync():
    """Synchronous wrapper to get just cash"""
    connector = TradeRepublicConnector()
    if not connector.enabled:
        return None
    
    async def run():
        return await connector.get_cash_balance()
    
    return asyncio.run(run())

if __name__ == "__main__":
    # Test
    cash, portfolio = get_portfolio_sync()
    print(f"Cash: {cash}€")
    print(f"Positions: {len(portfolio)}")
