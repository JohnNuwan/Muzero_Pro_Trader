import MetaTrader5 as mt5
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_comments():
    # Retrieve credentials from environment variables
    LOGIN = os.getenv("MT5_LOGIN")
    PASSWORD = os.getenv("MT5_PASSWORD")
    SERVER = os.getenv("MT5_SERVER")

    if not LOGIN or not PASSWORD or not SERVER:
        print("Error: MT5 credentials not found in environment variables.")
        return

    try:
        LOGIN = int(LOGIN)
    except ValueError:
        print("Error: MT5_LOGIN must be an integer.")
        return
    
    if not mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER):
        print(f"MT5 Init Failed: {mt5.last_error()}")
        return

    from_date = datetime.now() - timedelta(days=30)
    deals = mt5.history_deals_get(from_date, datetime.now())
    
    if deals is None:
        print("No deals found")
        return
        
    print(f"Found {len(deals)} deals")
    
    count = 0
    for deal in deals:
        if deal.entry == mt5.DEAL_ENTRY_IN:
            print(f"Ticket: {deal.ticket}, Symbol: {deal.symbol}, Comment: '{deal.comment}'")
            count += 1
            if count > 20: break
            
    mt5.shutdown()

if __name__ == "__main__":
    check_comments()
