import MetaTrader5 as mt5
from datetime import datetime, timedelta
import json

def check_comments():
    with open("gemini_config.json", "r") as f:
        config = json.load(f)
        
    LOGIN = 51162779
    PASSWORD = " m8hJ!cK9"
    SERVER = "FTMO-Demo"
    
    if not mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER):
        print("MT5 Init Failed")
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
