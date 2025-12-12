from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from gemini_core import gemini
import threading
import time
import uvicorn
from database import init_db

app = FastAPI(title="Gemini V12 Command Center")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Background Thread for Trading Loop
def trading_loop():
    while True:
        if gemini.running:
            try:
                gemini.tick()
            except Exception as e:
                print(f"Loop Error: {e}")
        time.sleep(1)

@app.on_event("startup")
async def startup_event():
    init_db()
    t = threading.Thread(target=trading_loop, daemon=True)
    t.start()
    gemini.start()  # 🚀 AUTO-START BOT!
    print("🚀 GEMINI V12 API & BOT STARTED!")

# --- ENDPOINTS ---

@app.get("/api/status")
def get_status():
    stats = gemini.get_daily_stats()
    account = gemini.get_account_info()
    return {
        "running": gemini.running,
        "daily_stats": stats,
        "account": account,
        "evolving": list(gemini.evolving_symbols.keys())
    }

@app.get("/api/positions")
def get_positions():
    return gemini.get_positions()

@app.get("/api/market")
def get_market():
    return gemini.get_market_state()

@app.get("/api/logs")
def get_logs():
    return gemini.logs

@app.get("/api/history")
def get_history():
    return gemini.get_recent_trades(limit=50)

@app.get("/api/history/analysis")
def get_history_analysis(period: str = "all"):
    return gemini.get_history_analysis(period)

@app.get("/api/config")
def get_config():
    return gemini.get_config()

@app.post("/api/config/update")
def update_config(config: dict):
    gemini.update_config(config)
    return {"status": "updated"}

@app.post("/api/control/start")
def start_bot():
    gemini.start()
    return {"status": "started"}

@app.post("/api/control/stop")
def stop_bot():
    gemini.stop()
    return {"status": "stopped"}

@app.post("/api/control/evolve/{symbol}")
def trigger_evolution(symbol: str):
    if gemini.trigger_evolution(symbol):
        return {"status": "triggered", "symbol": symbol}
    return {"status": "failed", "reason": "Already evolving or invalid"}

@app.post("/api/control/import_history")
def import_history(days: int = 30):
    gemini.import_history(days)
    return {"status": "imported", "days": days}

@app.get("/api/strategies")
def get_strategies():
    return gemini.get_strategy_performance()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
