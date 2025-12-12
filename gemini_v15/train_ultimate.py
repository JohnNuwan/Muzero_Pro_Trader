import os
import sys
import time
import MetaTrader5 as mt5
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from gemini_v15.environment.deep_trinity_env import DeepTrinityEnv
from gemini_v15.environment.risk_trinity_env import RiskTrinityEnv

# Major Forex, Crypto, Gold, Indices
SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", 
    "BTCUSD", "ETHUSD", 
    "XAUUSD", 
    "US30.cash", "GER40.cash", "US500.cash", "US100.cash"
]
TRAIN_STEPS = 20000 # Increase to 1,000,000 for "Grand Master" level

def train_symbol(symbol):
    print(f"\n🌟 Starting Training for {symbol}...")
    
    # --- 1. Train Trader (PPO) ---
    print(f"  🤖 Training Trader (PPO) for {symbol}...")
    try:
        env_trader = DeepTrinityEnv(symbol=symbol, lookback=2000)
        vec_env_trader = DummyVecEnv([lambda: env_trader])
        
        model_trader = PPO("MlpPolicy", vec_env_trader, verbose=0, learning_rate=0.0003)
        model_trader.learn(total_timesteps=TRAIN_STEPS)
        
        # Save
        path_trader = os.path.join(current_dir, "models", f"ppo_v15_{symbol}")
        model_trader.save(path_trader)
        print(f"  ✅ Trader Saved: {path_trader}.zip")
        
    except Exception as e:
        print(f"  ❌ Trader Training Failed: {e}")

    # --- 2. Train Risk Manager (SAC) ---
    print(f"  🛡️ Training Risk Manager (SAC) for {symbol}...")
    try:
        env_risk = RiskTrinityEnv(symbol=symbol, lookback=2000)
        vec_env_risk = DummyVecEnv([lambda: env_risk])
        
        model_risk = SAC("MlpPolicy", vec_env_risk, verbose=0)
        model_risk.learn(total_timesteps=TRAIN_STEPS)
        
        # Save
        path_risk = os.path.join(current_dir, "models", f"sac_v15_{symbol}")
        model_risk.save(path_risk)
        print(f"  ✅ Risk Manager Saved: {path_risk}.zip")
        
    except Exception as e:
        print(f"  ❌ Risk Manager Training Failed: {e}")

def main():
    print("🚀 LAUNCHING ULTIMATE TRAINING SESSION V15 🚀")
    print(f"Symbols: {SYMBOLS}")
    print(f"Steps per Model: {TRAIN_STEPS}")
    
    if not mt5.initialize():
        print("❌ Failed to initialize MT5")
        return

    start_total = time.time()
    
    for symbol in SYMBOLS:
        train_symbol(symbol)
        
    end_total = time.time()
    duration = (end_total - start_total) / 60
    print(f"\n🎉 GRAND TRAINING COMPLETE in {duration:.2f} minutes!")
    mt5.shutdown()

if __name__ == "__main__":
    main()
