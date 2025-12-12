import os
import sys
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from gemini_v15.environment.commission_trinity_env import CommissionTrinityEnv
import MetaTrader5 as mt5

# Configuration
SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", 
    "BTCUSD", "ETHUSD", 
    "XAUUSD", 
    "US30.cash", "GER40.cash", "US500.cash", "US100.cash"
]
TRAIN_STEPS = 50000 
MODELS_DIR = os.path.join(current_dir, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_v16():
    if not mt5.initialize():
        print("❌ MT5 Init Failed")
        return

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting V16 Training (Commission-Aware) on {device.upper()}...")
    if device == "cuda":
        print(f"🔥 GPU Detected: {torch.cuda.get_device_name(0)}")
    
    print(f"🎯 Rules: SL=10, TP=100, Reward=+0.25%, Penalty=-0.15%")
    print(f"🎰 Super Bonuses: +0.5% (+5pts), +1.0% (+10pts)")

    for symbol in SYMBOLS:
        print(f"\n🧠 Training V16 Agent for {symbol}...")
        
        try:
            # Create Env
            env = DummyVecEnv([lambda: CommissionTrinityEnv(symbol, lookback=2000)])
            
            # Initialize PPO
            model = PPO(
                "MlpPolicy", 
                env, 
                verbose=1,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                gamma=0.99,
                ent_coef=0.01,
                device="auto" 
            )
            
            # Train
            model.learn(total_timesteps=TRAIN_STEPS)
            
            # Save
            path = os.path.join(MODELS_DIR, f"ppo_v16_{symbol}")
            model.save(path)
            print(f"✅ Saved V16 Model: {path}")
            
        except Exception as e:
            print(f"❌ Error training {symbol}: {e}")
            import traceback
            traceback.print_exc()

    mt5.shutdown()
    print("\n🎉 V16 Training Complete!")

if __name__ == "__main__":
    train_v16()
