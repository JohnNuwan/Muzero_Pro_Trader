import torch
import os
import sys
import time
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.adversarial_env import AdversarialEnv
from training.train_alphazero import train_alphazero
from utils.config import TRAINING_CONFIG

def train_adversarial(symbol="EURUSD", run_name="alpha_v19_adversarial"):
    print(f"🛡️ Starting Adversarial Training: {run_name}")
    
    # 1. Create Adversarial Env
    # Start with low strength and increase
    env = AdversarialEnv(symbol=symbol, adversary_strength=0.1)
    
    # 2. Run Training Loop
    # We reuse the AlphaZero training loop but with the adversarial env
    # The 'train_alphazero' function handles the network, MCTS, and self-play
    
    # We might want to curriculum learning: increase strength over iterations
    # But train_alphazero is a single call. 
    # Let's just run it with the fixed strength for now, or modify train_alphazero to accept a callback.
    # For V19 Phase 4, simple integration is fine.
    
    train_alphazero(env, run_name=run_name)
    
if __name__ == "__main__":
    # Check if MT5 is initialized (needed for Env)
    import MetaTrader5 as mt5
    if not mt5.initialize():
        print("❌ MT5 Init Failed")
        sys.exit(1)
        
    train_adversarial()
