import os
import sys

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from gemini_v15.environment.deep_trinity_env import DeepTrinityEnv
import MetaTrader5 as mt5
import time

def train():
    print("🚀 Starting Gemini V15 Training Session...")
    
    # Initialize MT5 (Required for Data Loader)
    if not mt5.initialize():
        print("❌ Failed to initialize MT5")
        return

    try:
        # 1. Create Environment
        # We use a larger lookback for training to give the agent enough history
        print("Creating DeepTrinityEnv...")
        env = DeepTrinityEnv(symbol="EURUSD", lookback=2000)
        
        # Vectorize the environment (Required for SB3)
        vec_env = DummyVecEnv([lambda: env])
        
        # 2. Initialize Agent (PPO)
        # MlpPolicy = Multi-Layer Perceptron (Standard Neural Net)
        # verbose=1 shows progress
        print("Initializing PPO Agent...")
        model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64)
        
        # 3. Train
        print("🧠 Training Started...")
        start_time = time.time()
        
        # Train for 10,000 timesteps (Short run for verification)
        # In production, this would be 1M+
        model.learn(total_timesteps=10000)
        
        end_time = time.time()
        print(f"✅ Training Completed in {end_time - start_time:.2f} seconds")
        
        # 4. Save Model
        models_dir = os.path.join(current_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "ppo_v15_eurusd")
        model.save(model_path)
        print(f"💾 Model saved to {model_path}.zip")
        
        # 5. Quick Evaluation
        print("\n🔍 Running Evaluation Episode...")
        obs = vec_env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 100:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = vec_env.step(action)
            total_reward += rewards[0]
            steps += 1
            if dones[0]:
                done = True
                
        print(f"Evaluation Result: Total Reward = {total_reward:.2f} over {steps} steps")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    train()
