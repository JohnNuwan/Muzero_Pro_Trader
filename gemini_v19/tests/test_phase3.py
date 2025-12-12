import torch
import numpy as np
import sys
import os
import shutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.replay_buffer import ReplayBuffer
from training.self_play import SelfPlayWorker
from training.train_alphazero import train_alphazero
from models.alphazero_net import AlphaZeroTradingNet
from utils.config import MCTS_CONFIG

class MockEnv:
    """Simple Mock Environment for Self-Play Testing"""
    def __init__(self):
        self.state = np.random.randn(84).astype(np.float32)
        self.steps = 0
        
    def step(self, action):
        self.steps += 1
        next_state = self.state + np.random.randn(84).astype(np.float32) * 0.1
        reward = 1.0 if action == 1 else 0.0
        done = self.steps >= 5 # Short episode
        return next_state, reward, done, {}
        
    def reset(self):
        self.steps = 0
        self.state = np.random.randn(84).astype(np.float32)
        return self.state

def test_phase3():
    print("🧪 Testing V19 Phase 3: Self-Play & Training")
    
    # 1. Test Replay Buffer
    buffer = ReplayBuffer(capacity=100)
    buffer.push(np.zeros(84), np.ones(5)/5, 1.0)
    assert len(buffer) == 1
    s, p, v = buffer.sample(1)
    assert s.shape == (1, 84)
    print("✅ Replay Buffer OK")
    
    # 2. Test Self-Play Worker
    env = MockEnv()
    net = AlphaZeroTradingNet(84, 5)
    # Reduce simulations for speed
    fast_config = MCTS_CONFIG.copy()
    fast_config['n_simulations'] = 5
    
    worker = SelfPlayWorker(net, env, mcts_config=fast_config)
    samples, total_reward = worker.play_episode()
    
    print(f"   Samples collected: {len(samples)}")
    print(f"   Total Reward: {total_reward}")
    
    assert len(samples) == 5, "Should have 5 samples for 5 steps"
    assert len(samples[0]) == 3, "Sample should be (state, policy, value)"
    print("✅ Self-Play Worker OK")
    
    # 3. Test Training Loop (Integration)
    print("   Running short training loop...")
    # Override config for speed
    import training.train_alphazero as train_module
    train_module.TRAINING_CONFIG['self_play_episodes'] = 1
    train_module.TRAINING_CONFIG['epochs_per_iteration'] = 1
    train_module.TRAINING_CONFIG['batch_size'] = 4
    train_module.TRAINING_CONFIG['checkpoint_dir'] = "gemini_v19/tests/tmp_models"
    
    try:
        train_alphazero(env, run_name="test_run")
        print("✅ Training Loop OK")
    except Exception as e:
        print(f"❌ Training Loop Failed: {e}")
        raise e
    finally:
        # Cleanup
        if os.path.exists("gemini_v19/tests/tmp_models"):
            shutil.rmtree("gemini_v19/tests/tmp_models")
            
    print("\n🎉 Phase 3 Complete: Self-Play & Training functional!")

if __name__ == "__main__":
    test_phase3()
