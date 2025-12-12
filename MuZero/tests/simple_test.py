import torch
import numpy as np
import sys
sys.path.insert(0, 'c:/Users/nandi/Desktop/test')

from MuZero.config import MuZeroConfig
from MuZero.agents.muzero_agent import MuZeroAgent

class MockEnv:
    def __init__(self):
        self.observation_space_shape = (84,)
        self.action_space_n = 5
        self.step_count = 0
        
    def reset(self):
        self.step_count = 0
        return np.zeros(self.observation_space_shape, dtype=np.float32), {}
        
    def step(self, action):
        self.step_count += 1
        obs = np.random.randn(84).astype(np.float32)
        reward = np.random.randn()
        done = self.step_count >= 10  # Short episode
        info = {}
        return obs, reward, done, False, info

print("1. Creating config...")
config = MuZeroConfig()
config.num_simulations = 3
config.batch_size = 2

print("2. Creating agent...")
agent = MuZeroAgent(config)

print("3. Creating environment...")
env = MockEnv()

print("4. Playing one game...")
try:
    reward = agent.play_game(env)
    print(f"✅ Game completed! Total reward: {reward:.2f}")
    print(f"   Games in buffer: {len(agent.replay_buffer.buffer)}")
except Exception as e:
    print(f"❌ Error during play_game: {e}")
    import traceback
    traceback.print_exc()

print("\n5. Testing replay buffer sampling...")
if len(agent.replay_buffer.buffer) >= 2:
    print("   Playing another game...")
    agent.play_game(env)
    
    try:
        batch = agent.replay_buffer.sample_batch()
        print(f"✅ Batch sampled successfully!")
        print(f"   Observation batch shape: {batch[0].shape}")
        print(f"   Action batch shape: {batch[1].shape}")
    except Exception as e:
        print(f"❌ Error during sampling: {e}")
        import traceback
        traceback.print_exc()
else:
    print("   Skipping (need at least 2 games)")
