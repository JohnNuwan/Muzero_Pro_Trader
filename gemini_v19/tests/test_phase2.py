import torch
import numpy as np
import sys
import os
import copy

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mcts.alphazero_mcts import AlphaZeroMCTS
from models.alphazero_net import AlphaZeroTradingNet

class MockEnv:
    """Simple Mock Environment for MCTS Testing"""
    def __init__(self):
        self.state = np.random.randn(84).astype(np.float32)
        self.steps = 0
        
    def step(self, action):
        self.steps += 1
        # Simple state transition: just add noise
        next_state = self.state + np.random.randn(84).astype(np.float32) * 0.1
        reward = 1.0 if action == 1 else -0.1 # Dummy reward
        done = self.steps >= 10
        return next_state, reward, done, {}
        
    def reset(self):
        self.steps = 0
        self.state = np.random.randn(84).astype(np.float32)
        return self.state

def test_phase2():
    print("🧪 Testing V19 Phase 2: MCTS + PUCT")
    
    # 1. Setup
    input_dim = 84
    action_dim = 5
    net = AlphaZeroTradingNet(input_dim, action_dim)
    env = MockEnv()
    
    # 2. Init MCTS
    mcts = AlphaZeroMCTS(net, env, n_simulations=10, c_puct=1.5)
    print("✅ MCTS Initialized")
    
    # 3. Run Search
    root_state = env.reset()
    policy, root_node = mcts.search(root_state, temperature=1.0)
    
    print(f"   Policy: {policy}")
    print(f"   Root Visits: {root_node.visit_count}")
    
    # 4. Assertions
    assert len(policy) == action_dim, "Policy dimension mismatch"
    assert np.isclose(sum(policy), 1.0), "Policy sum is not 1.0"
    assert root_node.visit_count >= 10, "Root visits < n_simulations" # Root is visited in every sim + init
    
    # Check children
    children_count = len(root_node.children)
    print(f"   Root Children: {children_count}")
    assert children_count > 0, "Root was not expanded"
    
    print("✅ MCTS Search OK")
    
    print("\n🎉 Phase 2 Complete: MCTS is functional!")

if __name__ == "__main__":
    test_phase2()
