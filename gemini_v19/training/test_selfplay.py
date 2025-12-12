import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from gemini_v19.training.simulated_market import SimulatedMarket
from gemini_v19.models.alphazero_net import AlphaZeroTradingNet
from gemini_v19.utils.config import NETWORK_CONFIG
from gemini_v19.mcts.alphazero_mcts import AlphaZeroMCTS

def test():
    print("🧪 Testing Self-Play Components...")
    
    # 1. Test SimulatedMarket
    print("1. Testing SimulatedMarket...")
    env = SimulatedMarket("EURUSD", "H1")
    state = env.reset()
    print(f"   State Shape: {state.shape}")
    print(f"   State Type: {state.dtype}")
    print(f"   State Sample: {state[:5]}")
    
    if state.shape != (84,):
        print(f"❌ State shape mismatch! Expected (84,), got {state.shape}")
        return
        
    # 2. Test Network
    print("\n2. Testing AlphaZeroTradingNet...")
    net = AlphaZeroTradingNet(**NETWORK_CONFIG)
    net.eval()
    
    # Convert state to tensor
    tensor_state = torch.FloatTensor([state])
    print(f"   Tensor Shape: {tensor_state.shape}")
    
    try:
        policy, value = net(tensor_state)
        print(f"   Network Output: Policy {policy.shape}, Value {value.shape}")
    except Exception as e:
        print(f"❌ Network Forward Failed: {e}")
        return

    # 3. Test MCTS Expansion
    print("\n3. Testing MCTS Expansion...")
    mcts = AlphaZeroMCTS(net, env)
    root_node = mcts.search(state, temperature=1.0)[1]
    print("   MCTS Search Complete.")
    
    # 4. Test Step
    print("\n4. Testing Env Step...")
    next_state, reward, done, truncated, info = env.step(1)
    print(f"   Step Result: Reward={reward}, Done={done}")
    
    print("\n✅ All Tests Passed!")

if __name__ == "__main__":
    test()
