import torch
import torch.optim as optim
import numpy as np
import os
import sys
import time

# Add project root to path (must be at 'test' level)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from gemini_v19.models.alphazero_net import AlphaZeroTradingNet
from gemini_v19.models.loss import AlphaZeroLoss
from gemini_v19.training.replay_buffer import ReplayBuffer
from gemini_v19.training.self_play import SelfPlayEngine
from gemini_v19.utils.config import NETWORK_CONFIG, TRAINING_CONFIG, MCTS_CONFIG

def train_alphazero(env, run_name="alpha_v19_test"):
    print(f"🚀 Starting AlphaZero Training: {run_name}")
    
    # 1. Init Components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    network = AlphaZeroTradingNet(**NETWORK_CONFIG).to(device)
    optimizer = optim.Adam(network.parameters(), lr=TRAINING_CONFIG['learning_rate'], weight_decay=TRAINING_CONFIG['weight_decay'])
    loss_fn = AlphaZeroLoss()
    
    replay_buffer = ReplayBuffer(capacity=TRAINING_CONFIG['replay_buffer_size'])
    worker = SelfPlayEngine(network, device)
    
    # 2. Training Loop
    n_iterations = 5 # Short run for testing, normally 100+
    episodes_per_iter = TRAINING_CONFIG['self_play_episodes']
    epochs = TRAINING_CONFIG['epochs_per_iteration']
    batch_size = TRAINING_CONFIG['batch_size']
    
    for iteration in range(n_iterations):
        start_time = time.time()
        print(f"\n🔄 Iteration {iteration+1}/{n_iterations}")
        
        # A. Self-Play (Data Collection)
        network.eval()
        total_reward = 0
        new_samples = 0
        
        print(f"   Playing {episodes_per_iter} episodes...")
        print(f"   Playing {episodes_per_iter} episodes...")
        samples = worker.generate_games(n_games=episodes_per_iter)
        new_samples = len(samples)
            
        for sample in samples:
            replay_buffer.push(*sample)
                
        print(f"   Collected {new_samples} samples.")
        
        # B. Training (Network Update)
        network.train()
        total_loss = 0
        
        print(f"   Training for {epochs} epochs...")
        for epoch in range(epochs):
            if len(replay_buffer) < batch_size:
                continue
                
            states, policies, values = replay_buffer.sample(batch_size)
            
            # To Tensor
            states = torch.FloatTensor(states).to(device)
            policies = torch.FloatTensor(policies).to(device)
            values = torch.FloatTensor(values).to(device)
            
            # Forward
            policy_pred, value_pred = network(states)
            
            # Loss
            loss, p_loss, v_loss = loss_fn(policy_pred, value_pred, policies, values)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / epochs
        duration = time.time() - start_time
        print(f"   Iteration Complete in {duration:.2f}s. Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        if (iteration + 1) % 1 == 0:
            save_path = os.path.join(TRAINING_CONFIG['checkpoint_dir'], f"{run_name}_iter_{iteration+1}.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(network.state_dict(), save_path)
            print(f"   💾 Model saved to {save_path}")

if __name__ == "__main__":
    # Mock Env for standalone testing
    class MockEnv:
        def reset(self): return np.random.randn(84).astype(np.float32)
        def step(self, action): 
            next_state = np.random.randn(84).astype(np.float32)
            reward = np.random.randn()
            done = np.random.rand() < 0.1 # 10% chance done
            return next_state, reward, done, {}
            
    train_alphazero(MockEnv())
