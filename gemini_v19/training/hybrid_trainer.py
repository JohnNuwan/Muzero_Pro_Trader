import torch
import torch.optim as optim
import numpy as np
from gemini_v19.models.loss import AlphaZeroLoss
from gemini_v19.utils.selfplay_config import SELF_PLAY_CONFIG
from gemini_v19.utils.config import TRAINING_CONFIG

class HybridTrainer:
    """
    Trains AlphaZero network on mixed data (Self-Play + Real Trades).
    """
    def __init__(self, device):
        self.device = device
        self.config = SELF_PLAY_CONFIG
        self.training_config = TRAINING_CONFIG
        
    def train(self, model, self_play_data, real_data, epochs=None):
        """
        Train the model on mixed dataset.
        """
        if epochs is None:
            # Use retrain_epochs from config or default
            epochs = 50 # Default if not specified
            
        print(f"🧠 Starting Hybrid Training ({epochs} epochs)...")
        print(f"   Self-Play Data: {len(self_play_data)}")
        print(f"   Real Data: {len(real_data)}")
        
        optimizer = optim.Adam(model.parameters(), lr=self.training_config['learning_rate'])
        loss_fn = AlphaZeroLoss()
        
        model.train()
        
        # Prepare datasets
        # Self-play data structure: (state, policy, value)
        # Real data structure: (state, policy, value) - assumed compatible
        
        # Convert to lists if not already
        sp_states, sp_policies, sp_values = zip(*self_play_data) if self_play_data else ([], [], [])
        real_states, real_policies, real_values = zip(*real_data) if real_data else ([], [], [])
        
        # Convert to numpy for easy sampling
        sp_states = np.array(sp_states) if sp_states else np.empty((0, 0)) # Handle empty
        sp_policies = np.array(sp_policies) if sp_policies else np.empty((0, 0))
        sp_values = np.array(sp_values) if sp_values else np.empty((0, 0))
        
        real_states = np.array(real_states) if real_states else np.empty((0, 0))
        real_policies = np.array(real_policies) if real_policies else np.empty((0, 0))
        real_values = np.array(real_values) if real_values else np.empty((0, 0))
        
        batch_size = self.training_config['batch_size']
        
        # Calculate samples per batch based on weights
        sp_weight = self.config['self_play_weight']
        real_weight = self.config['real_data_weight']
        
        # Normalize weights if they don't sum to 1
        total_weight = sp_weight + real_weight
        sp_weight /= total_weight
        real_weight /= total_weight
        
        sp_batch_size = int(batch_size * sp_weight)
        real_batch_size = batch_size - sp_batch_size
        
        # Check if we have enough data
        if len(sp_states) < sp_batch_size and len(sp_states) > 0:
            sp_batch_size = len(sp_states)
            real_batch_size = batch_size - sp_batch_size
        
        if len(real_states) < real_batch_size and len(real_states) > 0:
            real_batch_size = len(real_states)
            sp_batch_size = batch_size - real_batch_size
            
        # Training Loop
        for epoch in range(epochs):
            total_loss = 0
            batches = 0
            
            # Determine number of batches (based on larger dataset or fixed?)
            # Let's run for a fixed number of batches per epoch or cover the data.
            # Usually we want to cover the data.
            # Let's base it on the total size / batch size.
            total_samples = len(sp_states) + len(real_states)
            num_batches = max(1, total_samples // batch_size)
            
            for _ in range(num_batches):
                # Sample from Self-Play
                batch_states = []
                batch_policies = []
                batch_values = []
                
                if len(sp_states) > 0:
                    sp_indices = np.random.choice(len(sp_states), sp_batch_size)
                    batch_states.append(sp_states[sp_indices])
                    batch_policies.append(sp_policies[sp_indices])
                    batch_values.append(sp_values[sp_indices])
                    
                # Sample from Real Data
                if len(real_states) > 0:
                    real_indices = np.random.choice(len(real_states), real_batch_size)
                    batch_states.append(real_states[real_indices])
                    batch_policies.append(real_policies[real_indices])
                    batch_values.append(real_values[real_indices])
                
                if not batch_states:
                    continue
                    
                # Concatenate
                # Check if elements are arrays or lists of arrays
                # If batch_states is list of arrays, vstack
                
                # Handle case where one source is empty
                if len(batch_states) == 1:
                    X = torch.FloatTensor(batch_states[0]).to(self.device)
                    target_policy = torch.FloatTensor(batch_policies[0]).to(self.device)
                    target_value = torch.FloatTensor(batch_values[0]).reshape(-1, 1).to(self.device)
                else:
                    X = torch.FloatTensor(np.concatenate(batch_states)).to(self.device)
                    target_policy = torch.FloatTensor(np.concatenate(batch_policies)).to(self.device)
                    target_value = torch.FloatTensor(np.concatenate(batch_values)).reshape(-1, 1).to(self.device)
                
                # Forward
                policy_pred, value_pred = model(X)
                loss, _, _ = loss_fn(policy_pred, value_pred, target_policy, target_value)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batches += 1
                
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / batches if batches > 0 else 0
                print(f"   Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
                
        return model
