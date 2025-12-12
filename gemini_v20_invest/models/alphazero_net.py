import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaZeroTradingNet(nn.Module):
    """
    Dual-Head Network for AlphaZero Trading
    Input: State Vector (84 features)
    Output: 
        - Policy Head: Probability distribution over actions (Softmax)
        - Value Head: Expected return/value of the state (Tanh)
    """
    def __init__(self, input_dim=84, action_dim=5, hidden_dims=[256, 256, 256], dropout=0.1, activation='relu', use_batch_norm=True, **kwargs):
        super(AlphaZeroTradingNet, self).__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        
        # --- Shared Trunk (ResNet-style could be added here, using simple MLP for now) ---
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
            
        self.shared_trunk = nn.Sequential(*layers)
        
        # --- Policy Head (Actor) ---
        # Output: Probability distribution over actions (π)
        self.policy_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
            # Softmax is applied in the forward pass or loss function
        )
        
        # --- Value Head (Critic) ---
        # Output: Scalar value estimate (V) in range [-1, 1]
        self.value_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        
    def forward(self, state):
        """
        Forward pass
        Args:
            state: Tensor of shape (batch_size, input_dim)
        Returns:
            policy: Tensor of shape (batch_size, action_dim) - Logits or Probabilities
            value: Tensor of shape (batch_size, 1) - Value estimate
        """
        # Ensure state is float
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
            
        features = self.shared_trunk(state)
        
        # Policy (Logits)
        policy_logits = self.policy_head(features)
        policy = F.softmax(policy_logits, dim=-1)
        
        # Value
        value = self.value_head(features)
        
        return policy, value

if __name__ == "__main__":
    # Simple Test
    net = AlphaZeroTradingNet(input_dim=84, action_dim=5)
    dummy_input = torch.randn(4, 84) # Batch of 4
    policy, value = net(dummy_input)
    
    print("Policy Shape:", policy.shape) # Should be (4, 5)
    print("Value Shape:", value.shape)   # Should be (4, 1)
    print("Policy Sum:", policy.sum(dim=1)) # Should be approx 1.0
