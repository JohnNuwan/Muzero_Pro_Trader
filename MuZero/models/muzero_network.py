
import torch
import torch.nn as nn
import torch.nn.functional as F
from MuZero.config import MuZeroConfig

class RepresentationNetwork(nn.Module):
    def __init__(self, config: MuZeroConfig):
        super().__init__()
        self.input_dim = config.observation_shape[0]
        self.hidden_dims = config.network_hidden_dims
        self.output_dim = config.hidden_state_size
        
        layers = []
        in_dim = self.input_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if config.network_hidden_dims: # Use config for batchnorm/dropout if needed, here simple
                layers.append(nn.ReLU())
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, self.output_dim))
        # Normalize hidden state to range [0, 1] or similar to keep dynamics stable
        # MinMax scaling or Tanh is common in MuZero
        layers.append(nn.Tanh()) 
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class DynamicsNetwork(nn.Module):
    def __init__(self, config: MuZeroConfig):
        super().__init__()
        self.input_dim = config.hidden_state_size + config.action_space_size
        self.hidden_dims = config.network_hidden_dims
        self.hidden_state_dim = config.hidden_state_size
        
        # Common layers
        layers = []
        in_dim = self.input_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
            
        self.common = nn.Sequential(*layers)
        
        # Heads
        self.next_state_head = nn.Sequential(
            nn.Linear(in_dim, self.hidden_state_dim),
            nn.Tanh() # Keep state normalized
        )
        
        self.reward_head = nn.Sequential(
            nn.Linear(in_dim, 1)
        )

    def forward(self, hidden_state, action):
        # Action should be one-hot encoded
        x = torch.cat([hidden_state, action], dim=1)
        common_out = self.common(x)
        
        next_hidden_state = self.next_state_head(common_out)
        reward = self.reward_head(common_out)
        
        return next_hidden_state, reward

class PredictionNetwork(nn.Module):
    def __init__(self, config: MuZeroConfig):
        super().__init__()
        self.input_dim = config.hidden_state_size
        self.hidden_dims = config.network_hidden_dims
        self.action_dim = config.action_space_size
        
        # Common layers
        layers = []
        in_dim = self.input_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
            
        self.common = nn.Sequential(*layers)
        
        # Heads
        self.policy_head = nn.Sequential(
            nn.Linear(in_dim, self.action_dim),
            nn.Softmax(dim=1) 
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(in_dim, 1)
        )

    def forward(self, hidden_state):
        common_out = self.common(hidden_state)
        
        policy = self.policy_head(common_out)
        value = self.value_head(common_out)
        
        return policy, value

class MuZeroNet(nn.Module):
    def __init__(self, config: MuZeroConfig):
        super().__init__()
        self.config = config
        self.representation = RepresentationNetwork(config)
        self.dynamics = DynamicsNetwork(config)
        self.prediction = PredictionNetwork(config)

    def initial_inference(self, observation):
        hidden_state = self.representation(observation)
        policy, value = self.prediction(hidden_state)
        return hidden_state, policy, value

    def recurrent_inference(self, hidden_state, action):
        next_hidden_state, reward = self.dynamics(hidden_state, action)
        policy, value = self.prediction(next_hidden_state)
        return next_hidden_state, reward, policy, value
