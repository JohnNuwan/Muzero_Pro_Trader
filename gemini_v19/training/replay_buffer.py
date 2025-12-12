import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """
    Replay Buffer for AlphaZero training
    Stores (state, policy, value) tuples
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, policy, value):
        """
        Add a sample to the buffer
        state: np.array
        policy: np.array (probability distribution)
        value: float (target return)
        """
        self.buffer.append((state, policy, value))
        
    def sample(self, batch_size):
        """
        Sample a batch of experiences
        """
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, policies, values = zip(*batch)
        
        return (
            np.array(states),
            np.array(policies),
            np.array(values).reshape(-1, 1)
        )
        
    def __len__(self):
        return len(self.buffer)
