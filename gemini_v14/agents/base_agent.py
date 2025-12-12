import numpy as np
import pickle
import os

class BaseAgent:
    """
    Base class for all V14 agents.
    """
    def __init__(self, name):
        self.name = name
        self.training = True
        
    def act(self, observation):
        raise NotImplementedError
        
    def learn(self, *args):
        pass
        
    def save(self, path):
        """Generic save method using pickle."""
        try:
            with open(path, 'wb') as f:
                pickle.dump(self, f)
            print(f"Saved agent {self.name} to {path}")
        except Exception as e:
            print(f"Error saving agent {self.name}: {e}")

    def load(self, path):
        """Generic load method."""
        try:
            with open(path, 'rb') as f:
                loaded_agent = pickle.load(f)
                self.__dict__.update(loaded_agent.__dict__)
            print(f"Loaded agent {self.name} from {path}")
        except Exception as e:
            print(f"Error loading agent {self.name}: {e}")

class TraderAgent(BaseAgent):
    """
    The Sniper.
    Input: Market State
    Output: 0=Wait, 1=Buy, 2=Sell
    """
    def __init__(self, name):
        super().__init__(name)
        
    def act(self, observation):
        # observation is trader_obs from TrinityEnv
        return 0 # Wait by default

class ManagerAgent(BaseAgent):
    """
    The Guardian.
    Input: Trade State
    Output: 0=Hold, 1=Close 50%, 2=Close 100%
    """
    def __init__(self, name):
        super().__init__(name)
        
    def act(self, observation):
        # observation is manager_obs from TrinityEnv
        return 0 # Hold by default

class CriticAgent(BaseAgent):
    """
    The Judge.
    Input: State, Action, Reward, Next State
    Output: Value / Advantage
    """
    def __init__(self, name):
        super().__init__(name)
        
    def evaluate(self, state, action):
        return 0.0
