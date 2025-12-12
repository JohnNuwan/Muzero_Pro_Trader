
import ray
import torch
import os

@ray.remote
class SharedStorage:
    """
    Holds the latest model weights and training metrics.
    Workers pull weights from here. Trainer pushes weights here.
    """
    def __init__(self, config):
        self.config = config
        self.weights = None
        self.infos = {
            "total_loss": [],
            "policy_loss": [],
            "value_loss": [],
            "reward_loss": [],
            "training_step": 0,
            "games_played": 0,
            "total_rewards": [] # Store recent rewards
        }
        
    def get_weights(self):
        return self.weights
        
    def set_weights(self, weights):
        self.weights = weights
        
    def get_infos(self):
        return self.infos
        
    def set_infos(self, key, value):
        if key in self.infos:
            if isinstance(self.infos[key], list):
                self.infos[key].append(value)
            else:
                self.infos[key] = value
                
    def increment_games_played(self):
        self.infos["games_played"] += 1
        return self.infos["games_played"]
        
    def log_reward(self, reward):
        self.infos["total_rewards"].append(reward)
        if len(self.infos["total_rewards"]) > 100: # Keep last 100
            self.infos["total_rewards"].pop(0)
            
    def get_avg_reward(self):
        if not self.infos["total_rewards"]:
            return 0.0
        return sum(self.infos["total_rewards"]) / len(self.infos["total_rewards"])
        
    def get_training_step(self):
        return self.infos["training_step"]
        
    def set_training_step(self, step):
        self.infos["training_step"] = step
