
import ray
import torch
import numpy as np
from MuZero.agents.muzero_agent import MuZeroAgent
from MuZero.environment.commission_trinity_env import CommissionTrinityEnv

@ray.remote # No GPU for workers on Windows (CPU only)
class SelfPlayWorker:
    """
    Plays games continuously and sends data to the shared replay buffer.
    """
    def __init__(self, config, seed, shared_storage, replay_buffer):
        self.config = config
        self.seed = seed
        self.shared_storage = shared_storage
        self.replay_buffer = replay_buffer
        
        # Initialize Agent (Model only for inference)
        self.agent = MuZeroAgent(config)
        self.agent.network.eval() # Inference mode
        
        # Initialize Environment
        # We can pick a random symbol or cycle through them
        self.symbol = config.symbols[seed % len(config.symbols)]
        self.env = CommissionTrinityEnv(symbol=self.symbol)
        
    def continuous_self_play(self):
        """
        Loop:
        1. Pull latest weights from SharedStorage
        2. Play a game
        3. Send game history to ReplayBuffer
        """
        while True:
            # 1. Update Weights
            weights = ray.get(self.shared_storage.get_weights.remote())
            if weights:
                self.agent.network.load_state_dict(weights)
                
            # 2. Play Game
            # We need to modify agent.play_game to NOT save to its own buffer,
            # but return the history so we can send it to the shared buffer.
            # Or we can just use the agent's buffer and send it.
            
            # Let's use a modified play loop here to be explicit
            game_history = self.agent.play_game_ray(self.env) # We need to add this method
            
            # 3. Send to Shared Replay Buffer
            self.replay_buffer.save_game.remote(game_history)
            
            # Log
            total_reward = sum(game_history.rewards)
            self.shared_storage.increment_games_played.remote()
            self.shared_storage.log_reward.remote(total_reward)
