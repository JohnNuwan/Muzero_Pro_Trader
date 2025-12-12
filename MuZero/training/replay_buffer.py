
import numpy as np
import random
from MuZero.config import MuZeroConfig

class GameHistory:
    """
    Stores the history of a single game/episode.
    """
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.policies = []
        self.values = []
        self.dones = []

    def store(self, obs, action, reward, policy, value, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.actions)

class ReplayBuffer:
    """
    Stores multiple GameHistory objects and samples sequences for training.
    """
    def __init__(self, config: MuZeroConfig):
        self.config = config
        self.buffer = []
        self.max_size = config.window_size

    def save_game(self, game_history: GameHistory):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(game_history)

    def sample_batch(self):
        """
        Samples a batch of sequences for training.
        Returns:
            (observation_batch, action_batch, reward_target_batch, value_target_batch, policy_target_batch)
        """
        batch_size = self.config.batch_size
        unroll_steps = self.config.num_unroll_steps
        td_steps = self.config.td_steps
        
        games = random.choices(self.buffer, k=batch_size)
        
        obs_batch = []
        action_batch = []
        reward_target_batch = []
        value_target_batch = []
        policy_target_batch = []
        
        for game in games:
            game_len = len(game)
            # Pick a start index. We need at least unroll_steps after it? 
            # Actually we can pad if needed, but easier to pick valid range.
            # We need to unroll K steps.
            start_index = random.randint(0, game_len - 1)
            
            # 1. Initial Observation
            obs_batch.append(game.observations[start_index])
            
            # 2. Unroll
            actions = []
            rewards = []
            values = []
            policies = []
            
            for k in range(unroll_steps):
                current_index = start_index + k
                
                # Action to take from current state (needed for dynamics)
                if current_index < game_len:
                    actions.append(game.actions[current_index])
                else:
                    # Random action or 0 if out of bounds (should be masked)
                    actions.append(random.randint(0, self.config.action_space_size - 1))
                
                # Target Reward (u_{t+k}) - Reward received AFTER taking action
                if current_index < game_len:
                    rewards.append(game.rewards[current_index])
                else:
                    rewards.append(0.0)
                    
                # Target Policy (pi_{t+k})
                if current_index < game_len:
                    policies.append(game.policies[current_index])
                else:
                    policies.append(np.zeros(self.config.action_space_size)) # Uniform?
                    
                # Target Value (z_{t+k}) - Bootstrapped n-step return
                # z_t = u_{t+1} + ... + u_{t+n} + v_{t+n}
                if current_index < game_len:
                    value = 0.0
                    # Bootstrap
                    bootstrap_index = current_index + td_steps
                    if bootstrap_index < game_len:
                        value = game.values[bootstrap_index] # Use search value
                        # Add rewards up to bootstrap
                        for i in range(current_index, bootstrap_index):
                            value += game.rewards[i] * (self.config.discount ** (i - current_index))
                    else:
                        # End of game
                        for i in range(current_index, game_len):
                            value += game.rewards[i] * (self.config.discount ** (i - current_index))
                            
                    values.append(value)
                else:
                    values.append(0.0)

            action_batch.append(actions)
            reward_target_batch.append(rewards)
            value_target_batch.append(values)
            policy_target_batch.append(policies)
            
        return (
            np.array(obs_batch),
            np.array(action_batch),
            np.array(reward_target_batch),
            np.array(value_target_batch),
            np.array(policy_target_batch)
        )
