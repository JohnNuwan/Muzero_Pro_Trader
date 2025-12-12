
import torch
import numpy as np
from MuZero.config import MuZeroConfig
from MuZero.models.muzero_network import MuZeroNet
from MuZero.agents.muzero_mcts import MuZeroMCTS
from MuZero.training.replay_buffer import GameHistory, ReplayBuffer

class MuZeroAgent:
    def __init__(self, config: MuZeroConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = MuZeroNet(config).to(self.device)
        self.replay_buffer = ReplayBuffer(config)
        
    def play_game(self, env, render=False):
        """
        Play one full game (episode) and store it in the replay buffer.
        """
        game_history = GameHistory()
        obs, _ = env.reset()
        done = False
        
        steps = 0
        with torch.no_grad():
            while not done and steps < self.config.max_moves:
                steps += 1
                # 1. MCTS to select action
                # Normalize obs if needed? V19 env returns numpy array.
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Get initial hidden state
                root_state = self.network.representation(obs_tensor)
                
                # Run MCTS
                mcts = MuZeroMCTS(self.config, self.network)
                root = mcts.run(root_state, add_exploration_noise=True)
                
                # Select action based on visit counts
                action = self._select_action(root)
                
                # Store statistics
                policy = self._get_policy_distribution(root)
                value = root.value()
                
                # Execute action
                next_obs, reward, done, _, _ = env.step(action)
                
                # Store in history
                game_history.store(obs, action, reward, policy, value, done)
                
                obs = next_obs
                
        self.replay_buffer.save_game(game_history)
        return sum(game_history.rewards)

    def play_game_ray(self, env, render=False):
        """
        Play one full game (episode) and RETURN the game history (for Ray).
        """
        game_history = GameHistory()
        obs, _ = env.reset()
        done = False
        
        steps = 0
        with torch.no_grad():
            while not done and steps < self.config.max_moves:
                steps += 1
                # 1. MCTS to select action
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Get initial hidden state
                root_state = self.network.representation(obs_tensor)
                
                # Run MCTS
                mcts = MuZeroMCTS(self.config, self.network)
                root = mcts.run(root_state, add_exploration_noise=True)
                
                # Select action based on visit counts
                action = self._select_action(root)
                
                # Store statistics
                policy = self._get_policy_distribution(root)
                value = root.value()
                
                # Execute action
                next_obs, reward, done, _, _ = env.step(action)
                
                # Store in history
                game_history.store(obs, action, reward, policy, value, done)
                
                obs = next_obs
                
        return game_history

    def _select_action(self, root):
        # Temperature based selection
        # For training, we usually sample. For eval, we take max.
        # Here we assume training (exploration).
        visit_counts = [(action, child.visit_count) for action, child in root.children.items()]
        actions = [x[0] for x in visit_counts]
        counts = [x[1] for x in visit_counts]
        
        # Softmax with temperature
        # For simplicity, just proportional to counts (temp=1)
        probs = np.array(counts) / sum(counts)
        action = np.random.choice(actions, p=probs)
        return action

    def _get_policy_distribution(self, root):
        policy = np.zeros(self.config.action_space_size)
        for action, child in root.children.items():
            policy[action] = child.visit_count
        policy /= sum(policy)
        return policy

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.device))
