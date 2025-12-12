import numpy as np
import math
import copy
import torch

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state # Observation
        self.parent = parent
        self.action = action # Action taken to reach this state
        self.children = {} # Map action -> MCTSNode
        self.visits = 0
        self.value = 0.0 # Total reward
        self.prior = 0.0 # Policy probability (P(s, a))

    def is_fully_expanded(self, action_space_size):
        return len(self.children) == action_space_size

    def best_child(self, c_puct=1.41):
        # UCB1 / PUCT Selection
        best_score = -float('inf')
        best_node = None
        
        for action, child in self.children.items():
            # PUCT Formula: Q(s,a) + U(s,a)
            # U(s,a) = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            
            q_value = child.value / (child.visits + 1e-9)
            u_value = c_puct * child.prior * math.sqrt(self.visits) / (1 + child.visits)
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_node = child
                
        return best_node

class MCTSAgent:
    """
    Monte Carlo Tree Search Agent.
    Uses a trained PPO Policy for priors and rollouts.
    """
    def __init__(self, policy_model, env, simulations=50, depth=10):
        self.policy = policy_model
        self.env = env # Must be a copy or lightweight simulator
        self.simulations = simulations
        self.depth = depth
        self.action_space_size = env.action_space.n

    def search(self, root_state):
        root = MCTSNode(root_state)
        
        # 1. Expand Root (Get Priors from Policy)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(root_state).unsqueeze(0).to(self.policy.device)
            distribution = self.policy.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.cpu().numpy()[0]
            
        for action in range(self.action_space_size):
            child = MCTSNode(root_state, parent=root, action=action)
            child.prior = probs[action]
            root.children[action] = child
            
        # 2. Run Simulations
        for _ in range(self.simulations):
            node = root
            
            # Fast Clone
            if hasattr(self.env, 'clone'):
                sim_env = self.env.clone()
            else:
                sim_env = copy.deepcopy(self.env)
            
            # Selection
            while node.is_fully_expanded(self.action_space_size) and node.children:
                node = node.best_child()
                _, _, done, _, _ = sim_env.step(node.action)
                if done: break
            
            # Expansion
            if not node.is_fully_expanded(self.action_space_size) and not done:
                # We already expanded root, but for deeper nodes...
                # For simplicity in this "Light" version, we just pick a random unexpanded child
                # Or we can use the policy again.
                # Let's just step the env with the node's action
                pass 
            
            # Simulation (Rollout)
            cumulative_reward = 0
            for _ in range(self.depth):
                # Random Policy or Fast Policy
                action = sim_env.action_space.sample() 
                _, reward, done, _, _ = sim_env.step(action)
                cumulative_reward += reward
                if done: break
                
            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value += cumulative_reward
                node = node.parent
                
        # 3. Select Best Action (Most Visited)
        best_action = max(root.children.items(), key=lambda item: item[1].visits)[0]
        return best_action
