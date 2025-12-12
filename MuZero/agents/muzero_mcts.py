
import math
import numpy as np
import torch
from MuZero.config import MuZeroConfig

class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MuZeroMCTS:
    def __init__(self, config: MuZeroConfig, network):
        self.config = config
        self.network = network

    def run(self, root_state, legal_actions=None, add_exploration_noise=False):
        """
        Run MCTS simulations.
        root_state: The initial hidden state from the Representation Network.
        """
        root = Node(0)
        root.hidden_state = root_state
        
        # Initial expansion
        policy, value = self.network.prediction(root_state)
        self._expand_node(root, policy, legal_actions)
        
        if add_exploration_noise:
            self._add_exploration_noise(root)

        min_max_stats = MinMaxStats()

        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]
            
            # Selection
            while node.expanded():
                action, node = self._select_child(node, min_max_stats)
                search_path.append(node)

            # Expansion and Evaluation
            parent = search_path[-2]
            action = search_path[-1].prior # Wait, we need the action that led to this node
            # Actually, _select_child returns (action, child)
            # We need to store the action in the search path or re-derive it.
            # Let's adjust the loop.
            
            # Re-implementation of Selection to track actions
            node = root
            search_path = [node]
            actions_path = []
            
            while node.expanded():
                action, node = self._select_child(node, min_max_stats)
                search_path.append(node)
                actions_path.append(action)
                
            # Now node is a leaf (not expanded).
            # We need to expand it using the Dynamics Network.
            # But wait, the root is already expanded with Representation Network.
            # Subsequent nodes are expanded with Dynamics Network.
            
            parent = search_path[-2]
            last_action = actions_path[-1]
            
            # Prepare action for network (one-hot encoding)
            action_one_hot = torch.zeros((1, self.config.action_space_size)).to(parent.hidden_state.device)
            action_one_hot[0, last_action] = 1.0
            
            # Dynamics Inference: compute next state using parent's hidden state and action
            next_hidden_state, reward, policy, value = self.network.recurrent_inference(
                parent.hidden_state, action_one_hot
            )
            
            node.hidden_state = next_hidden_state
            node.reward = reward.item()
            
            self._expand_node(node, policy, legal_actions)
            
            # Backpropagation
            self._backpropagate(search_path, value.item(), min_max_stats)

        return root

    def _select_child(self, node, min_max_stats):
        max_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            score = self._ucb_score(node, child, min_max_stats)
            if score > max_score:
                max_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child

    def _ucb_score(self, parent, child, min_max_stats):
        pb_c = math.log((parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base) + self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        
        if child.visit_count > 0:
            value_score = min_max_stats.normalize(child.reward + self.config.discount * child.value())
        else:
            value_score = 0
            
        return prior_score + value_score

    def _expand_node(self, node, policy_logits, legal_actions):
        # policy_logits is (1, action_dim)
        policy = policy_logits[0].detach().cpu().numpy()
        # Softmax if not already? My network has Softmax.
        
        for action in range(self.config.action_space_size):
            if legal_actions and action not in legal_actions:
                continue
            node.children[action] = Node(policy[action])

    def _backpropagate(self, search_path, value, min_max_stats):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.value())
            
            value = node.reward + self.config.discount * value

    def _add_exploration_noise(self, node):
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.config.root_dirichlet_alpha] * len(actions))
        frac = self.config.root_exploration_fraction
        for i, action in enumerate(actions):
            node.children[action].prior = node.children[action].prior * (1 - frac) + noise[i] * frac

class MinMaxStats:
    def __init__(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
