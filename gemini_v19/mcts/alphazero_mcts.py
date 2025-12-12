import torch
import numpy as np
import copy
from .mcts_node import MCTSNode
from .puct import select_child

class AlphaZeroMCTS:
    """
    Monte Carlo Tree Search using AlphaZero algorithm
    """
    def __init__(self, network, env, n_simulations=50, c_puct=1.5, dirichlet_alpha=0.3, exploration_fraction=0.25):
        self.network = network
        self.env = env  # Need a copy or clone capability for simulation
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        
    def search(self, root_state, temperature=1.0):
        """
        Run MCTS simulations and return action probabilities
        """
        self.network.eval() # Ensure eval mode for BatchNorm with batch_size=1
        root = MCTSNode(root_state, prior_p=1.0)
        
        # Add Dirichlet noise to root priors for exploration
        self._expand_root(root)
        
        for _ in range(self.n_simulations):
            node = root
            search_path = [node]
            
            # 1. Selection
            # Simulate environment steps as we descend
            # Note: In a real trading env, we might need a lightweight simulator model
            # For now, we assume we can clone the env or just use the state for expansion
            # Limitation: Without a perfect simulator, we can't predict next_state perfectly.
            # AlphaZero assumes a perfect simulator (like Chess board).
            # For Trading, we use the "Dynamics Model" approach implicitly or just tree search on Policy/Value.
            # Here, we will assume a simplified expansion where we don't actually step the env 
            # deep into the future unless we have a simulator. 
            # WAIT: AlphaZero requires stepping to get next_state. 
            # If we don't have a simulator, we can only do 1-step lookahead or need a learned model (MuZero).
            # For V19, we will use the CommissionTrinityEnv as the simulator.
            
            # Clone env for simulation path
            sim_env = copy.deepcopy(self.env) 
            
            while node.is_expanded():
                action, node = select_child(node, self.c_puct)
                search_path.append(node)
                # Step the sim env
                _, _, done, _, _ = sim_env.step(action)
                if done:
                    break
            
            # 2. Expansion
            # If not terminal, expand
            # We need the state of the leaf node. 
            # Since we stepped the sim_env, we can get it.
            # Note: This deepcopy is expensive. For optimization, we might need a faster simulator.
            
            leaf_state = node.state # This should be updated by the step? 
            # Actually, the node stores the state. But we need to compute next_state from action.
            # In standard MCTS, nodes represent states.
            # If we just descended, 'node' is the child we picked. It already has a state?
            # Yes, if we created it during expansion.
            
            # But wait, we only create children in expansion.
            # So if we are at a leaf (unexpanded), we expand it.
            
            # 3. Evaluation (Simulation)
            value = self._evaluate(node, sim_env)
            
            # 4. Backpropagation
            self._backpropagate(search_path, value)
            
        # Calculate policy from visit counts
        visit_counts = np.array([child.visit_count for child in root.children.values()])
        actions = list(root.children.keys())
        
        if temperature == 0:
            best_idx = np.argmax(visit_counts)
            policy = np.zeros(len(actions))
            policy[best_idx] = 1.0
        else:
            # Softmax with temperature
            visit_counts = visit_counts ** (1.0 / temperature)
            policy = visit_counts / np.sum(visit_counts)
            
        # Map back to full action space size (5)
        full_policy = np.zeros(5)
        for i, action in enumerate(actions):
            full_policy[action] = policy[i]
            
        return full_policy, root
        
    def _expand_root(self, root):
        """Special expansion for root with Dirichlet noise"""
        with torch.no_grad():
            # Debug shape
            # print(f"DEBUG: Root state shape: {root.state.shape}")
            device = next(self.network.parameters()).device
            policy_logits, _ = self.network(torch.FloatTensor([root.state]).to(device))
            policy = policy_logits[0].cpu().numpy() # Softmax already applied in net? Yes.
            
        # Add noise
        noise = np.random.dirichlet([self.dirichlet_alpha] * 5)
        policy = (1 - self.exploration_fraction) * policy + self.exploration_fraction * noise
        
        for action in range(5):
            # Create children but don't compute their states yet (lazy expansion)
            # Wait, we need next_state to create the node.
            # If we use lazy expansion, we compute state only when visiting.
            # Let's do eager expansion for now (simpler).
            
            # Step env to get next state
            # Note: This modifies the env passed to this method? No, we should use a clone.
            # But expanding root is done once.
            
            # CRITICAL: We need a way to get next_state(s, a) without ruining the main env.
            # We will use a clone.
            sim_env = copy.deepcopy(self.env)
            next_state, _, _, _, _ = sim_env.step(action)
            
            root.children[action] = MCTSNode(next_state, parent=root, prior_p=policy[action], action=action)

    def _evaluate(self, node, sim_env):
        """
        Evaluate a leaf node using the network.
        Also expands the node (adds children).
        """
        # Check if terminal
        # If sim_env is done, return actual reward? 
        # For now, rely on Value Head.
        
        with torch.no_grad():
            device = next(self.network.parameters()).device
            policy_logits, value = self.network(torch.FloatTensor([node.state]).to(device))
            policy = policy_logits[0].cpu().numpy()
            value = value.item()
            
        # Expand
        # We need to create children for all valid actions
        for action in range(5):
            # We need next_state for each action.
            # This requires N steps of the environment.
            # This is the heavy part of MCTS with a complex env.
            
            child_env = copy.deepcopy(sim_env)
            next_state, _, done, _, _ = child_env.step(action)
            
            if not done:
                node.children[action] = MCTSNode(next_state, parent=node, prior_p=policy[action], action=action)
                
        return value

    def _backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            # Value flip? In 2-player games yes. In single player (trading), no.
            # We want to maximize our own return.
            # So we keep value as is.
