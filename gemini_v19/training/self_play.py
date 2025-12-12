import numpy as np
import torch
from gemini_v19.mcts.alphazero_mcts import AlphaZeroMCTS
from gemini_v19.training.simulated_market import SimulatedMarket
from gemini_v19.utils.selfplay_config import SELF_PLAY_CONFIG

class SelfPlayEngine:
    """
    Generates self-play games using AlphaZero MCTS in a simulated market.
    """
    def __init__(self, network, device):
        self.network = network
        self.device = device
        self.config = SELF_PLAY_CONFIG
        
    def generate_games(self, n_games=None):
        """
        Generate N self-play games.
        Returns list of (state, policy, value) tuples.
        """
        if n_games is None:
            n_games = self.config['n_games']
            
        all_data = []
        
        print(f"🎮 Generating {n_games} self-play games...")
        
        for i in range(n_games):
            # Pick random symbol
            symbol = np.random.choice(self.config['symbols'])
            env = SimulatedMarket(symbol, self.config['timeframe'])
            
            game_data = self._play_one_game(env)
            all_data.extend(game_data)
            
            if (i+1) % 10 == 0:
                print(f"   Game {i+1}/{n_games} complete ({len(game_data)} steps)")
                
        return all_data
    
    def _play_one_game(self, env):
        """
        Play one complete game episode.
        """
        state = env.reset()
        done = False
        game_memory = [] # Stores (state, policy, value_placeholder)
        steps = 0
        
        while not done and steps < self.config['max_steps']:
            # MCTS Search
            # We need to create a new MCTS instance or reset it? 
            # AlphaZeroMCTS usually takes network and env.
            # Ideally we clone the env for MCTS simulations if MCTS modifies it.
            # But our MCTS implementation might rely on copy/deepcopy of env.
            # Let's assume MCTS handles env copying or we pass a copy.
            
            # Note: AlphaZeroMCTS in V19 likely expects an env that can be copied.
            # SimulatedMarket should be copyable.
            
            mcts = AlphaZeroMCTS(self.network, env, n_simulations=self.config['mcts_simulations'])
            policy, root_node = mcts.search(state, temperature=self.config['temperature'])
            
            # Select Action (Sample from policy)
            action = np.random.choice(len(policy), p=policy)
            
            # Execute
            next_state, reward, done, _, _ = env.step(action)
            
            # Store
            game_memory.append({
                'state': state,
                'policy': policy,
                'reward': reward,
                'value': 0 # Placeholder
            })
            
            state = next_state
            steps += 1
            
        # Calculate Returns (Value Target)
        # We use n-step return or Monte Carlo return (sum of future rewards)
        # For trading, reward is change in equity.
        # Value should be normalized [-1, 1] or scaled.
        # Let's use cumulative reward from state t to end.
        
        processed_data = []
        future_return = 0
        
        # Backpropagate returns
        for t in reversed(range(len(game_memory))):
            reward = game_memory[t]['reward']
            future_return += reward
            
            # Normalize return? 
            # Trading rewards can be large (PnL).
            # AlphaZero usually expects [-1, 1].
            # We can tanh the return.
            
            value_target = np.tanh(future_return / 100.0) # Scale factor 100 (e.g. $100 gain = 0.76)
            
            processed_data.append((
                game_memory[t]['state'],
                game_memory[t]['policy'],
                value_target
            ))
            
        return processed_data # Reversed list, but order doesn't matter for training buffer
