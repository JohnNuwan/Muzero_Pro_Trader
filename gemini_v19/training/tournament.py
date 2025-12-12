import numpy as np
import copy
from gemini_v19.training.simulated_market import SimulatedMarket
from gemini_v19.mcts.alphazero_mcts import AlphaZeroMCTS
from gemini_v19.utils.selfplay_config import SELF_PLAY_CONFIG

class TournamentManager:
    """
    Manages tournaments between two models (Candidate vs Champion).
    """
    def __init__(self, device):
        self.device = device
        self.config = SELF_PLAY_CONFIG
        
    def run_tournament(self, candidate_net, champion_net, n_games=None):
        """
        Run a tournament between candidate and champion.
        Returns: (win_rate, sharpe_candidate, sharpe_champion)
        """
        if n_games is None:
            n_games = self.config['tournament_games']
            
        print(f"⚔️ Starting Tournament ({n_games} games)...")
        
        wins_candidate = 0
        sharpes_candidate = []
        sharpes_champion = []
        
        for i in range(n_games):
            # Pick random symbol
            symbol = np.random.choice(self.config['symbols'])
            
            # Create identical environments for fair comparison
            # We can't easily seed the random walk in SimulatedMarket unless we modify it.
            # But SimulatedMarket loads real data and picks a random start point.
            # We can instantiate one env, pick a start point, and force it on both.
            
            env_base = SimulatedMarket(symbol, self.config['timeframe'])
            start_step = env_base.current_step # Randomly initialized in __init__ -> reset
            # Actually reset() is called to randomize.
            # Let's call reset once and get the step.
            env_base.reset()
            seed_step = env_base.current_step
            
            # Create two envs with same start
            env_cand = SimulatedMarket(symbol, self.config['timeframe'])
            env_cand.current_step = seed_step
            env_cand.balance = self.config['initial_balance']
            env_cand.equity = self.config['initial_balance']
            
            env_champ = SimulatedMarket(symbol, self.config['timeframe'])
            env_champ.current_step = seed_step
            env_champ.balance = self.config['initial_balance']
            env_champ.equity = self.config['initial_balance']
            
            # Run Episodes
            sharpe_cand = self._run_episode(candidate_net, env_cand)
            sharpe_champ = self._run_episode(champion_net, env_champ)
            
            sharpes_candidate.append(sharpe_cand)
            sharpes_champion.append(sharpe_champ)
            
            if sharpe_cand > sharpe_champ:
                wins_candidate += 1
                
            if (i+1) % 10 == 0:
                print(f"   Game {i+1}/{n_games} - Candidate Wins: {wins_candidate}")
                
        win_rate = wins_candidate / n_games
        avg_sharpe_cand = np.mean(sharpes_candidate)
        avg_sharpe_champ = np.mean(sharpes_champion)
        
        return win_rate, avg_sharpe_cand, avg_sharpe_champ
        
    def _run_episode(self, network, env):
        """
        Run a single episode and return Sharpe Ratio.
        """
        state = env._get_state() # Initial state
        done = False
        returns = []
        steps = 0
        
        while not done and steps < self.config['max_steps']:
            # MCTS Search (Fast mode for tournament)
            # Use fewer simulations for speed? Or same?
            # Config has 'mcts_simulations'.
            mcts = AlphaZeroMCTS(network, env, n_simulations=self.config['mcts_simulations'])
            
            # Deterministic choice (temperature=0) for evaluation
            policy, _ = mcts.search(state, temperature=0.0)
            action = np.argmax(policy)
            
            next_state, reward, done, _, _ = env.step(action)
            
            # Track returns (percentage change in equity)
            # Reward in env is change in equity value.
            # Return % = reward / prev_equity
            prev_equity = env.equity - reward
            if prev_equity > 0:
                ret = reward / prev_equity
                returns.append(ret)
            
            state = next_state
            steps += 1
            
        # Calculate Sharpe
        if len(returns) < 2:
            return 0.0
            
        returns = np.array(returns)
        std = np.std(returns)
        if std == 0:
            return 0.0
            
        sharpe = np.mean(returns) / std * np.sqrt(252 * 24) # Annualized (hourly)
        return sharpe
