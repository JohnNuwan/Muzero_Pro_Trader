"""
MuZero V2 Configuration - FTMO Optimized
- More aggressive learning rate
- Higher quality trade rewards
- Final account growth bonus
"""

class MuZeroConfigV2:
    def __init__(self):
        # Network
        self.observation_shape = (84,)
        self.action_space_size = 5     # HOLD, BUY, SELL, SPLIT, CLOSE
        self.hidden_state_size = 64
        self.network_hidden_dims = [256, 256]
        
        # Symbols (Same as V1)
        self.symbols = [
            "EURUSD", "XAUUSD", "BTCUSD",
            "US30.cash", "US500.cash", "USDJPY",
            "GBPUSD", "USDCAD", "USDCHF",
            "GER40.cash", "US100.cash"
        ]
        
        # Training - AGGRESSIVE
        self.batch_size = 128
        self.learning_rate = 5e-5      # ×5 faster than V1 (was 1e-5)
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.training_steps = 20000    # 2× longer training
        self.checkpoint_interval = 100
        self.num_unroll_steps = 5
        self.td_steps = 10
        
        # MCTS - Enhanced
        self.num_simulations = 150     # Increased from 100
        self.discount = 0.99
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        
        # Replay Buffer
        self.window_size = 150000      # ×1.5 bigger buffer
        self.batch_size = 128
        
        # Self-Play
        self.max_moves = 500
        
        # Paths
        self.results_path = "MuZero/results_v2"
        self.weights_path = "MuZero/weights_v2"
        
        # V2 Specific - FTMO Rewards
        self.quality_trade_bonus = 5.0      # +1% trade = +5 points (was +2)
        self.final_growth_bonus = 100.0    # +10% final account = +100 points (NEW!)
        self.final_growth_threshold = 0.10  # 10% growth threshold
        
    def visit_softmax_temperature_fn(self, trained_steps):
        """More aggressive temperature decay for faster convergence"""
        if trained_steps < 0.4 * self.training_steps:
            return 1.0
        elif trained_steps < 0.7 * self.training_steps:
            return 0.5
        else:
            return 0.1  # Lower final temp for more exploitation
