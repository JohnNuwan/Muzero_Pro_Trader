"""
MuZero V3 Configuration - Pro Trader Edition

Enhanced Features:
- SLBE (Stop Loss Break Even) system
- Time-based drawdown penalties  
- Smart SPLIT/CLOSE rewards
- Dynamic position scaling
"""

class MuZeroConfigV3:
    def __init__(self):
        # Network
        self.observation_shape = (142,)  # DeepTrinity (136) + V3 (pos, pnl, slbe) + Time (hr, day, vol) = 142
        self.action_space_size = 5
        self.hidden_state_size = 64
        self.network_hidden_dims = [256, 256]
        
        # Symbols
        self.symbols = [
            "EURUSD", "XAUUSD", "BTCUSD",
            "US30.cash", "US500.cash", "USDJPY",
            "GBPUSD", "USDCAD", "USDCHF",
            "GER40.cash", "US100.cash"
        ]
        
        # Training - AGGRESSIVE
        self.batch_size = 128
        self.learning_rate = 5e-5
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.training_steps = 30000  # Longer for V3
        self.checkpoint_interval = 100
        self.num_unroll_steps = 5
        self.td_steps = 10
        
        # MCTS
        self.num_simulations = 150
        self.discount = 0.99
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.50 # Electrochoc V3.1 (was 0.25)
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        
        # Replay Buffer
        self.window_size = 200000  # Bigger for V3
        self.batch_size = 128
        
        # Self-Play
        self.max_moves = 500
        
        # Paths
        self.results_path = "MuZero/results_v3"
        self.weights_path = "MuZero/weights_v3"
        
        # V3 Specific - Pro Trader Rewards (HUNGER MODE V3.1)
        self.quality_trade_bonus = 10.0     # +10 pts per 1% trade (DOUBLED from 5.0)
        self.final_growth_bonus = 50.0      # +50 pts bonus (REACTIVATED from 0.0)
        self.final_growth_threshold = 0.10
       
        # V3 NEW: SLBE Rewards (HUNGER MODE - DOUBLED)
        self.slbe_activation_bonus = 6.0    # +6 pts for activating SLBE (was 3.0)
        self.split_with_profit_bonus = 10.0 # +10 pts for smart SPLIT (was 5.0)
        self.close_big_winner_bonus = 15.0  # +15 pts for CLOSE >+2% (was 7.5)
        
        # V3 NEW: Time Penalties (INCREASED)
        self.drawdown_time_penalty_rate = 0.2   # -0.2 per 20 steps (was 0.05)
        self.max_drawdown_penalty = 10.0        # -10 pts for >5% DD (was 3.0)
        self.loss_penalty_multiplier = 2.0      # 2.0x penalty for losses (was 1.5x)
        
    def visit_softmax_temperature_fn(self, trained_steps):
        """More aggressive temperature decay"""
        if trained_steps < 0.3 * self.training_steps:
            return 1.0
        elif trained_steps < 0.6 * self.training_steps:
            return 0.5
        else:
            return 0.1
