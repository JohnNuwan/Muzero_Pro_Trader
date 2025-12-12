
# MuZero Configuration

class MuZeroConfig:
    def __init__(self):
        # Network
        self.observation_shape = (84,) # From V19 Config
        self.action_space_size = 5     # HOLD, BUY, SELL, SPLIT, CLOSE
        self.hidden_state_size = 64    # Size of the hidden state in MuZero
        self.network_hidden_dims = [256, 256] # For internal MLPs
        
        # Symbols (Same as V19)
        self.symbols = [
            "EURUSD",
            "XAUUSD",     # GOLD
            "BTCUSD",
            "US30.cash",
            "US500.cash",
            "USDJPY",
            "GBPUSD",
            "USDCAD",
            "USDCHF",
            "GER40.cash", # DAX
            "US100.cash"  # NASDAQ
        ]
        
        # Training (MINIMAL for fast testing)
        self.batch_size = 128          # Increased for stability
        self.learning_rate = 1e-5  # Reduced from 1e-4 to stabilize training
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.training_steps = 10000
        self.checkpoint_interval = 100
        self.num_unroll_steps = 5      
        self.td_steps = 10             
        
        # MCTS (MINIMAL for speed)
        self.num_simulations = 100     # Increased for better planning
        self.discount = 0.99
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        
        # Replay Buffer
        self.window_size = 100000      # Increased to prevent forgetting
        self.batch_size = 128          # Match training
        
        # Self-Play
        self.max_moves = 500            # Very short episodes

        # Paths
        self.results_path = "MuZero/results"
        self.weights_path = "MuZero/weights"

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (visited the most) is selected.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25
