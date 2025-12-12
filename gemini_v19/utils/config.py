# Gemini V19 Configuration

# Network Configuration
NETWORK_CONFIG = {
    'input_dim': 84,        # 78 features + 4 time + 2 position
    'action_dim': 5,        # HOLD, BUY, SELL, SPLIT, CLOSE
    'hidden_dims': [256, 256, 256],
    'activation': 'relu',
    'dropout': 0.1,
    'use_batch_norm': True
}

# MCTS Configuration
MCTS_CONFIG = {
    'n_simulations': 50,
    'c_puct': 1.5,          # Exploration constant
    'dirichlet_alpha': 0.3, # Root noise for exploration
    'exploration_fraction': 0.25,
    'temperature': 1.0      # Temperature for policy sampling (1.0 for training, 0.1 for eval)
}

# Training Configuration
TRAINING_CONFIG = {
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'batch_size': 64,
    'epochs_per_iteration': 10,
    'self_play_episodes': 100,
    'replay_buffer_size': 10000,
    'validation_episodes': 20,
    'checkpoint_dir': 'models/champions',
    'log_dir': 'logs'
}

# Environment Configuration
ENV_CONFIG = {
    'symbol': 'EURUSD',
    'timeframe': 'M5',
    'lookback': 1000,
    'initial_balance': 10000.0,
    'leverage': 100,
    'commission': 7.0,      # per lot
    'spread': 1.0,          # pips
    'reward_scaling': 1.0
}

# Continuous Learning Configuration
CONTINUOUS_LEARNING_CONFIG = {
    'retrain_time': '22:35',    # Start at 10 PM → Finish ~04:00 AM
    'lookback_trades': 1000,
    'retrain_epochs': 300,  # Augmenté de 50 → 300 pour apprentissage robuste
    'improvement_threshold': 1.05  # 5% improvement required to deploy
}
