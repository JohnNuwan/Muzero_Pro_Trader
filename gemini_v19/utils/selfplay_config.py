"""
V19 Self-Play Configuration
"""

SELF_PLAY_CONFIG = {
    # Generation
    'n_games': 50,              # Games per iteration (quick test: was 500)
    'max_steps': 100,            # Max steps per game
    'mcts_simulations': 50,      # MCTS simulations per move
    'temperature': 1.0,          # Exploration temperature
    
    # Mixing
    'self_play_weight': 0.6,     # 60% self-play data
    'real_data_weight': 0.4,     # 40% real trades data
    
    # Tournament
    'tournament_games': 50,      # Validation games
    'win_rate_threshold': 0.55,  # 55% win rate required to replace champion
    'sharpe_improvement': 1.05,  # 5% Sharpe improvement required
    
    # Environment
    'initial_balance': 10000.0,
    'symbols': ['EURUSD', 'XAUUSD', 'BTCUSD', 'US30.cash', 'US500.cash', 
                'USDJPY', 'GBPUSD', 'USDCAD', 'USDCHF', 'GER40.cash', 'US100.cash'],  # All 11 V19 symbols
    'timeframe': 'M15',     # M15 = Better granularity, faster convergence
    'data_lookback': 2000,  # Historical bars to load for simulation
}
