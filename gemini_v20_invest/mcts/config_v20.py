"""
V20-Invest MCTS Configuration
Optimized for long-term investment (D1, W1, M1 timeframes)
"""

# MCTS Parameters
SIMULATIONS = 50  # Plus précis que V19 (qui a 5)
C_PUCT = 1.0      # Exploration constant

# Horizon & Actions
HORIZON_DAYS = 30  # 30 jours de simulation en D1
ACTIONS = [
    'BUY_25%',    # Acheter 25% du cash
    'BUY_50%',    # Acheter 50% du cash
    'BUY_100%',   # Acheter 100% du cash (all-in)
    'SELL_25%',   # Vendre 25% de la position
    'SELL_50%',   # Vendre 50% de la position
    'SELL_100%',  # Vendre 100% de la position
    'HOLD'        # Ne rien faire
]

# Timeframes
TIMEFRAME_PRIMARY = 'D1'    # Daily (principal)
TIMEFRAME_SECONDARY = 'W1'  # Weekly (confirmation)
TIMEFRAME_TERTIARY = 'M1'   # Monthly (tendance)

# Model
INPUT_SIZE = 46  # Indicateurs réellement disponibles dans les CSV MT5
HIDDEN_SIZE = 256
OUTPUT_POLICY_SIZE = len(ACTIONS)  # 7 actions
OUTPUT_VALUE_SIZE = 1

# Training
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TEMPERATURE = 1.0  # Pour exploration during training

# Reward Function
REWARD_HOLDING_COST = 0.0001  # Coût d'opportunité par jour
REWARD_DIVIDEND_BOOST = 0.05  # Bonus dividendes
REWARD_WIN_THRESHOLD = 0.10   # +10% = success

if __name__ == "__main__":
    print("V20 MCTS Config:")
    print(f"- Simulations: {SIMULATIONS}")
    print(f"- Actions: {len(ACTIONS)}")
    print(f"- Horizon: {HORIZON_DAYS} days")
    print(f"- Timeframe: {TIMEFRAME_PRIMARY}")
