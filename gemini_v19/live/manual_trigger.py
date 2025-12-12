import sys
import os

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import configs to override
import gemini_v19.utils.config as config
import gemini_v19.utils.selfplay_config as sp_config

print("🧪 Configuring Manual Verification Run...")

# Override for speed
config.CONTINUOUS_LEARNING_CONFIG['retrain_epochs'] = 1
sp_config.SELF_PLAY_CONFIG['n_games'] = 10
sp_config.SELF_PLAY_CONFIG['tournament_games'] = 4
sp_config.SELF_PLAY_CONFIG['mcts_simulations'] = 10 # Faster MCTS

print(f"   Epochs: {config.CONTINUOUS_LEARNING_CONFIG['retrain_epochs']}")
print(f"   Self-Play Games: {sp_config.SELF_PLAY_CONFIG['n_games']}")
print(f"   Tournament Games: {sp_config.SELF_PLAY_CONFIG['tournament_games']}")

from gemini_v19.live.continuous_learner import ContinuousLearner

if __name__ == "__main__":
    print("\n🚀 Launching ContinuousLearner...")
    learner = ContinuousLearner()
    learner.retrain()
    print("\n✅ Manual Run Complete. System is healthy.")
