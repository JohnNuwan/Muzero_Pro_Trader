import json
import os
import numpy as np

log_file = "MuZero/results/training_log_20251130_063637.json"

with open(log_file, 'r') as f:
    data = json.load(f)

avg_rewards = data.get("avg_reward", [])
steps = data.get("steps", [])

# Note: avg_reward is logged per game, steps are logged per training step.
# They might not align perfectly index-wise if they are logged at different frequencies.
# Let's check the length.
print(f"Steps: {len(steps)}")
print(f"Avg Rewards: {len(avg_rewards)}")

# If lengths differ, we need to be careful.
# Usually avg_reward is updated every game played.
# Let's just find the max avg_reward and its index.
if avg_rewards:
    max_reward = max(avg_rewards)
    max_index = avg_rewards.index(max_reward)
    print(f"Max Avg Reward: {max_reward}")
    print(f"Index: {max_index}")
    
    # We need to map this back to a checkpoint.
    # Checkpoints are saved every 100 steps.
    # We need to know which step corresponded to this reward.
    # This is tricky if they are not aligned.
    
    # Let's assume the user wants the checkpoint CLOSEST to this peak.
    # If we can't map it exactly, we'll pick the latest one if the curve is flat/rising.
    
else:
    print("No rewards found.")
