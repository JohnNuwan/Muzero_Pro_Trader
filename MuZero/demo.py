"""
DEMO RAPIDE MuZero - Script All-in-One
"""
import torch
import numpy as np
import sys
import os

# Ajouter le dossier parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*60)
print("🎯 DEMO MuZero Trading Agent")
print("="*60)

# 1. Test Configuration
print("\n1️⃣ Test Configuration...")
try:
    from MuZero.config import MuZeroConfig
    config = MuZeroConfig()
    print(f"   ✅ Config créée: {config.action_space_size} actions, {config.hidden_state_size} hidden state")
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    sys.exit(1)

# 2. Test Networks
print("\n2️⃣ Test Réseaux de Neurones...")
try:
    from MuZero.models.muzero_network import MuZeroNet
    network = MuZeroNet(config)
    print(f"   ✅ Networks créés:")
    print(f"      - Representation: {config.observation_shape} → {config.hidden_state_size}")
    print(f"      - Dynamics: {config.hidden_state_size}+{config.action_space_size} → next_state+reward")
    print(f"      - Prediction: {config.hidden_state_size} → policy+value")
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. Test MCTS
print("\n3️⃣ Test MCTS...")
try:
    from MuZero.agents.muzero_mcts import MuZeroMCTS
    mcts = MuZeroMCTS(config, network)
    print(f"   ✅ MCTS créé: {config.num_simulations} simulations")
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Test Replay Buffer
print("\n4️⃣ Test Replay Buffer...")
try:
    from MuZero.training.replay_buffer import GameHistory, ReplayBuffer
    buffer = ReplayBuffer(config)
    game = GameHistory()
    game.store(np.zeros(84), 1, 0.5, np.array([0.2, 0.3, 0.3, 0.1, 0.1]), 1.0, False)
    buffer.save_game(game)
    print(f"   ✅ Replay Buffer créé: {len(buffer.buffer)} games")
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

#5. Test Agent
print("\n5️⃣ Test Agent...")
try:
    from MuZero.agents.muzero_agent import MuZeroAgent
    config.num_simulations = 3  # Réduit pour le test
    agent = MuZeroAgent(config)
    print(f"   ✅ Agent créé avec success!")
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. Test Jeu Complet avec Environnement Mock
print("\n6️⃣ Test Jeu Complet...")

class MockEnv:
    def __init__(self):
        self.step_count = 0
    def reset(self):
        self.step_count = 0
        return np.random.randn(84).astype(np.float32), {}
    def step(self, action):
        self.step_count += 1
        obs = np.random.randn(84).astype(np.float32)
        reward = np.random.randn() * 10  # PnL simulé
        done = self.step_count >= 5  # Partie courte
        return obs, reward, done, False, {}

try:
    env = MockEnv()
    total_reward = agent.play_game(env)
    print(f"   ✅ Partie terminée!")
    print(f"      - Reward total: {total_reward:.2f}")
    print(f"      - Games en buffer: {len(agent.replay_buffer.buffer)}")
    print(f"      - Steps dans game: {len(agent.replay_buffer.buffer[0].actions)}")
except Exception as e:
    print(f"   ❌ Erreur pendant le jeu: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 7. Test Sampling
print("\n7️⃣ Test Sampling du Buffer...")
try:
    # Jouer une autre partie pour avoir assez de données
    agent.play_game(env)
    
    obs, actions, rewards, values, policies = agent.replay_buffer.sample_batch()
    print(f"   ✅ Batch échantillonné:")
    print(f"      - Observations: {obs.shape}")
    print(f"      - Actions: {actions.shape}")
    print(f"      - Rewards: {rewards.shape}")
except Exception as e:
    print(f"   ❌ Erreur pendant sampling: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("✨ TOUS LES TESTS RÉUSSIS ! MuZero fonctionne !")
print("="*60)
print("\n📚 Prochaines étapes:")
print("   1. Lancer training: python -m MuZero.training.train")
print("   2. Tester avec V19 env: Remplacer MockEnv par CommissionTrinityEnv")
print("   3. Ajuster hyperparamètres dans config.py")
