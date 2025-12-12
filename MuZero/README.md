# MuZero Trading Agent 🎯

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

> Implementation of MuZero for Algorithmic Trading based on V19 Architecture

## 📁 Structure du Projet

```
MuZero/
├── config.py                    # Configuration centrale
├── models/                      # Réseaux de Neurones
│   ├── __init__.py
│   └── muzero_network.py       # Representation, Dynamics, Prediction
├── agents/                      # Agents et MCTS
│   ├── __init__.py
│   ├── muzero_agent.py         # Agent principal
│   └── muzero_mcts.py          # MCTS avec modèle appris
├── training/                    # Entraînement
│   ├── __init__.py
│   ├── replay_buffer.py        # Replay Buffer
│   └── train.py                # Boucle d'entraînement
├── tests/                       # Tests
│   ├── __init__.py
│   ├── test_integration.py     # Tests d'intégration
│   └── simple_test.py          # Test rapide
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md
│   ├── TRAINING_LOOP.md
│   └── README.md
├── results/                     # Résultats d'entraînement
└── weights/                     # Poids des modèles
```

## 🚀 Quick Start

### Installation

```bash
# Activer l'environnement virtuel
.\venv\Scripts\activate

# Les dépendances sont déjà installées (torch, numpy, etc.)
```

### Test Rapide

```bash
# Test simple (1 minute)
python MuZero/tests/simple_test.py

# Tests complets
python -m unittest MuZero.tests.test_integration
```

### Entraînement

```bash
python -m MuZero.training.train
```

### GPU Support (CUDA)

MuZero utilise automatiquement le GPU si disponible:

```bash
# Vérifier GPU
python MuZero/check_gpu.py

# Installer PyTorch avec CUDA (si nécessaire)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Performance GPU**:
- RTX 2060: ~2ms par forward pass
- Avec Mixed Precision (AMP) activé automatiquement
- Accélération 10-50x vs CPU

## 🧠 Architecture MuZero

MuZero utilise 3 réseaux de neurones :

| Réseau | Fonction | Input | Output |
|--------|----------|-------|--------|
| **Representation** (`h`) | Encode l'observation | État marché (84) | Hidden state (64) |
| **Dynamics** (`g`) | Modèle du monde | Hidden state + Action | Next state + Reward |
| **Prediction** (`f`) | Stratégie | Hidden state | Policy + Value |

### Différence avec V19 (AlphaZero)

- **V19**: Utilise l'environnement réel pour la planification MCTS
- **MuZero**: Utilise un modèle neuronal appris (Dynamics Network)
- **Avantage**: Planification 100x plus rapide, apprentissage sans règles

## 📊 Configuration

Voir [config.py](config.py) pour tous les hyperparamètres :

```python
# MCTS
num_simulations = 50
discount = 0.99

# Training
learning_rate = 1e-3
batch_size = 64
num_unroll_steps = 5

# Network
hidden_state_size = 64
network_hidden_dims = [256, 256]
```

## 📚 Documentation Complète

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - Architecture détaillée
- [TRAINING_LOOP.md](docs/TRAINING_LOOP.md) - Cycle d'entraînement
- [Walkthrough](../../.gemini/antigravity/brain/f023137b-4878-4a2d-ad4f-39341c85a516/walkthrough.md) - Implémentation complète

## 🧪 Tests

```bash
# Test d'initialisation
python -c "from MuZero.agents import MuZeroAgent; from MuZero.config import MuZeroConfig; agent = MuZeroAgent(MuZeroConfig()); print('✅ Agent OK')"

# Tests unitaires
python -m unittest discover MuZero/tests
```

## 🔧 Utilisation

```python
from MuZero.config import MuZeroConfig
from MuZero.agents import MuZeroAgent

# Configuration
config = MuZeroConfig()
config.num_simulations = 50

# Créer l'agent
agent = MuZeroAgent(config)

# Jouer une partie
from gemini_v15.environment.commission_trinity_env import CommissionTrinityEnv
env = CommissionTrinityEnv(symbol="EURUSD")
total_reward = agent.play_game(env)
```

## 📈 Status Actuel

✅ **Entraînement en cours** avec GPU (NVIDIA RTX 2060)
- Device: **CUDA activé**
- Mixed Precision (AMP): **Enabled**
- Symboles: 11 paires (EURUSD, XAUUSD, BTCUSD, indices)
- Configuration: 64 batch_size, 50 simulations, 500 max_moves

## 🎯 Prochaines Étapes

- [x] Entraînement initial lancé (nuit)
- [ ] Monitoring des checkpoints
- [ ] Comparaison performances vs V19
- [ ] Intégration live trading
- [ ] Multi-symboles optimisé
- [ ] Dashboard de monitoring

## 🤝 Basé sur

- **V19**: Architecture AlphaZero pour le trading
- **MuZero**: DeepMind's model-based RL
- **Environnement**: CommissionTrinityEnv (V15)

---

Made with ❤️ for algorithmic trading
