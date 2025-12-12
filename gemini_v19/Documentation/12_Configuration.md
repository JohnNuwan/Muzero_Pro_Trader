# 12 - Configuration Détaillée

## 📚 Introduction

V19 est hautement configurable via plusieurs fichiers de configuration centralisés.

---

## ⚙️ NETWORK_CONFIG

```python
NETWORK_CONFIG = {
    'input_dim': 84,                # Features count
    'action_dim': 5,                # HOLD, BUY, SELL, SPLIT, CLOSE
    'hidden_dims': [256, 256, 256], # Shared trunk
    'activation': 'relu',
    'dropout': 0.1,
    'use_batch_norm': True
}
```

### Tuning Guidelines

- **hidden_dims** : Augmenter si underfitting (128 → 512)
- **dropout** : Baisser si underfitting (0.1 → 0.05)

---

## 🌲 MCTS_CONFIG

```python
MCTS_CONFIG = {
    'n_simulations': 50,            # Nombre de sims/search
    'c_puct': 1.5,                  # Exploration constant
    'dirichlet_alpha': 0.3,         # Noise sparsity
    'exploration_fraction': 0.25,   # Noise fraction
    'temperature': 1.0              # Train: 1.0, Eval: 0.1
}
```

### Tuning Guidelines

- ↑ **n_simulations** (50 → 100) : Plus précis mais plus lent
- ↑ **c_puct** (1.5 → 2.0) : Plus d'exploration
- ↓ **temperature** (1.0 → 0.5) : Moins stochastique

---

## 🧠 TRAINING_CONFIG

```python
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
```

---

## 🎮 SELF_PLAY_CONFIG

```python
SELF_PLAY_CONFIG = {
    'n_games': 500,                 # Parties/nuit
    'max_steps': 100,               # Steps max/partie
    'mcts_simulations': 50,
    'temperature': 1.0,
    'self_play_weight': 0.6,        # 60% synthetic
    'real_data_weight': 0.4,        # 40% real
    'tournament_games': 50,
    'win_rate_threshold': 0.55,     # 55% pour deploy
    'sharpe_improvement': 1.05,     # +5%
    'initial_balance': 10000.0,
    'symbols': [11 symbols],
    'timeframe': 'M15',
    'data_lookback': 2000
}
```

### Tuning Guidelines

- ↑ **n_games** (500 → 1000) : Plus de données mais ~2× plus long
- Adjust **mixing ratio** selon performance (60/40 → 70/30)

---

## 🪜 PYRAMID_CONFIG

```python
PYRAMID_CONFIG = {
    'max_pyramids': 3,              # Max 3 pyramides/position
    'pyramid_volume_ratio': 0.5,    # 50% volume principal
    'min_confidence': 0.6,          # MCTS confidence ≥ 60%
    'sl_trigger_profit_pct': 0.001  # 0.10% profit → SL to BE
}
```

---

## ⏰ CONTINUOUS_LEARNING_CONFIG

```python
CONTINUOUS_LEARNING_CONFIG = {
    'retrain_time': '02:00',        # Heure de retrain
    'lookback_trades': 1000,        # Trades à charger
    'retrain_epochs': 300,          # Epochs (~6h)
    'improvement_threshold': 1.05   # +5% requis
}
```

---

## 🎯 Best Practices

1. **Ne pas modifier** `input_dim`, `action_dim` (architecture)
2. **Tester prudemment** les changements MCTS (impact fort)
3. **Monitorer** Sharpe/Win Rate après chaque modification
4. **Sauvegarder** configs avant modifications majeures

---

## 📊 Réglages Recommandés

| Usage | n_sims | temperature | epochs |
|-------|--------|-------------|--------|
| **Development** | 10 | 1.0 | 50 |
| **Testing** | 30 | 0.5 | 100 |
| **Production** | 50 | 0.1 | 300 |

---

**Fin de la Documentation Technique V19**
