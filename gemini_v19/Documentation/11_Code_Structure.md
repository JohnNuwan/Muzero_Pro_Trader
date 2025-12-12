# 11 - Structure du Code

## 📚 Organisation

```
gemini_v19/
├── live/              # Production trading
│   ├── main_v19_multi.py       # Main trader (11 symbols)
│   ├── continuous_learner.py   # Nightly retraining
│   ├── scheduler.py            # Cron scheduler
│   ├── pyramiding.py           # Pyramid manager
│   └── replay_db.py            # Experience replay
│
├── training/          # Self-play & data collection
│   ├── self_play.py            # Self-play engine
│   ├── hybrid_trainer.py       # Hybrid training
│   ├── tournament.py           # Tournament validation
│   ├── simulated_market.py     # Simulated env
│   └── collect_m15_data.py     # MT5 data fetcher
│
├── mcts/              # MCTS algorithm
│   ├── alphazero_mcts.py       # Main MCTS
│   ├── mcts_node.py            # Node class
│   └── puct.py                 # PUCT selection
│
├── models/            # Neural networks
│   ├── alphazero_net.py        # Dual-head network
│   ├── loss.py                 # Loss functions
│   └── champions/              # Saved models
│
├── environment/       # Trading environments
│   └── (inherited from gemini_v15)
│
├── utils/             # Utilities
│   ├── config.py               # Global config
│   ├── selfplay_config.py      # Self-play config
│   ├── pyramid_config.py       # Pyramid config
│   ├── telegram_notifier.py    # Notifications
│   └── logger.py               # Logging
│
└── Documentation/     # Technical docs
    ├── README.md
    ├── 01_AlphaZero_Theory.md
    ├── ...
    └── 12_Configuration.md
```

---

## 🔄 Data Flow

```
MT5 → Environment → State (84) → MCTS → Network → Policy/Value
                                    ↓
                                 Action
                                    ↓
                              Environment.step()
                                    ↓
                            Reward + Next State
                                    ↓
                              Replay Database
                                    ↓
                          ContinuousLearner (nightly)
                                    ↓
                           Self-Play + Hybrid Training
                                    ↓
                              Tournament
                                    ↓
                          Deploy New Champion
```

---

## 🎯 Design Patterns

### 1. Strategy Pattern

**PyramidManager** encapsule la logique de pyramiding.

### 2. Observer Pattern

**TelegramNotifier** observe les events de trading.

### 3. Singleton Pattern

**ReplayDatabase** instance unique partagée.

---

**Prochaine section** : [12_Configuration.md](12_Configuration.md)
