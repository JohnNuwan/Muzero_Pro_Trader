# MuZero V3.1 "Hunger Mode" - Pro Trading Agent 🚀💰

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![MT5](https://img.shields.io/badge/MetaTrader5-5.0%2B-green.svg)](https://www.metatrader5.com/)

> Advanced MuZero implementation with Hybrid Continuous Learning, Multi-Asset Trading, and Aggressive Reward Shaping for maximum profitability.

---

## 🎯 Features

### Core Architecture
- **MuZero Algorithm**: Model-based reinforcement learning with learned dynamics
- **142-Feature Observation Space**: Multi-timeframe indicators (M1-D1) + Position state + Market context (Hour, Day, Volatility)
- **5-Action Space**: HOLD, BUY, SELL, SPLIT (take profit 50%), CLOSE
- **150 MCTS Simulations**: Deep planning for optimal decision-making

### V3.1 "Hunger Mode" Enhancements
- **🔥 Doubled Reward Bonuses**:
  - Quality Trade (+1%): **+10 pts** (was +5)
  - SLBE Activation: **+6 pts** (was +3)
  - Smart SPLIT: **+10 pts** (was +5)
  - Big CLOSE (+2%): **+15 pts** (was +7.5)
  - Final Growth (+10%): **+50 pts** (reactivated)
- **Stable Penalties**: Unchanged (maintains risk discipline)
- **Result**: Agent aggressively seeks high-probability, high-reward trades

### Advanced Systems
- **Hybrid Continuous Learning**: Live trades automatically feed back into training
- **Multi-Asset Support**: 11 instruments (Forex, Crypto, Indices, Gold)
- **SLBE (Stop Loss Break Even)**: Automatic risk elimination after +1% gain
- **Inactivity Penalty**: Forces action (no "bunker" behavior)
- **Dynamic Position Sizing**: Equity-based lot calculation with symbol-specific limits

---

## 📂 Project Structure

```
MuZero/
├── config_v3.py                     # V3.1 Hunger Mode Configuration
├── models/
│   └── muzero_network.py           # Representation + Dynamics + Prediction Networks
├── agents/
│   ├── muzero_agent.py             # Training Agent
│   └── muzero_mcts.py              # Monte Carlo Tree Search
├── environment/
│   ├── commission_trinity_env_v3.py # V3.1 Trading Environment (142 features)
│   └── deep_trinity_env.py          # Base Multi-Timeframe Logic
├── training/
│   ├── train_v3.py                  # V3.1 Training Script (with Hybrid Learning)
│   ├── replay_buffer.py             # Experience Replay
│   └── logger.py                    # TensorBoard + Metrics Persistence
├── live/
│   └── live_muzero.py               # Live Trading Script (MT5 Integration)
├── utils/
│   ├── indicators.py                # Technical Indicators (RSI, MA, ATR, etc.)
│   └── mtf_data_loader.py           # Multi-Timeframe Data Loader
├── results_v3/                      # Training Logs + TensorBoard
├── weights_v3/                      # Model Checkpoints
└── README.md                        # This File
```

---

## 🚀 Quick Start

### 1. Prerequisites
```bash
# Windows with MT5 installed
# Python 3.10+
# NVIDIA GPU recommended (CUDA support)
```

### 2. Environment Setup
```bash
# Clone repository
cd c:\Users\nandi\Desktop\test

# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure MT5 Credentials
Create `.env` file in project root:
```env
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=Your-Broker-Server
```

### 4. Start Training
```bash
# Launch V3.1 Hunger Mode Training
python -m MuZero.training.train_v3

# Monitor with TensorBoard (in separate terminal)
tensorboard --logdir=MuZero/results_v3/runs
```

### 5. Live Trading (Demo/Real)
```bash
# Ensure MT5 is running and connected
python MuZero\live\live_muzero.py
```

---

## 📊 Training Configuration (V3.1 Hunger Mode)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Observation Shape** | 142 features | Multi-timeframe + Position + Context |
| **MCTS Simulations** | 150 | Deep planning |
| **Learning Rate** | 5e-5 | Stable convergence |
| **Batch Size** | 128 | Balanced gradient estimates |
| **Replay Buffer** | 100,000 | Long-term memory |
| **Training Steps** | 30,000 | ~24-30 hours on GPU |
| **Exploration (root_exploration_fraction)** | 0.50 | Aggressive exploration ("Electrochoc") |

### Reward Structure
| Event | Reward | Rationale |
|-------|--------|-----------|
| Trade +1% | +10 pts | Doubled to chase quality gains |
| SLBE Activated | +6 pts | Encourage risk protection |
| SPLIT @ Profit | +10 pts | Reward profit-taking discipline |
| CLOSE +2% | +15 pts | Massive bonus for big wins |
| Final Growth +10% | +50 pts | Jackpot for episode-level success |
| Trade -1% | -10 pts | 2× asymmetric penalty (unchanged) |
| Drawdown >5% | -10 pts | Hard limit on risk (unchanged) |
| Inactivity (100 steps) | -1 pts/step | Force action (unchanged) |

---

## 🧠 Hybrid Continuous Learning

### How It Works
1. **Live Trading**: `live_muzero.py` executes trades and records experiences into `GameHistory` objects.
2. **Auto-Ingestion**: `train_v3.py` scans `results_v3/live_buffer/` every few steps.
3. **Replay Integration**: Live games are added to the replay buffer and trained on alongside simulated games.
4. **Continuous Adaptation**: Model learns from real market conditions in real-time.

### Benefits
- ✅ Adapts to market regime changes (e.g., high volatility events)
- ✅ Reduces overfitting to historical data
- ✅ Self-improving system without manual retraining

---

## 🎮 Live Trading Features

### "Health Bar" Visualization
```
🔍 EURUSD     Val:  2.33 | Life: ████████░░ (85%) | Pol: ['0.95', '0.01', ...']
🔍 BTCUSD     Val:  2.48 | Life: ██████████ (100%) | Pol: ['0.97', '0.01', ...']
```
- **Life**: Steps since last trade (depletes over time)
- **KICK Mechanism**: Forces action when Life hits 0% (prevents excessive passivity)

### Dynamic Filling Mode
- Auto-detects broker's supported order filling modes (IOC/FOK)
- Fixes `SYMBOL_FILLING_IOC` AttributeError on older MT5 versions

### Cooldown System
- 30-minute cooldown after opening a position (prevents churning)

---

## 📈 Expected Performance (V3.1 Hunger Mode)

| Metric | V3.0 (Conservative) | V3.1 Hunger (Target) |
|--------|---------------------|----------------------|
| **Avg Reward/Episode** | ~40 pts | **70-90 pts** |
| **Win Rate** | ~55% | **52-58%** |
| **Trades/Episode** | 8-12 | **5-8** (more selective) |
| **Avg Profit/Trade** | ~0.5% | **~1.5%** (big fish hunting) |
| **Monthly Return** | ~12% | **~20-30%** (aggressive) |

---

## 🛠️ Development Tools

### TensorBoard Monitoring
```bash
tensorboard --logdir=MuZero/results_v3/runs
# Open http://localhost:6006
```
**Graphs Available**:
- Total Loss (Blue)
- Loss Breakdown (Value, Policy, Reward)
- Reward Evolution
- Reward Distribution

### Checkpoints
- Saved every 100 steps in `MuZero/weights_v3/`
- `best_model_v3.pth`: Best performing model (auto-selected)

---

## 🔒 Security & Git

### Credentials
- **Never commit `.env`** (contains MT5 passwords)
- Use `.env.example` as template for collaborators

### Version Control
```bash
# Initialized with:
git init
git add .
git commit -m "V3.1 Hunger Mode - Production Ready"
```

**Files Excluded** (`.gitignore`):
- `.env` (credentials)
- `venv/` (dependencies)
- `results_v3/` (large training data)
- `weights_v3/*.pth` (model files, except best)

---

## 🧪 Testing

### Quick Validation
```bash
# Test environment loading
python -c "from MuZero.environment.commission_trinity_env_v3 import CommissionTrinityEnvV3; env = CommissionTrinityEnvV3('EURUSD'); print('✅ Env OK')"

# Test agent initialization
python -c "from MuZero.agents.muzero_agent import MuZeroAgent; from MuZero.config_v3 import MuZeroConfigV3; agent = MuZeroAgent(MuZeroConfigV3()); print('✅ Agent OK')"
```

---

## 📚 Documentation

- **Architecture**: 3-network design (Representation, Dynamics, Prediction)
- **MCTS Algorithm**: Upper Confidence Bound for Trees (PUCT)
- **Reward Shaping**: Asymmetric penalties (2× for losses), big bonuses for wins
- **142 Features**: Multi-timeframe indicators + Position state + Time context

---

## 🎯 Roadmap

### Current Phase: V3.1 Hunger Mode Training
- [x] Hunger Mode configuration (2025-12-13)
- [/] Training to 30,000 steps (~24-30h)
- [ ] Validate on demo account (1-2 weeks)

### Next Phase: FTMO Challenge Preparation
- [ ] Add news filter (no trading 30min before/after events)
- [ ] Implement strict 1% risk per trade
- [ ] Backtest in "FTMO Mode" (Max DD <8%)

### Future Enhancements
- [ ] Ensemble Models (3× MuZero voting)
- [ ] Meta-Learning (adaptive hyperparameters)
- [ ] Multi-GPU training

---

## 🤝 Credits

- **MuZero Algorithm**: DeepMind (Schrittwieser et al., 2019)
- **Trading Environment**: Custom Trinity Architecture
- **MT5 Integration**: MetaQuotes MetaTrader5 Python API

---

**Made with 🔥 and 🧠 for professional algorithmic trading.**

> ⚠️ **Disclaimer**: Forex/CFD trading carries high risk. This software is for educational/research purposes. Trade responsibly.
