# 🏛️ Gemini V19: Architecture Technique

## 🎯 Vue d'Ensemble

V19 est une implémentation AlphaZero-style pour le trading, combinant :
- **Dual-Head Neural Network** (Policy + Value)
- **Monte Carlo Tree Search (MCTS)** avec PUCT
- **Self-Play** pour continuous learning
- **Adversarial Training** pour robustesse

---

## 🧠 Architecture du Réseau

### Dual-Head Network

```
Input: State Vector (78 features)
    ↓
Shared Trunk (ResNet-style)
    ├── Conv1D → BatchNorm → ReLU
    ├── Conv1D → BatchNorm → ReLU
    └── Linear (256 units)
    ↓
Split into 2 heads
    ├─→ Policy Head
    │     └── Linear(256, 5) → Softmax → π(a|s)
    │
    └─→ Value Head
          └── Linear(256, 1) → Tanh → V(s) ∈ [-1, 1]
```

### Input Features (78 total)

#### Multi-Timeframe Features (6 TF × 13 features = 78)
Timeframes: M1, M5, M15, H1, H4, D1

Pour chaque timeframe :
1. **RSI** (Relative Strength Index)
2. **MFI** (Money Flow Index)
3. **ADX** (Average Directional Index)
4. **Z-Score** (Normalized price)
5. **Trend Score** (Custom metric)
6. **Linear Regression Angle**
7. **Fibonacci Position** (0-1 scale)
8. **Distance to Resistance**
9. **Distance to Support**
10. **Skewness** (Distribution)
11. **Kurtosis** (Tails)
12. **Shannon Entropy** (Uncertainty)
13. **Hurst Exponent** (Trend persistence)

#### Time Features (4)
1. `sin(2π × hour / 24)` : Hour cyclical encoding
2. `cos(2π × hour / 24)`
3. `sin(2π × day / 7)` : Day of week encoding
4. `cos(2π × day / 7)`

#### Position Features (2)
1. **Position State** : -1 (short), 0 (flat), 1 (long)
2. **PnL %** : Current unrealized P&L percentage

**Total** : 78 + 4 + 2 = **84 features** (final input dim)

### Output Actions (5)

0. **HOLD** : Ne rien faire
1. **BUY** : Ouvrir/ajouter une position longue
2. **SELL** : Ouvrir/ajouter une position courte
3. **SPLIT** : Fermer 50% de la position
4. **CLOSE** : Fermer 100% de la position

---

## 🌲 MCTS Architecture

### Node Structure
```python
class MCTSNode:
    state: np.array          # State vector
    parent: MCTSNode         # Parent node
    children: Dict[int, MCTSNode]  # {action: child_node}
    
    # Stats
    visit_count: int         # N(s,a)
    value_sum: float         # W(s,a)
    prior: float             # P(s,a) from policy network
    
    # Methods
    Q(): float               # Mean value = W / N
    U(c_puct): float         # Exploration bonus
    PUCT(c_puct): float      # Selection score = Q + U
```

### MCTS Algorithm (4 steps)

#### 1. Selection
Descendre l'arbre en choisissant à chaque nœud l'action avec le plus haut score PUCT :
```
PUCT(s,a) = Q(s,a) + c_puct × P(s,a) × √N(s) / (1 + N(s,a))
```
- `Q(s,a)` : Valeur moyenne empirique
- `P(s,a)` : Prior du policy network
- `c_puct` : Constante d'exploration (~ 1.5)

#### 2. Expansion
Au nœud terminal, créer tous les enfants (actions légales) avec leurs priors :
```python
policy, _ = network(state)
for action in range(5):
    node.children[action] = MCTSNode(state, prior=policy[action])
```

#### 3. Simulation
Évaluer le nœud avec le value network (pas de rollout) :
```python
_, value = network(state)
```

#### 4. Backpropagation
Remonter le chemin et mettre à jour les stats :
```python
for node in reversed(search_path):
    node.visit_count += 1
    node.value_sum += value
    value = -value  # Flip pour adversaire
```

### Output Policy
Après 50 simulations, la policy améliorée est :
```
π_MCTS(a|s) = N(s,a) / Σ N(s,·)
```
(Proportionnelle aux visites, pas aux Q-values)

---

## 🎮 Self-Play Architecture

### Self-Play Loop

```
Initialize network θ
Initialize replay buffer D = ∅

for iteration = 1 to N:
    # Phase 1: Self-Play (génération de données)
    for episode = 1 to K:
        states, policies, values = play_episode(θ)
        D ← D ∪ {(s, π, z)}
    
    # Phase 2: Training (mise à jour du réseau)
    for epoch = 1 to E:
        batch = sample(D, batch_size=64)
        loss = alphazero_loss(batch)
        θ ← θ - α ∇_θ loss
    
    # Phase 3: Validation
    if val_performance(θ) > best_performance:
        save_model(θ)
```

### Loss Function
```python
def alphazero_loss(policy_pred, value_pred, policy_target, value_target):
    # Policy loss (cross-entropy)
    L_policy = -Σ π_target × log(π_pred)
    
    # Value loss (MSE)
    L_value = (z_target - v_pred)²
    
    # Combined
    L_total = L_policy + L_value
    return L_total
```

---

## 🥊 Adversarial Training Architecture

### Adversarial Environment

L'environnement adversarial injecte du bruit/slippage/gaps avec une probabilité donnée :

```python
class AdversarialEnv(CommissionTrinityEnv):
    def __init__(self, adversary_strength=0.2):
        super().__init__()
        self.adversary_strength = adversary_strength
    
    def step(self, action):
        state, reward, done, info = super().step(action)
        
        # Adversary intervenes
        if random.random() < self.adversary_strength:
            # Random event
            event = random.choice(['slippage', 'gap', 'news'])
            
            if event == 'slippage':
                reward *= 0.8  # 20% perte de reward
            elif event == 'gap':
                reward *= 0.5  # 50% perte
            elif event == 'news':
                reward *= 0.3  # 70% perte
            
            info['adversarial_event'] = event
        
        return state, reward, done, info
```

### Training Strategy

**Phase 1** : Entraîner agent principal sur env normal  
**Phase 2** : Entraîner adversaire à maximiser les pertes de l'agent  
**Phase 3** : Réentraîner agent principal sur env adversarial  
**Repeat** : Itérer jusqu'à convergence

---

## 🔄 Continuous Learning Architecture

### Replay Database Schema (SQLite)

```sql
CREATE TABLE live_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    symbol TEXT NOT NULL,
    state BLOB NOT NULL,        -- Pickled state vector
    action INTEGER NOT NULL,     -- 0-4
    reward FLOAT NOT NULL,
    done BOOLEAN NOT NULL,
    
    -- Metadata
    equity FLOAT,
    lot_size FLOAT,
    sl_pips INTEGER,
    tp_pips INTEGER
);

CREATE INDEX idx_timestamp ON live_trades(timestamp);
CREATE INDEX idx_symbol ON live_trades(symbol);
```

### Nightly Retraining Pipeline

```
2:00 AM (Marchés fermés)
    ↓
Load last 1000 trades from DB
    ↓
Create new training batch (state, policy, value)
    ↓
Clone current_model → new_model
    ↓
Train new_model for 50 epochs
    ↓
Validate on validation_env (500 episodes)
    ↓
If Sharpe(new) > Sharpe(old) × 1.05:
    Deploy new_model
    Archive old_model
Else:
    Discard new_model
    Log "No improvement"
```

---

## 🚀 Live Deployment Architecture

### Main Orchestrator Flow

```
Initialize AlphaZeroOrchestrator
    ├── Load champion network (alphazero_champion.pth)
    ├── Initialize MCTS (50 simulations)
    ├── Connect to MT5
    └── Initialize Replay DB

Main Loop (every 60 seconds):
    for symbol in SYMBOLS:
        1. Get current state (78 features)
        2. Run MCTS search (50 simulations)
        3. Get best action
        4. Execute trade (via MT5)
        5. Store (state, action, reward) in Replay DB
        6. Update UI (Rich dashboard)
    
    Sleep 60 seconds

Background Thread (Continuous Learning):
    if current_time == "02:00":
        nightly_retrain()
```

### Parallel Processing

Pour 10 symboles, on peut paralléliser :
```python
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(process_symbol, s): s for s in SYMBOLS}
    for future in as_completed(futures):
        symbol = futures[future]
        action = future.result()
        execute_trade(symbol, action)
```

---

## 📊 Métriques et Monitoring

### Metrics Collectées

#### Performance
- **Sharpe Ratio** : (Mean Return) / (Std Return)
- **Max Drawdown** : Max % loss from peak
- **Profit Factor** : Total Win / Total Loss
- **Win Rate** : % de trades gagnants
- **Average Win** : Moyenne des gains
- **Average Loss** : Moyenne des pertes

#### MCTS Stats
- **Average Simulations** : Nombre moyen de simulations par décision
- **Search Time** : Temps moyen de MCTS
- **Policy Entropy** : Entropie de π_MCTS (diversité)
- **Value Accuracy** : MAE entre V(s) et reward réel

#### Training Stats
- **Loss Policy** : Cross-entropy loss
- **Loss Value** : MSE loss
- **Loss Total** : Combined loss
- **Gradient Norm** : Pour détecter exploding/vanishing gradients

---

## 🛠️ Configuration et Hyperparamètres

### Network Config
```python
NETWORK_CONFIG = {
    'input_dim': 84,
    'action_dim': 5,
    'hidden_dims': [128, 256, 256, 128],
    'activation': 'relu',
    'dropout': 0.1,
}
```

### MCTS Config
```python
MCTS_CONFIG = {
    'n_simulations': 50,
    'c_puct': 1.5,
    'dirichlet_alpha': 0.3,  # Root exploration noise
    'exploration_fraction': 0.25,
}
```

### Training Config
```python
TRAINING_CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs_per_iteration': 10,
    'self_play_episodes': 100,
    'replay_buffer_size': 10000,
    'validation_episodes': 500,
}
```

### Continuous Learning Config
```python
CONTINUOUS_LEARNING_CONFIG = {
    'retrain_time': '02:00',
    'lookback_trades': 1000,
    'retrain_epochs': 50,
    'improvement_threshold': 1.05,  # 5% Sharpe improvement
}
```

---

## 🔐 Sécurité et Robustesse

### Fail-Safes

1. **Max Daily Drawdown** : Si DD > 5%, arrêt automatique
2. **Max Position Size** : Lot size limité à 2% de equity
3. **MCTS Timeout** : Si search > 10s, fallback to policy head only
4. **Network Validation** : Avant déploiement, validation sur test set

### Error Handling

```python
try:
    action = self.mcts.search(state)
except Exception as e:
    logger.error(f"MCTS failed: {e}")
    # Fallback to policy head
    policy, _ = self.network(state)
    action = torch.argmax(policy).item()
```

---

## 📚 Technologies et Dépendances

### Core
- **PyTorch** : Neural networks
- **Gymnasium** : RL environment
- **NumPy** : Numerical computing

### Trading
- **MetaTrader5** : Broker interface
- **pandas** : Data manipulation
- **TA-Lib** : Technical indicators

### Visualization
- **Rich** : Terminal UI
- **Matplotlib** : Plotting
- **TensorBoard** : Training curves

### Database
- **SQLite** : Replay database
- **SQLAlchemy** : ORM

---

## 🎯 Next Steps

Voir `ROADMAP.md` pour le plan de développement détaillé.
