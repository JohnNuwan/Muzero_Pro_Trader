# 03 - Architecture du Réseau Neuronal

## 📚 Introduction

Le réseau neuronal **AlphaZeroTradingNet** est au cœur du système V19. C'est un réseau **dual-head** qui prédit simultanément :

1. **Policy π(a|s)** - Distribution de probabilité sur les actions
2. **Value V(s)** - Estimation de la valeur de l'état

---

## 🏗️ Architecture Globale

```
Input (84,)
    ↓
[Shared Trunk: 3×256 MLP + BatchNorm + ReLU + Dropout]
    ↓
    ├──→ [Policy Head] → π(a|s) ∈ R^5 (proba sur 5 actions)
    └──→ [Value Head]  → V(s) ∈ [-1, 1] (scalar)
```

---

## 📥 Input Layer

### Input Dimensions: **84 features**

Décomposition :

```
84 = 6 timeframes × 13 indicators + 4 time + 2 position

= 78 (multi-timeframe indicators)
+ 4  (temporal encoding)
+ 2  (position state)
```

#### 1. Multi-Timeframe Indicators (78)

Pour chaque timeframe (M1, M5, M15, H1, H4, D1), **13 indicateurs** :

```python
features_per_tf = [
    'rsi',           # Relative Strength Index
    'mfi',           # Money Flow Index
    'adx',           # Average Directional Index
    'z_score',       # Statistical Z-score
    'trend_score',   # Proprietary trend indicator
    'linreg_angle',  # Linear regression angle
    'fibo_pos',      # Position relative to Fibonacci levels
    'dist_to_res',   # Distance to resistance
    'dist_to_sup',   # Distance to support
    'skew',          # Statistical skewness
    'kurtosis',      # Statistical kurtosis
    'entropy',       # Shannon entropy
    'hurst'          # Hurst exponent
]
```

**Total** : 6 × 13 = 78 features

#### 2. Temporal Features (4)

Encodage cyclique du temps :

```python
hour = current_time.hour
day = current_time.dayofweek

time_features = [
    sin(2π * hour / 24),      # Heure (cyclique)
    cos(2π * hour / 24),
    sin(2π * day / 7),        # Jour semaine (cyclique)
    cos(2π * day / 7)
]
```

**Pourquoi cyclique ?** Pour capturer la périodicité (17h ≈ 18h, mais 23h ≈ 0h).

#### 3. Position State (2)

```python
pos_state = {
    -1.0  si position SHORT
     0.0  si position FLAT
    +1.0  si position LONG
}

pnl_pct = (current_price - entry_price) / entry_price  si position != 0
          0.0                                           sinon
```

**Total** : 2 features

---

## 🧱 Shared Trunk

Le **Shared Trunk** est un MLP à 3 couches hidden, partagé entre les deux heads.

### Architecture

```python
self.shared_trunk = nn.Sequential(
    nn.Linear(84, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.1),
    
    nn.Linear(256, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.1),
    
    nn.Linear(256, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.1)
)
```

### Forward Pass

```
h₀ = Input         # (batch, 84)
h₁ = ReLU(BN(W₁ h₀ + b₁)) + Dropout
h₂ = ReLU(BN(W₂ h₁ + b₂)) + Dropout
h₃ = ReLU(BN(W₃ h₂ + b₃)) + Dropout
```

Output : **h₃ ∈ R^256** (shared representation)

### Batch Normalization

**Formule** :

```
BN(x) = γ * (x - μ) / √(σ² + ε) + β
```

Où :
- **μ, σ²** : Mean et variance du batch
- **γ, β** : Paramètres apprenables (scale & shift)
- **ε** : Petite constante pour stabilité numérique (1e-5)

**Avantages** :
- ✅ Accélère la convergence
- ✅ Régularisation (effet similaire au dropout)
- ✅ Permet learning rates plus élevés

### Dropout

**Formule** :

```
Dropout(x, p=0.1) = {
    x / (1-p)  avec probabilité (1-p)
    0          avec probabilité p
}
```

**Effet** :
- **Training** : Désactive aléatoirement 10% des neurones
- **Inference** : Pas de dropout (mode eval)

**Avantage** : Prévient l'overfitting en forçant redundancy.

---

## 🎭 Policy Head

Le **Policy Head** prédit une distribution de probabilité sur les actions.

### Architecture

```python
self.policy_head = nn.Sequential(
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 5)
    # Softmax appliqué dans forward()
)
```

### Forward Pass

```
p_logits = W_p2 * ReLU(W_p1 * h₃ + b_p1) + b_p2
π(a|s) = Softmax(p_logits)
```

**Softmax** :

```
π(a|s) = exp(p_logits[a]) / Σ_i exp(p_logits[i])
```

**Propriétés** :
- Σ_a π(a|s) = 1 (distribution de probabilité valide)
- π(a|s) ∈ [0, 1] pour tout a
- Max likelihood training via cross-entropy

### Output Dimensions

**π(a|s) ∈ R^5** :

```python
actions = {
    0: HOLD    # Pas de changement
    1: BUY     # Ouvrir ou pyramider long
    2: SELL    # Ouvrir ou pyramider short
    3: SPLIT   # Fermer 50% de la position
    4: CLOSE   # Fermer 100% de la position
}
```

---

## 💰 Value Head

Le **Value Head** prédit la valeur espérée de l'état.

### Architecture

```python
self.value_head = nn.Sequential(
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Tanh()
)
```

### Forward Pass

```
v = W_v2 * ReLU(W_v1 * h₃ + b_v1) + b_v2
V(s) = Tanh(v)
```

**Tanh** :

```
Tanh(x) = (e^x - e^-x) / (e^x + e^-x)
```

**Propriétés** :
- V(s) ∈ [-1, 1]
- Symétrique autour de 0
- Saturation pour grandes valeurs (|x| > 3)

### Interprétation

- **V(s) ≈ +1** : État très favorable (gains attendus)
- **V(s) ≈ 0** : État neutre
- **V(s) ≈ -1** : État défavorable (pertes attendues)

---

## 📊 Nombre de Paramètres

### Calcul

| Layer | Input | Output | Weights | Biases | BatchNorm | **Total** |
|-------|-------|--------|---------|--------|-----------|-----------|
| Linear1 | 84 | 256 | 21,504 | 256 | 512 | **22,272** |
| Linear2 | 256 | 256 | 65,536 | 256 | 512 | **66,304** |
| Linear3 | 256 | 256 | 65,536 | 256 | 512 | **66,304** |
| Policy1 | 256 | 128 | 32,768 | 128 | 0 | **32,896** |
| Policy2 | 128 | 5 | 640 | 5 | 0 | **645** |
| Value1 | 256 | 128 | 32,768 | 128 | 0 | **32,896** |
| Value2 | 128 | 1 | 128 | 1 | 0 | **129** |

**Total** : ~221,446 paramètres

---

## 🔄 Forward Pass Complet

```python
def forward(self, state):
    # Input validation
    if not isinstance(state, torch.Tensor):
        state = torch.FloatTensor(state)
    
    # Shared trunk
    features = self.shared_trunk(state)  # (batch, 256)
    
    # Policy head
    policy_logits = self.policy_head(features)  # (batch, 5)
    policy = F.softmax(policy_logits, dim=-1)
    
    # Value head
    value = self.value_head(features)  # (batch, 1)
    
    return policy, value
```

**Dimensions** :

```
Input:    (batch, 84)
          ↓
Features: (batch, 256)
          ↓
Policy:   (batch, 5)   # Softmax probabilities
Value:    (batch, 1)   # Tanh ∈ [-1, 1]
```

---

## 🎯 Loss Function

### Formule Complète

```
L(θ) = L_policy(θ) + L_value(θ) + L_reg(θ)
```

#### 1. Policy Loss (Cross-Entropy)

```
L_policy = -Σ_a π_target(a) * log(π_θ(a|s) + ε)
```

**Code** :

```python
policy_loss = -torch.sum(target_policy * torch.log(pred_policy + 1e-8), dim=1)
policy_loss = torch.mean(policy_loss)
```

#### 2. Value Loss (MSE)

```
L_value = (z - V_θ(s))²
```

**Code** :

```python
value_loss = (target_value - pred_value) ** 2
value_loss = torch.mean(value_loss)
```

#### 3. Regularization Loss (L2)

```
L_reg = λ * Σ_i θ_i²
```

**Code** :

```python
l2_reg = sum(p.pow(2).sum() for p in model.parameters())
reg_loss = weight_decay * l2_reg
```

**Total dans V19** :

```python
total_loss = policy_loss + value_loss + reg_loss
```

---

## ⚙️ Optimisation

### Optimizer : Adam

**Paramètres** :

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,          # Learning rate
    weight_decay=1e-4 # L2 regularization
)
```

**Adam Update Rule** :

```
m_t = β₁ * m_{t-1} + (1 - β₁) * ∇L
v_t = β₂ * v_{t-1} + (1 - β₂) * (∇L)²

θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
```

Où :
- **m_t** : Premier moment (moyenne des gradients)
- **v_t** : Second moment (variance des gradients)
- **β₁, β₂** : Decay rates (0.9, 0.999)
- **α** : Learning rate (1e-3)

---

## 🚀 Techniques d'Entraînement

### 1. Learning Rate Schedule

```python
# Initial training
lr = 1e-3

# Fine-tuning (continuous learning)
lr = 1e-3 * 0.1 = 1e-4
```

### 2. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Effet** : Évite les explosions de gradients.

### 3. Batch Size

- **Initial Training** : 64
- **Evaluation** : 1 (inference single state)

---

## 📈 Performance & Inference

### Temps d'Inférence

- **CPU** : ~10 ms par forward pass
- **GPU (CUDA)** : ~2 ms par forward pass

### Memory Footprint

- **Model Size** : ~850 KB (221k params × 4 bytes)
- **Activation Memory** (batch=64) : ~200 KB

---

## 🔬 Ablation Study

### Impact des Composantes

| Variante | Sharpe Ratio | Win Rate | Notes |
|----------|--------------|----------|-------|
| **Full Model** | 1.8 | 62% | Baseline |
| Sans BatchNorm | 1.5 | 58% | Convergence plus lente |
| Sans Dropout | 1.6 | 59% | Léger overfitting |
| 2 Layers (128) | 1.4 | 56% | Capacité insuffisante |
| 4 Layers (512) | 1.7 | 61% | Pas de gain significatif |

**Conclusion** : 3×256 avec BN et Dropout est optimal.

---

**Prochaine section** : [04_Environment.md](04_Environment.md)
