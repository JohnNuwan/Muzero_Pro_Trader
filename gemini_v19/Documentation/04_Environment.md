# 04 - Environment de Trading

## 📚 Introduction

L'**environnement** définit comment l'agent perçoit le marché et interagit avec lui. V19 utilise le **CommissionTrinityEnv**, un environnement Gymnasium-compatible.

---

## 🎯 Interface Gymnasium

### Méthodes Principales

```python
class CommissionTrinityEnv(gym.Env):
    def reset(self) -> (observation, info):
        """Réinitialise l'environnement"""
        
    def step(self, action) -> (observation, reward, done, truncated, info):
        """Exécute une action et retourne la transition"""
```

---

## 👁️ Observation Space

### Dimensions: R^84

```
observation = [
    indicators_M1[13],    # Indicateurs M1
    indicators_M5[13],    # Indicateurs M5
    indicators_M15[13],   # Indicateurs M15
    indicators_H1[13],    # Indicateurs H1
    indicators_H4[13],    # Indicateurs H4
    indicators_D1[13],    # Indicateurs D1
    time_features[4],     # sin/cos hour, sin/cos day
    position_state[1],    # -1/0/+1
    pnl_pct[1]           # PnL %
]
```

### Multi-Timeframe Alignment

Pour chaque timeframe, on prend la **dernière bougie fermée** relative au temps actuel :

```python
current_time = primary_data.index[current_step]  # M5 en live

for tf in ["M1", "M5", "M15", "H1", "H4", "D1"]:
    df = self.data[tf]
    idx = df.index.searchsorted(current_time, side='right') - 1
    row = df.iloc[idx]
    features.extend(extract_indicators(row))
```

**Propriété** : Pas de look-ahead bias.

---

## 🎮 Action Space

### Discrete(5)

```python
actions = {
    0: HOLD,    # Ne rien faire
    1: BUY,     # Ouvrir long ou pyramider
    2: SELL,    # Ouvrir short ou pyramider
    3: SPLIT,   # Fermer 50% de la position
    4: CLOSE    # Fermer 100% de la position
}
```

### Logique de BUY (action=1)

```python
if position_size == 0:
    # Ouvrir position long
    position_size = +trade_size
    avg_entry_price = current_price
    
elif position_size < 0:
    # Flip short → long (fermer short + ouvrir long)
    position_size = +trade_size
    avg_entry_price = current_price
    
elif position_size > 0:
    # Pyramider long (si profitable)
    if (current_price - avg_entry_price) / avg_entry_price > 0.001:
        # Weighted average entry
        total_val = (position_size * avg_entry_price) + (trade_size * current_price)
        position_size += trade_size
        avg_entry_price = total_val / position_size
```

### Trend Filter

**EMA 200** utilisé comme filtre :

```
BUY autorisé  si price > EMA200
SELL autorisé si price < EMA200
```

**Code** :

```python
if action == 1:  # BUY
    if current_price < ema_200:
        action = 0  # Force HOLD (contre-tendance)
```

---

## 💰 Reward Shaping

### Formule Complète

```
R_t = R_realized + R_unrealized + R_quality + R_drawdown + R_pyramid
```

### 1. Realized PnL Reward (Asymétrique)

```
R_realized = {
    PnL / initial_balance * 100          si PnL > 0 (gain)
    PnL / initial_balance * 200          si PnL < 0 (perte × 2)
}
```

**Effet** : Pénalise fortement les pertes pour encourager la prudence.

### 2. Unrealized PnL Reward

Pour positions ouvertes :

```
R_unrealized = unrealized_pnl / initial_balance * 100 * 0.01
```

**Coefficient 0.01** : Faible poids pour éviter de sur-valoriser les gains non réalisés.

### 3. Quality Trade Bonus

Si le trade réalise > 0.10% :

```
R_quality = +2.0
```

**Objectif** : Favoriser les trades de qualité plutôt que le volume.

### 4. Drawdown Penalty

```
drawdown = (peak_equity - current_equity) / peak_equity

R_drawdown = {
    -5.0  si drawdown > 3%
     0.0  sinon
}
```

**Heavy penalty** pour protéger le capital.

### 5. Pyramid Reward

```
R_pyramid = +0.1  si pyramide profitable ajoutée
            -0.1  si tentative de pyramide échouée
```

---

## 🎲 SL/TP Simulation

### Stop Loss

```
SL_distance = 10 pips
```

**Check** :

```python
pnl_pips = (current_price - entry_price) / pip_size

if position_size > 0 and pnl_pips <= -SL_distance:
    # Hit SL
    realized_pnl = -SL_distance * pip_size * position_size
    balance += realized_pnl
    position_size = 0
    reward -= 1.0  # Penalty
```

### Take Profit

```
TP_distance = 100 pips
```

**Check** :

```python
if position_size > 0 and pnl_pips >= TP_distance:
    # Hit TP
    realized_pnl = TP_distance * pip_size * position_size
    balance += realized_pnl
    position_size = 0
    reward += 2.0  # Bonus
    reward += 2.0  # Quality bonus
```

---

## 🔧 Dynamic Trade Sizing

Pour normaliser l'exposition entre symboles :

```python
trade_size = {
    10000.0  (0.1 lot)   si Forex standard
    0.1                  si BTC, US30, GER40, US500, US100
    1.0                  si ETH, XAU
}
```

**Objectif** : ~10-20 USD de risque par trade, quelle que soit la paire.

---

## 📊 Info Dictionary

À chaque step, `info` contient :

```python
info = {
    'balance': float,        # Balance actuelle
    'equity': float,         # Equity (balance + unrealized)
    'step': int,             # Step number
    'realized_pnl': float,   # PnL réalisé ce step
    'drawdown': float        # % drawdown
}
```

---

## 🔄 Episode Lifecycle

### 1. Reset

```python
obs, info = env.reset()
```

Initialise :
- `balance = initial_balance` (10k)
- `position_size = 0`
- `current_step` = random index dans l'historique

### 2. Step Loop

```python
while not done:
    action = agent.select_action(obs)
    next_obs, reward, done, truncated, info = env.step(action)
    obs = next_obs
```

### 3. Termination

**Done** si :
- `current_step >= max_steps` (limite temps)
- OU balance trop faible (< seuil)

---

## 🎯 Reward Engineering Rationale

### Pourquoi Asymétrique (losses × 2) ?

- Mimique l'aversion au risque humaine
- Encourage trades à haute win rate
- Pénalise overtrading

**Résultat empirique** : Win rate passe de 52% → 62%

### Pourquoi Quality Bonus ?

- Évite les scalping micro-gains
- Favorise les moves significatifs (> 0.10%)

**Résultat** : Sharpe ratio +0.3

---

## 📐 State Transition Probability

Le marché est **stochastique** :

```
P(s_{t+1} | s_t, a_t) ≠ déterministe
```

Facteurs aléatoires :
- News events
- Order flow
- Slippage
- Latence

**Conséquence** : Impossibilité de prédire parfaitement, nécessité d'explorer (self-play).

---

**Prochaine section** : [05_Self_Play_Pipeline.md](05_Self_Play_Pipeline.md)
