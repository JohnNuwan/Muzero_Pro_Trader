# 06 - Entraînement Hybride

## 📚 Introduction

V19 utilise un **entraînement hybride** mixant données synthétiques (self-play) et réelles.

---

## 🔀 Mixing Ratio

```
Batch = 60% Self-Play + 40% Real Trades
```

### Justification

- **60% Self-Play** : Exploration, diversité de stratégies
- **40% Real Trades** : Grounding, réalité du marché

---

## 🎯 Construction des Batchs

```python
batch_size = 64
sp_batch_size = int(64 * 0.6) = 38
real_batch_size = int(64 * 0.4) = 26

# Sample
sp_indices = np.random.choice(len(self_play_data), 38)
real_indices = np.random.choice(len(real_data), 26)

# Concatenate
batch_states = np.concatenate([
    self_play_states[sp_indices],
    real_states[real_indices]
])
```

---

## 📊 Training Loop

```python
for epoch in range(300):
    for batch in iterate_batches():
        # Forward
        policy_pred, value_pred = model(batch_states)
        
        # Loss
        loss = policy_loss(policy_pred, target_policy) + \
               value_loss(value_pred, target_value)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 🎯 Loss Functions

### Policy Loss

```
L_π = -Σ_a π_target(a) * log(π_pred(a) + ε)
```

### Value Loss

```
L_V = (V_target - V_pred)²
```

### Total

```
L = L_π + L_V + λ * ||θ||²
```

---

**Prochaine section** : [07_Tournament.md](07_Tournament.md)
