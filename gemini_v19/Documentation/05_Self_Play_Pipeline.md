# 05 - Pipeline Self-Play

## 📚 Introduction

Le **Self-Play** génère des parties synthétiques en faisant jouer le modèle contre lui-même dans un environnement simulé.

---

## 🎮 SimulatedMarket

### Caractéristiques

- Charge données historiques M15 (2 ans)
- Simule l'exécution de trades
- Calcule rewards réalistes
- Compatible avec MCTS

### Initialisation

```python
env = Sim

ulatedMarket(symbol="EURUSD", timeframe="M15")
state = env.reset()
```

**Reset** :
- Pick random start index (évite overfitting à un range)
- Reset balance, position

---

## 🔄 Génération de Parties

### Algorithme

```python
def generate_game(network, env):
    state = env.reset()
    trajectory = []
    
    for step in range(max_steps):
        # MCTS search
        mcts = AlphaZeroMCTS(network, env, n_sims=50)
        policy, _ = mcts.search(state, temperature=1.0)
        
        # Sample action
        action = np.random.choice(5, p=policy)
        
        # Execute
        next_state, reward, done, _, _ = env.step(action)
        
        # Store
        trajectory.append((state, policy, reward))
        
        if done:
            break
            
        state = next_state
    
    return trajectory
```

### Monte Carlo Returns

```python
def compute_returns(trajectory, gamma=0.99):
    G = 0
    returns = []
    
    for (s, π, r) in reversed(trajectory):
        G = r + gamma * G
        returns.insert(0, tanh(G / 100.0))  # Normalize
    
    return [(s, π, z) for (s, π, _), z in zip(trajectory, returns)]
```

**Tanh normalization** : Limite les valeurs dans [-1, 1].

---

## 📊 Configuration V19

```python
SELF_PLAY_CONFIG = {
    'n_games': 500,              # 500 parties/nuit
    'max_steps': 100,            # 100 steps max/partie
    'mcts_simulations': 50,
    'temperature': 1.0,          # Exploration
    'symbols': [11 symboles],
    'timeframe': 'M15'
}
```

**Output** : ~500 × 100 = 50,000 positions étiquetées.

---

## 🎯 Équation Clé

```
G_t = Σ(k=0 to T-t) γ^k * r_{t+k}
```

Avec **γ = 0.99** (discount factor).

---

**Prochaine section** : [06_Hybrid_Training.md](06_Hybrid_Training.md)
