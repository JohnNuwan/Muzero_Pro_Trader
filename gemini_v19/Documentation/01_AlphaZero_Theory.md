# 01 - Théorie AlphaZero

## 📚 Introduction

AlphaZero est un algorithme d'apprentissage par renforcement révolutionnaire développé par DeepMind en 2017. Il combine trois concepts puissants :

1. **Monte Carlo Tree Search (MCTS)** - Exploration efficace de l'espace des décisions
2. **Deep Neural Networks** - Approximation de la politique et de la valeur
3. **Self-Play** - Apprentissage autonome sans données étiquetées

---

## 🧮 Fondements Mathématiques

### 1. Processus de Décision Markovien (MDP)

Le trading est modélisé comme un **MDP** défini par le tuple **(S, A, P, R, γ)** :

- **S** : Espace des états (observations du marché)
- **A** : Espace des actions {HOLD, BUY, SELL, SPLIT, CLOSE}
- **P** : Fonction de transition P(s'|s,a) (stochastique)
- **R** :  Fonction de récompense R(s,a,s')
- **γ** : Facteur de discount (γ ∈ [0,1])

**Équation de Bellman** :

```
V^π(s) = E_π [ Σ(t=0 to ∞) γ^t * r_t | s_0 = s ]
```

Où :
- **V^π(s)** est la valeur de l'état s sous la politique π
- **r_t** est la récompense au temps t

### 2. Fonction de Politique

La politique **π(a|s)** est une distribution de probabilité sur les actions :

```
π(a|s) = P(A_t = a | S_t = s)
```

**Objectif** : Trouver la politique optimale **π*** qui maximise le retour espéré :

```
π* = argmax_π E[Σ γ^t * r_t]
```

### 3. Fonction Action-Valeur (Q-Function)

La Q-function **Q^π(s,a)** estime le retour espéré en prenant l'action a dans l'état s, puis en suivant π :

```
Q^π(s,a) = E_π [ r + γ * V^π(s') ]
```

**Équation de Bellman pour Q** :

```
Q*(s,a) = E [ r + γ * max_a' Q*(s', a') ]
```

---

## 🎮 Paradigme Self-Play

### Concept

Au lieu d'apprendre à partir de données étiquetées, AlphaZero **joue contre lui-même** :

1. Le modèle actuel génère des parties
2. Ces parties servent de données d'entraînement
3. Un nouveau modèle est entraîné sur ces données
4. Le nouveau modèle remplace l'ancien s'il est meilleur

### Avantages

- ✅ **Pas de biais humain** : Le système découvre des stratégies novatrices
- ✅ **Données infinies** : Génération continue de nouvelles parties
- ✅ **Amélioration garantie** : Validation par tournoi

---

## 🧠 Architecture de Réseau

AlphaZero utilise un **réseau dual-head** :

### Input Layer
```
État s ∈ R^84
```

### Shared Trunk
```
h = ReLU(BatchNorm(Linear(s, 256)))
h = ReLU(BatchNorm(Linear(h, 256)))
h = ReLU(BatchNorm(Linear(h, 256)))
```

### Policy Head (π)
```
p_logits = Linear(h, 128)
p_logits = ReLU(p_logits)
p_logits = Linear(p_logits, 5)
π(a|s) = Softmax(p_logits)
```

**Output** : Distribution de probabilité sur les 5 actions

### Value Head (V)
```
v = Linear(h, 128)
v = ReLU(v)
v = Linear(v, 1)
V(s) = Tanh(v)
```

**Output** : Valeur estimée de l'état ∈ [-1, 1]

---

## 📊 Fonction de Perte (Loss Function)

AlphaZero minimise une loss composite :

```
L(θ) = (z - v_θ(s))² - π^T log p_θ(s) + c * ||θ||²
```

Où :
- **z** : Valeur cible (Monte Carlo return ou reward réel)
- **v_θ(s)** : Prédiction de valeur du réseau
- **π** : Politique cible (vecteur de probabilités)
- **p_θ(s)** : Prédiction de politique du réseau
- **c** : Coefficient de régularisation L2
- **θ** : Paramètres du réseau

### Composantes

#### 1. Value Loss (MSE)
```
L_value = (z - v_θ(s))²
```

Minimise l'erreur quadratique entre la valeur prédite et la cible.

#### 2. Policy Loss (Cross-Entropy)
```
L_policy = -Σ_a π(a) * log p_θ(a|s)
```

Maximise la log-likelihood de la politique cible.

#### 3. Regularization Loss
```
L_reg = c * Σ θ_i²
```

Prévient l'overfitting en pénalisant les grands poids.

---

## 🔄 Algorithme de Self-Play

### Pseudocode

```python
def self_play():
    state = env.reset()
    trajectory = []
    
    while not done:
        # MCTS Search
        policy, value = mcts_search(state, network, n_sims=50)
        
        # Sample action
        action = sample(policy)
        
        # Execute
        next_state, reward, done = env.step(action)
        
        # Store
        trajectory.append((state, policy, reward))
        state = next_state
    
    # Backpropagate returns
    returns = compute_monte_carlo_returns(trajectory)
    
    return [(s, π, G) for (s, π, r), G in zip(trajectory, returns)]
```

### Compute Monte Carlo Returns

```python
def compute_monte_carlo_returns(trajectory, gamma=0.99):
    G = 0
    returns = []
    
    for (s, π, r) in reversed(trajectory):
        G = r + gamma * G
        returns.insert(0, G)
    
    return returns
```

---

## 🎯 Équation PUCT (Predictor + UCT)

Pendant la recherche MCTS, chaque nœud est évalué avec :

```
Score(s,a) = Q(s,a) + U(s,a)
```

Où :

```
U(s,a) = c_puct * P(s,a) * √(Σ_b N(s,b)) / (1 + N(s,a))
```

**Termes** :
- **Q(s,a)** : Valeur moyenne de l'action a (exploitation)
- **P(s,a)** : Probabilité a priori de la politique réseau
- **N(s,a)** : Nombre de visites de (s,a)
- **c_puct** : Constante d'exploration (1.5 dans V19)

**Propriétés** :
- Actions peu visitées ont un U(s,a) élevé → Exploration
- Actions avec haute probabilité P(s,a) sont favorisées
- Balance exploration-exploitation de façon optimale

---

## 📈 Convergence et Garanties

### Théorème (Silver et al., 2017)

Sous certaines conditions (ergodicity, sufficient exploration), la suite de politiques générée par AlphaZero converge vers un **Nash Equilibrium** du jeu.

### En pratique pour Trading

Le marché n'est pas un jeu à somme nulle et non-stationnaire, donc :
- ❌ Pas de garantie de convergence stricte
- ✅ Amélioration continue observée empiriquement
- ✅ Validation par tournoi assure non-régression

---

## 🔬 Innovations de V19

### 1. Hybrid Training (MuZero-inspired)

Au lieu de seulement du self-play, V19 mixe :
- **60% Self-Play** : Exploration de nouvelles stratégies
- **40% Real Trades** : Grounding dans la réalité du marché

**Formule** :

```
Batch = Sample(60%, Self_Play_Buffer) ∪ Sample(40%, Real_Trades_Buffer)
```

### 2. Multi-Timeframe State

State vector intègre 6 timeframes (M1, M5, M15, H1, H4, D1) :

```
s = [f_M1, f_M5, f_M15, f_H1, f_H4, f_D1, t_features, pos_features]
```

Où **f_tf** est un vecteur de 13 indicateurs techniques par timeframe.

### 3. Asymmetric Reward

Les pertes pèsent 2× plus que les gains :

```
R(s,a,s') = {
    PnL           si PnL > 0
    2 * PnL       si PnL < 0
}
```

**Justification** : Mimique l'aversion au risque humaine et incentivise les trades de qualité.

---

## 📚 Références

1. **Silver, D., et al. (2017)** - "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
2. **Schrittwieser, J., et al. (2020)** - "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (MuZero)
3. **Sutton & Barto (2018)** - "Reinforcement Learning: An Introduction"

---

**Prochaine section** : [02_MCTS_Algorithm.md](02_MCTS_Algorithm.md)
