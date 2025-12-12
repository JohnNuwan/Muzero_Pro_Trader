# 02 - Algorithme MCTS (Monte Carlo Tree Search)

## 📚 Introduction

Le **Monte Carlo Tree Search** est l'algorithme central de décision dans AlphaZero. Il combine :
- Recherche arborescente guidée par heuristique
- Évaluation Monte Carlo (sampling)
- Réseau neuronal pour l'initialisation et l'évaluation

---

## 🌳 Structure de l'Arbre

### Nœud MCTS

Chaque nœud représente un **état du jeu** (ou état du marché en trading).

**Attributs** :
```python
class MCTSNode:
    state: np.ndarray        # État observé (84 features)
    parent: MCTSNode | None  # Nœud parent
    children: dict[int, MCTSNode]  # Actions → Nœuds enfants
    
    visit_count: int         # N(s)
    value_sum: float         # W(s)
    prior_p: float           # P(s,a) - Proba a priori de la policy
```

**Valeur moyenne Q** :

```
Q(s) = W(s) / N(s)
```

Où :
- **W(s)** : Somme cumulée des valeurs backpropagées
- **N(s)** : Nombre total de visites

---

## 🔄 Les 4 Phases de MCTS

### 1️⃣ Selection

**Objectif** : Descendre dans l'arbre jusqu'à une feuille non-étendue.

**Algorithme** :
```python
def select(node):
    while node.is_expanded():
        action, node = select_child(node, c_puct=1.5)
    return node
```

**Critère PUCT** (Predictor + UCT) :

```
a* = argmax_a [ Q(s,a) + U(s,a) ]
```

Où :

```
U(s,a) = c_puct * P(s,a) * √(Σ_b N(s,b)) / (1 + N(s,a))
```

**Décomposition** :

- **Q(s,a)** : Exploitation - Valeur moyenne observée
  ```
  Q(s,a) = W(s,a) / N(s,a)
  ```

- **U(s,a)** : Exploration - Bonus pour actions peu visitées
  - ↑ si **P(s,a)** élevée (confiance du réseau)
  - ↑ si **N(s,a)** faible (peu exploré)
  - ↓ quand le parent est très visité (exploration normalisée)

**Code V19** :

```python
def select_child(node, c_puct):
    best_score = -np.inf
    best_action = None
    best_child = None
    
    # Total visits parent
    total_n = sum(child.visit_count for child in node.children.values())
    
    for action, child in node.children.items():
        # Q value
        q_value = child.value_sum / child.visit_count if child.visit_count > 0 else 0
        
        # U value (exploration bonus)
        u_value = c_puct * child.prior_p * np.sqrt(total_n) / (1 + child.visit_count)
        
        # PUCT score
        score = q_value + u_value
        
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    
    return best_action, best_child
```

---

### 2️⃣ Expansion

**Objectif** : Créer les nœuds enfants pour une feuille.

**Processus** :

1. Prédire la politique avec le réseau neuronal :
   ```
   π(·|s), V(s) = Network_θ(s)
   ```

2. Pour chaque action légale **a ∈ A** :
   - Simuler `s' = Env.step(s, a)`
   - Créer nœud enfant avec `prior_p = π(a|s)`

**Code V19** :

```python
def expand(node, network, env):
    # Network inference
    with torch.no_grad():
        policy_logits, value = network(torch.FloatTensor([node.state]))
        policy = policy_logits[0].numpy()  # Softmax already applied
        value = value.item()
    
    # Create children for all valid actions
    for action in range(5):  # HOLD, BUY, SELL, SPLIT, CLOSE
        # Clone env to simulate
        child_env = copy.deepcopy(env)
        next_state, _, done, _, _ = child_env.step(action)
        
        if not done:
            node.children[action] = MCTSNode(
                state=next_state,
                parent=node,
                prior_p=policy[action],
                action=action
            )
    
    return value
```

---

### 3️⃣ Simulation / Evaluation

Dans AlphaZero, la simulation est **remplacée par le Value Head du réseau** :

```
V(s_leaf) ≈ E[ Σ γ^t r_t | s_leaf ]
```

**Avantage** :
- ❌ Pas besoin de rollouts aléatoires (lent et bruité)
- ✅ Évaluation apprise par le réseau (rapide et précise)

**Dans V19** :

```python
value = network.value_head(node.state)  # ∈ [-1, 1]
```

Cette valeur est ensuite backpropagée.

---

### 4️⃣ Backpropagation

**Objectif** : Remonter la valeur jusqu'à la racine et mettre à jour les statistiques.

**Algorithme** :

```python
def backpropagate(search_path, value):
    for node in reversed(search_path):
        node.visit_count += 1
        node.value_sum += value
        # Note: Pas de flip de signe car trading = single player
```

**Mise à jour des Q-values** :

Après backprop, les Q-values sont automatiquement recalculées :

```
Q(s,a) = W(s,a) / N(s,a)
```

---

## 🎲 Dirichlet Noise (Exploration Root)

Pour encourager l'exploration au nœud racine, AlphaZero ajoute du **bruit de Dirichlet** :

```
P_root(a) = (1 - ε) * π(a) + ε * η_a
```

Où :
- **ε** : Fraction du bruit (0.25 dans V19)
- **η** ~ **Dir(α)** : Vecteur Dirichlet avec α = 0.3

**Propriétés Dirichlet** :

```
η ~ Dir(α₁, ..., α_K)
    Σ η_i = 1
    η_i ∈ [0, 1]
```

Pour **α < 1** (sparse) : Certains η_i sont proches de 1, d'autres proches de 0  
Pour **α > 1** (dense) : η_i uniformément répartis

**Code V19** :

```python
def add_dirichlet_noise(policy, alpha=0.3, epsilon=0.25):
    noise = np.random.dirichlet([alpha] * len(policy))
    return (1 - epsilon) * policy + epsilon * noise
```

---

## 📊 Policy de Sortie

Après **N simulations**, la politique finale est dérivée des **visit counts** :

```
π(a|s_root) = N(s_root, a)^(1/τ) / Σ_b N(s_root, b)^(1/τ)
```

Où **τ** est la **température** :

- **τ → 0** : Déterministe, π(a*) = 1 où a* = argmax N(s,a)
- **τ = 1** : Proportionnel aux visites, π(a) ∝ N(s,a)
- **τ > 1** : Plus uniforme (plus d'exploration)

**Dans V19** :

- **Training** : τ = 1.0 (stochastique pour exploration)
- **Evaluation** : τ = 0.1 (quasi-déterministe)

**Code** :

```python
def compute_policy(root, temperature=1.0):
    visit_counts = np.array([child.visit_count for child in root.children.values()])
    actions = list(root.children.keys())
    
    if temperature == 0:
        # Deterministic
        best_idx = np.argmax(visit_counts)
        policy = np.zeros(len(actions))
        policy[best_idx] = 1.0
    else:
        # Stochastic
        visit_counts = visit_counts ** (1.0 / temperature)
        policy = visit_counts / np.sum(visit_counts)
    
    # Map to full action space
    full_policy = np.zeros(5)
    for i, action in enumerate(actions):
        full_policy[action] = policy[i]
    
    return full_policy
```

---

## ⚙️ Paramètres MCTS V19

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `n_simulations` | 50 | Nombre de simulations par search |
| `c_puct` | 1.5 | Constante d'exploration PUCT |
| `dirichlet_alpha` | 0.3 | Paramètre du bruit Dirichlet |
| `exploration_fraction` | 0.25 | Fraction du bruit au root (ε) |
| `temperature` | 1.0 / 0.1 | Train / Eval |

---

## 🔬 Complexité et Performance

### Complexité Temporelle

Pour **N** simulations et **b** branches par nœud (≈5 dans V19) :

- **Selection** : O(depth) ≈ O(log N)
- **Expansion** : O(b) = O(5) = O(1)
- **Evaluation** : O(1) (forward pass réseau)
- **Backpropagation** : O(depth) ≈ O(log N)

**Total par simulation** : O(log N)

**Total pour N simulations** : O(N log N)

### Complexité Spatiale

Arbre MCTS : O(N × b) ≈ O(250) nœuds pour N=50, b=5

### Temps Réel V19

- **50 MCTS sims** : ~55 secondes
- **1 forward pass** : ~10 ms (CPU)
- **Bottleneck** : Environment stepping (copy.deepcopy)

---

## 🎯 Comparaison MCTS vs Minimax

| Aspect | MCTS | Minimax |
|--------|------|---------|
| **Exploration** | Selective (PUCT) | Exhaustive |
| **Heuristique** | Réseau neuronal | Fonction d'évaluation manuelle |
| **Profondeur** | Adaptive | Fixe ou iterative deepening |
| **Complexité** | O(N log N) | O(b^d) exponential |
| **Trading** | ✅ Excellent | ❌ Trop lent |

---

## 📈 Convergence de MCTS

### Théorème (Kocsis & Szepesvári, 2006)

Avec UCB1 (ancêtre de PUCT), MCTS converge vers la meilleure action avec probabilité 1 quand **N → ∞**.

### En Pratique

- **N = 50** dans V19 est suffisant pour des décisions fiables
- Trade-off exploration-exploitation bien calibré avec c_puct = 1.5
- Dirichlet noise évite les modes locaux

---

## 🚀 Optimisations V19

### 1. Lazy Expansion

Créer les enfants seulement quand visités :

```python
if action not in node.children:
    node.children[action] = create_child(node, action)
```

**Gain** : Réduit mémoire et temps si certaines branches jamais explorées.

### 2. Virtual Loss (Multi-threading)

Pour paralléliser MCTS, ajouter une perte virtuelle :

```python
node.virtual_loss += 1  # Avant simulation
# ... simulation ...
node.virtual_loss -= 1  # Après backprop
```

**Effet** : Décourage les threads de suivre le même chemin simultanément.

*Note : V19 n'utilise pas cette optimisation (single-threaded).*

---

## 🔗 Lien avec le Réseau

Le réseau neuronal **guide** MCTS via :

1. **Policy Head** → Priors P(s,a)
   - Initialise les probabilités avant exploration
   - Réduit l'espace de recherche

2. **Value Head** → Evaluation V(s)
   - Remplace les rollouts aléatoires
   - Apprentissage par self-play améliore progressivement

**Boucle Vertueuse** :

```
Meilleur Réseau → Meilleure Guidance MCTS → Meilleurs Données → Meilleur Réseau
```

---

**Prochaine section** : [03_Network_Architecture.md](03_Network_Architecture.md)
