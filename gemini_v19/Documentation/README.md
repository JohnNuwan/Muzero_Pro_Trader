# 📚 Gemini V19 - Documentation Technique

Bienvenue dans la documentation technique complète de Gemini V19 AlphaZero Trading System.

---

## 📖 Table des Matières

### 1️⃣ Fondements Théoriques

- **[01_AlphaZero_Theory.md](01_AlphaZero_Theory.md)** - Théorie mathématique d'AlphaZero
  - Équations fondamentales
  - Fonction de valeur et politique
  - Algorithme d'apprentissage par renforcement

- **[02_MCTS_Algorithm.md](02_MCTS_Algorithm.md)** - Monte Carlo Tree Search
  - Algorithme PUCT
  - Sélection, expansion, simulation, backpropagation
  - Formules mathématiques détaillées

### 2️⃣ Architecture Système

- **[03_Network_Architecture.md](03_Network_Architecture.md)** - Architecture du réseau neuronal
  - Shared Trunk
  - Policy Head
  - Value Head
  - Dimensions et activation functions

- **[04_Environment.md](04_Environment.md)** - Environment de trading
  - Commission Trinity Env
  - Observation space (84 features)
  - Action space (5 actions)
  - Reward shaping

### 3️⃣ Pipeline d'Entraînement

- **[05_Self_Play_Pipeline.md](05_Self_Play_Pipeline.md)** - Self-Play et génération de données
  - Simulated Market
  - Génération de trajectoires
  - Monte Carlo returns

- **[06_Hybrid_Training.md](06_Hybrid_Training.md)** - Entraînement hybride
  - Mixing ratio (60/40)
  - Loss function
  - Optimisation

- **[07_Tournament.md](07_Tournament.md)** - Validation par tournoi
  - Head-to-head evaluation
  - Métriques (Sharpe, Win Rate)
  - Decision criteria

### 4️⃣ Indicateurs Techniques

- **[08_Indicators.md](08_Indicators.md)** - Les 78 indicateurs utilisés
  - Trend (SMA, EMA, MACD, ADX, etc.)
  - Momentum (RSI, Stochastic, etc.)
  - Volatility (ATR, Bollinger, etc.)
  - Volume (OBV, VWAP, etc.)
  - Formules mathématiques de chaque indicateur

### 5️⃣ Stratégies Avancées

- **[09_Pyramiding.md](09_Pyramiding.md)** - Stratégie de pyramiding
  - Conditions d'entrée
  - Gestion du Stop Loss
  - Risk management

- **[10_Risk_Management.md](10_Risk_Management.md)** - Gestion des risques
  - Position sizing
  - Drawdown control
  - Asymmetric rewards

### 6️⃣ Implémentation

- **[11_Code_Structure.md](11_Code_Structure.md)** - Structure du code
  - Organisation des modules
  - Flow de données
  - Design patterns

- **[12_Configuration.md](12_Configuration.md)** - Configuration détaillée
  - Tous les paramètres expliqués
  - Tuning guidelines
  - Best practices

---

## 🎯 Comment Utiliser Cette Documentation

### Pour les Développeurs
1. Commencez par `01_AlphaZero_Theory.md` pour comprendre les fondements
2. Lisez `02_MCTS_Algorithm.md` pour maîtriser le moteur de décision
3. Consultez `03_Network_Architecture.md` pour l'architecture
4. Parcourez les autres documents selon vos besoins

### Pour les Chercheurs
- Focus sur les documents 01-07 pour la théorie et les algorithmes
- `08_Indicators.md` pour les features engineering
- `06_Hybrid_Training.md` pour les innovations

### Pour les Traders
- `04_Environment.md` pour comprendre comment le système "voit" le marché
- `09_Pyramiding.md` et `10_Risk_Management.md` pour les stratégies
- `12_Configuration.md` pour ajuster les paramètres

---

## 📐 Notation Mathématique

Les documents utilisent la notation suivante :

- **s** : État (state)
- **a** : Action
- **π(a|s)** : Politique (probabilité de l'action a dans l'état s)
- **V(s)** : Fonction de valeur
- **Q(s,a)** : Fonction action-valeur
- **r** : Récompense (reward)
- **γ** : Facteur de discount
- **θ** : Paramètres du réseau neuronal

---

## 🔬 Équations Clés

### AlphaZero Loss
```
L(θ) = (z - v)² - π^T log p + c||θ||²
```

### PUCT Score
```
U(s,a) = Q(s,a) + c_puct * P(s,a) * √(Σ N(s,b)) / (1 + N(s,a))
```

### Monte Carlo Return
```
G_t = Σ(k=0 to T-t) γ^k * r_(t+k)
```

---

**Version** : V19 Enhanced  
**Dernière mise à jour** : 26 Novembre 2025  
**Contributeurs** : Gemini AI Trading Team
