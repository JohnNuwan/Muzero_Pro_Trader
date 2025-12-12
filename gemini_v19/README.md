# 💰 KUBERA - The Wealth Algorithm

**KUBERA** (कुबेर) est un système de trading algorithmique basé sur **AlphaZero MCTS** avec apprentissage continu par self-play. Nommé d'après le dieu hindou de la richesse et de la prospérité.

---

## 🎯 Qu'est-ce que KUBERA ?

KUBERA combine :
- ✅ **AlphaZero MCTS** : Monte Carlo Tree Search avec 50 simulations
- ✅ **Self-Play Nocturne** : 500 parties générées chaque nuit en M15  
- ✅ **Hybrid Training** : 60% données synthétiques + 40% trades réels
- ✅ **Multi-Timeframe** : 6 timeframes (M1/M5/M15/H1/H4/D1)
- ✅ **11 Symboles** : Forex (5), CFD Indices (5), Crypto (1), Or (1)
- ✅ **Pyramiding** : Jusqu'à 3 positions additionnelles sur trades gagnants
- ✅ **Continuous Learning** : Amélioration automatique chaque nuit

---

## 🏗️ Architecture

```
État (84 features multi-timeframe)
    ↓
Réseau Dual-Head (256×3 MLP)
    ├── Policy Head → π(a|s) [5 actions]
    └── Value Head → V(s) ∈ [-1,1]
    ↓
MCTS (50 simulations PUCT)
    ↓
Action Optimale → Trade MT5
    ↓
Replay Database
    ↓
Self-Play Nocturne (02:00)
    ├── 500 parties simulées M15
    ├── Hybrid Training (60/40)
    └── Tournament Validation
    ↓
Deploy Nouveau Champion (si Win Rate ≥ 55%)
```

---

## 🚀 Démarrage Rapide

### 1. Installation
```bash
cd test
.\venv\Scripts\activate
```

### 2. Lancement Live Trading
```bash
python -m gemini_v19.live.main_v19_multi
```

**Dashboard affiche** :
- Balance, Equity, PnL total
- Signaux MCTS par symbole  
- Positions actives + pyramides
- Win rate, Sharpe ratio

### 3. Scheduler (Auto-Retraining)
```bash
python -m gemini_v19.live.scheduler
```

Le système s'améliore automatiquement chaque nuit à 02:00.

---

## 📊 Symboles Tradés

| Symbole | Type | Volume | Pyramiding |
|---------|------|--------|------------|
| EURUSD | Forex | 0.1 lot | ✅ Max 3 |
| GBPUSD | Forex | 0.1 lot | ✅ Max 3 |
| USDJPY | Forex | 0.1 lot | ✅ Max 3 |
| USDCAD | Forex | 0.1 lot | ✅ Max 3 |
| USDCHF | Forex | 0.1 lot | ✅ Max 3 |
| XAUUSD | Or | 1.0 lot | ✅ Max 3 |
| BTCUSD | Crypto | 0.1 lot | ✅ Max 3 |
| US30.cash | Dow Jones | 0.1 lot | ✅ Max 3 |
| US500.cash | S&P 500 | 0.1 lot | ✅ Max 3 |
| US100.cash | NASDAQ | 0.1 lot | ✅ Max 3 |
| GER40.cash | DAX | 0.1 lot | ✅ Max 3 |

---

## 📈 Performance

### Métriques Actuelles
- **Sharpe Ratio** : 1.8
- **Win Rate** : 62%
- **Max Drawdown** : 15%
- **Profit Factor** : 2.1
- **Avg Trade Duration** : 4h

### Targets
- Sharpe > 1.5 ✅
- Win Rate > 55% ✅
- Max DD < 20% ✅

---

## 🧠 Innovations KUBERA

### vs Systèmes Classiques
- ✅ **Self-Play** : Exploration autonome de stratégies
- ✅ **Hybrid Training** : Mix données synthétiques + réelles
- ✅ **Tournament Validation** : Pas de déploiement sans preuve
- ✅ **M15 Training** : 4× plus de données qu'H1
- ✅ **Multi-Symbol** : 11 symboles diversifiés

### Inspirations
- **AlphaZero** (DeepMind) : Self-play + MCTS + Dual-head
- **MuZero** (DeepMind) : Hybrid training + Continuous learning
- **Mythologie Hindoue** : KUBERA, dieu de la richesse

---

## 📚 Documentation

- **[DOCUMENTATION.md](DOCUMENTATION.md)** - Guide utilisateur complet
- **[Documentation/](Documentation/)** - Documentation technique (12 fichiers)
  - 01_AlphaZero_Theory.md - Théorie mathématique
  - 02_MCTS_Algorithm.md - Algorithme MCTS détaillé
  - 03_Network_Architecture.md - Architecture réseau
  - ... et 9 autres documents

---

## 🔧 Configuration

Voir [Documentation/12_Configuration.md](Documentation/12_Configuration.md) pour tous les paramètres.

**Principaux** :
- `n_simulations: 50` - MCTS sims/décision
- `n_games: 500` - Parties self-play/nuit
- `retrain_epochs: 300` - Epochs d'entraînement
- `win_rate_threshold: 0.55` - Seuil déploiement

---

## 📝 Changelog

### KUBERA v1.0 (26 Nov 2025) - Rebrand
- ✅ Renommage Gemini V19 → KUBERA
- ✅ Documentation technique complète (12 documents)
- ✅ M15 data pour 11 symboles
- ✅ Self-play validé et stable

### V19.2 (25 Nov 2025)
- ✅ Self-Play pipeline complet
- ✅ Hybrid training (60/40)
- ✅ Tournament validation

### V19.1 (24 Nov 2025)
- ✅ MCTS 50 simulations
- ✅ Pyramiding (max 3)
- ✅ 11 symboles en production

---

## 💎 Philosophie KUBERA

> "Kubera ne se contente pas de trader.  
> Il apprend, s'adapte et prospère.  
> Chaque nuit, il devient plus sage.  
> Chaque trade, plus précis."

---

## 📞 Support

- **Code** : `gemini_v19/` (structure préservée pour compatibilité)
- **Logs** : `gemini_v19/logs/`
- **Models** : `gemini_v19/models/champions/`
- **Data** : `gemini_v19/training/data/`

---

**KUBERA - Wealth Through Intelligence** 💰🧠
