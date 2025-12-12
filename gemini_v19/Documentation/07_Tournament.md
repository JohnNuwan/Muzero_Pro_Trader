# 07 - Validation par Tournoi

## 📚 Introduction

Le **tournoi** valide qu'un nouveau modèle surpasse le champion actuel.

---

## ⚔️ Head-to-Head

```
Candidate vs Champion : 50 parties
```

**Fairness** : Mêmes seeds pour les deux modèles.

---

## 📊 Métriques

### 1. Win Rate

```
WR = Wins_candidate / Total_games
```

### 2. Sharpe Ratio

**Formule** :

```
Sharpe = (μ_returns / σ_returns) * √(252 * 24)
```

Où :
- **μ_returns** : Moyenne des returns horaires
- **σ_returns** : Écart-type des returns
- **252 * 24** : Annualisation (252 jours × 24 heures)

---

## 🎯 Critères de Déploiement

```
Deploy si:
  - Win Rate ≥ 55%
  ET
  - Sharpe_new ≥ Sharpe_old × 1.05
```

**Si NON** : Champion retient son titre.

---

## 🔬 Calcul du Sharpe

```python
returns = []
for step in episode:
    ret = reward / prev_equity
    returns.append(ret)

sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)
```

---

**Prochaine section** : [08_Indicators.md](08_Indicators.md)
