# 10 - Gestion des Risques

## 📚 Introduction

La **gestion des risques** est cruciale pour la survie à long terme.

---

## 💰 Position Sizing

### Fixed Fractional

```
Position_Size = Balance × Risk_Fraction
```

**V19** : Risk fixe par trade (~0.1%)

### Kelly Criterion (Optionnel)

```
f* = (p × b - q) / b
```

Où :
- **p** : Win rate
- **q** : 1 - p (loss rate)
- **b** : Avg_Win / Avg_Loss

**Exemple** :
- p = 0.62
- Avg_Win / Avg_Loss = 1.5

```
f* = (0.62 × 1.5 - 0.38) / 1.5 = 0.367
```

→ Risquer **36.7%** du capital (trop agressif!)

**Fraction de Kelly** : f*/2 = 18% (plus raisonnable)

---

## 📉 Drawdown Control

### Max Drawdown Limit

```
Max_DD = 20%
```

**Action si DD > 20%** :
- Pause trading
- Reduce position size
- Re-evaluate stratégie

### Drawdown Calculation

```
DD(t) = (Peak_Equity - Current_Equity) / Peak_Equity
```

---

## ⚖️ Asymmetric Rewards

### Reward Shaping

```
R = {
    +PnL           si gain
    -2 × PnL       si perte
}
```

**Effet** : Encourage high win rate + large avg wins.

---

## 🛡️ Stop Loss Placement

### ATR-Based

```
SL_Distance = 2 × ATR(14)
```

**Adaptatif** : Se resserre en faible volatilité, s'élargit en haute volatilité.

---

## 📊 Risk Metrics

### Sharpe Ratio

```
Sharpe = μ / σ × √T
```

**Target** : > 1.5

### Max Consecutive Losses

**Monitoring** : Alerte si > 5 pertes consécutives.

---

**Prochaine section** : [11_Code_Structure.md](11_Code_Structure.md)
