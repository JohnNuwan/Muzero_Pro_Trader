# 09 - Stratégie de Pyramiding

## 📚 Introduction

Le **pyramiding** permet d'ajouter jusqu'à **3 positions** sur un trade gagnant pour maximiser les profits.

---

## 🎯 Conditions d'Entrée

### Pyramide Autorisée si :

```
1. Position principale PROFITABLE (PnL > 0)
2. Signal MCTS = Direction position principale
3. Confidence MCTS ≥ 60%
4. Nombre pyramides actuelles < 3
```

**Code** :

```python
def can_pyramid(main_pos, signal_dir, confidence):
    if main_pos.pnl <= 0:
        return False
    if main_pos.direction != signal_dir:
        return False
    if confidence < 0.60:
        return False
    if len(main_pos.pyramids) >= 3:
        return False
    return True
```

---

## 📏 Volume de Pyramide

```
Pyramid_Volume = Main_Position_Volume × 0.5
```

**Exemple** :
- Position principale : 0.10 lot
- Pyramide 1 : 0.05 lot
- Pyramide 2 : 0.05 lot
- Pyramide 3 : 0.05 lot

**Total max** : 0.25 lot

---

## 🛡️ Gestion du Stop Loss

### SL Initial

```
SL_pyramid = Entry_Price - (SL_Distance × 2)
```

Plus conservateur que position principale.

### Move to Break Even

**Trigger** : Pyramide atteint +0.10% profit

```python
if pyramid_profit_pct >= 0.001:  # 0.10%
    new_SL = pyramid_entry + spread
    modify_SL(pyramid, new_SL)
```

**Effet** : Risk-free pyramid dès +10 pips.

---

## 📊 Équations

### PnL Total

```
Total_PnL = Main_PnL + Σ(Pyramid_i_PnL)
```

### Weighted Entry Price

```
Avg_Entry = (V₁×P₁ + V₂×P₂ + ... + Vₙ×Pₙ) / (V₁ + V₂ + ... + Vₙ)
```

---

## ⚠️ Risques

- ↑ Exposition si marché reverse
- **Mitigation** : SL rapide à BE

---

**Prochaine section** : [10_Risk_Management.md](10_Risk_Management.md)
