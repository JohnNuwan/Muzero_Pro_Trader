# 08 - Indicateurs Techniques

## 📚 Introduction

V19 utilise **13 indicateurs** par timeframe, soit **78 features** multi-timeframe.

---

## 📊 Les 13 Indicateurs

### 1. RSI (Relative Strength Index)

**Formule** :

```
RS = Avg_Gain(14) / Avg_Loss(14)
RSI = 100 - (100 / (1 + RS))
```

**Interprétation** :
- RSI > 70 : Surachat
- RSI < 30 : Survente

---

### 2. MFI (Money Flow Index)

**Formule** :

```
Typical_Price = (High + Low + Close) / 3
Raw_Money_Flow = Typical_Price × Volume

Positive_Flow = Σ(Raw_MF si price ↑)
Negative_Flow = Σ(Raw_MF si price ↓)

MFI = 100 - (100 / (1 + Positive_Flow / Negative_Flow))
```

---

### 3. ADX (Average Directional Index)

**Formule** :

```
+DM = max(High - prev_High, 0)
-DM = max(prev_Low - Low, 0)

+DI = SMA(+DM, 14) / ATR(14) × 100
-DI = SMA(-DM, 14) / ATR(14) × 100

DX = |+DI - -DI| / (+DI + -DI) × 100
ADX = SMA(DX, 14)
```

**Interprétation** :
- ADX > 25 : Tendance forte
- ADX < 20 : Range/consolidation

---

### 4. Z-Score

**Formule** :

```
Z = (Price - μ(20)) / σ(20)
```

**Interprétation** :
- Z > 2 : Extrême haut (potentiel retour à la moyenne)
- Z < -2 : Extrême bas

---

### 5. Trend Score (Propriétaire)

**Formule** :

```
Trend = sign(SMA(20) - SMA(50)) + sign(EMA(12) - EMA(26)) + sign(Close - SMA(200))
```

Chaque terme ∈ {-1, 0, +1}  
**Range** : [-3, +3]

---

### 6. Linear Regression Angle

**Formule** :

```
Slope = Cov(time, price) / Var(time)
Angle = arctan(Slope) × (180 / π)
```

**Interprétation** :
- Angle > 45° : Tendance haussière forte
- Angle < -45° : Tendance baissière forte

---

### 7. Fibonacci Position

**Formule** :

```
Range = High(100) - Low(100)
Fibo_Levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

Fibo_Pos = (Close - Low) / Range
```

---

### 8. Distance to Resistance

**Formule** :

```
Resistance = max(High(lookback))
Dist_Res = (Resistance - Close) / Close
```

---

### 9. Distance to Support

**Formule** :

```
Support = min(Low(lookback))
Dist_Sup = (Close - Support) / Close
```

---

### 10. Skewness (Asymétrie)

**Formule** :

```
Skew = E[(X - μ)³] / σ³
```

**Interprétation** :
- Skew > 0 : Queue à droite (outliers hauts)
- Skew < 0 : Queue à gauche (outliers bas)

---

### 11. Kurtosis (Aplatissement)

**Formule** :

```
Kurt = E[(X - μ)⁴] / σ⁴ - 3
```

**Interprétation** :
- Kurt > 0 : Fat tails (plus de valeurs extrêmes)
- Kurt < 0 : Thin tails

---

### 12. Shannon Entropy

**Formule** :

```
Returns = price_change / price
Bins = histogram(Returns, bins=10)
P(i) = Bins[i] / Σ Bins

Entropy = -Σ P(i) * log₂(P(i))
```

**Interprétation** :
- Entropy élevée : Marché incertain/volatile
- Entropy faible : Marché directionnel

---

### 13. Hurst Exponent

**Formule (R/S Analysis)** :

```
R/S(τ) = Range(cumsum(returns - mean)) / StdDev(returns)

Hurst = slope(log(R/S) vs log(τ))
```

**Interprétation** :
- H > 0.5 : Trending (momentum)
- H = 0.5 : Random walk
- H < 0.5 : Mean-reverting

---

## 🔢 Normalisation

Certains indicateurs sont normalisés :

```python
normalized = (value - mean) / std
# ou
normalized = 2 * (value - min) / (max - min) - 1  # [-1, 1]
```

---

**Prochaine section** : [09_Pyramiding.md](09_Pyramiding.md)
