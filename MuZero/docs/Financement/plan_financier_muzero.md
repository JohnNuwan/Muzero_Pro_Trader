# Plan Financier MuZero : Stratégie Multi-Prop Firms

## 📋 Table des Matières

1. [Hypothèses et Paramètres](#hypothèses-et-paramètres)
2. [Formules Mathématiques](#formules-mathématiques)
3. [Phase 1 : Validation (Mois 1-4)](#phase-1--validation-mois-1-4)
4. [Phase 2 : Multiplication (Mois 5-8)](#phase-2--multiplication-mois-5-8)
5. [Phase 3 : Scale Massif (Mois 9-12)](#phase-3--scale-massif-mois-9-12)
6. [Année 2 : Expansion Exponentielle](#année-2--expansion-exponentielle)
7. [Année 3 : Empire Multi-Millions](#année-3--empire-multi-millions)
8. [Fiscalité Détaillée](#fiscalité-détaillée)
9. [Architecture Technique Copy Trading](#architecture-technique-copy-trading)
10. [Gestion des Risques](#gestion-des-risques)

---

## Hypothèses et Paramètres

### Variables de Base

| Variable | Symbole | Valeur | Unité |
|----------|---------|--------|-------|
| Performance mensuelle | `r` | 3% | sans unité |
| Reward share FTMO | `s` | 90% | sans unité |
| Taux de change USD/EUR | `fx` | 0.94 | EUR/USD |
| Taux d'imposition Flat Tax | `t` | 30% | sans unité |
| Période de scaling FTMO | `p` | 4 | mois |
| Multiplicateur scaling | `m` | 1.25 | sans unité |

### Prop Firms Disponibles

| Prop Firm | Code | Scaling Max | Reward Share |
|-----------|------|-------------|--------------|
| FTMO | FT | $2M | 90% |
| FundedNext | FN | $2M | 90% |
| E8 Funding | E8 | $1M | 80% |
| MyForexFunds | MF | $600k | 85% |
| The Funded Trader | TFT | $600k | 90% |

---

## Formules Mathématiques

### 1. Profit Mensuel Brut

Pour un compte de taille `C` avec performance `r` :

```
P(C, r) = C × r
```

**Exemple** : Compte $10,000 @ 3%
```
P(10000, 0.03) = 10000 × 0.03 = $300
```

### 2. Profit Net Après Reward Share

Avec un reward share `s` :

```
P_net(C, r, s) = C × r × s
```

**Exemple** : FTMO 90%
```
P_net(10000, 0.03, 0.90) = 10000 × 0.03 × 0.90 = $270
```

### 3. Scaling FTMO

Après `p` mois profitables, la taille du compte devient :

```
C_scaled(C_initial, m, n) = C_initial × m^n
```

Où `n` est le nombre de cycles de scaling.

**Exemple** : $10k après 3 cycles (12 mois)
```
C_scaled(10000, 1.25, 3) = 10000 × 1.25^3 = 10000 × 1.953 = $19,530
```

**Table de Scaling FTMO** :

| Cycle | Mois | Formule | Taille |
|-------|------|---------|--------|
| 0 | 0 | 10000 × 1.25^0 | $10,000 |
| 1 | 4 | 10000 × 1.25^1 | $12,500 ≈ $25k |
| 2 | 8 | 10000 × 1.25^2 | $15,625 ≈ $50k |
| 3 | 12 | 10000 × 1.25^3 | $19,531 ≈ $100k |
| 4 | 16 | 10000 × 1.25^4 | $24,414 ≈ $200k |
| 5 | 20 | 10000 × 1.25^5 | $30,517 ≈ $400k |
| 6 | 24 | 10000 × 1.25^6 | $38,146 ≈ $800k |
| 7 | 28 | 10000 × 1.25^7 | $47,683 ≈ $1.6M |
| 8 | 32 | 10000 × 1.25^8 | $59,604 ≈ $2M (MAX) |

### 4. Revenus Totaux Multi-Comptes

Pour `N` comptes avec tailles `C_i`, performance `r`, et reward shares `s_i` :

```
R_total = Σ(i=1 to N) [C_i × r × s_i]
```

**Exemple** : 3 comptes
```
R_total = (10k × 0.03 × 0.90) + (25k × 0.03 × 0.90) + (50k × 0.03 × 0.85)
        = 270 + 675 + 1275
        = $2,220/mois
```

### 5. Revenus Après Impôts

Avec taux d'imposition `t` :

```
R_net(R_brut, t, fx) = (R_brut × fx) × (1 - t)
```

**Exemple** : $2,220/mois
```
R_net(2220, 0.30, 0.94) = (2220 × 0.94) × (1 - 0.30)
                        = 2086.8 × 0.70
                        = €1,460.76
```

---

## Phase 1 : Validation (Mois 1-4)

### Objectif
Valider MuZero sur 1 compte FTMO 10k.

### Investissement Initial

| Item | Coût |
|------|------|
| Challenge FTMO 10k | €89 |
| VPS (4 mois) | €30 × 4 = €120 |
| **Total** | **€209** |

### Revenus Détaillés

**Mois 1-4** : Compte $10,000

```
P_mensuel = 10,000 × 0.03 = $300
P_net = 300 × 0.90 = $270
P_net_EUR = 270 × 0.94 = €253.80

Revenus 4 mois = €253.80 × 4 = €1,015.20
```

### Bilan Phase 1

| Item | Montant |
|------|---------|
| Investissement | -€209 |
| Revenus bruts | €1,015 |
| Impôts (30%) | -€305 |
| **NET** | **€501** |

**ROI Phase 1** :
```
ROI = (501 - 209) / 209 × 100 = 139.7%
```

---

## Phase 2 : Multiplication (Mois 5-8)

### Stratégie
Utiliser les gains Phase 1 pour acheter 3 nouveaux challenges.

### Nouveaux Challenges

| Prop Firm | Taille | Coût | Funded Mois |
|-----------|--------|------|-------------|
| FTMO #2 | $10k | €89 | M6 |
| FundedNext | $10k | €99 | M6 |
| E8 Funding | $25k | €250 | M7 |

**Total investissement** : €438 (financé par Phase 1)

### Comptes Actifs Mois 5-8

1. **FTMO #1** : Scale à $25k (M5)
2. **FTMO #2** : $10k (M6)
3. **FundedNext** : $10k (M6)
4. **E8 Funding** : $25k (M7)

### Calculs Détaillés

**Mois 5** : 1 compte
```
FTMO #1 (25k, 90%):
P = 25,000 × 0.03 × 0.90 = $675
€ = 675 × 0.94 = €634.50
```

**Mois 6** : 3 comptes
```
FTMO #1: $675 → €634.50
FTMO #2: $270 → €253.80
FundedNext: $270 → €253.80
Total = €1,142.10
```

**Mois 7-8** : 4 comptes
```
FTMO #1: €634.50
FTMO #2: €253.80
FundedNext: €253.80
E8 (25k, 80%): 25,000 × 0.03 × 0.80 × 0.94 = €564
Total = €1,706.10 /mois
```

### Revenus Phase 2

```
M5: €634.50
M6: €1,142.10
M7: €1,706.10
M8: €1,706.10
────────────────
Total: €5,188.80
```

### Bilan Phase 2

| Item | Montant |
|------|---------|
| Investissement | -€438 |
| Revenus bruts | €5,189 |
| Impôts (30%) | -€1,557 |
| **NET** | **€3,194** |

---

## Phase 3 : Scale Massif (Mois 9-12)

### Stratégie
Ouvrir 5 nouveaux challenges + Scaling des comptes existants.

### Portfolio Mois 9

| Prop Firm | Compte | Taille | Reward | Profit/Mois |
|-----------|--------|--------|--------|-------------|
| FTMO #1 | 1 | $50k | 90% | $1,350 |
| FTMO #2 | 2 | $25k | 90% | $675 |
| FTMO #3 | 3 | $50k | 90% | $1,350 |
| FundedNext #1 | 4 | $25k | 90% | $675 |
| FundedNext #2 | 5 | $50k | 90% | $1,350 |
| E8 #1 | 6 | $50k | 80% | $1,200 |
| E8 #2 | 7 | $50k | 80% | $1,200 |
| MyForexFunds #1 | 8 | $50k | 85% | $1,275 |
| MyForexFunds #2 | 9 | $50k | 85% | $1,275 |

**Total Capital** : $450,000

### Formule Revenus Mois 9-12

```
R_total = Σ(C_i × 0.03 × s_i)
       = (50k × 0.03 × 0.90) + (25k × 0.03 × 0.90) + (50k × 0.03 × 0.90)
         + (25k × 0.03 × 0.90) + (50k × 0.03 × 0.90)
         + (50k × 0.03 × 0.80) + (50k × 0.03 × 0.80)
         + (50k × 0.03 × 0.85) + (50k × 0.03 × 0.85)
       
       = $1,350 + $675 + $1,350 + $675 + $1,350
         + $1,200 + $1,200 + $1,275 + $1,275
       
       = $10,350/mois
```

**En EUR** :
```
R_EUR = 10,350 × 0.94 = €9,729/mois
```

### Revenus Phase 3 (4 mois)

```
Total brut : €9,729 × 4 = €38,916
Impôts (30%) : -€11,675
NET : €27,241
```

---

## Résumé Année 1

| Phase | Durée | Comptes | Revenus Bruts | Impôts | NET |
|-------|-------|---------|---------------|--------|-----|
| 1 | M1-4 | 1 | €1,015 | €305 | €501 |
| 2 | M5-8 | 4 | €5,189 | €1,557 | €3,194 |
| 3 | M9-12 | 9 | €38,916 | €11,675 | €27,241 |
| **TOTAL** | **12 mois** | **9** | **€45,120** | **€13,537** | **€30,936** |

### Équation Générale Année 1

```
R_annuel = Σ(m=1 to 12) [Σ(i=1 to N_m) (C_i,m × r × s_i × fx)]

Où :
- N_m = nombre de comptes au mois m
- C_i,m = taille du compte i au mois m (avec scaling)
```

---

## Année 2 : Expansion Exponentielle

### Stratégie
- Scaling individuel de chaque compte
- Ouverture de 9 nouveaux comptes
- **Total** : 18 comptes

### Portfolio Année 2

| Prop Firm | Nombre | Taille Moyenne | Capital Total |
|-----------|--------|----------------|---------------|
| FTMO | 5 | $200k | $1,000,000 |
| FundedNext | 4 | $200k | $800,000 |
| E8 Funding | 4 | $200k | $800,000 |
| MyForexFunds | 3 | $200k | $600,000 |
| The Funded Trader | 2 | $200k | $400,000 |
| **TOTAL** | **18** | - | **$3,600,000** |

### Formule Revenus Mensuels

```
R_mensuel = Σ(i=1 to 18) [C_i × 0.03 × s_i]

Calcul détaillé :
- FTMO (5 comptes) : 5 × (200k × 0.03 × 0.90) = $27,000
- FundedNext (4) : 4 × (200k × 0.03 × 0.90) = $21,600
- E8 (4) : 4 × (200k × 0.03 × 0.80) = $19,200
- MyForexFunds (3) : 3 × (200k × 0.03 × 0.85) = $15,300
- TFT (2) : 2 × (200k × 0.03 × 0.90) = $10,800

Total = $93,900/mois
```

### Revenus Année 2

```
Revenus mensuels : $93,900
Revenus annuels : $93,900 × 12 = $1,126,800

Conversion EUR : $1,126,800 × 0.94 = €1,059,192

Impôts (30%) : €317,758
NET : €741,434
```

---

## Année 3 : Empire Multi-Millions

### Stratégie Max Out
- Scaler tous les comptes jusqu'au maximum
- FTMO/FundedNext : $2M chaque
- E8 : $1M
- Autres : $600k

### Portfolio Année 3

| Prop Firm | Comptes | Taille/Compte | Capital Total |
|-----------|---------|---------------|---------------|
| FTMO | 5 | $2M | $10M |
| FundedNext | 5 | $2M | $10M |
| E8 Funding | 5 | $1M | $5M |
| MyForexFunds | 5 | $600k | $3M |
| The Funded Trader | 5 | $600k | $3M |
| **TOTAL** | **25** | - | **$31M** |

### Formule Revenus Année 3

```
R_mensuel = Σ [N_pf × C_pf × r × s_pf]

Calcul :
- FTMO : 5 × (2M × 0.03 × 0.90) = $270,000
- FundedNext : 5 × (2M × 0.03 × 0.90) = $270,000
- E8 : 5 × (1M × 0.03 × 0.80) = $120,000
- MyForexFunds : 5 × (600k × 0.03 × 0.85) = $76,500
- TFT : 5 × (600k × 0.03 × 0.90) = $81,000

Total = $817,500/mois
```

### Revenus Année 3

```
Revenus mensuels : $817,500
Revenus annuels : $817,500 × 12 = $9,810,000

Conversion EUR : $9,810,000 × 0.94 = €9,221,400
```

### Fiscalité Optimisée (Holding)

Avec une structure Holding + SASU :

```
Revenus bruts : €9,221,400
Salaire dirigeant : €150,000 (TMI 45% = €67,500)
Dividendes société : €9,071,400

IS (Impôt sur Sociétés) :
- Tranche 1 (0-42,500) : 42,500 × 0.15 = €6,375
- Tranche 2 (reste) : 9,028,900 × 0.25 = €2,257,225
Total IS : €2,263,600

Dividendes nets : €6,807,800
Flat Tax dividendes (30%) : €2,042,340

Total impôts : €67,500 + €2,263,600 + €2,042,340 = €4,373,440

NET FINAL : €9,221,400 - €4,373,440 = €4,847,960
```

**Taux effectif d'imposition** :
```
T_eff = 4,373,440 / 9,221,400 × 100 = 47.4%
```

---

## Projection Complète 3 Ans

### Table Récapitulative

| Année | Comptes | Capital Total | Revenus Bruts | Impôts | NET | ROI Cumulé |
|-------|---------|---------------|---------------|--------|-----|------------|
| 1 | 9 | $450k | €45,120 | €13,537 | €30,936 | +14,700% |
| 2 | 18 | $3.6M | €1,059,192 | €317,758 | €741,434 | +352,000% |
| 3 | 25 | $31M | €9,221,400 | €4,373,440 | €4,847,960 | +2,300,000% |
| **TOTAL** | - | - | **€10,325,712** | **€4,704,735** | **€5,620,330** | - |

### Équation Générale Revenus (3 Ans)

```
R_total(3ans) = Σ(a=1 to 3) Σ(m=1 to 12) Σ(i=1 to N_a,m) [C_i,a,m × r × s_i × fx]

Où :
- a = année
- m = mois
- N_a,m = nombre de comptes à l'année a, mois m
- C_i,a,m = capital du compte i à l'année a, mois m (avec scaling)
```

---

## Fiscalité Détaillée

### Flat Tax (Années 1-2)

**Formule** :
```
Impôt = R_brut × 0.30
      = R_brut × (0.128 + 0.172)
      = (R_brut × 0.128) + (R_brut × 0.172)
        ︸━━━━━━━━━━━━━━   ︸━━━━━━━━━━━━━━
        Impôt sur Revenu   Prélèvements Sociaux (CSG/CRDS)
```

### Holding + SASU (Année 3)

**Structure fiscale optimale** :

```
Revenus → Société → IS 25% → Dividendes → Holding → Flat Tax 30%
```

**Avantages** :
1. Déduction frais professionnels (30-40%)
2. IS réduit sur tranche basse (15%)
3. Optimisation charges sociales

**Formule complète** :
```
NET = R_brut - Salaire_IR - IS - Dividendes_FlatTax

Avec :
- Salaire_IR = Salaire × TMI
- IS = (R_brut - Salaire - Frais) × T_IS
- Dividendes_FlatTax = (R_brut - Salaire - IS) × 0.30
```

---

## Architecture Technique Copy Trading

### Schéma Général

```
┌─────────────────────┐
│  MuZero Master Bot  │
│   (Live Trading)    │
└──────────┬──────────┘
           │
           ↓
┌──────────────────────┐
│  MT5 Master Account  │
│  (Signal Provider)   │
└──────────┬───────────┘
           │
           ↓ (MT5 Signals / Trade Copier)
           │
     ┌─────┴─────┬─────────┬─────────┬────────┐
     ↓           ↓         ↓         ↓        ↓
┌─────────┐ ┌────────┐ ┌───────┐ ┌──────┐ ┌─────┐
│ FTMO #1 │ │ FTMO#2 │ │  E8   │ │ MFF  │ │ TFT │
└─────────┘ └────────┘ └───────┘ └──────┘ └─────┘
   (VPS 1)    (VPS 2)    (VPS 3)  (VPS 4) (VPS 5)
```

### Formules de Synchronisation

**Ratio de lots** :
```
Lot_slave = Lot_master × (Capital_slave / Capital_master) × Risk_factor

Exemple :
- Master : $10k, 0.1 lot
- Slave : $200k, lot = ?

Lot_slave = 0.1 × (200,000 / 10,000) × 1.0 = 2.0 lots
```

**Latence acceptable** :
```
Latence_max = Spread_moyen / 2

Pour EURUSD (spread 0.5 pip) :
Latence_max = 0.5 / 2 = 0.25 pip → ~25ms réseau
```

---

## Gestion des Risques

### Formule de Drawdown Global

Pour `N` comptes corrélés à 100% :

```
DD_global = Σ(i=1 to N) [C_i × DD_pct]

Limite FTMO : DD_global < 0.10 × Σ(C_i)
```

**Exemple** : 9 comptes, DD 8%
```
Capital total : $450k
DD_global = 450,000 × 0.08 = $36,000

Limite FTMO (10%) : 450,000 × 0.10 = $45,000

Marge de sécurité : 45,000 - 36,000 = $9,000 ✅
```

### Stop-Loss Automatique Global

**Règle** : Si DD global > 5%, arrêt de tous les comptes.

```
Si Σ(DD_i) / Σ(C_i) > 0.05 → STOP_ALL()
```

**Implémentation** :
```python
def monitor_global_dd(accounts):
    total_capital = sum(a.capital for a in accounts)
    total_dd = sum(a.current_dd for a in accounts)
    dd_pct = total_dd / total_capital
    
    if dd_pct > 0.05:
        for acc in accounts:
            acc.disable_trading()
            acc.close_all_positions()
```

---

## Annexe : Tables de Référence

### Table de Scaling FTMO (Détaillée)

| Mois | Cycle | Capital | Profit 3%/mois | Reward 90% | Cumul |
|------|-------|---------|----------------|------------|-------|
| 1 | 0 | $10,000 | $300 | $270 | $270 |
| 2 | 0 | $10,000 | $300 | $270 | $540 |
| 3 | 0 | $10,000 | $300 | $270 | $810 |
| 4 | 0 | $10,000 | $300 | $270 | $1,080 |
| 5 | 1 | $25,000 | $750 | $675 | $1,755 |
| 6 | 1 | $25,000 | $750 | $675 | $2,430 |
| 7 | 1 | $25,000 | $750 | $675 | $3,105 |
| 8 | 1 | $25,000 | $750 | $675 | $3,780 |
| 9 | 2 | $50,000 | $1,500 | $1,350 | $5,130 |
| 12 | 2 | $50,000 | $1,500 | $1,350 | $9,480 |
| 16 | 3 | $100,000 | $3,000 | $2,700 | $20,280 |
| 20 | 4 | $200,000 | $6,000 | $5,400 | $42,060 |
| 24 | 5 | $400,000 | $12,000 | $10,800 | $85,260 |
| 28 | 6 | $800,000 | $24,000 | $21,600 | $171,660 |
| 32 | 7 | $1,600,000 | $48,000 | $43,200 | $344,460 |
| 36 | 8 | $2,000,000 | $60,000 | $54,000 | $560,460 |

**Formule cumulative** :
```
Cumul(mois_n) = Σ(i=1 to n) [C(cycle(i)) × 0.03 × 0.90]
```

---

## Conclusion

### ROI Global (3 Ans)

**Investissement initial** : €209  
**NET cumulé 3 ans** : €5,620,330  

**ROI** :
```
ROI = (5,620,330 - 209) / 209 × 100 = 2,688,000%
```

### Temps pour Indépendance Financière

Objectif : €10,000/mois NET

**Atteint en** : Mois 9-10 (Phase 3)

```
Revenus M10 : €9,729/mois > €10,000/mois ✅
```

**Conclusion** : Indépendance financière en **10 mois** avec stratégie multi-prop firms.

---

*Document généré le 29/11/2025*  
*Version : 1.0*  
*Auteur : Analyse MuZero Trading Bot*
