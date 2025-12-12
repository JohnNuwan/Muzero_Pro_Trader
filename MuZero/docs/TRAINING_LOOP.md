# Cycle d'Entraînement MuZero

L'entraînement d'un agent de type MuZero est plus complexe que celui d'AlphaZero car nous devons entraîner non seulement la stratégie (politique/valeur), mais aussi le modèle du monde (dynamique). L'entraînement se fait de manière cyclique.

Le processus global est le suivant : **Jouer -> Stocker -> Entraîner**.

---

## 1. Phase de Jeu (Collecte de Données)

L'agent interagit avec l'environnement (en backtest ou en direct) pour générer des données d'expérience.

1.  À chaque pas de temps, l'agent observe l'état réel du marché (`s_t`).
2.  Il utilise son MCTS (alimenté par ses réseaux `h`, `g`, et `f` actuels) pour décider de la meilleure action à prendre (`a_t`).
3.  L'action est exécutée dans l'environnement.
4.  L'environnement retourne la récompense réelle obtenue (`r_{t+1}`) et la nouvelle observation réelle (`s_{t+1}`).
5.  L'agent stocke cette transition complète `(s_t, a_t, r_{t+1}, s_{t+1})` ainsi que la politique et la valeur calculées par le MCTS dans une base de données d'expérience (`ReplayDatabase`).

Ce processus est répété pour générer un grand nombre de parties ou de longues séquences de trading.

---

## 2. Phase d'Entraînement (Apprentissage)

Périodiquement, après avoir collecté suffisamment de données, la phase d'entraînement commence.

1.  **Échantillonnage**: On tire un batch de trajectoires de la `ReplayDatabase`. Une trajectoire est une séquence de `K` étapes : `(s_t, a_t, r_{t+1}, ... s_{t+K})`.

2.  **Apprentissage "Unrolled" sur K étapes**: Pour chaque trajectoire du batch, l'algorithme va "dérouler" la prédiction sur plusieurs étapes :
    *   **Étape 0 (Initiale)**:
        *   Le `Representation Network` encode l'état initial réel `s_t` pour obtenir le `hidden_state_0`.
        *   Le `Prediction Network` prédit une politique `p_0` et une valeur `v_0` à partir de `hidden_state_0`. La perte est calculée en comparant `p_0` et `v_0` à la politique et au résultat final de la partie stockés dans la base de données.

    *   **Étape k (1 à K)**:
        *   Le `Dynamics Network` prend `hidden_state_{k-1}` et l'action `a_{t+k-1}` pour prédire une récompense `r_k` et un nouvel état `hidden_state_k`.
        *   La perte sur la récompense est calculée en comparant `r_k` à la récompense réelle `r_{t+k}` stockée.
        *   Le `Prediction Network` prend ce nouvel état *imaginaire* `hidden_state_k` pour prédire une politique `p_k` et une valeur `v_k`.
        *   La perte est à nouveau calculée pour `p_k` et `v_k` en les comparant aux données cibles de la base de données.

3.  **Calcul de la Perte Totale**: La perte totale est la somme des pertes à chaque étape `k`:
    *   Perte sur la **valeur** (comparer la valeur prédite au résultat final réel du trade/de la partie).
    *   Perte sur la **politique** (comparer la politique prédite à la politique améliorée du MCTS).
    *   Perte sur la **récompense** (comparer la récompense prédite par `g` à la récompense réelle).

4.  **Rétropropagation**: La perte totale est rétropropagée à travers les trois réseaux (`h`, `g`, `f`) pour mettre à jour leurs poids simultanément.

Ce cycle garantit que l'agent améliore à la fois sa compréhension du monde, sa capacité à prédire les conséquences de ses actions, et sa stratégie de décision.
