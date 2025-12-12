# Architecture MuZero pour le Trading Algorithmique

L'architecture de notre agent MuZero s'articule autour de trois composantes neuronales distinctes qui travaillent de concert. L'objectif est de passer d'un modèle réactif (état -> action) à un modèle prédictif basé sur un modèle du monde appris (état -> planification via modèle interne -> action).

---

## 1. Representation Network (Fonction `h`)

Cette première brique est responsable de la compréhension de l'état actuel du marché.

*   **Rôle**: Compresser l'observation brute et complexe du marché en un vecteur de caractéristiques abstrait et de taille fixe, appelé `hidden_state`.
*   **Input**: Les données brutes fournies par l'environnement. Cela inclut les séries temporelles (OHLCV) sur différentes périodes, ainsi que tous les indicateurs techniques (RSI, MACD, Bandes de Bollinger, etc.) que nous utilisons dans la v19.
*   **Output**: Un `hidden_state` (vecteur PyTorch).
*   **Intuition**: Ce réseau apprend à extraire l'essence de la situation actuelle du marché, en ignorant le bruit et en se concentrant sur les signaux pertinents.

---

## 2. Dynamics Network (Fonction `g`) - Le "World Model"

C'est le cœur de l'approche MuZero et la nouveauté la plus significative. Ce réseau apprend les "règles du jeu" du marché.

*   **Rôle**: Prédire comment le `hidden_state` et la récompense évoluent si une certaine action est entreprise.
*   **Input**: Un `hidden_state` (fourni par le Representation Network) et une `action` hypothétique (0: HOLD, 1: BUY, 2: SELL).
*   **Output**: 
    1.  Le `next_hidden_state` prédit (le nouvel état abstrait du marché après l'action).
    2.  La `reward` prédite (le gain ou la perte immédiat(e) associé(e) à cette action).
*   **Intuition**: Ce réseau est notre "simulateur de marché" interne. Il permet à l'agent de se demander : "Si le marché est dans cette situation et que j'achète, à quoi ressemblera la situation juste après et quel sera mon PnL immédiat ?" sans avoir à interagir avec l'environnement réel.

---

## 3. Prediction Network (Fonction `f`)

Ce réseau est similaire à celui utilisé dans AlphaZero. Il est responsable de la stratégie.

*   **Rôle**: Évaluer un état et suggérer les meilleures actions possibles.
*   **Input**: Un `hidden_state` (qu'il provienne du Representation Network ou qu'il soit un état futur imaginé par le Dynamics Network).
*   **Output**:
    1.  Une `policy` (distribution de probabilité sur les actions possibles).
    2.  Une `value` (un scalaire représentant l'espérance de gains futurs à partir de cet état).
*   **Intuition**: Ce réseau répond à la question : "Étant donné cette situation (réelle ou imaginée), quelles sont les meilleures manœuvres et à quel point cette situation est-elle prometteuse ?"

---

## Intégration dans le Monte Carlo Tree Search (MCTS)

La magie opère lorsque le MCTS utilise ces trois réseaux. Lors de la phase de planification (la recherche dans l'arbre), l'agent va :
1.  Encoder l'état réel du marché une seule fois avec le **Representation Network**.
2.  Pour explorer l'arbre, il va simuler des séquences d'actions (`action_1`, `action_2`, ...). À chaque nœud de l'arbre :
    *   Il utilise le **Prediction Network** pour évaluer la situation et savoir quelles actions explorer.
    *   Il utilise le **Dynamics Network** pour générer le prochain état imaginaire de la simulation, sans plus jamais faire appel à l'environnement.

Cela permet une planification beaucoup plus profonde et rapide, car l'agent peut explorer des milliers de futurs possibles "dans sa tête".
