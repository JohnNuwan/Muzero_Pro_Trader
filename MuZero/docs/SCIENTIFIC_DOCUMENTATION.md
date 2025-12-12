# Documentation Scientifique de l'Algorithme MuZero

## 1. Introduction

MuZero est un algorithme d'apprentissage par renforcement profond développé par DeepMind. Sa principale innovation est sa capacité à maîtriser des environnements complexes sans en connaître les règles. Pour ce faire, MuZero apprend un modèle de l'environnement, ce qui lui permet de planifier ses actions en simulant les résultats possibles "dans sa tête".

Cette approche combine un **modèle appris** avec une **recherche arborescente Monte Carlo (MCTS)**, guidée par trois fonctions neuronales fondamentales : une fonction de représentation, une fonction de dynamique et une fonction de prédiction.

## 2. Les Trois Fonctions Clés

L'architecture de MuZero repose sur trois réseaux de neurones qui travaillent de concert.

### 2.1. Fonction de Représentation (`h`)

La fonction de représentation, notée `h`, prend en entrée une séquence d'observations brutes de l'environnement et la compresse en un **état caché** (ou *hidden state*), noté `s`. Cet état caché est une représentation abstraite et compacte de l'état actuel de l'environnement, apprise par le modèle.

**Équation :**
```latex
s_t = h(o_1, o_2, ..., o_t)
```
Où :
- `o_t` est l'observation à l'instant `t`.
- `s_t` est l'état caché à l'instant `t`.

Cette abstraction permet à l'agent de se concentrer sur les informations pertinentes et d'ignorer le "bruit" des observations brutes.

### 2.2. Fonction de Dynamique (`g`)

La fonction de dynamique, notée `g`, est le **modèle du monde** appris par MuZero. Elle prédit comment l'état et la récompense évoluent lorsqu'une action est entreprise.

**Équation :**
```latex
r_{k+1}, s_{k+1} = g(s_k, a_{k+1})
```
Où :
- `s_k` est un état caché (réel ou simulé).
- `a_{k+1}` est une action hypothétique.
- `r_{k+1}` est la **récompense** prédite pour cette action.
- `s_{k+1}` est le **nouvel état caché** prédit.

Cette fonction est au cœur de la planification, car elle permet à l'agent de simuler des trajectoires futures sans interagir avec l'environnement réel.

### 2.3. Fonction de Prédiction (`f`)

La fonction de prédiction, notée `f`, évalue un état caché et guide la stratégie de l'agent.

**Équation :**
```latex
p_k, v_k = f(s_k)
```
Où :
- `s_k` est un état caché.
- `p_k` est la **politique** (un vecteur de probabilités sur les actions possibles).
- `v_k` est la **valeur** (un scalaire représentant l'espérance des récompenses futures cumulées à partir de cet état).

Ces deux sorties sont cruciales pour guider la recherche arborescente Monte Carlo.

## 3. Planification avec Monte Carlo Tree Search (MCTS)

Le MCTS de MuZero est une version améliorée de celui d'AlphaZero. Il utilise les trois fonctions (`h`, `g`, `f`) pour construire un arbre de recherche et sélectionner la meilleure action.

Le processus se déroule en plusieurs simulations. Chaque simulation part de la racine (l'état actuel) et descend dans l'arbre en sélectionnant des actions jusqu'à atteindre une feuille. Le critère de sélection d'une action `a` dans un nœud `s` est basé sur la formule **PUCT (Polynomial Upper Confidence for Trees)** :

**Équation :**
```latex
a_t = \arg\max_a \left( Q(s, a) + P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \cdot (c_1 + \log\frac{\sum_b N(s, b) + c_2 + 1}{c_2}) \right)
```
Où :
- `Q(s, a)` est la **valeur moyenne** de l'action `a` depuis l'état `s`.
- `P(s, a)` est la **probabilité a priori** de choisir l'action `a` (donnée par la politique `p` de la fonction `f`).
- `N(s, a)` est le **nombre de fois** que l'action `a` a été choisie depuis `s`.
- `c_1` et `c_2` sont des hyperparamètres qui contrôlent l'équilibre entre l'**exploitation** (choisir des actions prometteuses) et l'**exploration** (essayer des actions moins explorées).

Une fois une feuille atteinte, la fonction de prédiction `f` est appelée pour évaluer le nœud, et la valeur `v` est remontée pour mettre à jour les statistiques (`Q` et `N`) des nœuds parents.

## 4. Entraînement

L'entraînement vise à optimiser les trois fonctions neuronales en comparant leurs prédictions aux données réelles issues de parties jouées par l'agent.

L'agent joue des parties et stocke les trajectoires `(s_t, a_t, r_{t+1}, \pi_t, z_t)` dans un buffer de rejeu, où `\pi_t` est la politique MCTS améliorée et `z_t` le résultat final de la partie.

Pour l'entraînement, on échantillonne une trajectoire et on "déroule" le modèle sur `K` étapes. La fonction de perte `L` est la somme des pertes à chaque étape `k` :

**Équation :**
```latex
L_t(\theta) = \sum_{k=0}^{K} \left( l^r(r_{t+k}, \hat{r}_{t+k}) + l^v(z_{t+k}, \hat{v}_{t+k}) + l^p(\pi_{t+k}, \hat{p}_{t+k}) \right)
```
Où :
- `\theta` représente les poids des trois réseaux.
- **Perte sur la récompense (`l^r`)**: Compare la récompense prédite `\hat{r}` par la fonction `g` à la récompense réelle `r`.
- **Perte sur la valeur (`l^v`)**: Compare la valeur prédite `\hat{v}` par la fonction `f` au résultat de la partie `z`.
- **Perte sur la politique (`l^p`)**: Compare la politique prédite `\hat{p}` par la fonction `f` à la politique de recherche MCTS `\pi`.

Typiquement, ces pertes sont calculées par une erreur quadratique moyenne pour la valeur et la récompense, et par une entropie croisée pour la politique. Cette perte totale est ensuite rétropropagée à travers les trois réseaux pour les mettre à jour.
