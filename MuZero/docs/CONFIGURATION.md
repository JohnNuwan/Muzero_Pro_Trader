# Documentation de la Configuration (`config.py`)

Le fichier `config.py` centralise tous les hyperparamètres et réglages nécessaires au fonctionnement et à l'entraînement de l'agent MuZero. Comprendre ces paramètres est crucial pour optimiser les performances et adapter l'agent à différents environnements ou stratégies.

La classe `MuZeroConfig` contient les attributs suivants, regroupés par catégorie :

## 1. Paramètres du Réseau (Network)

*   `observation_shape` (tuple) :
    *   **Description** : La forme (dimensions) de l'observation brute de l'environnement que le réseau de représentation reçoit en entrée.
    *   **Exemple** : `(84,)` pourrait signifier un vecteur de 84 caractéristiques.
*   `action_space_size` (int) :
    *   **Description** : Le nombre total d'actions discrètes que l'agent peut entreprendre.
    *   **Exemple** : `5` pourrait correspondre à `HOLD, BUY, SELL, SPLIT, CLOSE`.
*   `hidden_state_size` (int) :
    *   **Description** : La dimension du vecteur de l'état caché produit par le réseau de représentation et utilisé par les réseaux de dynamique et de prédiction. C'est la taille de la représentation abstraite de l'environnement.
*   `network_hidden_dims` (list of int) :
    *   **Description** : Une liste définissant l'architecture des couches cachées (nombre de neurones par couche) pour les MLP (Multi-Layer Perceptrons) internes des réseaux (représentation, dynamique, prédiction).

## 2. Symboles de Trading

*   `symbols` (list of str) :
    *   **Description** : Liste des symboles de trading (paires de devises, indices, matières premières, etc.) que l'agent est configuré pour gérer. Identique à la configuration V19.

## 3. Paramètres d'Entraînement (Training)

*   `batch_size` (int) :
    *   **Description** : La taille des mini-lots de données échantillonnés du buffer de rejeu pour chaque étape de mise à jour du réseau.
*   `learning_rate` (float) :
    *   **Description** : Le taux d'apprentissage utilisé par l'optimiseur pour ajuster les poids du réseau.
*   `weight_decay` (float) :
    *   **Description** : Terme de régularisation L2 ajouté à la fonction de perte pour prévenir le surapprentissage.
*   `momentum` (float) :
    *   **Description** : Paramètre utilisé par certains optimiseurs (comme SGD) pour accélérer la convergence et stabiliser l'entraînement.
*   `training_steps` (int) :
    *   **Description** : Le nombre total d'étapes de mise à jour des poids du réseau durant l'entraînement.
*   `checkpoint_interval` (int) :
    *   **Description** : La fréquence (en nombre d'étapes d'entraînement) à laquelle le modèle est sauvegardé.
*   `num_unroll_steps` (int) :
    *   **Description** : Le nombre d'étapes futures sur lesquelles le modèle de dynamique est déroulé (`unrolled`) pendant l'entraînement. Ceci permet de calculer des pertes multi-pas.
*   `td_steps` (int) :
    *   **Description** : Le nombre d'étapes utilisées pour le calcul des cibles de valeur (Temporal Difference) par bootstrapping. Définit la profondeur de la somme des récompenses futures pour estimer la valeur d'un état.

## 4. Paramètres MCTS (Monte Carlo Tree Search)

*   `num_simulations` (int) :
    *   **Description** : Le nombre de simulations complètes effectuées par le MCTS pour chaque décision d'action. Un nombre plus élevé conduit à une meilleure planification mais est plus coûteux en calcul.
*   `discount` (float) :
    *   **Description** : Le facteur d'actualisation utilisé pour les récompenses futures. Une valeur proche de 1 accorde une importance quasi égale aux récompenses immédiates et futures.
*   `root_dirichlet_alpha` (float) :
    *   **Description** : Paramètre `alpha` pour le bruit de Dirichlet ajouté à la politique à la racine de l'arbre MCTS. Cela favorise l'exploration au début de la recherche.
*   `root_exploration_fraction` (float) :
    *   **Description** : La fraction du bruit de Dirichlet à ajouter à la politique de la racine.
*   `pb_c_base` (int) :
    *   **Description** : Un paramètre constant utilisé dans la formule PUCT pour la sélection des actions pendant le MCTS.
*   `pb_c_init` (float) :
    *   **Description** : Un autre paramètre constant pour la formule PUCT.

## 5. Buffer de Rejeu (Replay Buffer)

*   `window_size` (int) :
    *   **Description** : La taille maximale (en nombre de jeux/trajectoires) du buffer de rejeu. Les jeux les plus anciens sont supprimés lorsque le buffer est plein.
*   `batch_size` (int) :
    *   **Description** : *Dupliqué avec le paramètre d'entraînement.* C'est la taille des mini-lots de données échantillonnés du buffer de rejeu pour l'entraînement. Il est important que cette valeur corresponde au `batch_size` d'entraînement.

## 6. Self-Play

*   `max_moves` (int) :
    *   **Description** : Le nombre maximal de mouvements (actions) permis dans un seul épisode de self-play. Utile pour limiter la durée des épisodes.

## 7. Chemins (Paths)

*   `results_path` (str) :
    *   **Description** : Le chemin du répertoire où les résultats de l'entraînement, les journaux et d'autres sorties seront sauvegardés.
*   `weights_path` (str) :
    *   **Description** : Le chemin du répertoire où les poids du modèle MuZero entraîné seront sauvegardés.

## Fonction `visit_softmax_temperature_fn`

*   **Description** : Cette fonction retourne une valeur de "température" utilisée pour altérer la distribution des comptages de visites lors de la sélection d'actions. Elle permet de rendre la sélection d'actions plus déterministe (plus "greedy") à mesure que l'entraînement progresse. Une température plus basse concentre la probabilité sur les actions les plus visitées.
    *   **`trained_steps < 0.5 * self.training_steps`** : Température de `1.0` (plus d'exploration).
    *   **`trained_steps < 0.75 * self.training_steps`** : Température de `0.5` (moins d'exploration).
    *   **`else`** : Température de `0.25` (très peu d'exploration, actions presque déterministes).
