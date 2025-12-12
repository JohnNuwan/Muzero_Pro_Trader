# Feuille de Route (Roadmap) pour le Développement

Ce document décompose l'implémentation de la version MuZero en phases et tâches concrètes.

---

## Phase 1 : Définition des Modèles PyTorch

L'objectif est de créer les briques de base de l'architecture neuronale.

- [ ] **Tâche 1.1 : `RepresentationNetwork`**
    - Créer un fichier `muzero/networks/representation.py`.
    - Définir une classe PyTorch qui prend en entrée les observations du marché et les transforme en `hidden_state`.
- [ ] **Tâche 1.2 : `DynamicsNetwork`**
    - Créer un fichier `muzero/networks/dynamics.py`.
    - Définir une classe PyTorch qui prend un `hidden_state` et une `action` et retourne un `next_hidden_state` et une `reward`.
- [ ] **Tâche 1.3 : `PredictionNetwork`**
    - Créer un fichier `muzero/networks/prediction.py`.
    - Définir une classe PyTorch qui prend un `hidden_state` et retourne une `policy` et une `value`.
- [ ] **Tâche 1.4 : `MuZeroNetwork`**
    - Créer un modèle principal qui assemble les trois réseaux précédents en un seul objet facile à manipuler.
    - Ce modèle exposera trois méthodes : `represent`, `dynamics`, et `predict`.

---

## Phase 2 : Adaptation du Monte Carlo Tree Search (MCTS)

Le MCTS doit être modifié pour utiliser notre modèle appris au lieu de l'environnement réel pour la simulation.

- [ ] **Tâche 2.1 : Créer `MuZeroMCTS`**
    - Dupliquer et adapter l'implémentation existante `AlphaZeroMCTS`.
- [ ] **Tâche 2.2 : Modifier l'algorithme de recherche**
    - Remplacer les appels à `env.step()` dans la boucle de recherche par des appels au `DynamicsNetwork`.
    - L'arbre MCTS sera maintenant construit sur des états imaginaires (`hidden_state`).

---

## Phase 3 : Script d'Entraînement

C'est la partie la plus complexe, qui orchestre l'apprentissage des modèles.

- [ ] **Tâche 3.1 : Créer `train_muzero.py`**
    - Mettre en place la boucle principale `Jouer -> Stocker -> Entraîner`.
- [ ] **Tâche 3.2 : Implémenter le `ReplayBuffer`**
    - Adapter la `ReplayDatabase` pour stocker des trajectoires complètes (séquences d'étapes) plutôt que des transitions uniques.
- [ ] **Tâche 3.3 : Définir les fonctions de perte**
    - Implémenter la logique de calcul de perte "déroulée" sur `K` étapes, comme décrit dans `TRAINING_LOOP.md`.
    - La perte totale combinera les erreurs de prédiction de la valeur, de la politique et de la récompense.
- [ ] **Tâche 3.4 : Configurer l'optimiseur**
    - Mettre en place un optimiseur (ex: Adam) pour mettre à jour les poids du `MuZeroNetwork`.

---

## Phase 4 : Intégration et Déploiement

- [ ] **Tâche 4.1 : Mettre à jour l'agent de trading**
    - Créer une classe `MuZeroTrader` qui utilise le `MuZeroNetwork` et le `MuZeroMCTS` pour prendre des décisions.
- [ ] **Tâche 4.2 : Mettre à jour l'orchestrateur**
    - Adapter `MultiSymbolOrchestrator` pour qu'il puisse lancer des `MuZeroTrader`.
- [ ] **Tâche 4.3 : Tests et Validation**
    - Conduire des backtests approfondis pour comparer les performances avec la v19.
    - Valider le comportement de l'agent en paper trading avant tout déploiement en argent réel.
