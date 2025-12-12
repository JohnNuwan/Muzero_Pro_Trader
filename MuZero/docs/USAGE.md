# Guide d'Utilisation

Ce document décrit comment exécuter les différentes composantes du projet MuZero pour le trading algorithmique, y compris l'entraînement, la démonstration et l'évaluation.

## 1. Entraînement d'un Modèle MuZero

Le processus d'entraînement implique la collecte de données (self-play) et la mise à jour des réseaux neuronaux.

*   **Point d'entrée principal** : Le script `training/train.py` est supposé être le point d'entrée pour lancer le processus d'entraînement.
*   **Configuration** : Les paramètres d'entraînement sont définis dans `config.py`. Assurez-vous de modifier ce fichier ou de fournir des arguments en ligne de commande si nécessaire.

Pour lancer un entraînement :

```bash
python training/train.py --config=<votre_fichier_config>.py # Exemple si plusieurs configs
```

**Options Courantes :**
*   `--episodes <nombre>` : Spécifie le nombre d'épisodes (parties/séquences de trading) à jouer pour l'entraînement.
*   `--steps <nombre>` : Spécifie le nombre maximal de pas par épisode.
*   `--save_interval <nombre>` : Fréquence de sauvegarde du modèle (par exemple, toutes les `X` itérations).
*   `--weights <chemin>` : Chemin vers des poids de modèle pré-existants pour reprendre un entraînement.

Exemple :
```bash
python training/train.py --episodes 100 --save_interval 10
```

Les modèles entraînés sont généralement sauvegardés dans le dossier `weights/` ou `results/weights/`.

## 2. Exécution d'une Démonstration

Le script `demo.py` est fourni pour exécuter une démonstration de l'agent MuZero, potentiellement en utilisant un modèle pré-entraîné.

Pour lancer la démonstration :

```bash
python demo.py --weights <chemin_vers_modele.pth> --env <nom_environnement>
```

**Options Courantes :**
*   `--weights <chemin>` : Chemin vers les poids du modèle MuZero à utiliser pour la démonstration.
*   `--env <nom>` : Nom de l'environnement de trading à utiliser (par exemple, "StockEnv", "CryptoEnv").
*   `--render` : (Optionnel) Afficher une visualisation de la simulation si l'environnement le supporte.

Exemple avec un modèle pré-entraîné dans `weights/best_model.pth` :
```bash
python demo.py --weights weights/best_model.pth --env StockEnv --render
```

## 3. Lancement d'un Test d'Intégration ou d'un Simple Test

Le dossier `tests/` contient des scripts pour vérifier le bon fonctionnement de certaines parties du système.

*   Pour lancer un test simple :
    ```bash
    python tests/simple_test.py
    ```

*   Pour lancer les tests d'intégration :
    ```bash
    python tests/test_integration.py
    ```

Ces scripts sont utiles pour s'assurer que les modifications n'ont pas introduit de régressions ou que les composants fonctionnent comme prévu.

## 4. Vérification de l'Installation GPU

Le script `check_gpu.py` permet de vérifier si PyTorch détecte et utilise correctement votre carte graphique (GPU).

```bash
python check_gpu.py
```

Ceci est crucial pour l'entraînement à grande échelle.
