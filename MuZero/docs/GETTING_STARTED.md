# Guide de Démarrage

Ce guide vous aidera à configurer l'environnement nécessaire pour exécuter le projet MuZero pour le trading algorithmique.

## Prérequis

Assurez-vous que les éléments suivants sont installés sur votre système :

*   **Python 3.8+** : Il est recommandé d'utiliser une version de Python 3.8 ou supérieure.
*   **pip** : Le gestionnaire de paquets de Python, généralement inclus avec Python.
*   **Git** : Pour cloner le dépôt du projet.

## 1. Cloner le Dépôt

Ouvrez votre terminal ou invite de commandes et exécutez la commande suivante pour cloner le projet :

```bash
git clone https://github.com/votre_utilisateur/MuZero.git # Remplacez par l'URL réelle de votre dépôt
cd MuZero
```

## 2. Créer un Environnement Virtuel (Recommandé)

Il est fortement conseillé de travailler dans un environnement virtuel Python pour éviter les conflits de dépendances avec d'autres projets.

```bash
python -m venv venv
```

Activez l'environnement virtuel :

*   **Sur Windows :**
    ```bash
    .\venv\Scripts\activate
    ```
*   **Sur macOS/Linux :**
    ```bash
    source venv/bin/activate
    ```

## 3. Installer les Dépendances

Comme il n'y a pas de `requirements.txt` fourni, vous devrez installer les dépendances manuellement ou les créer. Le projet MuZero utilise généralement des bibliothèques telles que PyTorch, NumPy, etc.

Voici une liste *probable* des dépendances (à adapter selon les besoins réels du projet) :

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu # ou l'URL pour votre GPU
pip install numpy pandas gym # ou d'autres bibliothèques de trading/environnement si utilisées
pip install ray # Si le projet utilise Ray pour la distribution
```
*(**Note** : Remplacez l'URL de PyTorch par celle correspondant à votre configuration GPU (cu118, cu121, etc.) ou utilisez `--index-url https://download.pytorch.org/whl/cpu` pour la version CPU.)*

## 4. Vérifier l'Installation

Vous pouvez exécuter un script simple pour vérifier que tout est correctement configuré. Par exemple, le script `check_gpu.py` peut être utilisé pour vérifier la configuration de PyTorch et la disponibilité du GPU :

```bash
python check_gpu.py
```

Si le script s'exécute sans erreur majeure et que les informations sur le GPU sont correctes (si vous en avez un), votre environnement est prêt.
