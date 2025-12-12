# Guide de Contribution

Nous apprécions grandement toute contribution à ce projet MuZero pour le trading algorithmique ! Que ce soit pour signaler un bug, suggérer une nouvelle fonctionnalité, ou soumettre des modifications de code, votre aide est précieuse.

Ce document décrit les lignes directrices pour contribuer au projet.

## 1. Signaler un Bug

Si vous trouvez un bug, veuillez le signaler en ouvrant une issue sur le dépôt GitHub du projet. Avant de créer une nouvelle issue, veuillez vérifier si une issue similaire n'existe pas déjà.

Lors de la création d'un rapport de bug, incluez autant de détails que possible :

*   **Description claire et concise** du problème.
*   **Étapes pour reproduire** le bug.
*   **Comportement attendu** vs **comportement observé**.
*   Votre **environnement** (OS, version de Python, dépendances installées).
*   Tout **message d'erreur** ou trace de pile (stack trace) pertinent.

## 2. Suggérer une Fonctionnalité

Nous sommes ouverts aux suggestions de nouvelles fonctionnalités ! Si vous avez une idée, veuillez ouvrir une issue pour la discuter.

Lors de la suggestion d'une fonctionnalité, décrivez :

*   La **fonctionnalité** que vous souhaitez ajouter.
*   Le **cas d'utilisation** (pourquoi cette fonctionnalité est utile).
*   Toute **solution alternative** que vous avez envisagée.

## 3. Contribuer au Code

### 3.1. Cloner le Dépôt

Commencez par cloner le dépôt et créer une nouvelle branche pour vos modifications :

```bash
git clone https://github.com/votre_utilisateur/MuZero.git # Remplacez par l'URL réelle
cd MuZero
git checkout -b ma-nouvelle-fonctionnalite
```

### 3.2. Environnement de Développement

Assurez-vous d'avoir configuré votre environnement de développement comme décrit dans `GETTING_STARTED.md`.

### 3.3. Directives de Code

*   **Style de Code** : Suivez le style de code existant dans le projet (PEP 8 pour Python).
*   **Documentation** : Commentez votre code là où c'est nécessaire. Mettez à jour la documentation pertinente (y compris les docstrings des fonctions/classes) pour refléter vos changements.
*   **Tests** : Ajoutez des tests unitaires ou d'intégration pour couvrir vos nouvelles fonctionnalités ou corrections de bugs. Assurez-vous que tous les tests existants passent.
*   **Commit Messages** : Utilisez des messages de commit clairs et concis, décrivant la nature de vos changements.

### 3.4. Soumettre une Pull Request (PR)

Une fois vos modifications prêtes :

1.  **Mettez à jour votre branche** avec les dernières modifications du dépôt principal :
    ```bash
    git fetch origin
    git rebase origin/main # ou la branche principale
    ```
2.  **Poussez votre branche** vers votre fork :
    ```bash
    git push origin ma-nouvelle-fonctionnalite
    ```
3.  Ouvrez une **Pull Request** sur le dépôt principal.

Dans la description de la Pull Request, incluez :

*   Une **description claire** de ce que la PR fait et pourquoi.
*   Les **numéros d'issues** qu'elle résout (par exemple, "Closes #123").
*   Toute **information supplémentaire** pertinente pour l'examinateur.

Nous examinerons votre PR dès que possible et vous fournirons des commentaires.

Merci de votre contribution !
