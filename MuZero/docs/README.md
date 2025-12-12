# Project Gemini - MuZero Version

## Objectif

Ce projet vise à faire évoluer l'agent de trading `v19`, basé sur une architecture de type AlphaZero, vers un agent de nouvelle génération inspiré par MuZero.

## Idée Fondamentale

Contrairement à AlphaZero qui nécessite la connaissance des règles de l'environnement, MuZero apprend lui-même un modèle interne de son environnement. Dans le contexte du trading, cela signifie que notre agent ne se contentera plus de réagir à l'état actuel du marché, mais apprendra à **modéliser les dynamiques du marché** pour anticiper les mouvements futurs et planifier ses actions de manière beaucoup plus profonde.

Cette version représente un bond en avant en termes de complexité et de potentiel de performance.

## Structure de la Documentation

*   **[ARCHITECTURE.md](./ARCHITECTURE.md)**: Décrit l'architecture des trois réseaux de neurones qui forment le cœur de l'agent MuZero.
*   **[TRAINING_LOOP.md](./TRAINING_LOOP.md)**: Explique le processus d'entraînement cyclique nécessaire pour entraîner les modèles.
*   **[ROADMAP.md](./ROADMAP.md)**: Détaille les étapes de développement pour implémenter cette nouvelle version.
