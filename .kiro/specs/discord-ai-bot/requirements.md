# Requirements Document

## Introduction

Ce projet vise à créer un bot Discord nommé "Jean_De_AI" avec une intelligence artificielle GPT construite from scratch. Le bot sera capable de converser naturellement avec les utilisateurs sur Discord en utilisant un modèle de langage entraîné sur des conversations personnalisées. Le système inclura également une mémoire pour maintenir le contexte des conversations.

## Requirements

### Requirement 1: Architecture du Bot Discord

**User Story:** En tant qu'utilisateur Discord, je veux interagir avec un bot nommé "Jean_De_AI" qui répond à mes messages et participe activement aux discussions, afin d'avoir des conversations naturelles.

#### Acceptance Criteria

1. WHEN le bot est démarré THEN il SHALL se connecter à Discord avec le nom "Jean_De_AI"
2. WHEN un utilisateur envoie un message dans un canal de discussion THEN le bot SHALL détecter le message
3. WHEN le bot détecte un message THEN il SHALL décider s'il doit répondre (peut répondre spontanément sans être mentionné)
4. WHEN le bot décide de répondre THEN il SHALL traiter le message et générer une réponse en utilisant le modèle GPT
5. WHEN le bot génère une réponse THEN il SHALL l'envoyer dans le même canal Discord
6. WHEN personne n'a parlé dans le canal pendant 10 minutes THEN le bot SHALL envoyer un message comme "hello? ya quelqu'un? jsuis tous seul j'ai peur"
7. WHEN le bot participe à une discussion THEN il SHALL pouvoir intervenir naturellement sans toujours attendre d'être mentionné

### Requirement 2: Modèle GPT From Scratch

**User Story:** En tant que développeur, je veux un modèle GPT implémenté from scratch en Python, afin d'avoir un contrôle total sur l'architecture et l'entraînement.

#### Acceptance Criteria

1. WHEN le système est initialisé THEN il SHALL implémenter une architecture Transformer avec attention multi-têtes
2. WHEN le modèle est créé THEN il SHALL inclure des couches d'embedding, des blocs Transformer, et une couche de sortie
3. WHEN le modèle reçoit du texte THEN il SHALL tokenizer l'entrée en utilisant un vocabulaire personnalisé
4. WHEN le modèle génère du texte THEN il SHALL utiliser un mécanisme de décodage (greedy, beam search, ou sampling)
5. IF le modèle est entraîné THEN il SHALL sauvegarder les poids dans un fichier

### Requirement 3: Système de Dataset et Entraînement

**User Story:** En tant que développeur, je veux pouvoir entraîner le modèle sur des conversations personnalisées, afin que le bot parle comme une personne normale.

#### Acceptance Criteria

1. WHEN des conversations sont fournies THEN le système SHALL les stocker dans le dossier datasets
2. WHEN l'entraînement commence THEN le système SHALL charger et prétraiter les données du dataset
3. WHEN le modèle s'entraîne THEN il SHALL optimiser les poids en utilisant la backpropagation
4. WHEN l'entraînement progresse THEN le système SHALL afficher les métriques de loss et de perplexité
5. IF l'entraînement est terminé THEN le système SHALL sauvegarder le modèle entraîné

### Requirement 4: Système de Mémoire Conversationnelle

**User Story:** En tant qu'utilisateur, je veux que le bot se souvienne du contexte de nos conversations, afin d'avoir des échanges plus cohérents et personnalisés.

#### Acceptance Criteria

1. WHEN un utilisateur envoie un message THEN le système SHALL stocker le message dans la mémoire
2. WHEN le bot génère une réponse THEN il SHALL inclure le contexte des N derniers messages
3. WHEN une conversation devient trop longue THEN le système SHALL gérer la fenêtre de contexte (truncation ou summarization)
4. IF le bot redémarre THEN il SHALL pouvoir charger l'historique des conversations depuis le stockage persistant
5. WHEN un utilisateur démarre une nouvelle conversation THEN le système SHALL créer un nouveau contexte mémoire

### Requirement 5: Structure du Projet et Organisation

**User Story:** En tant que développeur, je veux une structure de projet claire et organisée, afin de faciliter le développement et la maintenance.

#### Acceptance Criteria

1. WHEN le projet est créé THEN il SHALL avoir la structure suivante:
   - bot/ (code du bot Discord)
   - docs/ (documentation)
   - python/ (code du modèle GPT)
   - python/datasets/ (données d'entraînement)
   - python/exemples/ (exemples d'utilisation)
   - python/tests/ (tests unitaires)
2. WHEN le code est écrit THEN il SHALL être modulaire et réutilisable
3. WHEN des dépendances sont nécessaires THEN elles SHALL être listées dans requirements.txt
4. WHEN le projet est documenté THEN il SHALL inclure un README avec instructions d'installation et d'utilisation

### Requirement 6: Tests et Validation

**User Story:** En tant que développeur, je veux tester les composants du système, afin de garantir leur bon fonctionnement.

#### Acceptance Criteria

1. WHEN le modèle GPT est implémenté THEN il SHALL avoir des tests unitaires pour chaque composant
2. WHEN le tokenizer est créé THEN il SHALL avoir des tests de validation
3. WHEN le bot Discord est développé THEN il SHALL avoir des exemples de test de connexion
4. WHEN le système de mémoire est implémenté THEN il SHALL avoir des tests de stockage et récupération

### Requirement 7: Configuration et Déploiement

**User Story:** En tant qu'utilisateur, je veux pouvoir configurer et démarrer le bot facilement, afin de l'utiliser rapidement.

#### Acceptance Criteria

1. WHEN le bot est configuré THEN il SHALL utiliser un fichier de configuration pour le token Discord
2. WHEN le modèle est chargé THEN il SHALL utiliser des chemins configurables pour les fichiers de poids
3. WHEN le bot démarre THEN il SHALL valider la configuration avant de se connecter
4. IF la configuration est invalide THEN le système SHALL afficher des messages d'erreur clairs
