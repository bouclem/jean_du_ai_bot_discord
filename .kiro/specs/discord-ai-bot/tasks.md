# Implementation Plan

- [x] 1. Créer la structure du projet et les fichiers de configuration





  - Créer l'arborescence complète des dossiers (bot/, python/, docs/, etc.)
  - Créer requirements.txt avec les dépendances (discord.py, numpy)
  - Créer .env.example pour la configuration
  - Créer .gitignore pour exclure les fichiers sensibles
  - _Requirements: 5.1, 7.1_

- [x] 2. Implémenter le tokenizer




  - [x] 2.1 Créer la classe Tokenizer de base


    - Implémenter __init__, encode(), decode()
    - Implémenter build_vocab() pour construire le vocabulaire à partir de textes
    - Gérer les tokens spéciaux ([PAD], [UNK], [BOS], [EOS])
    - _Requirements: 2.3_
  

  - [x] 2.2 Ajouter la persistence du vocabulaire


    - Implémenter save_vocab() et load_vocab()
    - Utiliser JSON pour stocker le vocabulaire
    - _Requirements: 2.5_
  

  - [x] 2.3 Créer les tests unitaires du tokenizer

    - Tester encoding/decoding basique
    - Tester les tokens spéciaux
    - Tester la sauvegarde/chargement du vocabulaire
    - _Requirements: 6.1, 6.2_

- [x] 3. Implémenter l'architecture Transformer




  - [x] 3.1 Créer la classe MultiHeadAttention


    - Implémenter le mécanisme d'attention scaled dot-product
    - Implémenter les projections Q, K, V
    - Implémenter la concaténation des têtes multiples
    - _Requirements: 2.1_


  
  - [x] 3.2 Créer la classe FeedForward

    - Implémenter le réseau feed-forward à 2 couches


    - Ajouter l'activation GELU ou ReLU
    - _Requirements: 2.1_
  

  - [x] 3.3 Créer la classe TransformerBlock


    - Combiner MultiHeadAttention et FeedForward
    - Ajouter layer normalization et connexions résiduelles
    - Implémenter dropout
    - _Requirements: 2.1, 2.2_
  
  - [x] 3.4 Créer les tests unitaires des composants Transformer

    - Tester MultiHeadAttention avec des entrées simples
    - Tester FeedForward
    - Tester TransformerBlock complet
    - _Requirements: 6.1_

- [x] 4. Implémenter le modèle GPT complet




  - [x] 4.1 Créer la classe GPTModel


    - Implémenter les embeddings (token + positional)
    - Empiler N TransformerBlocks
    - Ajouter la couche de sortie (projection vers vocabulaire)
    - Implémenter forward()
    - _Requirements: 2.1, 2.2_
  
  - [x] 4.2 Implémenter la génération de texte

    - Créer generate() avec sampling par température
    - Implémenter greedy decoding
    - Gérer la longueur maximale de génération
    - _Requirements: 2.4_
  
  - [x] 4.3 Ajouter la sauvegarde/chargement du modèle

    - Implémenter save() pour sauvegarder les poids
    - Implémenter load() pour charger les poids
    - Utiliser pickle ou numpy pour la persistence
    - _Requirements: 2.5_
  
  - [x] 4.4 Créer les tests unitaires du modèle GPT


    - Tester forward pass avec des entrées aléatoires
    - Tester la génération de texte
    - Tester save/load
    - _Requirements: 6.1_

- [x] 5. Implémenter le système d'entraînement





  - [x] 5.1 Créer la classe ConversationDataset


    - Implémenter __init__, __len__, __getitem__
    - Charger les conversations depuis JSON
    - Prétraiter les données (tokenization, padding)
    - _Requirements: 3.2_
  
  - [x] 5.2 Créer la classe GPTTrainer


    - Implémenter la boucle d'entraînement
    - Calculer la loss (cross-entropy)
    - Implémenter la backpropagation (gradient descent)
    - Afficher les métriques (loss, perplexité)
    - _Requirements: 3.3, 3.4_
  


  - [x] 5.3 Ajouter les checkpoints d'entraînement





    - Sauvegarder le modèle périodiquement
    - Sauvegarder l'état de l'optimiseur


    - _Requirements: 3.5_
  


  - [x] 5.4 Créer un exemple d'entraînement



    - Créer python/exemples/exemple_training.py
    - Charger un dataset d'exemple
    - Entraîner le modèle sur quelques epochs
    - Sauvegarder le modèle entraîné
    - _Requirements: 5.2, 6.2_

- [ ] 6. Implémenter le système de mémoire conversationnelle
  - [ ] 6.1 Créer la classe ConversationMemory
    - Implémenter add_message() pour ajouter des messages
    - Implémenter get_context() pour récupérer l'historique
    - Gérer la fenêtre de contexte (limiter à N messages)
    - Organiser par channel_id
    - _Requirements: 4.1, 4.2, 4.3_
  
  - [ ] 6.2 Ajouter la persistence de la mémoire
    - Implémenter save_to_disk() en JSON
    - Implémenter load_from_disk()
    - Gérer les erreurs de fichier corrompu
    - _Requirements: 4.4_
  
  - [ ] 6.3 Créer les tests unitaires de la mémoire
    - Tester ajout et récupération de messages
    - Tester la gestion de la fenêtre de contexte
    - Tester la persistence
    - _Requirements: 6.4_

- [ ] 7. Implémenter le moniteur d'inactivité
  - [ ] 7.1 Créer la classe InactivityMonitor
    - Tracker le dernier message par canal (timestamp)
    - Implémenter check_inactivity() pour vérifier le timeout
    - Implémenter update_activity() pour mettre à jour le timestamp
    - _Requirements: 1.6_
  
  - [ ] 7.2 Ajouter les messages de solitude
    - Créer get_lonely_message() qui retourne des variantes
    - Exemples: "hello? ya quelqu'un? jsuis tous seul j'ai peur"
    - _Requirements: 1.6_

- [ ] 8. Implémenter le bot Discord
  - [ ] 8.1 Créer la classe BotConfig
    - Charger les variables d'environnement
    - Valider la configuration
    - Gérer les valeurs par défaut
    - _Requirements: 7.1, 7.2, 7.3_
  
  - [ ] 8.2 Créer la classe JeanDeAIBot
    - Initialiser le client Discord
    - Charger le modèle GPT et la mémoire
    - Implémenter on_ready() pour confirmer la connexion
    - _Requirements: 1.1, 7.4_
  
  - [ ] 8.3 Implémenter la détection et traitement des messages
    - Implémenter on_message() pour détecter les messages
    - Filtrer les messages du bot lui-même
    - Implémenter should_respond() avec logique de probabilité
    - _Requirements: 1.2, 1.3, 1.7_
  
  - [ ] 8.4 Implémenter la génération de réponses
    - Créer generate_response() qui utilise le modèle GPT
    - Récupérer le contexte depuis ConversationMemory
    - Formater le prompt avec l'historique
    - Ajouter la réponse à la mémoire
    - _Requirements: 1.4, 4.2_
  
  - [ ] 8.5 Envoyer les réponses sur Discord
    - Envoyer la réponse générée dans le canal
    - Gérer les erreurs d'envoi (rate limiting, permissions)
    - _Requirements: 1.5_

- [ ] 9. Intégrer le moniteur d'inactivité au bot
  - Initialiser InactivityMonitor dans JeanDeAIBot
  - Mettre à jour l'activité dans on_message()
  - Lancer check_inactivity() en tâche asynchrone
  - Envoyer le message de solitude si timeout
  - _Requirements: 1.6_

- [ ] 10. Créer les exemples d'utilisation
  - [ ] 10.1 Créer exemple_generation.py
    - Charger un modèle entraîné
    - Générer du texte à partir de prompts
    - Afficher les résultats
    - _Requirements: 5.2_
  
  - [ ] 10.2 Créer exemple_conversation.py
    - Simuler une conversation avec le modèle
    - Utiliser ConversationMemory
    - Afficher l'historique
    - _Requirements: 5.2_

- [ ] 11. Créer les tests d'intégration
  - Créer test_bot_integration.py
  - Mocker le client Discord
  - Tester le pipeline complet: message → contexte → génération → réponse
  - Tester le moniteur d'inactivité
  - _Requirements: 6.1_

- [ ] 12. Créer la documentation
  - [ ] 12.1 Créer docs/README.md
    - Expliquer le projet
    - Instructions d'installation
    - Configuration du token Discord
    - Comment lancer le bot
    - _Requirements: 5.4_
  
  - [ ] 12.2 Créer docs/TRAINING.md
    - Expliquer comment préparer un dataset
    - Format des conversations JSON
    - Comment entraîner le modèle
    - Paramètres d'entraînement
    - _Requirements: 5.4_
  
  - [ ] 12.3 Créer docs/DEPLOYMENT.md
    - Configuration de l'environnement
    - Variables d'environnement
    - Déploiement sur un serveur
    - _Requirements: 5.4_

- [ ] 13. Créer un dataset d'exemple
  - Créer python/datasets/conversations.json
  - Ajouter des conversations d'exemple en français naturel
  - Utiliser un style conversationnel casual
  - Inclure des variantes de salutations, questions, réponses
  - _Requirements: 3.1_

- [ ] 14. Créer le script de lancement principal
  - Créer bot/main.py comme point d'entrée
  - Charger la configuration
  - Initialiser et lancer le bot
  - Gérer les erreurs de connexion
  - Ajouter logging
  - _Requirements: 7.3, 7.4_
