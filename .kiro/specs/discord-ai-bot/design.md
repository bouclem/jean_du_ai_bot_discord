# Design Document - Jean_De_AI Discord Bot

## Overview

Jean_De_AI est un bot Discord intelligent qui utilise un modèle GPT implémenté from scratch en Python. Le bot peut participer naturellement aux conversations, se souvenir du contexte, et même initier des interactions lorsque le canal est silencieux. L'architecture est divisée en trois composants principaux : le bot Discord, le modèle GPT, et le système de mémoire.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Discord Server                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Discord Bot Layer                          │
│  - Message Detection                                         │
│  - Response Decision Logic                                   │
│  - Inactivity Timer (10 min)                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Memory System                              │
│  - Conversation History                                      │
│  - Context Window Management                                 │
│  - Persistent Storage                                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   GPT Model (From Scratch)                   │
│  - Tokenizer                                                 │
│  - Transformer Architecture                                  │
│  - Text Generation                                           │
└─────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Discord Bot Component (`bot/`)

#### bot/main.py
Point d'entrée principal du bot Discord.

**Responsabilités:**
- Initialiser le client Discord
- Gérer les événements Discord (on_ready, on_message)
- Coordonner entre le système de mémoire et le modèle GPT

**Interface:**
```python
class JeanDeAIBot:
    def __init__(self, token: str, model_path: str)
    async def on_ready()
    async def on_message(message: discord.Message)
    async def should_respond(message: discord.Message) -> bool
    async def generate_response(message: discord.Message) -> str
    def run()
```

#### bot/inactivity_monitor.py
Surveille l'inactivité des canaux.

**Responsabilités:**
- Tracker le dernier message par canal
- Déclencher un message après 10 minutes d'inactivité

**Interface:**
```python
class InactivityMonitor:
    def __init__(self, bot: JeanDeAIBot, timeout_minutes: int = 10)
    def update_activity(self, channel_id: str)
    async def check_inactivity()
    def get_lonely_message() -> str
```

#### bot/config.py
Gestion de la configuration.

**Interface:**
```python
class BotConfig:
    discord_token: str
    model_path: str
    memory_path: str
    response_probability: float  # Probabilité de répondre spontanément
```

### 2. GPT Model Component (`python/`)

#### python/model.py
Implémentation du modèle Transformer GPT.

**Architecture du Modèle:**
- Embedding Layer (token + positional embeddings)
- N Transformer Blocks (self-attention + feed-forward)
- Output Layer (projection vers vocabulaire)

**Interface:**
```python
class GPTModel:
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, 
                 n_layers: int, max_seq_len: int)
    def forward(self, x: np.ndarray) -> np.ndarray
    def generate(self, prompt: str, max_length: int, 
                 temperature: float = 1.0) -> str
    def save(self, path: str)
    def load(self, path: str)
```

#### python/transformer.py
Composants du Transformer.

**Interface:**
```python
class MultiHeadAttention:
    def __init__(self, d_model: int, n_heads: int)
    def forward(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray

class FeedForward:
    def __init__(self, d_model: int, d_ff: int)
    def forward(self, x: np.ndarray) -> np.ndarray

class TransformerBlock:
    def __init__(self, d_model: int, n_heads: int, d_ff: int)
    def forward(self, x: np.ndarray) -> np.ndarray
```

#### python/tokenizer.py
Tokenization du texte.

**Responsabilités:**
- Convertir texte → tokens (IDs)
- Convertir tokens → texte
- Gérer le vocabulaire

**Interface:**
```python
class Tokenizer:
    def __init__(self, vocab_path: str = None)
    def build_vocab(self, texts: List[str])
    def encode(self, text: str) -> List[int]
    def decode(self, tokens: List[int]) -> str
    def save_vocab(self, path: str)
    def load_vocab(self, path: str)
```

#### python/trainer.py
Entraînement du modèle.

**Interface:**
```python
class GPTTrainer:
    def __init__(self, model: GPTModel, learning_rate: float)
    def train(self, dataset: Dataset, epochs: int, batch_size: int)
    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float
    def backward(self, loss: float)
    def save_checkpoint(self, epoch: int, path: str)
```

### 3. Memory System Component (`python/memory.py`)

**Responsabilités:**
- Stocker l'historique des conversations par canal
- Gérer la fenêtre de contexte
- Persister les données

**Interface:**
```python
class ConversationMemory:
    def __init__(self, max_context_length: int = 10, storage_path: str = None)
    def add_message(self, channel_id: str, user: str, message: str)
    def get_context(self, channel_id: str) -> List[Dict[str, str]]
    def clear_context(self, channel_id: str)
    def save_to_disk(self)
    def load_from_disk(self)
```

### 4. Dataset Component (`python/datasets/`)

#### python/datasets/dataset.py
Gestion des données d'entraînement.

**Interface:**
```python
class ConversationDataset:
    def __init__(self, data_path: str, tokenizer: Tokenizer)
    def load_conversations(self, file_path: str)
    def preprocess(self)
    def __len__(self) -> int
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]
```

#### python/datasets/conversations.json
Format des données:
```json
{
  "conversations": [
    {
      "messages": [
        {"role": "user", "content": "salut ça va?"},
        {"role": "assistant", "content": "yo ça roule et toi?"},
        {"role": "user", "content": "tranquille, tu fais quoi?"}
      ]
    }
  ]
}
```

## Data Models

### Message Model
```python
@dataclass
class Message:
    channel_id: str
    user: str
    content: str
    timestamp: datetime
    is_bot: bool
```

### ModelConfig
```python
@dataclass
class ModelConfig:
    vocab_size: int = 10000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1
```

### TrainingConfig
```python
@dataclass
class TrainingConfig:
    learning_rate: float = 0.0001
    batch_size: int = 32
    epochs: int = 10
    warmup_steps: int = 1000
    save_interval: int = 1000
```

## Error Handling

### Discord Bot Errors
- **Connection Errors**: Retry avec backoff exponentiel
- **Rate Limiting**: Respecter les limites Discord API
- **Invalid Token**: Afficher message d'erreur clair et arrêter

### Model Errors
- **Out of Memory**: Réduire batch size automatiquement
- **Model Not Found**: Afficher chemin attendu et instructions
- **Generation Timeout**: Limiter le nombre de tokens générés

### Memory Errors
- **Disk Full**: Logger erreur et continuer en mémoire seulement
- **Corrupted Data**: Réinitialiser la mémoire pour ce canal
- **Context Overflow**: Tronquer automatiquement les vieux messages

## Testing Strategy

### Unit Tests (`python/tests/`)

#### test_tokenizer.py
- Test encoding/decoding
- Test vocabulaire
- Test cas spéciaux (emojis, caractères spéciaux)

#### test_model.py
- Test forward pass
- Test génération de texte
- Test sauvegarde/chargement

#### test_transformer.py
- Test attention mechanism
- Test feed-forward
- Test layer normalization

#### test_memory.py
- Test ajout de messages
- Test récupération de contexte
- Test persistence

### Integration Tests (`python/tests/integration/`)

#### test_bot_integration.py
- Test connexion Discord (mock)
- Test pipeline complet: message → réponse
- Test inactivity monitor

### Examples (`python/exemples/`)

#### exemple_training.py
Script d'exemple pour entraîner le modèle:
```python
# Charger dataset
# Initialiser modèle
# Entraîner
# Sauvegarder
```

#### exemple_generation.py
Script d'exemple pour tester la génération:
```python
# Charger modèle
# Générer texte à partir d'un prompt
# Afficher résultat
```

#### exemple_conversation.py
Script d'exemple pour tester une conversation:
```python
# Charger modèle et mémoire
# Simuler conversation
# Afficher historique
```

## Deployment and Configuration

### Environment Variables
```
DISCORD_TOKEN=your_token_here
MODEL_PATH=./models/gpt_model.pkl
MEMORY_PATH=./data/memory.json
RESPONSE_PROBABILITY=0.3
```

### File Structure
```
.
├── bot/
│   ├── main.py
│   ├── inactivity_monitor.py
│   └── config.py
├── python/
│   ├── model.py
│   ├── transformer.py
│   ├── tokenizer.py
│   ├── trainer.py
│   ├── memory.py
│   ├── datasets/
│   │   ├── dataset.py
│   │   └── conversations.json
│   ├── exemples/
│   │   ├── exemple_training.py
│   │   ├── exemple_generation.py
│   │   └── exemple_conversation.py
│   └── tests/
│       ├── test_tokenizer.py
│       ├── test_model.py
│       ├── test_transformer.py
│       ├── test_memory.py
│       └── integration/
│           └── test_bot_integration.py
├── docs/
│   ├── README.md
│   ├── TRAINING.md
│   └── DEPLOYMENT.md
├── models/
│   └── (saved model files)
├── data/
│   └── (memory and logs)
├── requirements.txt
└── .env
```

## Performance Considerations

### Model Size
- Commencer avec un modèle petit (d_model=256, n_layers=6)
- Peut être agrandi si nécessaire et si ressources disponibles

### Response Time
- Génération: ~1-3 secondes pour 50 tokens
- Utiliser cache pour les embeddings fréquents
- Limiter max_length pour éviter timeouts Discord

### Memory Management
- Limiter contexte à 10 derniers messages par défaut
- Nettoyer vieux canaux inactifs périodiquement
- Sauvegarder sur disque toutes les 5 minutes

## Security Considerations

- Token Discord dans variable d'environnement (jamais dans le code)
- Valider tous les inputs utilisateur
- Limiter la longueur des messages générés
- Filtrer contenu inapproprié si nécessaire
- Rate limiting pour éviter spam
