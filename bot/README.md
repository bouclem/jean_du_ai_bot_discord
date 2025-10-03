# Jean_De_AI Discord Bot

Discord bot implementation with GPT-based conversation capabilities.

## Components

### bot/config.py
Configuration management for the bot:
- Loads environment variables
- Validates configuration
- Provides default values
- Clear error messages for invalid config

### bot/main.py
Main bot implementation:
- `JeanDeAIBot` class that integrates Discord client, GPT model, and conversation memory
- Message detection and filtering
- Response decision logic (mentions + probability-based spontaneous responses)
- Response generation using GPT with conversation context
- Error handling for Discord API (rate limiting, permissions)

### bot/inactivity_monitor.py
Monitors channel inactivity and sends "lonely" messages after timeout.

## Usage

1. Set up environment variables (see `.env.example`):
```bash
DISCORD_TOKEN=your_token_here
MODEL_PATH=./models/gpt_model.pkl
VOCAB_PATH=./models/vocab.json
MEMORY_PATH=./data/memory.json
RESPONSE_PROBABILITY=0.3
```

2. Run the bot:
```python
from bot.config import BotConfig
from bot.main import JeanDeAIBot

# Load configuration
config = BotConfig.from_env()
config.validate()

# Create and run bot
bot = JeanDeAIBot(config)
bot.run()
```

## Features

- ✅ Connects to Discord with proper intents
- ✅ Detects and processes messages
- ✅ Filters out bot messages
- ✅ Responds when mentioned
- ✅ Spontaneous responses based on probability
- ✅ Uses conversation context from memory
- ✅ Generates responses with GPT model
- ✅ Handles Discord API errors gracefully
- ✅ Saves conversation memory to disk

## Testing

Run tests with:
```bash
pytest python/tests/test_bot_config.py -v
pytest python/tests/test_bot_integration.py -v
```

All 15 tests pass successfully.
