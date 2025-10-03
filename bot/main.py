"""
Jean_De_AI Discord Bot - Main bot implementation.
Integrates GPT model, conversation memory, and Discord client.
"""

import discord
from discord.ext import commands
import os
import sys
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from python.model import GPTModel
from python.tokenizer import Tokenizer
from python.memory import ConversationMemory
from bot.config import BotConfig


class JeanDeAIBot:
    """
    Jean_De_AI Discord bot with GPT-based conversation capabilities.
    """
    
    def __init__(self, config: BotConfig):
        """
        Initialize the bot with configuration.
        
        Args:
            config: BotConfig instance with all settings
        """
        self.config = config
        
        # Initialize Discord client with intents
        intents = discord.Intents.default()
        intents.message_content = True  # Required to read message content
        intents.messages = True
        
        self.client = discord.Client(intents=intents)
        
        # Load tokenizer
        print(f"Chargement du vocabulaire depuis {config.vocab_path}...")
        self.tokenizer = Tokenizer(vocab_path=config.vocab_path)
        print(f"Vocabulaire chargé: {self.tokenizer.vocab_size} tokens")
        
        # Load GPT model
        print(f"Chargement du modèle depuis {config.model_path}...")
        self.model = GPTModel(
            vocab_size=self.tokenizer.vocab_size,
            d_model=256,
            n_heads=8,
            n_layers=6,
            d_ff=1024,
            max_seq_len=512,
            dropout=0.1
        )
        
        # Load model weights if file exists
        if os.path.exists(config.model_path):
            self.model.load(config.model_path)
            print("Modèle chargé avec succès!")
        else:
            print(f"ATTENTION: Fichier modèle non trouvé à {config.model_path}")
            print("Le bot utilisera un modèle non entraîné.")
        
        # Initialize conversation memory
        print(f"Initialisation de la mémoire (max {config.max_context_length} messages)...")
        self.memory = ConversationMemory(
            max_context_length=config.max_context_length,
            storage_path=config.memory_path
        )
        
        # Load memory from disk if exists
        try:
            self.memory.load_from_disk()
            print("Mémoire chargée depuis le disque")
        except Exception as e:
            print(f"Impossible de charger la mémoire: {e}")
            print("Démarrage avec une mémoire vide")
        
        # Register event handlers
        self._register_events()
    
    def _register_events(self):
        """Register Discord event handlers."""
        
        @self.client.event
        async def on_ready():
            """Called when the bot successfully connects to Discord."""
            await self.on_ready()
        
        @self.client.event
        async def on_message(message):
            """Called when a message is received."""
            await self.on_message(message)
    
    async def on_ready(self):
        """
        Handler called when bot connects to Discord.
        Confirms connection and displays bot information.
        """
        print("=" * 50)
        print(f"✓ Bot connecté en tant que: {self.client.user.name}")
        print(f"✓ ID: {self.client.user.id}")
        print(f"✓ Serveurs: {len(self.client.guilds)}")
        print("=" * 50)
        print("Jean_De_AI est prêt à converser!")
        print("=" * 50)
    
    async def on_message(self, message: discord.Message):
        """
        Handler called when a message is received.
        Detects messages, filters bot's own messages, and decides whether to respond.
        
        Args:
            message: Discord message object
        """
        # Ignore messages from the bot itself
        if message.author == self.client.user:
            return
        
        # Ignore messages from other bots
        if message.author.bot:
            return
        
        # Add message to memory
        channel_id = str(message.channel.id)
        user_name = message.author.name
        content = message.content
        
        self.memory.add_message(
            channel_id=channel_id,
            user=user_name,
            message=content,
            is_bot=False
        )
        
        # Decide if bot should respond
        should_respond = await self.should_respond(message)
        
        if should_respond:
            # Generate response
            response = await self.generate_response(message)
            
            if response:
                # Send response to Discord
                await self.send_response(message.channel, response)
    
    async def should_respond(self, message: discord.Message) -> bool:
        """
        Determine if the bot should respond to a message.
        
        The bot responds if:
        - It is mentioned directly
        - Random probability check passes (spontaneous response)
        
        Args:
            message: Discord message object
            
        Returns:
            True if bot should respond, False otherwise
        """
        # Always respond if bot is mentioned
        if self.client.user.mentioned_in(message):
            print(f"[{message.channel.name}] Mentionné par {message.author.name}")
            return True
        
        # Respond with configured probability (spontaneous response)
        if random.random() < self.config.response_probability:
            print(f"[{message.channel.name}] Réponse spontanée à {message.author.name}")
            return True
        
        return False
    
    async def generate_response(self, message: discord.Message) -> str:
        """
        Generate a response using the GPT model with conversation context.
        
        Args:
            message: Discord message object
            
        Returns:
            Generated response text, or empty string if generation fails
        """
        try:
            channel_id = str(message.channel.id)
            
            # Get conversation context from memory
            context = self.memory.get_context(channel_id)
            
            # Format prompt with conversation history
            prompt = self._format_prompt(context)
            
            print(f"Génération de réponse pour: {message.content[:50]}...")
            
            # Generate response using GPT model
            response = self.model.generate(
                prompt=prompt,
                tokenizer=self.tokenizer,
                max_length=100,
                temperature=0.8,
                method='sample'
            )
            
            # Clean up response (remove prompt if present)
            response = response.strip()
            
            # Add bot's response to memory
            if response:
                self.memory.add_message(
                    channel_id=channel_id,
                    user=self.client.user.name,
                    message=response,
                    is_bot=True
                )
                
                # Save memory to disk periodically
                try:
                    self.memory.save_to_disk()
                except Exception as e:
                    print(f"Erreur lors de la sauvegarde de la mémoire: {e}")
            
            return response
            
        except Exception as e:
            print(f"Erreur lors de la génération: {e}")
            return ""
    
    def _format_prompt(self, context: list) -> str:
        """
        Format conversation context into a prompt for the model.
        
        Args:
            context: List of message dictionaries from memory
            
        Returns:
            Formatted prompt string
        """
        if not context:
            return ""
        
        # Build conversation history
        lines = []
        for msg in context[-5:]:  # Use last 5 messages for prompt
            user = msg['user']
            content = msg['content']
            lines.append(f"{user}: {content}")
        
        # Join with newlines
        prompt = "\n".join(lines)
        
        return prompt
    
    async def send_response(self, channel: discord.TextChannel, response: str):
        """
        Send a response message to a Discord channel.
        Handles errors like rate limiting and permission issues.
        
        Args:
            channel: Discord channel to send message to
            response: Response text to send
        """
        try:
            # Limit response length to Discord's limit (2000 characters)
            if len(response) > 2000:
                response = response[:1997] + "..."
            
            # Send the message
            await channel.send(response)
            print(f"✓ Réponse envoyée dans #{channel.name}")
            
        except discord.errors.Forbidden:
            print(f"✗ Erreur: Pas de permission pour envoyer dans #{channel.name}")
        
        except discord.errors.HTTPException as e:
            if e.status == 429:  # Rate limit
                print(f"✗ Rate limit atteint. Retry après {e.retry_after}s")
            else:
                print(f"✗ Erreur HTTP lors de l'envoi: {e}")
        
        except Exception as e:
            print(f"✗ Erreur inattendue lors de l'envoi: {e}")
    
    def run(self):
        """
        Start the Discord bot.
        
        Raises:
            discord.LoginFailure: If token is invalid
            discord.HTTPException: If connection fails
        """
        print("Démarrage de Jean_De_AI...")
        print(f"Configuration:")
        print(f"  - Probabilité de réponse: {self.config.response_probability}")
        print(f"  - Timeout d'inactivité: {self.config.inactivity_timeout_minutes} min")
        print(f"  - Contexte max: {self.config.max_context_length} messages")
        print()
        
        try:
            self.client.run(self.config.discord_token)
        except discord.LoginFailure:
            print("ERREUR: Token Discord invalide!")
            print("Vérifiez votre variable d'environnement DISCORD_TOKEN")
            raise
        except Exception as e:
            print(f"ERREUR lors du démarrage: {e}")
            raise
