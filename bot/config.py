"""
Configuration management for the Discord bot.
Loads and validates environment variables with default values.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class BotConfig:
    """Configuration for Jean_De_AI Discord bot."""
    
    discord_token: str
    model_path: str
    vocab_path: str
    memory_path: str
    response_probability: float
    inactivity_timeout_minutes: int
    max_context_length: int
    
    @classmethod
    def from_env(cls) -> 'BotConfig':
        """
        Load configuration from environment variables.
        
        Returns:
            BotConfig instance with loaded values
            
        Raises:
            ValueError: If required configuration is missing or invalid
        """
        # Load required variables
        discord_token = os.getenv('DISCORD_TOKEN')
        if not discord_token:
            raise ValueError(
                "DISCORD_TOKEN est requis. "
                "Veuillez définir la variable d'environnement DISCORD_TOKEN."
            )
        
        # Load optional variables with defaults
        model_path = os.getenv('MODEL_PATH', './models/gpt_model.pkl')
        vocab_path = os.getenv('VOCAB_PATH', './models/vocab.json')
        memory_path = os.getenv('MEMORY_PATH', './data/memory.json')
        
        # Load numeric configurations with validation
        try:
            response_probability = float(os.getenv('RESPONSE_PROBABILITY', '0.3'))
            if not 0.0 <= response_probability <= 1.0:
                raise ValueError("RESPONSE_PROBABILITY doit être entre 0.0 et 1.0")
        except ValueError as e:
            raise ValueError(f"RESPONSE_PROBABILITY invalide: {e}")
        
        try:
            inactivity_timeout_minutes = int(os.getenv('INACTIVITY_TIMEOUT_MINUTES', '10'))
            if inactivity_timeout_minutes <= 0:
                raise ValueError("INACTIVITY_TIMEOUT_MINUTES doit être positif")
        except ValueError as e:
            raise ValueError(f"INACTIVITY_TIMEOUT_MINUTES invalide: {e}")
        
        try:
            max_context_length = int(os.getenv('MAX_CONTEXT_LENGTH', '10'))
            if max_context_length <= 0:
                raise ValueError("MAX_CONTEXT_LENGTH doit être positif")
        except ValueError as e:
            raise ValueError(f"MAX_CONTEXT_LENGTH invalide: {e}")
        
        return cls(
            discord_token=discord_token,
            model_path=model_path,
            vocab_path=vocab_path,
            memory_path=memory_path,
            response_probability=response_probability,
            inactivity_timeout_minutes=inactivity_timeout_minutes,
            max_context_length=max_context_length
        )
    
    def validate(self) -> None:
        """
        Validate the configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate token
        if not self.discord_token or len(self.discord_token) < 10:
            raise ValueError("DISCORD_TOKEN semble invalide (trop court)")
        
        # Validate paths
        if not self.model_path:
            raise ValueError("MODEL_PATH ne peut pas être vide")
        
        if not self.vocab_path:
            raise ValueError("VOCAB_PATH ne peut pas être vide")
        
        if not self.memory_path:
            raise ValueError("MEMORY_PATH ne peut pas être vide")
        
        # Validate numeric values
        if not 0.0 <= self.response_probability <= 1.0:
            raise ValueError("response_probability doit être entre 0.0 et 1.0")
        
        if self.inactivity_timeout_minutes <= 0:
            raise ValueError("inactivity_timeout_minutes doit être positif")
        
        if self.max_context_length <= 0:
            raise ValueError("max_context_length doit être positif")
