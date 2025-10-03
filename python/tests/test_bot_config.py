"""
Tests for bot configuration.
"""

import os
import sys
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from bot.config import BotConfig


def test_bot_config_from_env_with_valid_token(monkeypatch):
    """Test loading configuration from environment variables."""
    monkeypatch.setenv('DISCORD_TOKEN', 'test_token_123456789')
    
    config = BotConfig.from_env()
    
    assert config.discord_token == 'test_token_123456789'
    assert config.model_path == './models/gpt_model.pkl'
    assert config.vocab_path == './models/vocab.json'
    assert config.memory_path == './data/memory.json'
    assert config.response_probability == 0.3
    assert config.inactivity_timeout_minutes == 10
    assert config.max_context_length == 10


def test_bot_config_missing_token(monkeypatch):
    """Test that missing token raises ValueError."""
    monkeypatch.delenv('DISCORD_TOKEN', raising=False)
    
    with pytest.raises(ValueError, match="DISCORD_TOKEN est requis"):
        BotConfig.from_env()


def test_bot_config_custom_values(monkeypatch):
    """Test loading custom configuration values."""
    monkeypatch.setenv('DISCORD_TOKEN', 'custom_token')
    monkeypatch.setenv('MODEL_PATH', './custom/model.pkl')
    monkeypatch.setenv('VOCAB_PATH', './custom/vocab.json')
    monkeypatch.setenv('RESPONSE_PROBABILITY', '0.5')
    monkeypatch.setenv('INACTIVITY_TIMEOUT_MINUTES', '15')
    monkeypatch.setenv('MAX_CONTEXT_LENGTH', '20')
    
    config = BotConfig.from_env()
    
    assert config.model_path == './custom/model.pkl'
    assert config.vocab_path == './custom/vocab.json'
    assert config.response_probability == 0.5
    assert config.inactivity_timeout_minutes == 15
    assert config.max_context_length == 20


def test_bot_config_invalid_probability(monkeypatch):
    """Test that invalid probability raises ValueError."""
    monkeypatch.setenv('DISCORD_TOKEN', 'test_token')
    monkeypatch.setenv('RESPONSE_PROBABILITY', '1.5')
    
    with pytest.raises(ValueError, match="RESPONSE_PROBABILITY"):
        BotConfig.from_env()


def test_bot_config_invalid_timeout(monkeypatch):
    """Test that invalid timeout raises ValueError."""
    monkeypatch.setenv('DISCORD_TOKEN', 'test_token')
    monkeypatch.setenv('INACTIVITY_TIMEOUT_MINUTES', '-5')
    
    with pytest.raises(ValueError, match="INACTIVITY_TIMEOUT_MINUTES"):
        BotConfig.from_env()


def test_bot_config_validate():
    """Test configuration validation."""
    config = BotConfig(
        discord_token='valid_token_123',
        model_path='./models/model.pkl',
        vocab_path='./models/vocab.json',
        memory_path='./data/memory.json',
        response_probability=0.3,
        inactivity_timeout_minutes=10,
        max_context_length=10
    )
    
    # Should not raise
    config.validate()


def test_bot_config_validate_short_token():
    """Test that short token fails validation."""
    config = BotConfig(
        discord_token='short',
        model_path='./models/model.pkl',
        vocab_path='./models/vocab.json',
        memory_path='./data/memory.json',
        response_probability=0.3,
        inactivity_timeout_minutes=10,
        max_context_length=10
    )
    
    with pytest.raises(ValueError, match="DISCORD_TOKEN semble invalide"):
        config.validate()
