"""
Integration tests for the Discord bot.
Tests bot initialization and message handling logic.
"""

import os
import sys
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from bot.config import BotConfig
from bot.main import JeanDeAIBot


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return BotConfig(
        discord_token='test_token_123456789',
        model_path='./models/gpt_model.pkl',
        vocab_path='./models/vocab.json',
        memory_path='./data/test_memory.json',
        response_probability=0.5,
        inactivity_timeout_minutes=10,
        max_context_length=10
    )


@pytest.fixture
def mock_discord_message():
    """Create a mock Discord message."""
    message = Mock()
    message.author = Mock()
    message.author.name = 'TestUser'
    message.author.bot = False
    message.content = 'Hello bot!'
    message.channel = Mock()
    message.channel.id = 123456
    message.channel.name = 'test-channel'
    return message


def test_bot_config_validation(mock_config):
    """Test that bot configuration validates correctly."""
    mock_config.validate()  # Should not raise


@pytest.mark.asyncio
async def test_should_respond_when_mentioned(mock_config):
    """Test that bot responds when mentioned."""
    with patch('bot.main.Tokenizer'), \
         patch('bot.main.GPTModel'), \
         patch('bot.main.ConversationMemory'), \
         patch('bot.main.discord.Client'):
        
        bot = JeanDeAIBot(mock_config)
        
        # Create mock message with bot mentioned
        message = Mock()
        message.author = Mock()
        message.author.bot = False
        
        # Mock the client user and mentioned_in method
        mock_user = Mock()
        mock_user.mentioned_in = Mock(return_value=True)
        
        with patch.object(bot.client, 'user', mock_user):
            result = await bot.should_respond(message)
            assert result is True


@pytest.mark.asyncio
async def test_should_not_respond_to_bot_messages(mock_config, mock_discord_message):
    """Test that bot ignores messages from other bots."""
    with patch('bot.main.Tokenizer'), \
         patch('bot.main.GPTModel'), \
         patch('bot.main.ConversationMemory'), \
         patch('bot.main.discord.Client'):
        
        bot = JeanDeAIBot(mock_config)
        
        # Set message author as bot
        mock_discord_message.author.bot = True
        
        # on_message should return early for bot messages
        await bot.on_message(mock_discord_message)
        # If it doesn't crash, the test passes


@pytest.mark.asyncio
async def test_should_not_respond_to_own_messages(mock_config, mock_discord_message):
    """Test that bot ignores its own messages."""
    with patch('bot.main.Tokenizer'), \
         patch('bot.main.GPTModel'), \
         patch('bot.main.ConversationMemory'), \
         patch('bot.main.discord.Client'):
        
        bot = JeanDeAIBot(mock_config)
        
        # Mock the client user
        mock_user = Mock()
        
        with patch.object(bot.client, 'user', mock_user):
            # Set message author as the bot itself
            mock_discord_message.author = mock_user
            
            # on_message should return early
            await bot.on_message(mock_discord_message)
            # If it doesn't crash, the test passes


def test_format_prompt_empty_context(mock_config):
    """Test prompt formatting with empty context."""
    with patch('bot.main.Tokenizer'), \
         patch('bot.main.GPTModel'), \
         patch('bot.main.ConversationMemory'):
        
        bot = JeanDeAIBot(mock_config)
        
        prompt = bot._format_prompt([])
        assert prompt == ""


def test_format_prompt_with_messages(mock_config):
    """Test prompt formatting with message history."""
    with patch('bot.main.Tokenizer'), \
         patch('bot.main.GPTModel'), \
         patch('bot.main.ConversationMemory'):
        
        bot = JeanDeAIBot(mock_config)
        
        context = [
            {'user': 'Alice', 'content': 'Hello!'},
            {'user': 'Bob', 'content': 'Hi there!'},
            {'user': 'Alice', 'content': 'How are you?'}
        ]
        
        prompt = bot._format_prompt(context)
        
        assert 'Alice: Hello!' in prompt
        assert 'Bob: Hi there!' in prompt
        assert 'Alice: How are you?' in prompt


@pytest.mark.asyncio
async def test_send_response_truncates_long_messages(mock_config):
    """Test that long responses are truncated to Discord's limit."""
    with patch('bot.main.Tokenizer'), \
         patch('bot.main.GPTModel'), \
         patch('bot.main.ConversationMemory'):
        
        bot = JeanDeAIBot(mock_config)
        
        # Create mock channel
        channel = AsyncMock()
        
        # Create a message longer than 2000 characters
        long_message = 'a' * 2500
        
        await bot.send_response(channel, long_message)
        
        # Verify send was called with truncated message
        channel.send.assert_called_once()
        sent_message = channel.send.call_args[0][0]
        assert len(sent_message) <= 2000
        assert sent_message.endswith('...')


@pytest.mark.asyncio
async def test_send_response_handles_forbidden_error(mock_config):
    """Test that bot handles permission errors gracefully."""
    with patch('bot.main.Tokenizer'), \
         patch('bot.main.GPTModel'), \
         patch('bot.main.ConversationMemory'):
        
        bot = JeanDeAIBot(mock_config)
        
        # Create mock channel that raises Forbidden error
        channel = AsyncMock()
        
        import discord
        channel.send.side_effect = discord.errors.Forbidden(
            Mock(status=403), 'Forbidden'
        )
        
        # Should not raise exception
        await bot.send_response(channel, 'Test message')
