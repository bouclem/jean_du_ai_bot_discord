"""
Tests unitaires pour le moniteur d'inactivité.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from bot.inactivity_monitor import InactivityMonitor


class TestInactivityMonitor:
    """Tests pour la classe InactivityMonitor"""

    def test_init(self):
        """Test l'initialisation du moniteur"""
        bot = Mock()
        monitor = InactivityMonitor(bot, timeout_minutes=5)
        
        assert monitor.bot == bot
        assert monitor.timeout_minutes == 5
        assert monitor.last_activity == {}
        assert monitor._running is False

    def test_update_activity(self):
        """Test la mise à jour de l'activité d'un canal"""
        bot = Mock()
        monitor = InactivityMonitor(bot)
        
        channel_id = "123456789"
        before = datetime.now()
        monitor.update_activity(channel_id)
        after = datetime.now()
        
        assert channel_id in monitor.last_activity
        assert before <= monitor.last_activity[channel_id] <= after

    def test_get_lonely_message(self):
        """Test la génération de messages de solitude"""
        bot = Mock()
        monitor = InactivityMonitor(bot)
        
        # Générer plusieurs messages pour vérifier la variété
        messages = [monitor.get_lonely_message() for _ in range(10)]
        
        # Vérifier qu'on obtient des strings non vides
        assert all(isinstance(msg, str) for msg in messages)
        assert all(len(msg) > 0 for msg in messages)
        
        # Vérifier qu'il y a de la variété (au moins 2 messages différents sur 10)
        assert len(set(messages)) >= 2

    @pytest.mark.asyncio
    async def test_send_lonely_message(self):
        """Test l'envoi d'un message de solitude"""
        bot = Mock()
        channel = AsyncMock()
        bot.get_channel = Mock(return_value=channel)
        
        monitor = InactivityMonitor(bot)
        
        await monitor._send_lonely_message("123456789")
        
        # Vérifier que get_channel a été appelé
        bot.get_channel.assert_called_once_with(123456789)
        
        # Vérifier que send a été appelé sur le canal
        channel.send.assert_called_once()
        
        # Vérifier que le message est une string
        call_args = channel.send.call_args[0]
        assert isinstance(call_args[0], str)

    @pytest.mark.asyncio
    async def test_check_inactivity_timeout(self):
        """Test la détection d'inactivité après timeout"""
        bot = Mock()
        channel = AsyncMock()
        bot.get_channel = Mock(return_value=channel)
        
        monitor = InactivityMonitor(bot, timeout_minutes=0)  # Timeout immédiat pour le test
        
        # Simuler une activité ancienne
        channel_id = "123456789"
        monitor.last_activity[channel_id] = datetime.now() - timedelta(minutes=1)
        
        # Lancer check_inactivity dans une tâche
        task = asyncio.create_task(monitor.check_inactivity())
        
        # Attendre un peu pour que la vérification se fasse
        # (le check_inactivity fait une itération puis sleep 60s)
        await asyncio.sleep(0.2)
        
        # Arrêter le moniteur
        monitor.stop()
        
        # Annuler la tâche proprement
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Vérifier qu'un message a été envoyé
        assert channel.send.called

    def test_multiple_channels(self):
        """Test le tracking de plusieurs canaux"""
        bot = Mock()
        monitor = InactivityMonitor(bot)
        
        channels = ["111", "222", "333"]
        
        for channel_id in channels:
            monitor.update_activity(channel_id)
        
        assert len(monitor.last_activity) == 3
        assert all(ch in monitor.last_activity for ch in channels)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
