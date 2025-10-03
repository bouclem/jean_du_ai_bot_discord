"""
Moniteur d'inactivité pour le bot Discord.
Surveille l'activité des canaux et envoie des messages de solitude après un timeout.
"""

import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    import discord


class InactivityMonitor:
    """
    Surveille l'inactivité des canaux Discord et déclenche des messages
    après une période d'inactivité définie.
    """

    def __init__(self, bot, timeout_minutes: int = 10):
        """
        Initialise le moniteur d'inactivité.

        Args:
            bot: Instance du bot Discord (JeanDeAIBot)
            timeout_minutes: Durée d'inactivité en minutes avant d'envoyer un message (défaut: 10)
        """
        self.bot = bot
        self.timeout_minutes = timeout_minutes
        self.last_activity: Dict[str, datetime] = {}
        self._running = False
        self._task = None

    def update_activity(self, channel_id: str) -> None:
        """
        Met à jour le timestamp de la dernière activité pour un canal.

        Args:
            channel_id: ID du canal Discord
        """
        self.last_activity[channel_id] = datetime.now()

    async def check_inactivity(self) -> None:
        """
        Vérifie périodiquement l'inactivité des canaux et envoie des messages
        de solitude si le timeout est dépassé.
        
        Cette méthode doit être lancée comme une tâche asynchrone.
        """
        self._running = True
        
        while self._running:
            try:
                current_time = datetime.now()
                timeout_delta = timedelta(minutes=self.timeout_minutes)
                
                # Vérifier chaque canal pour l'inactivité
                for channel_id, last_time in list(self.last_activity.items()):
                    time_since_activity = current_time - last_time
                    
                    if time_since_activity >= timeout_delta:
                        # Le canal est inactif depuis trop longtemps
                        await self._send_lonely_message(channel_id)
                        
                        # Mettre à jour le timestamp pour éviter de spammer
                        self.update_activity(channel_id)
                
                # Attendre 1 minute avant la prochaine vérification
                await asyncio.sleep(60)
                
            except Exception as e:
                print(f"Erreur dans check_inactivity: {e}")
                await asyncio.sleep(60)

    async def _send_lonely_message(self, channel_id: str) -> None:
        """
        Envoie un message de solitude dans le canal spécifié.

        Args:
            channel_id: ID du canal Discord
        """
        try:
            channel = self.bot.get_channel(int(channel_id))
            if channel:
                message = self.get_lonely_message()
                await channel.send(message)
        except Exception as e:
            print(f"Erreur lors de l'envoi du message de solitude: {e}")

    def get_lonely_message(self) -> str:
        """
        Retourne un message de solitude aléatoire parmi plusieurs variantes.

        Returns:
            str: Message de solitude
        """
        messages = [
            "hello? ya quelqu'un? jsuis tous seul j'ai peur",
            "bon bah j'vais parler tout seul alors...",
            "c'est mort ici ou quoi?",
            "vous êtes où les gens?",
            "allo? y'a quelqu'un?",
            "jsuis abandonné ou quoi",
            "silence radio... ça fait peur",
            "bon ok j'ai compris, personne veut me parler",
            "vous me ghostez là non?",
            "j'commence à m'ennuyer tout seul ici"
        ]
        return random.choice(messages)

    def start(self) -> None:
        """
        Démarre le moniteur d'inactivité en créant une tâche asynchrone.
        """
        if not self._running:
            self._task = asyncio.create_task(self.check_inactivity())

    def stop(self) -> None:
        """
        Arrête le moniteur d'inactivité.
        """
        self._running = False
        if self._task:
            self._task.cancel()
