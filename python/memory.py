"""
Système de mémoire conversationnelle pour le bot Discord.
Gère l'historique des conversations par canal avec fenêtre de contexte.
"""

from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import json
import os


@dataclass
class Message:
    """Représente un message dans l'historique."""
    channel_id: str
    user: str
    content: str
    timestamp: str
    is_bot: bool


class ConversationMemory:
    """
    Gère la mémoire conversationnelle du bot.
    
    Stocke l'historique des messages par canal et gère la fenêtre de contexte
    pour limiter le nombre de messages conservés.
    """
    
    def __init__(self, max_context_length: int = 10, storage_path: Optional[str] = None):
        """
        Initialise le système de mémoire.
        
        Args:
            max_context_length: Nombre maximum de messages à conserver par canal
            storage_path: Chemin du fichier JSON pour la persistence
        """
        self.max_context_length = max_context_length
        self.storage_path = storage_path
        self.conversations: Dict[str, List[Message]] = {}
    
    def add_message(self, channel_id: str, user: str, message: str, is_bot: bool = False):
        """
        Ajoute un message à l'historique d'un canal.
        
        Args:
            channel_id: ID du canal Discord
            user: Nom de l'utilisateur
            message: Contenu du message
            is_bot: True si le message vient du bot
        """
        # Créer le canal s'il n'existe pas
        if channel_id not in self.conversations:
            self.conversations[channel_id] = []
        
        # Créer le message
        msg = Message(
            channel_id=channel_id,
            user=user,
            content=message,
            timestamp=datetime.now().isoformat(),
            is_bot=is_bot
        )
        
        # Ajouter le message
        self.conversations[channel_id].append(msg)
        
        # Gérer la fenêtre de contexte
        if len(self.conversations[channel_id]) > self.max_context_length:
            self.conversations[channel_id] = self.conversations[channel_id][-self.max_context_length:]
    
    def get_context(self, channel_id: str) -> List[Dict[str, str]]:
        """
        Récupère l'historique des messages pour un canal.
        
        Args:
            channel_id: ID du canal Discord
            
        Returns:
            Liste de dictionnaires contenant les messages
        """
        if channel_id not in self.conversations:
            return []
        
        # Convertir les messages en dictionnaires
        return [asdict(msg) for msg in self.conversations[channel_id]]
    
    def clear_context(self, channel_id: str):
        """
        Efface l'historique d'un canal.
        
        Args:
            channel_id: ID du canal Discord
        """
        if channel_id in self.conversations:
            del self.conversations[channel_id]
    
    def save_to_disk(self):
        """
        Sauvegarde la mémoire sur disque en JSON.
        
        Raises:
            ValueError: Si storage_path n'est pas défini
            IOError: Si l'écriture échoue
        """
        if not self.storage_path:
            raise ValueError("storage_path n'est pas défini")
        
        try:
            # Créer le dossier parent si nécessaire
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            # Convertir les conversations en format sérialisable
            data = {
                "max_context_length": self.max_context_length,
                "conversations": {
                    channel_id: [asdict(msg) for msg in messages]
                    for channel_id, messages in self.conversations.items()
                }
            }
            
            # Écrire dans le fichier
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            raise IOError(f"Erreur lors de la sauvegarde: {e}")
    
    def load_from_disk(self):
        """
        Charge la mémoire depuis le disque.
        
        Raises:
            ValueError: Si storage_path n'est pas défini
            IOError: Si la lecture échoue ou si le fichier est corrompu
        """
        if not self.storage_path:
            raise ValueError("storage_path n'est pas défini")
        
        if not os.path.exists(self.storage_path):
            # Fichier n'existe pas encore, c'est normal
            return
        
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Valider la structure
            if not isinstance(data, dict) or "conversations" not in data:
                raise ValueError("Structure de fichier invalide")
            
            # Charger max_context_length si présent
            if "max_context_length" in data:
                self.max_context_length = data["max_context_length"]
            
            # Reconstruire les conversations
            self.conversations = {}
            for channel_id, messages in data["conversations"].items():
                self.conversations[channel_id] = [
                    Message(**msg) for msg in messages
                ]
                
        except json.JSONDecodeError as e:
            raise IOError(f"Fichier JSON corrompu: {e}")
        except Exception as e:
            raise IOError(f"Erreur lors du chargement: {e}")
