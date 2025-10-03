"""
Tests unitaires pour le système de mémoire conversationnelle.
"""

import pytest
import json
import os
import tempfile
from python.memory import ConversationMemory, Message


class TestConversationMemory:
    """Tests pour la classe ConversationMemory."""
    
    def test_init(self):
        """Test l'initialisation de la mémoire."""
        memory = ConversationMemory(max_context_length=5)
        assert memory.max_context_length == 5
        assert memory.conversations == {}
    
    def test_add_message(self):
        """Test l'ajout d'un message."""
        memory = ConversationMemory()
        memory.add_message("channel1", "user1", "Hello!", is_bot=False)
        
        context = memory.get_context("channel1")
        assert len(context) == 1
        assert context[0]["user"] == "user1"
        assert context[0]["content"] == "Hello!"
        assert context[0]["is_bot"] is False
    
    def test_add_multiple_messages(self):
        """Test l'ajout de plusieurs messages."""
        memory = ConversationMemory()
        memory.add_message("channel1", "user1", "Message 1")
        memory.add_message("channel1", "user2", "Message 2")
        memory.add_message("channel1", "bot", "Message 3", is_bot=True)
        
        context = memory.get_context("channel1")
        assert len(context) == 3
        assert context[0]["content"] == "Message 1"
        assert context[1]["content"] == "Message 2"
        assert context[2]["content"] == "Message 3"
        assert context[2]["is_bot"] is True
    
    def test_multiple_channels(self):
        """Test la gestion de plusieurs canaux."""
        memory = ConversationMemory()
        memory.add_message("channel1", "user1", "Channel 1 message")
        memory.add_message("channel2", "user2", "Channel 2 message")
        
        context1 = memory.get_context("channel1")
        context2 = memory.get_context("channel2")
        
        assert len(context1) == 1
        assert len(context2) == 1
        assert context1[0]["content"] == "Channel 1 message"
        assert context2[0]["content"] == "Channel 2 message"
    
    def test_context_window_limit(self):
        """Test la limitation de la fenêtre de contexte."""
        memory = ConversationMemory(max_context_length=3)
        
        # Ajouter 5 messages
        for i in range(5):
            memory.add_message("channel1", f"user{i}", f"Message {i}")
        
        context = memory.get_context("channel1")
        
        # Seulement les 3 derniers messages doivent être conservés
        assert len(context) == 3
        assert context[0]["content"] == "Message 2"
        assert context[1]["content"] == "Message 3"
        assert context[2]["content"] == "Message 4"
    
    def test_get_context_empty_channel(self):
        """Test la récupération de contexte pour un canal vide."""
        memory = ConversationMemory()
        context = memory.get_context("nonexistent_channel")
        assert context == []
    
    def test_clear_context(self):
        """Test l'effacement du contexte d'un canal."""
        memory = ConversationMemory()
        memory.add_message("channel1", "user1", "Message 1")
        memory.add_message("channel1", "user2", "Message 2")
        
        assert len(memory.get_context("channel1")) == 2
        
        memory.clear_context("channel1")
        assert len(memory.get_context("channel1")) == 0
    
    def test_clear_nonexistent_channel(self):
        """Test l'effacement d'un canal qui n'existe pas."""
        memory = ConversationMemory()
        # Ne devrait pas lever d'erreur
        memory.clear_context("nonexistent_channel")
    
    def test_save_to_disk(self):
        """Test la sauvegarde sur disque."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            memory = ConversationMemory(max_context_length=5, storage_path=temp_path)
            memory.add_message("channel1", "user1", "Test message")
            memory.add_message("channel2", "user2", "Another message")
            
            memory.save_to_disk()
            
            # Vérifier que le fichier existe et contient les bonnes données
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert "conversations" in data
            assert "channel1" in data["conversations"]
            assert "channel2" in data["conversations"]
            assert len(data["conversations"]["channel1"]) == 1
            assert data["conversations"]["channel1"][0]["content"] == "Test message"
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_save_without_storage_path(self):
        """Test la sauvegarde sans storage_path défini."""
        memory = ConversationMemory()
        
        with pytest.raises(ValueError, match="storage_path n'est pas défini"):
            memory.save_to_disk()
    
    def test_load_from_disk(self):
        """Test le chargement depuis le disque."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            # Créer et sauvegarder une mémoire
            memory1 = ConversationMemory(max_context_length=5, storage_path=temp_path)
            memory1.add_message("channel1", "user1", "Saved message")
            memory1.save_to_disk()
            
            # Charger dans une nouvelle instance
            memory2 = ConversationMemory(storage_path=temp_path)
            memory2.load_from_disk()
            
            context = memory2.get_context("channel1")
            assert len(context) == 1
            assert context[0]["content"] == "Saved message"
            assert memory2.max_context_length == 5
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_load_nonexistent_file(self):
        """Test le chargement d'un fichier qui n'existe pas."""
        memory = ConversationMemory(storage_path="/nonexistent/path/file.json")
        # Ne devrait pas lever d'erreur
        memory.load_from_disk()
        assert memory.conversations == {}
    
    def test_load_without_storage_path(self):
        """Test le chargement sans storage_path défini."""
        memory = ConversationMemory()
        
        with pytest.raises(ValueError, match="storage_path n'est pas défini"):
            memory.load_from_disk()
    
    def test_load_corrupted_file(self):
        """Test le chargement d'un fichier JSON corrompu."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            f.write("{ invalid json }")
            temp_path = f.name
        
        try:
            memory = ConversationMemory(storage_path=temp_path)
            
            with pytest.raises(IOError, match="Fichier JSON corrompu"):
                memory.load_from_disk()
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_load_invalid_structure(self):
        """Test le chargement d'un fichier avec structure invalide."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump({"invalid": "structure"}, f)
            temp_path = f.name
        
        try:
            memory = ConversationMemory(storage_path=temp_path)
            
            with pytest.raises(IOError, match="Structure de fichier invalide"):
                memory.load_from_disk()
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_persistence_roundtrip(self):
        """Test un cycle complet sauvegarde/chargement."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            # Créer une mémoire avec plusieurs messages
            memory1 = ConversationMemory(max_context_length=10, storage_path=temp_path)
            memory1.add_message("channel1", "user1", "Message 1")
            memory1.add_message("channel1", "user2", "Message 2")
            memory1.add_message("channel2", "user3", "Message 3", is_bot=True)
            memory1.save_to_disk()
            
            # Charger dans une nouvelle instance
            memory2 = ConversationMemory(storage_path=temp_path)
            memory2.load_from_disk()
            
            # Vérifier que tout est identique
            context1 = memory2.get_context("channel1")
            context2 = memory2.get_context("channel2")
            
            assert len(context1) == 2
            assert len(context2) == 1
            assert context1[0]["content"] == "Message 1"
            assert context1[1]["content"] == "Message 2"
            assert context2[0]["content"] == "Message 3"
            assert context2[0]["is_bot"] is True
            assert memory2.max_context_length == 10
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
