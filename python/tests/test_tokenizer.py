"""
Unit tests for the Tokenizer class.
Tests encoding/decoding, special tokens, and vocabulary persistence.
"""
import unittest
import os
import tempfile
import json
from python.tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):
    """Test cases for Tokenizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = Tokenizer()
        self.sample_texts = [
            "bonjour",
            "salut ça va?",
            "hello world"
        ]
    
    def test_initialization(self):
        """Test tokenizer initialization with special tokens."""
        tokenizer = Tokenizer()
        
        # Check special tokens are present
        self.assertIn(Tokenizer.PAD_TOKEN, tokenizer.token_to_id)
        self.assertIn(Tokenizer.UNK_TOKEN, tokenizer.token_to_id)
        self.assertIn(Tokenizer.BOS_TOKEN, tokenizer.token_to_id)
        self.assertIn(Tokenizer.EOS_TOKEN, tokenizer.token_to_id)
        
        # Check special token IDs
        self.assertEqual(tokenizer.token_to_id[Tokenizer.PAD_TOKEN], 0)
        self.assertEqual(tokenizer.token_to_id[Tokenizer.UNK_TOKEN], 1)
        self.assertEqual(tokenizer.token_to_id[Tokenizer.BOS_TOKEN], 2)
        self.assertEqual(tokenizer.token_to_id[Tokenizer.EOS_TOKEN], 3)
    
    def test_build_vocab(self):
        """Test vocabulary building from texts."""
        tokenizer = Tokenizer()
        tokenizer.build_vocab(self.sample_texts)
        
        # Vocabulary should contain special tokens + unique characters
        self.assertGreater(tokenizer.vocab_size, 4)  # More than just special tokens
        
        # Check that common characters are in vocab
        self.assertIn('a', tokenizer.token_to_id)
        self.assertIn('o', tokenizer.token_to_id)
        self.assertIn(' ', tokenizer.token_to_id)
    
    def test_encode_basic(self):
        """Test basic text encoding."""
        tokenizer = Tokenizer()
        tokenizer.build_vocab(["hello"])
        
        # Encode with special tokens
        token_ids = tokenizer.encode("hello", add_special_tokens=True)
        
        # Should have BOS + characters + EOS
        self.assertGreater(len(token_ids), 2)
        self.assertEqual(token_ids[0], tokenizer.get_bos_token_id())
        self.assertEqual(token_ids[-1], tokenizer.get_eos_token_id())
    
    def test_encode_without_special_tokens(self):
        """Test encoding without special tokens."""
        tokenizer = Tokenizer()
        tokenizer.build_vocab(["hi"])
        
        token_ids = tokenizer.encode("hi", add_special_tokens=False)
        
        # Should not contain BOS or EOS
        self.assertNotIn(tokenizer.get_bos_token_id(), token_ids)
        self.assertNotIn(tokenizer.get_eos_token_id(), token_ids)
    
    def test_decode_basic(self):
        """Test basic token decoding."""
        tokenizer = Tokenizer()
        tokenizer.build_vocab(["hello"])
        
        # Encode then decode
        original_text = "hello"
        token_ids = tokenizer.encode(original_text, add_special_tokens=True)
        decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        self.assertEqual(decoded_text, original_text)
    
    def test_decode_with_special_tokens(self):
        """Test decoding with special tokens included."""
        tokenizer = Tokenizer()
        tokenizer.build_vocab(["test"])
        
        token_ids = tokenizer.encode("test", add_special_tokens=True)
        decoded_text = tokenizer.decode(token_ids, skip_special_tokens=False)
        
        # Should contain special tokens in output
        self.assertIn(Tokenizer.BOS_TOKEN, decoded_text)
        self.assertIn(Tokenizer.EOS_TOKEN, decoded_text)
    
    def test_unknown_token_handling(self):
        """Test handling of unknown tokens."""
        tokenizer = Tokenizer()
        tokenizer.build_vocab(["abc"])
        
        # Encode text with unknown character
        token_ids = tokenizer.encode("xyz", add_special_tokens=False)
        
        # All unknown characters should map to UNK token
        unk_id = tokenizer.get_unk_token_id()
        for token_id in token_ids:
            self.assertEqual(token_id, unk_id)
    
    def test_special_token_ids(self):
        """Test special token ID getters."""
        tokenizer = Tokenizer()
        
        self.assertEqual(tokenizer.get_pad_token_id(), 0)
        self.assertEqual(tokenizer.get_unk_token_id(), 1)
        self.assertEqual(tokenizer.get_bos_token_id(), 2)
        self.assertEqual(tokenizer.get_eos_token_id(), 3)
    
    def test_vocab_size(self):
        """Test vocabulary size property."""
        tokenizer = Tokenizer()
        
        # Initially should have 4 special tokens
        self.assertEqual(tokenizer.vocab_size, 4)
        
        # After building vocab, should increase
        tokenizer.build_vocab(["hello"])
        self.assertGreater(tokenizer.vocab_size, 4)
    
    def test_save_vocab(self):
        """Test vocabulary saving to file."""
        tokenizer = Tokenizer()
        tokenizer.build_vocab(self.sample_texts)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            # Save vocabulary
            tokenizer.save_vocab(temp_path)
            
            # Check file exists and is valid JSON
            self.assertTrue(os.path.exists(temp_path))
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            self.assertIn('token_to_id', vocab_data)
            self.assertIn('id_to_token', vocab_data)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_load_vocab(self):
        """Test vocabulary loading from file."""
        tokenizer = Tokenizer()
        tokenizer.build_vocab(self.sample_texts)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            # Save vocabulary
            original_vocab_size = tokenizer.vocab_size
            tokenizer.save_vocab(temp_path)
            
            # Create new tokenizer and load vocab
            new_tokenizer = Tokenizer()
            new_tokenizer.load_vocab(temp_path)
            
            # Check vocabulary matches
            self.assertEqual(new_tokenizer.vocab_size, original_vocab_size)
            self.assertEqual(new_tokenizer.token_to_id, tokenizer.token_to_id)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_save_load_roundtrip(self):
        """Test that save/load preserves functionality."""
        tokenizer = Tokenizer()
        tokenizer.build_vocab(self.sample_texts)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            # Encode with original tokenizer
            original_text = "salut"
            original_ids = tokenizer.encode(original_text)
            
            # Save and load
            tokenizer.save_vocab(temp_path)
            new_tokenizer = Tokenizer(vocab_path=temp_path)
            
            # Encode with loaded tokenizer
            new_ids = new_tokenizer.encode(original_text)
            
            # Should produce same token IDs
            self.assertEqual(original_ids, new_ids)
            
            # Decode should also work
            decoded = new_tokenizer.decode(new_ids, skip_special_tokens=True)
            self.assertEqual(decoded, original_text)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_encode_decode_with_spaces(self):
        """Test encoding/decoding text with spaces."""
        tokenizer = Tokenizer()
        tokenizer.build_vocab(["hello world"])
        
        original_text = "hello world"
        token_ids = tokenizer.encode(original_text, add_special_tokens=True)
        decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        self.assertEqual(decoded_text, original_text)
    
    def test_encode_decode_with_punctuation(self):
        """Test encoding/decoding text with punctuation."""
        tokenizer = Tokenizer()
        tokenizer.build_vocab(["ça va?", "hello!"])
        
        original_text = "ça va?"
        token_ids = tokenizer.encode(original_text, add_special_tokens=True)
        decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        self.assertEqual(decoded_text, original_text)
    
    def test_min_frequency_filtering(self):
        """Test that min_freq parameter filters rare tokens."""
        tokenizer = Tokenizer()
        
        # Build vocab with min_freq=2
        texts = ["aaa", "b"]  # 'a' appears 3 times, 'b' appears 1 time
        tokenizer.build_vocab(texts, min_freq=2)
        
        # 'a' should be in vocab (freq=3)
        self.assertIn('a', tokenizer.token_to_id)
        
        # 'b' should not be in vocab (freq=1 < min_freq=2)
        self.assertNotIn('b', tokenizer.token_to_id)


if __name__ == '__main__':
    unittest.main()
