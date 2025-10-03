"""
Tokenizer for converting text to token IDs and vice versa.
Supports special tokens: [PAD], [UNK], [BOS], [EOS]
"""
import json
from typing import List, Dict, Optional


class Tokenizer:
    """Tokenizer for text encoding and decoding with vocabulary management."""
    
    # Special tokens
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    
    def __init__(self, vocab_path: Optional[str] = None):
        """
        Initialize the tokenizer.
        
        Args:
            vocab_path: Optional path to load vocabulary from
        """
        # Initialize with special tokens
        self.token_to_id: Dict[str, int] = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.BOS_TOKEN: 2,
            self.EOS_TOKEN: 3
        }
        self.id_to_token: Dict[int, str] = {
            0: self.PAD_TOKEN,
            1: self.UNK_TOKEN,
            2: self.BOS_TOKEN,
            3: self.EOS_TOKEN
        }
        
        # Load vocabulary if path provided
        if vocab_path:
            self.load_vocab(vocab_path)
    
    def build_vocab(self, texts: List[str], min_freq: int = 1) -> None:
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts: List of text strings to build vocabulary from
            min_freq: Minimum frequency for a token to be included
        """
        # Count token frequencies
        token_freq: Dict[str, int] = {}
        
        for text in texts:
            # Simple character-level tokenization
            tokens = self._tokenize(text)
            for token in tokens:
                token_freq[token] = token_freq.get(token, 0) + 1
        
        # Start from index 4 (after special tokens)
        next_id = len(self.token_to_id)
        
        # Add tokens that meet minimum frequency
        for token, freq in sorted(token_freq.items()):
            if freq >= min_freq and token not in self.token_to_id:
                self.token_to_id[token] = next_id
                self.id_to_token[next_id] = token
                next_id += 1
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into individual tokens.
        Currently uses character-level tokenization.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        # Character-level tokenization (can be extended to word-level or BPE)
        return list(text)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            add_special_tokens: Whether to add BOS and EOS tokens
            
        Returns:
            List of token IDs
        """
        tokens = self._tokenize(text)
        
        # Convert tokens to IDs
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.token_to_id[self.BOS_TOKEN])
        
        for token in tokens:
            token_id = self.token_to_id.get(token, self.token_to_id[self.UNK_TOKEN])
            token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.token_to_id[self.EOS_TOKEN])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        special_ids = {
            self.token_to_id[self.PAD_TOKEN],
            self.token_to_id[self.UNK_TOKEN],
            self.token_to_id[self.BOS_TOKEN],
            self.token_to_id[self.EOS_TOKEN]
        }
        
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            
            token = self.id_to_token.get(token_id, self.UNK_TOKEN)
            tokens.append(token)
        
        # Join tokens (for character-level, just concatenate)
        return ''.join(tokens)
    
    def save_vocab(self, path: str) -> None:
        """
        Save vocabulary to a JSON file.
        
        Args:
            path: Path to save vocabulary file
        """
        vocab_data = {
            'token_to_id': self.token_to_id,
            'id_to_token': {str(k): v for k, v in self.id_to_token.items()}
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    def load_vocab(self, path: str) -> None:
        """
        Load vocabulary from a JSON file.
        
        Args:
            path: Path to vocabulary file
        """
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.token_to_id = vocab_data['token_to_id']
        self.id_to_token = {int(k): v for k, v in vocab_data['id_to_token'].items()}
    
    @property
    def vocab_size(self) -> int:
        """Get the size of the vocabulary."""
        return len(self.token_to_id)
    
    def get_pad_token_id(self) -> int:
        """Get the ID of the PAD token."""
        return self.token_to_id[self.PAD_TOKEN]
    
    def get_unk_token_id(self) -> int:
        """Get the ID of the UNK token."""
        return self.token_to_id[self.UNK_TOKEN]
    
    def get_bos_token_id(self) -> int:
        """Get the ID of the BOS token."""
        return self.token_to_id[self.BOS_TOKEN]
    
    def get_eos_token_id(self) -> int:
        """Get the ID of the EOS token."""
        return self.token_to_id[self.EOS_TOKEN]
