"""
Dataset class for loading and preprocessing conversation data for GPT training.
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from python.tokenizer import Tokenizer


class ConversationDataset:
    """
    Dataset for loading conversations from JSON and preparing them for training.
    """
    
    def __init__(self, data_path: str, tokenizer: Tokenizer, max_seq_len: int = 512):
        """
        Initialize the conversation dataset.
        
        Args:
            data_path: Path to JSON file containing conversations
            tokenizer: Tokenizer instance for encoding text
            max_seq_len: Maximum sequence length for padding/truncation
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.conversations = []
        self.processed_data = []
        
        # Load and preprocess data
        self.load_conversations(data_path)
        self.preprocess()
    
    def load_conversations(self, file_path: str) -> None:
        """
        Load conversations from JSON file.
        
        Expected format:
        {
            "conversations": [
                {
                    "messages": [
                        {"role": "user", "content": "salut"},
                        {"role": "assistant", "content": "yo ça va?"}
                    ]
                }
            ]
        }
        
        Args:
            file_path: Path to JSON file
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.conversations = data.get('conversations', [])
    
    def preprocess(self) -> None:
        """
        Preprocess conversations into training examples.
        Each conversation is converted to a sequence of tokens with input and target.
        """
        self.processed_data = []
        
        for conversation in self.conversations:
            messages = conversation.get('messages', [])
            
            # Concatenate all messages in the conversation
            full_text = ""
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                # Add role prefix for context
                full_text += f"{role}: {content}\n"
            
            # Encode the full conversation
            token_ids = self.tokenizer.encode(full_text, add_special_tokens=True)
            
            # Create training examples with sliding window
            # For each position, predict the next token
            if len(token_ids) > 1:
                # Truncate if too long
                if len(token_ids) > self.max_seq_len:
                    token_ids = token_ids[:self.max_seq_len]
                
                # Input: all tokens except the last
                # Target: all tokens except the first (shifted by 1)
                input_ids = token_ids[:-1]
                target_ids = token_ids[1:]
                
                # Pad sequences to max_seq_len
                input_ids = self._pad_sequence(input_ids)
                target_ids = self._pad_sequence(target_ids)
                
                self.processed_data.append({
                    'input_ids': np.array(input_ids, dtype=np.int32),
                    'target_ids': np.array(target_ids, dtype=np.int32)
                })
    
    def _pad_sequence(self, token_ids: List[int]) -> List[int]:
        """
        Pad sequence to max_seq_len with PAD tokens.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Padded list of token IDs
        """
        pad_id = self.tokenizer.get_pad_token_id()
        
        if len(token_ids) < self.max_seq_len:
            # Pad to max length
            padding = [pad_id] * (self.max_seq_len - len(token_ids))
            return token_ids + padding
        else:
            # Already at or above max length (should be truncated before)
            return token_ids[:self.max_seq_len]
    
    def __len__(self) -> int:
        """
        Get the number of training examples in the dataset.
        
        Returns:
            Number of examples
        """
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a training example by index.
        
        Args:
            idx: Index of the example
            
        Returns:
            Tuple of (input_ids, target_ids) as numpy arrays
        """
        example = self.processed_data[idx]
        return example['input_ids'], example['target_ids']
    
    def get_batch(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a batch of training examples.
        
        Args:
            indices: List of indices to retrieve
            
        Returns:
            Tuple of (input_batch, target_batch) as numpy arrays
        """
        input_batch = []
        target_batch = []
        
        for idx in indices:
            input_ids, target_ids = self[idx]
            input_batch.append(input_ids)
            target_batch.append(target_ids)
        
        return np.array(input_batch), np.array(target_batch)
