"""
GPT Model implementation from scratch.
Includes token and positional embeddings, transformer blocks, and output projection.
"""

import numpy as np
import pickle
from typing import Optional, Tuple
from python.transformer import TransformerBlock, LayerNorm
from python.tokenizer import Tokenizer


class GPTModel:
    """
    GPT (Generative Pre-trained Transformer) model implementation.
    """
    
    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 6, d_ff: int = 1024, max_seq_len: int = 512,
                 dropout: float = 0.1):
        """
        Initialize GPT model.
        
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of the model embeddings
            n_heads: Number of attention heads
            n_layers: Number of transformer blocks
            d_ff: Dimension of feed-forward layer
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        
        # Token embeddings
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        
        # Positional embeddings
        self.positional_embedding = self._create_positional_embeddings(max_seq_len, d_model)
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ]
        
        # Final layer normalization
        self.ln_f = LayerNorm(d_model)
        
        # Output projection to vocabulary
        self.output_projection = np.random.randn(d_model, vocab_size) * 0.02
        self.output_bias = np.zeros(vocab_size)
    
    def _create_positional_embeddings(self, max_seq_len: int, d_model: int) -> np.ndarray:
        """
        Create sinusoidal positional embeddings.
        
        Args:
            max_seq_len: Maximum sequence length
            d_model: Model dimension
            
        Returns:
            Positional embeddings of shape (max_seq_len, d_model)
        """
        position = np.arange(max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_embedding = np.zeros((max_seq_len, d_model))
        pos_embedding[:, 0::2] = np.sin(position * div_term)
        pos_embedding[:, 1::2] = np.cos(position * div_term)
        
        return pos_embedding
    
    def _create_causal_mask(self, seq_len: int) -> np.ndarray:
        """
        Create causal mask to prevent attending to future tokens.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Mask of shape (seq_len, seq_len) with -inf for future positions
        """
        mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
        return mask
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through the GPT model.
        
        Args:
            x: Input token IDs of shape (batch_size, seq_len)
            training: Whether in training mode
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        
        # Get token embeddings
        token_emb = self.token_embedding[x]  # (batch_size, seq_len, d_model)
        
        # Add positional embeddings
        pos_emb = self.positional_embedding[:seq_len, :]  # (seq_len, d_model)
        embeddings = token_emb + pos_emb  # Broadcasting
        
        # Create causal mask
        mask = self._create_causal_mask(seq_len)
        
        # Pass through transformer blocks
        hidden_states = embeddings
        for block in self.transformer_blocks:
            hidden_states = block.forward(hidden_states, mask=mask, training=training)
        
        # Final layer normalization
        hidden_states = self.ln_f.forward(hidden_states)
        
        # Project to vocabulary
        logits = np.dot(hidden_states, self.output_projection) + self.output_bias
        
        return logits
    
    def generate(self, prompt: str, tokenizer: Tokenizer, max_length: int = 50,
                 temperature: float = 1.0, method: str = 'sample') -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            tokenizer: Tokenizer instance
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            method: Generation method ('sample' or 'greedy')
            
        Returns:
            Generated text string
        """
        # Encode prompt
        token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        generated_ids = token_ids.copy()
        
        # Generate tokens one by one
        for _ in range(max_length):
            # Prepare input (limit to max_seq_len)
            input_ids = generated_ids[-self.max_seq_len:]
            input_array = np.array([input_ids])  # (1, seq_len)
            
            # Forward pass
            logits = self.forward(input_array, training=False)  # (1, seq_len, vocab_size)
            
            # Get logits for the last token
            next_token_logits = logits[0, -1, :]  # (vocab_size,)
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Sample or greedy decode
            if method == 'greedy':
                next_token_id = np.argmax(next_token_logits)
            else:  # sample
                # Apply softmax to get probabilities
                probs = self._softmax(next_token_logits)
                next_token_id = np.random.choice(len(probs), p=probs)
            
            # Add to generated sequence
            generated_ids.append(int(next_token_id))
            
            # Stop if EOS token is generated
            if next_token_id == tokenizer.get_eos_token_id():
                break
        
        # Decode generated tokens
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def save(self, path: str) -> None:
        """
        Save model weights to file.
        
        Args:
            path: Path to save model file
        """
        model_data = {
            'config': {
                'vocab_size': self.vocab_size,
                'd_model': self.d_model,
                'n_heads': self.n_heads,
                'n_layers': self.n_layers,
                'd_ff': self.d_ff,
                'max_seq_len': self.max_seq_len,
                'dropout': self.dropout
            },
            'token_embedding': self.token_embedding,
            'positional_embedding': self.positional_embedding,
            'output_projection': self.output_projection,
            'output_bias': self.output_bias,
            'ln_f_gamma': self.ln_f.gamma,
            'ln_f_beta': self.ln_f.beta,
            'transformer_blocks': []
        }
        
        # Save transformer block weights
        for block in self.transformer_blocks:
            block_data = {
                'attention': {
                    'W_q': block.attention.W_q,
                    'W_k': block.attention.W_k,
                    'W_v': block.attention.W_v,
                    'W_o': block.attention.W_o,
                    'b_q': block.attention.b_q,
                    'b_k': block.attention.b_k,
                    'b_v': block.attention.b_v,
                    'b_o': block.attention.b_o
                },
                'feed_forward': {
                    'W1': block.feed_forward.W1,
                    'b1': block.feed_forward.b1,
                    'W2': block.feed_forward.W2,
                    'b2': block.feed_forward.b2
                },
                'ln1': {
                    'gamma': block.ln1.gamma,
                    'beta': block.ln1.beta
                },
                'ln2': {
                    'gamma': block.ln2.gamma,
                    'beta': block.ln2.beta
                }
            }
            model_data['transformer_blocks'].append(block_data)
        
        # Save using pickle
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, path: str) -> None:
        """
        Load model weights from file.
        
        Args:
            path: Path to model file
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Load configuration
        config = model_data['config']
        self.vocab_size = config['vocab_size']
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.n_layers = config['n_layers']
        self.d_ff = config['d_ff']
        self.max_seq_len = config['max_seq_len']
        self.dropout = config['dropout']
        
        # Load embeddings and output layer
        self.token_embedding = model_data['token_embedding']
        self.positional_embedding = model_data['positional_embedding']
        self.output_projection = model_data['output_projection']
        self.output_bias = model_data['output_bias']
        
        # Load final layer norm
        self.ln_f = LayerNorm(self.d_model)
        self.ln_f.gamma = model_data['ln_f_gamma']
        self.ln_f.beta = model_data['ln_f_beta']
        
        # Recreate and load transformer blocks
        self.transformer_blocks = []
        for block_data in model_data['transformer_blocks']:
            block = TransformerBlock(self.d_model, self.n_heads, self.d_ff, self.dropout)
            
            # Load attention weights
            block.attention.W_q = block_data['attention']['W_q']
            block.attention.W_k = block_data['attention']['W_k']
            block.attention.W_v = block_data['attention']['W_v']
            block.attention.W_o = block_data['attention']['W_o']
            block.attention.b_q = block_data['attention']['b_q']
            block.attention.b_k = block_data['attention']['b_k']
            block.attention.b_v = block_data['attention']['b_v']
            block.attention.b_o = block_data['attention']['b_o']
            
            # Load feed-forward weights
            block.feed_forward.W1 = block_data['feed_forward']['W1']
            block.feed_forward.b1 = block_data['feed_forward']['b1']
            block.feed_forward.W2 = block_data['feed_forward']['W2']
            block.feed_forward.b2 = block_data['feed_forward']['b2']
            
            # Load layer norms
            block.ln1.gamma = block_data['ln1']['gamma']
            block.ln1.beta = block_data['ln1']['beta']
            block.ln2.gamma = block_data['ln2']['gamma']
            block.ln2.beta = block_data['ln2']['beta']
            
            self.transformer_blocks.append(block)
