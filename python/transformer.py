"""
Transformer components for GPT model implementation.
Includes MultiHeadAttention, FeedForward, and TransformerBlock.
"""

import numpy as np


class MultiHeadAttention:
    """
    Multi-head attention mechanism with scaled dot-product attention.
    """
    
    def __init__(self, d_model: int, n_heads: int):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Dimension of the model
            n_heads: Number of attention heads
        """
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        
        # Initialize projection matrices for Q, K, V and output
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01
        
        # Biases
        self.b_q = np.zeros(d_model)
        self.b_k = np.zeros(d_model)
        self.b_v = np.zeros(d_model)
        self.b_o = np.zeros(d_model)
    
    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Split the last dimension into (n_heads, d_k).
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor of shape (batch_size, n_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch, n_heads, seq_len, d_k)
    
    def merge_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Merge heads back to original shape.
        
        Args:
            x: Input tensor of shape (batch_size, n_heads, seq_len, d_k)
            
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(0, 2, 1, 3)  # (batch, seq_len, n_heads, d_k)
        return x.reshape(batch_size, seq_len, self.d_model)
    
    def scaled_dot_product_attention(self, q: np.ndarray, k: np.ndarray, 
                                     v: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Compute scaled dot-product attention.
        
        Args:
            q: Query tensor of shape (batch_size, n_heads, seq_len, d_k)
            k: Key tensor of shape (batch_size, n_heads, seq_len, d_k)
            v: Value tensor of shape (batch_size, n_heads, seq_len, d_k)
            mask: Optional mask tensor
            
        Returns:
            Attention output of shape (batch_size, n_heads, seq_len, d_k)
        """
        # Compute attention scores
        scores = np.matmul(q, k.transpose(0, 1, 3, 2))  # (batch, n_heads, seq_len, seq_len)
        scores = scores / np.sqrt(self.d_k)
        
        # Apply mask if provided (for causal attention)
        if mask is not None:
            scores = scores + mask
        
        # Apply softmax
        attention_weights = self._softmax(scores, axis=-1)
        
        # Apply attention to values
        output = np.matmul(attention_weights, v)  # (batch, n_heads, seq_len, d_k)
        
        return output
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax along specified axis."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def forward(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, 
                mask: np.ndarray = None) -> np.ndarray:
        """
        Forward pass of multi-head attention.
        
        Args:
            q: Query tensor of shape (batch_size, seq_len, d_model)
            k: Key tensor of shape (batch_size, seq_len, d_model)
            v: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Project Q, K, V
        q_proj = np.dot(q, self.W_q) + self.b_q
        k_proj = np.dot(k, self.W_k) + self.b_k
        v_proj = np.dot(v, self.W_v) + self.b_v
        
        # Split into multiple heads
        q_heads = self.split_heads(q_proj)
        k_heads = self.split_heads(k_proj)
        v_heads = self.split_heads(v_proj)
        
        # Apply scaled dot-product attention
        attention_output = self.scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)
        
        # Merge heads
        merged = self.merge_heads(attention_output)
        
        # Final linear projection
        output = np.dot(merged, self.W_o) + self.b_o
        
        return output



class FeedForward:
    """
    Position-wise feed-forward network with two linear layers.
    """
    
    def __init__(self, d_model: int, d_ff: int, activation: str = 'gelu'):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Dimension of the model
            d_ff: Dimension of the feed-forward layer (typically 4 * d_model)
            activation: Activation function ('gelu' or 'relu')
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        
        # Initialize weights for two linear layers
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros(d_model)
    
    def gelu(self, x: np.ndarray) -> np.ndarray:
        """
        Gaussian Error Linear Unit activation function.
        GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution function of the standard normal distribution.
        Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        """
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit activation function."""
        return np.maximum(0, x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # First linear layer
        hidden = np.dot(x, self.W1) + self.b1
        
        # Apply activation
        if self.activation == 'gelu':
            hidden = self.gelu(hidden)
        elif self.activation == 'relu':
            hidden = self.relu(hidden)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        
        # Second linear layer
        output = np.dot(hidden, self.W2) + self.b2
        
        return output



class LayerNorm:
    """
    Layer normalization.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Initialize layer normalization.
        
        Args:
            d_model: Dimension of the model
            eps: Small constant for numerical stability
        """
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Normalized tensor of same shape
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        normalized = (x - mean) / (std + self.eps)
        return self.gamma * normalized + self.beta


class TransformerBlock:
    """
    Transformer block combining multi-head attention and feed-forward network
    with layer normalization and residual connections.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize transformer block.
        
        Args:
            d_model: Dimension of the model
            n_heads: Number of attention heads
            d_ff: Dimension of feed-forward layer
            dropout: Dropout rate
        """
        self.d_model = d_model
        self.dropout = dropout
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, n_heads)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff)
        
        # Layer normalization
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
    
    def apply_dropout(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Apply dropout during training.
        
        Args:
            x: Input tensor
            training: Whether in training mode
            
        Returns:
            Tensor with dropout applied
        """
        if training and self.dropout > 0:
            mask = np.random.binomial(1, 1 - self.dropout, x.shape)
            return x * mask / (1 - self.dropout)
        return x
    
    def forward(self, x: np.ndarray, mask: np.ndarray = None, training: bool = True) -> np.ndarray:
        """
        Forward pass of transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            training: Whether in training mode
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Multi-head attention with residual connection and layer norm
        # Pre-norm architecture: LayerNorm -> Attention -> Dropout -> Residual
        attn_input = self.ln1.forward(x)
        attn_output = self.attention.forward(attn_input, attn_input, attn_input, mask)
        attn_output = self.apply_dropout(attn_output, training)
        x = x + attn_output  # Residual connection
        
        # Feed-forward with residual connection and layer norm
        # Pre-norm architecture: LayerNorm -> FeedForward -> Dropout -> Residual
        ff_input = self.ln2.forward(x)
        ff_output = self.feed_forward.forward(ff_input)
        ff_output = self.apply_dropout(ff_output, training)
        x = x + ff_output  # Residual connection
        
        return x
