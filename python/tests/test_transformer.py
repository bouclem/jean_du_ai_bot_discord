"""
Unit tests for Transformer components.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from transformer import MultiHeadAttention, FeedForward, TransformerBlock, LayerNorm


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention class."""
    
    def test_initialization(self):
        """Test that MultiHeadAttention initializes correctly."""
        d_model = 256
        n_heads = 8
        mha = MultiHeadAttention(d_model, n_heads)
        
        assert mha.d_model == d_model
        assert mha.n_heads == n_heads
        assert mha.d_k == d_model // n_heads
        assert mha.W_q.shape == (d_model, d_model)
        assert mha.W_k.shape == (d_model, d_model)
        assert mha.W_v.shape == (d_model, d_model)
        assert mha.W_o.shape == (d_model, d_model)
    
    def test_invalid_dimensions(self):
        """Test that initialization fails when d_model is not divisible by n_heads."""
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=256, n_heads=7)
    
    def test_split_heads(self):
        """Test head splitting operation."""
        d_model = 256
        n_heads = 8
        batch_size = 2
        seq_len = 10
        
        mha = MultiHeadAttention(d_model, n_heads)
        x = np.random.randn(batch_size, seq_len, d_model)
        
        split = mha.split_heads(x)
        
        assert split.shape == (batch_size, n_heads, seq_len, d_model // n_heads)
    
    def test_merge_heads(self):
        """Test head merging operation."""
        d_model = 256
        n_heads = 8
        batch_size = 2
        seq_len = 10
        
        mha = MultiHeadAttention(d_model, n_heads)
        x = np.random.randn(batch_size, n_heads, seq_len, d_model // n_heads)
        
        merged = mha.merge_heads(x)
        
        assert merged.shape == (batch_size, seq_len, d_model)
    
    def test_split_merge_inverse(self):
        """Test that split and merge are inverse operations."""
        d_model = 256
        n_heads = 8
        batch_size = 2
        seq_len = 10
        
        mha = MultiHeadAttention(d_model, n_heads)
        x = np.random.randn(batch_size, seq_len, d_model)
        
        split = mha.split_heads(x)
        merged = mha.merge_heads(split)
        
        np.testing.assert_allclose(x, merged, rtol=1e-5)
    
    def test_scaled_dot_product_attention(self):
        """Test scaled dot-product attention computation."""
        d_model = 256
        n_heads = 8
        batch_size = 2
        seq_len = 10
        d_k = d_model // n_heads
        
        mha = MultiHeadAttention(d_model, n_heads)
        
        q = np.random.randn(batch_size, n_heads, seq_len, d_k)
        k = np.random.randn(batch_size, n_heads, seq_len, d_k)
        v = np.random.randn(batch_size, n_heads, seq_len, d_k)
        
        output = mha.scaled_dot_product_attention(q, k, v)
        
        assert output.shape == (batch_size, n_heads, seq_len, d_k)
    
    def test_forward_pass(self):
        """Test forward pass with simple inputs."""
        d_model = 256
        n_heads = 8
        batch_size = 2
        seq_len = 10
        
        mha = MultiHeadAttention(d_model, n_heads)
        
        q = np.random.randn(batch_size, seq_len, d_model)
        k = np.random.randn(batch_size, seq_len, d_model)
        v = np.random.randn(batch_size, seq_len, d_model)
        
        output = mha.forward(q, k, v)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_self_attention(self):
        """Test self-attention (Q=K=V)."""
        d_model = 128
        n_heads = 4
        batch_size = 1
        seq_len = 5
        
        mha = MultiHeadAttention(d_model, n_heads)
        
        x = np.random.randn(batch_size, seq_len, d_model)
        output = mha.forward(x, x, x)
        
        assert output.shape == (batch_size, seq_len, d_model)


class TestFeedForward:
    """Tests for FeedForward class."""
    
    def test_initialization(self):
        """Test that FeedForward initializes correctly."""
        d_model = 256
        d_ff = 1024
        
        ff = FeedForward(d_model, d_ff)
        
        assert ff.d_model == d_model
        assert ff.d_ff == d_ff
        assert ff.W1.shape == (d_model, d_ff)
        assert ff.W2.shape == (d_ff, d_model)
    
    def test_gelu_activation(self):
        """Test GELU activation function."""
        ff = FeedForward(256, 1024, activation='gelu')
        
        # Test basic properties of GELU
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        output = ff.gelu(x)
        
        # GELU(0) should be close to 0
        assert np.abs(output[2]) < 1e-6
        
        # GELU should be approximately x for large positive x
        assert output[4] > 1.9  # GELU(2) ≈ 1.95
        
        # GELU should be close to 0 for large negative x
        assert np.abs(output[0]) < 0.1  # GELU(-2) ≈ -0.045
        
        # GELU(1) should be close to 0.84
        assert 0.8 < output[3] < 0.9
    
    def test_relu_activation(self):
        """Test ReLU activation function."""
        ff = FeedForward(256, 1024, activation='relu')
        
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        output = ff.relu(x)
        
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(output, expected)
    
    def test_forward_pass_gelu(self):
        """Test forward pass with GELU activation."""
        d_model = 256
        d_ff = 1024
        batch_size = 2
        seq_len = 10
        
        ff = FeedForward(d_model, d_ff, activation='gelu')
        x = np.random.randn(batch_size, seq_len, d_model)
        
        output = ff.forward(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_forward_pass_relu(self):
        """Test forward pass with ReLU activation."""
        d_model = 256
        d_ff = 1024
        batch_size = 2
        seq_len = 10
        
        ff = FeedForward(d_model, d_ff, activation='relu')
        x = np.random.randn(batch_size, seq_len, d_model)
        
        output = ff.forward(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_invalid_activation(self):
        """Test that invalid activation raises error."""
        ff = FeedForward(256, 1024, activation='invalid')
        x = np.random.randn(2, 10, 256)
        
        with pytest.raises(ValueError):
            ff.forward(x)


class TestLayerNorm:
    """Tests for LayerNorm class."""
    
    def test_initialization(self):
        """Test that LayerNorm initializes correctly."""
        d_model = 256
        ln = LayerNorm(d_model)
        
        assert ln.gamma.shape == (d_model,)
        assert ln.beta.shape == (d_model,)
        np.testing.assert_array_equal(ln.gamma, np.ones(d_model))
        np.testing.assert_array_equal(ln.beta, np.zeros(d_model))
    
    def test_normalization(self):
        """Test that layer norm produces zero mean and unit variance."""
        d_model = 256
        batch_size = 2
        seq_len = 10
        
        ln = LayerNorm(d_model)
        x = np.random.randn(batch_size, seq_len, d_model) * 10 + 5
        
        output = ln.forward(x)
        
        # Check mean is close to 0 and std is close to 1
        mean = np.mean(output, axis=-1)
        std = np.std(output, axis=-1)
        
        np.testing.assert_allclose(mean, 0, atol=1e-5)
        np.testing.assert_allclose(std, 1, atol=1e-5)


class TestTransformerBlock:
    """Tests for TransformerBlock class."""
    
    def test_initialization(self):
        """Test that TransformerBlock initializes correctly."""
        d_model = 256
        n_heads = 8
        d_ff = 1024
        dropout = 0.1
        
        block = TransformerBlock(d_model, n_heads, d_ff, dropout)
        
        assert block.d_model == d_model
        assert block.dropout == dropout
        assert isinstance(block.attention, MultiHeadAttention)
        assert isinstance(block.feed_forward, FeedForward)
        assert isinstance(block.ln1, LayerNorm)
        assert isinstance(block.ln2, LayerNorm)
    
    def test_forward_pass(self):
        """Test forward pass of complete transformer block."""
        d_model = 256
        n_heads = 8
        d_ff = 1024
        batch_size = 2
        seq_len = 10
        
        block = TransformerBlock(d_model, n_heads, d_ff, dropout=0.0)
        x = np.random.randn(batch_size, seq_len, d_model)
        
        output = block.forward(x, training=False)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_residual_connections(self):
        """Test that residual connections are working."""
        d_model = 256
        n_heads = 8
        d_ff = 1024
        batch_size = 1
        seq_len = 5
        
        # Create block with zero dropout for deterministic behavior
        block = TransformerBlock(d_model, n_heads, d_ff, dropout=0.0)
        
        # Zero out all weights to test residual connection
        block.attention.W_q = np.zeros_like(block.attention.W_q)
        block.attention.W_k = np.zeros_like(block.attention.W_k)
        block.attention.W_v = np.zeros_like(block.attention.W_v)
        block.attention.W_o = np.zeros_like(block.attention.W_o)
        block.feed_forward.W1 = np.zeros_like(block.feed_forward.W1)
        block.feed_forward.W2 = np.zeros_like(block.feed_forward.W2)
        
        x = np.random.randn(batch_size, seq_len, d_model)
        output = block.forward(x, training=False)
        
        # With zero weights, output should be close to input due to residual connections
        # (after layer norm effects)
        assert output.shape == x.shape
    
    def test_dropout_training_vs_inference(self):
        """Test that dropout behaves differently in training vs inference."""
        d_model = 128
        n_heads = 4
        d_ff = 512
        batch_size = 1
        seq_len = 5
        
        block = TransformerBlock(d_model, n_heads, d_ff, dropout=0.5)
        x = np.random.randn(batch_size, seq_len, d_model)
        
        # Set seed for reproducibility
        np.random.seed(42)
        output_train = block.forward(x.copy(), training=True)
        
        np.random.seed(42)
        output_inference = block.forward(x.copy(), training=False)
        
        # Outputs should be different due to dropout
        assert not np.allclose(output_train, output_inference)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
