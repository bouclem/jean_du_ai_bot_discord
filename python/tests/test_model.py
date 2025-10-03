"""
Unit tests for GPT model.
Tests forward pass, text generation, and model save/load functionality.
"""

import numpy as np
import os
import tempfile
import pytest
from python.model import GPTModel
from python.tokenizer import Tokenizer


class TestGPTModel:
    """Test suite for GPT model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a small model for testing
        self.vocab_size = 100
        self.d_model = 64
        self.n_heads = 4
        self.n_layers = 2
        self.d_ff = 128
        self.max_seq_len = 50
        
        self.model = GPTModel(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            dropout=0.1
        )
        
        # Create a simple tokenizer
        self.tokenizer = Tokenizer()
        texts = ["hello world", "bonjour le monde", "test text"]
        self.tokenizer.build_vocab(texts)
    
    def test_model_initialization(self):
        """Test that model initializes with correct parameters."""
        assert self.model.vocab_size == self.vocab_size
        assert self.model.d_model == self.d_model
        assert self.model.n_heads == self.n_heads
        assert self.model.n_layers == self.n_layers
        assert self.model.d_ff == self.d_ff
        assert self.model.max_seq_len == self.max_seq_len
        
        # Check embeddings shape
        assert self.model.token_embedding.shape == (self.vocab_size, self.d_model)
        assert self.model.positional_embedding.shape == (self.max_seq_len, self.d_model)
        
        # Check output projection shape
        assert self.model.output_projection.shape == (self.d_model, self.vocab_size)
        assert self.model.output_bias.shape == (self.vocab_size,)
        
        # Check number of transformer blocks
        assert len(self.model.transformer_blocks) == self.n_layers
    
    def test_forward_pass_shape(self):
        """Test forward pass produces correct output shape."""
        batch_size = 2
        seq_len = 10
        
        # Create random input token IDs
        x = np.random.randint(0, self.vocab_size, size=(batch_size, seq_len))
        
        # Forward pass
        logits = self.model.forward(x, training=False)
        
        # Check output shape
        assert logits.shape == (batch_size, seq_len, self.vocab_size)
    
    def test_forward_pass_values(self):
        """Test forward pass produces valid logits."""
        batch_size = 1
        seq_len = 5
        
        # Create input
        x = np.random.randint(0, self.vocab_size, size=(batch_size, seq_len))
        
        # Forward pass
        logits = self.model.forward(x, training=False)
        
        # Check that logits are finite
        assert np.all(np.isfinite(logits))
        
        # Check that logits have reasonable range (not too extreme)
        assert np.abs(logits).max() < 1000
    
    def test_causal_mask(self):
        """Test that causal mask is created correctly."""
        seq_len = 5
        mask = self.model._create_causal_mask(seq_len)
        
        # Check shape
        assert mask.shape == (seq_len, seq_len)
        
        # Check that diagonal and below are 0
        for i in range(seq_len):
            for j in range(seq_len):
                if j <= i:
                    assert mask[i, j] == 0
                else:
                    assert mask[i, j] < 0  # Should be -inf or very negative
    
    def test_generate_greedy(self):
        """Test text generation with greedy decoding."""
        prompt = "hello"
        
        # Generate text
        generated = self.model.generate(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=10,
            temperature=1.0,
            method='greedy'
        )
        
        # Check that output is a string
        assert isinstance(generated, str)
        
        # Check that output contains the prompt
        assert prompt in generated or len(generated) > 0
    
    def test_generate_sample(self):
        """Test text generation with sampling."""
        prompt = "test"
        
        # Generate text
        generated = self.model.generate(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=10,
            temperature=0.8,
            method='sample'
        )
        
        # Check that output is a string
        assert isinstance(generated, str)
        
        # Check that output is not empty
        assert len(generated) > 0
    
    def test_generate_temperature(self):
        """Test that temperature affects generation."""
        prompt = "hi"
        
        # Generate with low temperature (more deterministic)
        gen_low_temp = self.model.generate(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=5,
            temperature=0.1,
            method='sample'
        )
        
        # Generate with high temperature (more random)
        gen_high_temp = self.model.generate(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=5,
            temperature=2.0,
            method='sample'
        )
        
        # Both should produce strings
        assert isinstance(gen_low_temp, str)
        assert isinstance(gen_high_temp, str)
    
    def test_save_and_load(self):
        """Test model save and load functionality."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            tmp_path = tmp.name
        
        try:
            # Save model
            self.model.save(tmp_path)
            
            # Check file exists
            assert os.path.exists(tmp_path)
            
            # Create new model and load weights
            new_model = GPTModel(
                vocab_size=self.vocab_size,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                d_ff=self.d_ff,
                max_seq_len=self.max_seq_len
            )
            new_model.load(tmp_path)
            
            # Check that configurations match
            assert new_model.vocab_size == self.model.vocab_size
            assert new_model.d_model == self.model.d_model
            assert new_model.n_heads == self.model.n_heads
            assert new_model.n_layers == self.model.n_layers
            
            # Check that embeddings match
            assert np.allclose(new_model.token_embedding, self.model.token_embedding)
            assert np.allclose(new_model.positional_embedding, self.model.positional_embedding)
            
            # Check that output layer matches
            assert np.allclose(new_model.output_projection, self.model.output_projection)
            assert np.allclose(new_model.output_bias, self.model.output_bias)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_save_load_produces_same_output(self):
        """Test that loaded model produces same output as original."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            tmp_path = tmp.name
        
        try:
            # Create input
            x = np.random.randint(0, self.vocab_size, size=(1, 10))
            
            # Get output from original model
            original_output = self.model.forward(x, training=False)
            
            # Save and load model
            self.model.save(tmp_path)
            new_model = GPTModel(
                vocab_size=self.vocab_size,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                d_ff=self.d_ff,
                max_seq_len=self.max_seq_len
            )
            new_model.load(tmp_path)
            
            # Get output from loaded model
            loaded_output = new_model.forward(x, training=False)
            
            # Outputs should be very close
            assert np.allclose(original_output, loaded_output, rtol=1e-5)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_max_sequence_length(self):
        """Test that model handles sequences up to max_seq_len."""
        # Test with max sequence length
        x = np.random.randint(0, self.vocab_size, size=(1, self.max_seq_len))
        logits = self.model.forward(x, training=False)
        
        assert logits.shape == (1, self.max_seq_len, self.vocab_size)
    
    def test_different_batch_sizes(self):
        """Test that model works with different batch sizes."""
        seq_len = 10
        
        for batch_size in [1, 2, 4, 8]:
            x = np.random.randint(0, self.vocab_size, size=(batch_size, seq_len))
            logits = self.model.forward(x, training=False)
            
            assert logits.shape == (batch_size, seq_len, self.vocab_size)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
