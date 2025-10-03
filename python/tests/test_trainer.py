"""
Unit tests for GPTTrainer class, including checkpoint functionality.
"""

import unittest
import numpy as np
import os
import tempfile
import shutil
from python.trainer import GPTTrainer
from python.model import GPTModel
from python.tokenizer import Tokenizer
from python.datasets.dataset import ConversationDataset


class TestGPTTrainer(unittest.TestCase):
    """Test cases for GPTTrainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a small model for testing
        self.vocab_size = 100
        self.d_model = 32
        self.n_heads = 2
        self.n_layers = 2
        self.max_seq_len = 16
        
        self.model = GPTModel(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            max_seq_len=self.max_seq_len
        )
        
        self.trainer = GPTTrainer(self.model, learning_rate=0.001)
        
        # Create temporary directory for checkpoints
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_compute_loss(self):
        """Test loss computation."""
        batch_size = 2
        seq_len = 8
        
        # Create dummy logits and targets
        logits = np.random.randn(batch_size, seq_len, self.vocab_size)
        targets = np.random.randint(0, self.vocab_size, (batch_size, seq_len))
        
        loss = self.trainer.compute_loss(logits, targets)
        
        # Loss should be positive
        self.assertGreater(loss, 0)
        self.assertIsInstance(loss, float)
    
    def test_compute_perplexity(self):
        """Test perplexity computation."""
        loss = 2.5
        perplexity = self.trainer.compute_perplexity(loss)
        
        # Perplexity = exp(loss)
        expected = np.exp(loss)
        self.assertAlmostEqual(perplexity, expected, places=5)
    
    def test_backward(self):
        """Test gradient computation."""
        batch_size = 2
        seq_len = 8
        
        logits = np.random.randn(batch_size, seq_len, self.vocab_size)
        targets = np.random.randint(0, self.vocab_size, (batch_size, seq_len))
        
        gradients = self.trainer.backward(logits, targets)
        
        # Check gradient shape
        self.assertIn('grad_logits', gradients)
        self.assertEqual(gradients['grad_logits'].shape, logits.shape)
    
    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        checkpoint_path = os.path.join(self.temp_dir, 'test_checkpoint.ckpt')
        
        # Set some training state
        self.trainer.current_epoch = 5
        self.trainer.global_step = 100
        self.trainer.history['loss'] = [2.5, 2.3, 2.1]
        self.trainer.history['perplexity'] = [12.2, 10.0, 8.2]
        
        # Save checkpoint
        self.trainer.save_checkpoint(epoch=5, path=checkpoint_path)
        
        # Check files exist
        self.assertTrue(os.path.exists(checkpoint_path))
        model_path = checkpoint_path.replace('.ckpt', '_model.pkl')
        self.assertTrue(os.path.exists(model_path))
    
    def test_load_checkpoint(self):
        """Test checkpoint loading."""
        checkpoint_path = os.path.join(self.temp_dir, 'test_checkpoint.ckpt')
        
        # Set training state and save
        self.trainer.current_epoch = 5
        self.trainer.global_step = 100
        self.trainer.learning_rate = 0.0005
        self.trainer.history['loss'] = [2.5, 2.3, 2.1]
        self.trainer.history['perplexity'] = [12.2, 10.0, 8.2]
        
        self.trainer.save_checkpoint(epoch=5, path=checkpoint_path)
        
        # Create new trainer and load checkpoint
        new_model = GPTModel(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            max_seq_len=self.max_seq_len
        )
        new_trainer = GPTTrainer(new_model)
        
        # Load checkpoint
        new_trainer.load_checkpoint(checkpoint_path)
        
        # Verify state was restored
        self.assertEqual(new_trainer.current_epoch, 5)
        self.assertEqual(new_trainer.global_step, 100)
        self.assertEqual(new_trainer.learning_rate, 0.0005)
        self.assertEqual(len(new_trainer.history['loss']), 3)
        self.assertEqual(len(new_trainer.history['perplexity']), 3)
    
    def test_checkpoint_resume_training(self):
        """Test that training can be resumed from checkpoint."""
        checkpoint_path = os.path.join(self.temp_dir, 'resume_checkpoint.ckpt')
        
        # Train for a few steps
        self.trainer.current_epoch = 2
        self.trainer.global_step = 50
        self.trainer.history['loss'] = [3.0, 2.8]
        
        # Save checkpoint
        self.trainer.save_checkpoint(epoch=2, path=checkpoint_path)
        
        # Create new trainer and load
        new_model = GPTModel(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            max_seq_len=self.max_seq_len
        )
        new_trainer = GPTTrainer(new_model)
        new_trainer.load_checkpoint(checkpoint_path)
        
        # Verify training can continue from saved state
        self.assertEqual(new_trainer.current_epoch, 2)
        self.assertEqual(new_trainer.global_step, 50)
        
        # Simulate continuing training
        new_trainer.global_step += 10
        self.assertEqual(new_trainer.global_step, 60)


if __name__ == '__main__':
    unittest.main()
