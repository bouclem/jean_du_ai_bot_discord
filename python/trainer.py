"""
Trainer class for training the GPT model with backpropagation and optimization.
"""

import numpy as np
import pickle
import os
from typing import Optional, Dict, List
from python.model import GPTModel
from python.datasets.dataset import ConversationDataset


class GPTTrainer:
    """
    Trainer for GPT model with gradient descent optimization.
    """
    
    def __init__(self, model: GPTModel, learning_rate: float = 0.0001, 
                 clip_grad: float = 1.0):
        """
        Initialize the trainer.
        
        Args:
            model: GPT model to train
            learning_rate: Learning rate for gradient descent
            clip_grad: Gradient clipping threshold
        """
        self.model = model
        self.learning_rate = learning_rate
        self.clip_grad = clip_grad
        
        # Training history
        self.history = {
            'loss': [],
            'perplexity': []
        }
        
        # Gradients storage
        self.gradients = {}
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
    
    def compute_loss(self, logits: np.ndarray, targets: np.ndarray, 
                     pad_id: int = 0) -> float:
        """
        Compute cross-entropy loss.
        
        Args:
            logits: Model predictions of shape (batch_size, seq_len, vocab_size)
            targets: Target token IDs of shape (batch_size, seq_len)
            pad_id: ID of padding token to ignore in loss
            
        Returns:
            Average loss value
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Reshape for easier computation
        logits_flat = logits.reshape(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
        targets_flat = targets.reshape(-1)  # (batch_size * seq_len,)
        
        # Compute softmax probabilities
        probs = self._softmax(logits_flat)
        
        # Compute cross-entropy loss
        # Loss = -log(p(correct_class))
        loss = 0.0
        count = 0
        
        for i in range(len(targets_flat)):
            target_id = targets_flat[i]
            
            # Skip padding tokens
            if target_id == pad_id:
                continue
            
            # Add negative log probability of correct class
            prob = probs[i, target_id]
            loss += -np.log(prob + 1e-10)  # Add small epsilon for numerical stability
            count += 1
        
        # Average loss
        if count > 0:
            loss = loss / count
        
        return loss
    
    def compute_perplexity(self, loss: float) -> float:
        """
        Compute perplexity from loss.
        Perplexity = exp(loss)
        
        Args:
            loss: Cross-entropy loss
            
        Returns:
            Perplexity value
        """
        return np.exp(loss)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Compute softmax probabilities.
        
        Args:
            x: Input logits
            
        Returns:
            Softmax probabilities
        """
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def backward(self, logits: np.ndarray, targets: np.ndarray, 
                 pad_id: int = 0) -> Dict[str, np.ndarray]:
        """
        Compute gradients via backpropagation.
        
        This is a simplified implementation that computes gradients for the output layer.
        A full implementation would backpropagate through all layers.
        
        Args:
            logits: Model predictions of shape (batch_size, seq_len, vocab_size)
            targets: Target token IDs of shape (batch_size, seq_len)
            pad_id: ID of padding token to ignore
            
        Returns:
            Dictionary of gradients
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Reshape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        
        # Compute softmax probabilities
        probs = self._softmax(logits_flat)
        
        # Compute gradient of loss w.r.t. logits
        # For cross-entropy: d_loss/d_logits = probs - one_hot(targets)
        grad_logits = probs.copy()
        
        count = 0
        for i in range(len(targets_flat)):
            target_id = targets_flat[i]
            
            # Skip padding tokens
            if target_id == pad_id:
                grad_logits[i, :] = 0
                continue
            
            grad_logits[i, target_id] -= 1
            count += 1
        
        # Average gradients
        if count > 0:
            grad_logits = grad_logits / count
        
        # Reshape back
        grad_logits = grad_logits.reshape(batch_size, seq_len, vocab_size)
        
        return {'grad_logits': grad_logits}
    
    def update_parameters(self, gradients: Dict[str, np.ndarray], 
                         hidden_states: np.ndarray) -> None:
        """
        Update model parameters using computed gradients.
        
        This simplified version updates only the output projection layer.
        
        Args:
            gradients: Dictionary containing gradients
            hidden_states: Hidden states from the model
        """
        grad_logits = gradients['grad_logits']
        
        # Compute gradient for output projection
        # logits = hidden_states @ W + b
        # grad_W = hidden_states.T @ grad_logits
        # grad_b = sum(grad_logits)
        
        batch_size, seq_len, d_model = hidden_states.shape
        _, _, vocab_size = grad_logits.shape
        
        # Reshape for matrix multiplication
        hidden_flat = hidden_states.reshape(-1, d_model)  # (batch*seq, d_model)
        grad_flat = grad_logits.reshape(-1, vocab_size)  # (batch*seq, vocab_size)
        
        # Compute gradients
        grad_W = np.dot(hidden_flat.T, grad_flat)  # (d_model, vocab_size)
        grad_b = np.sum(grad_flat, axis=0)  # (vocab_size,)
        
        # Clip gradients
        grad_W = np.clip(grad_W, -self.clip_grad, self.clip_grad)
        grad_b = np.clip(grad_b, -self.clip_grad, self.clip_grad)
        
        # Update parameters
        self.model.output_projection -= self.learning_rate * grad_W
        self.model.output_bias -= self.learning_rate * grad_b
    
    def train_step(self, input_ids: np.ndarray, target_ids: np.ndarray, 
                   pad_id: int = 0) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            target_ids: Target token IDs of shape (batch_size, seq_len)
            pad_id: Padding token ID
            
        Returns:
            Dictionary with loss and perplexity
        """
        # Forward pass
        logits = self.model.forward(input_ids, training=True)
        
        # Compute loss
        loss = self.compute_loss(logits, target_ids, pad_id)
        perplexity = self.compute_perplexity(loss)
        
        # Backward pass
        gradients = self.backward(logits, target_ids, pad_id)
        
        # Get hidden states (output before final projection)
        # We need to recompute this - in a real implementation, we'd cache it
        hidden_states = self._get_hidden_states(input_ids)
        
        # Update parameters
        self.update_parameters(gradients, hidden_states)
        
        return {
            'loss': loss,
            'perplexity': perplexity
        }
    
    def _get_hidden_states(self, x: np.ndarray) -> np.ndarray:
        """
        Get hidden states before final projection.
        This recomputes the forward pass up to the last layer.
        
        Args:
            x: Input token IDs
            
        Returns:
            Hidden states
        """
        batch_size, seq_len = x.shape
        
        # Get embeddings
        token_emb = self.model.token_embedding[x]
        pos_emb = self.model.positional_embedding[:seq_len, :]
        embeddings = token_emb + pos_emb
        
        # Pass through transformer blocks
        mask = self.model._create_causal_mask(seq_len)
        hidden_states = embeddings
        for block in self.model.transformer_blocks:
            hidden_states = block.forward(hidden_states, mask=mask, training=True)
        
        # Final layer norm
        hidden_states = self.model.ln_f.forward(hidden_states)
        
        return hidden_states
    
    def save_checkpoint(self, epoch: int, path: str) -> None:
        """
        Save training checkpoint including model and optimizer state.
        
        Args:
            epoch: Current epoch number
            path: Path to save checkpoint
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'learning_rate': self.learning_rate,
            'clip_grad': self.clip_grad,
            'history': self.history,
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'd_model': self.model.d_model,
                'n_heads': self.model.n_heads,
                'n_layers': self.model.n_layers,
                'd_ff': self.model.d_ff,
                'max_seq_len': self.model.max_seq_len,
                'dropout': self.model.dropout
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Save model weights separately
        model_path = path.replace('.ckpt', '_model.pkl')
        self.model.save(model_path)
        
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load training checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.learning_rate = checkpoint['learning_rate']
        self.clip_grad = checkpoint['clip_grad']
        self.history = checkpoint['history']
        
        # Load model weights
        model_path = path.replace('.ckpt', '_model.pkl')
        self.model.load(model_path)
        
        print(f"Checkpoint loaded from {path}")
        print(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
    
    def train(self, dataset: ConversationDataset, epochs: int, 
              batch_size: int, print_every: int = 10, 
              save_every: int = 100, checkpoint_dir: str = './checkpoints') -> None:
        """
        Train the model on the dataset.
        
        Args:
            dataset: ConversationDataset instance
            epochs: Number of training epochs
            batch_size: Batch size for training
            print_every: Print metrics every N steps
            save_every: Save checkpoint every N steps
            checkpoint_dir: Directory to save checkpoints
        """
        pad_id = dataset.tokenizer.get_pad_token_id()
        n_samples = len(dataset)
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Dataset size: {n_samples} examples")
        print(f"Batch size: {batch_size}")
        print("-" * 50)
        
        for epoch in range(self.current_epoch, epochs):
            epoch_loss = 0.0
            epoch_perplexity = 0.0
            n_batches = 0
            
            # Shuffle indices
            indices = np.random.permutation(n_samples)
            
            # Train in batches
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                
                # Get batch
                input_batch, target_batch = dataset.get_batch(batch_indices.tolist())
                
                # Train step
                metrics = self.train_step(input_batch, target_batch, pad_id)
                
                epoch_loss += metrics['loss']
                epoch_perplexity += metrics['perplexity']
                n_batches += 1
                self.global_step += 1
                
                # Print progress
                if self.global_step % print_every == 0:
                    avg_loss = epoch_loss / n_batches
                    avg_perplexity = epoch_perplexity / n_batches
                    print(f"Epoch {epoch + 1}/{epochs} | Step {self.global_step} | "
                          f"Loss: {avg_loss:.4f} | Perplexity: {avg_perplexity:.4f}")
                
                # Save checkpoint
                if self.global_step % save_every == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{self.global_step}.ckpt')
                    self.save_checkpoint(epoch, checkpoint_path)
            
            # Epoch summary
            avg_loss = epoch_loss / n_batches
            avg_perplexity = epoch_perplexity / n_batches
            
            self.history['loss'].append(avg_loss)
            self.history['perplexity'].append(avg_perplexity)
            self.current_epoch = epoch + 1
            
            print(f"\nEpoch {epoch + 1} completed:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Average Perplexity: {avg_perplexity:.4f}")
            print("-" * 50)
            
            # Save checkpoint at end of epoch
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.ckpt')
            self.save_checkpoint(epoch + 1, checkpoint_path)
        
        print("\nTraining completed!")
