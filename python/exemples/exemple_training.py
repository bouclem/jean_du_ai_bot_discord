"""
Example script for training the GPT model on conversation data.
This demonstrates how to:
1. Load a dataset
2. Build vocabulary
3. Initialize and train the model
4. Save the trained model
"""

import os
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from python.tokenizer import Tokenizer
from python.model import GPTModel
from python.trainer import GPTTrainer
from python.datasets.dataset import ConversationDataset


def main():
    """Main training function."""
    
    print("=" * 60)
    print("GPT Model Training Example")
    print("=" * 60)
    
    # Configuration
    DATASET_PATH = 'python/datasets/conversations.json'
    VOCAB_PATH = 'models/vocab.json'
    MODEL_PATH = 'models/gpt_model.pkl'
    CHECKPOINT_DIR = 'checkpoints'
    
    # Model hyperparameters
    D_MODEL = 512  # Augmenté pour plus de capacité
    N_HEADS = 8
    N_LAYERS = 8
    D_FF = 2048
    MAX_SEQ_LEN = 256
    DROPOUT = 0.1
    
    # Training hyperparameters
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 4
    EPOCHS = 50  # Beaucoup plus d'epochs
    PRINT_EVERY = 10
    SAVE_EVERY = 50
    
    # Step 1: Initialize tokenizer
    print("\n[1/5] Initializing tokenizer...")
    tokenizer = Tokenizer()
    
    # Build vocabulary from dataset if vocab doesn't exist
    if not os.path.exists(VOCAB_PATH):
        print("Building vocabulary from dataset...")
        import json
        
        # Load conversations to build vocab
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        for conv in data.get('conversations', []):
            for msg in conv.get('messages', []):
                texts.append(msg.get('content', ''))
        
        tokenizer.build_vocab(texts, min_freq=1)
        
        # Save vocabulary
        os.makedirs(os.path.dirname(VOCAB_PATH), exist_ok=True)
        tokenizer.save_vocab(VOCAB_PATH)
        print(f"Vocabulary built and saved to {VOCAB_PATH}")
        print(f"Vocabulary size: {tokenizer.vocab_size}")
    else:
        tokenizer.load_vocab(VOCAB_PATH)
        print(f"Vocabulary loaded from {VOCAB_PATH}")
        print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Step 2: Load dataset
    print("\n[2/5] Loading dataset...")
    dataset = ConversationDataset(
        data_path=DATASET_PATH,
        tokenizer=tokenizer,
        max_seq_len=MAX_SEQ_LEN
    )
    print(f"Dataset loaded: {len(dataset)} training examples")
    
    # Step 3: Initialize model
    print("\n[3/5] Initializing GPT model...")
    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT
    )
    print(f"Model initialized with {N_LAYERS} layers, {N_HEADS} heads, d_model={D_MODEL}")
    
    # Step 4: Initialize trainer
    print("\n[4/5] Initializing trainer...")
    trainer = GPTTrainer(
        model=model,
        learning_rate=LEARNING_RATE,
        clip_grad=1.0
    )
    print(f"Trainer initialized with learning rate {LEARNING_RATE}")
    
    # Step 5: Train model
    print("\n[5/5] Starting training...")
    print("=" * 60)
    
    trainer.train(
        dataset=dataset,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        print_every=PRINT_EVERY,
        save_every=SAVE_EVERY,
        checkpoint_dir=CHECKPOINT_DIR
    )
    
    # Save final model
    print("\nSaving final model...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Test generation
    print("\n" + "=" * 60)
    print("Testing text generation...")
    print("=" * 60)
    
    test_prompts = [
        "salut",
        "ça va?",
        "tu fais quoi?"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        generated = model.generate(
            prompt=prompt,
            tokenizer=tokenizer,
            max_length=30,
            temperature=0.8,
            method='sample'
        )
        print(f"Generated: {generated}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
