"""
Training script for mHC-GPT model.

This script demonstrates training the mHC-GPT model on a simple text dataset.
For demonstration purposes, it uses a small synthetic or character-level dataset.
"""

import os
import math
import time
import argparse
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import mHCGPT, mhc_gpt_tiny


@dataclass
class TrainConfig:
    """Training configuration."""
    # Model
    vocab_size: int = 256  # Character-level
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    expansion_rate: int = 4
    ffn_ratio: int = 4
    max_seq_len: int = 128
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_iters: int = 5000
    warmup_iters: int = 100
    lr_decay_iters: int = 5000
    min_lr: float = 1e-5
    grad_clip: float = 1.0

    # Logging
    log_interval: int = 100
    eval_interval: int = 500
    eval_iters: int = 50

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class CharDataset(Dataset):
    """
    Simple character-level dataset.
    """

    def __init__(self, text: str, block_size: int = 128):
        self.block_size = block_size

        # Character-level tokenization
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}

        # Encode text
        self.data = torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

    def decode(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices])

    def encode(self, text):
        return torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)


def get_sample_text() -> str:
    """Generate sample training text."""
    # Simple patterns for testing
    text = """
    The quick brown fox jumps over the lazy dog.
    A journey of a thousand miles begins with a single step.
    To be or not to be, that is the question.
    All that glitters is not gold.
    The early bird catches the worm.
    Actions speak louder than words.
    Practice makes perfect.
    Knowledge is power.
    Time flies like an arrow.
    Where there is a will, there is a way.
    """ * 500  # Repeat for more data
    return text


def get_lr(it: int, config: TrainConfig) -> float:
    """Learning rate schedule with warmup and cosine decay."""
    # Warmup
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters

    # After decay
    if it > config.lr_decay_iters:
        return config.min_lr

    # Cosine decay
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig
) -> Tuple[float, float]:
    """Estimate train and validation loss."""
    model.eval()
    losses = {'train': [], 'val': []}

    for split, loader in [('train', train_loader), ('val', val_loader)]:
        for i, (x, y) in enumerate(loader):
            if i >= config.eval_iters:
                break
            x, y = x.to(config.device), y.to(config.device)
            _, loss = model(x, y)
            losses[split].append(loss.item())

    model.train()
    return (
        sum(losses['train']) / len(losses['train']) if losses['train'] else 0,
        sum(losses['val']) / len(losses['val']) if losses['val'] else 0
    )


def train(config: Optional[TrainConfig] = None):
    """Main training function."""
    if config is None:
        config = TrainConfig()

    print(f"Training on device: {config.device}")

    # Create dataset
    print("Creating dataset...")
    text = get_sample_text()
    full_dataset = CharDataset(text, config.max_seq_len)

    # Split into train/val
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    print(f"Dataset size: {len(full_dataset)}")
    print(f"Vocab size: {full_dataset.vocab_size}")
    print(f"Train size: {train_size}, Val size: {val_size}")

    # Create model
    print("Creating model...")
    model = mHCGPT(
        vocab_size=full_dataset.vocab_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        expansion_rate=config.expansion_rate,
        ffn_ratio=config.ffn_ratio,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    ).to(config.device)

    print(f"Model parameters: {model.count_parameters():,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)
    )

    # Training loop
    print("\nStarting training...")
    model.train()
    train_iter = iter(train_loader)
    best_val_loss = float('inf')

    pbar = tqdm(range(config.max_iters), desc="Training")
    for it in pbar:
        # Get batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(config.device), y.to(config.device)

        # Update learning rate
        lr = get_lr(it, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass
        _, loss = model(x, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()

        # Logging
        if it % config.log_interval == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr:.2e}'})

        # Evaluation
        if it % config.eval_interval == 0 and it > 0:
            train_loss, val_loss = estimate_loss(model, train_loader, val_loader, config)
            print(f"\nStep {it}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pt')
                print(f"Saved best model with val loss {val_loss:.4f}")

            # Generate sample
            model.eval()
            prompt = full_dataset.encode("The ")[:5].unsqueeze(0).to(config.device)
            generated = model.generate(prompt, max_new_tokens=50, temperature=0.8)
            generated_text = full_dataset.decode(generated[0].tolist())
            print(f"Sample: {generated_text}")
            model.train()

    # Final evaluation
    train_loss, val_loss = estimate_loss(model, train_loader, val_loader, config)
    print(f"\nFinal: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), 'final_model.pt')
    print("Saved final model")

    return model, full_dataset


def check_mhc_stability(model: mHCGPT):
    """Check mHC matrix stability metrics as mentioned in the paper."""
    print("\n" + "=" * 50)
    print("mHC Stability Analysis")
    print("=" * 50)

    for i, block in enumerate(model.blocks):
        for name, mhc in [('attn', block.mhc_attn), ('ffn', block.mhc_ffn)]:
            H_pre, H_res, H_post = mhc.get_matrices()

            # Spectral norm (should be <= 1 for doubly stochastic)
            # Move to CPU for SVD operation (not supported on MPS)
            spectral_norm = torch.linalg.norm(H_res.detach().cpu(), ord=2).item()

            # Row/column sum deviation from 1
            row_dev = (H_res.sum(dim=-1) - 1).abs().max().item()
            col_dev = (H_res.sum(dim=-2) - 1).abs().max().item()

            if i < 2 or i == len(model.blocks) - 1:  # Print first 2 and last layer
                print(f"Layer {i} {name}:")
                print(f"  H_res spectral norm: {spectral_norm:.6f}")
                print(f"  H_res row sum max deviation: {row_dev:.6f}")
                print(f"  H_res col sum max deviation: {col_dev:.6f}")
                print(f"  H_pre range: [{H_pre.min().item():.4f}, {H_pre.max().item():.4f}]")
                print(f"  H_post range: [{H_post.min().item():.4f}, {H_post.max().item():.4f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train mHC-GPT model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--expansion_rate", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = TrainConfig(
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        max_iters=args.max_iters,
        expansion_rate=args.expansion_rate
    )

    if args.device:
        config.device = args.device

    model, dataset = train(config)
    check_mhc_stability(model)
