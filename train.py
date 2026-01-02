"""
Training script for mHC-GPT model.

This script demonstrates training the mHC-GPT model on a simple text dataset.
For demonstration purposes, it uses a small synthetic or character-level dataset.
"""

import os
import math
import argparse
from typing import Optional, Tuple
from dataclasses import dataclass, field, asdict

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import mHCGPT


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
    betas: Tuple[float, float] = (0.9, 0.95)

    # Mixed precision
    use_amp: bool = True

    # Logging
    log_interval: int = 100
    eval_interval: int = 500
    eval_iters: int = 50

    # Checkpointing
    save_dir: str = "checkpoints"
    save_best: str = "best_model.pt"
    save_final: str = "final_model.pt"
    resume_from: Optional[str] = None

    # Data loading
    num_workers: int = 0  # 0 for main process only
    seed: int = 42

    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class CharDataset(Dataset):
    """
    Simple character-level dataset.
    Supports contiguous train/val split for proper language modeling evaluation.
    """

    def __init__(self, data: torch.Tensor, char_to_idx: dict, idx_to_char: dict, block_size: int = 128):
        self.block_size = block_size
        self.data = data
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.vocab_size = len(char_to_idx)

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

    def decode(self, indices):
        return ''.join([self.idx_to_char.get(i, '?') for i in indices])

    def encode(self, text):
        return torch.tensor([self.char_to_idx.get(c, 0) for c in text], dtype=torch.long)

    @classmethod
    def from_text(cls, text: str, block_size: int = 128, train_split: float = 0.9):
        """
        Create train and val datasets from text with contiguous split.

        Args:
            text: Raw text to tokenize
            block_size: Context window size
            train_split: Fraction of data for training

        Returns:
            Tuple of (train_dataset, val_dataset, vocab_size)
        """
        # Character-level tokenization
        chars = sorted(list(set(text)))
        char_to_idx = {c: i for i, c in enumerate(chars)}
        idx_to_char = {i: c for c, i in char_to_idx.items()}

        # Encode full text
        data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

        # Contiguous split (not random!)
        split_idx = int(len(data) * train_split)
        train_data = data[:split_idx]
        val_data = data[split_idx:]

        train_dataset = cls(train_data, char_to_idx, idx_to_char, block_size)
        val_dataset = cls(val_data, char_to_idx, idx_to_char, block_size)

        return train_dataset, val_dataset


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
        if config.warmup_iters == 0:
            return config.learning_rate
        return config.learning_rate * it / config.warmup_iters

    # After decay
    if it > config.lr_decay_iters:
        return config.min_lr

    # Cosine decay
    decay_iters = config.lr_decay_iters - config.warmup_iters
    if decay_iters == 0:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / decay_iters
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    use_amp: bool = False
) -> Tuple[float, float]:
    """Estimate train and validation loss."""
    model.eval()
    losses = {'train': [], 'val': []}

    for split, loader in [('train', train_loader), ('val', val_loader)]:
        # Use fresh iterator to avoid affecting loader state
        loader_iter = iter(loader)
        for i in range(config.eval_iters):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                break
            x, y = x.to(config.device), y.to(config.device)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    _, loss = model(x, y)
            else:
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

    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    print(f"Training on device: {config.device}")
    print(f"Random seed: {config.seed}")

    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)

    # Create dataset with contiguous split
    print("Creating dataset...")
    text = get_sample_text()
    train_dataset, val_dataset = CharDataset.from_text(text, config.max_seq_len)

    pin_memory = config.device == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory
    )

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    print(f"Vocab size: {train_dataset.vocab_size}")

    # Create model
    print("Creating model...")
    model = mHCGPT(
        vocab_size=train_dataset.vocab_size,
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
        betas=config.betas
    )

    # Mixed precision scaler (only for CUDA)
    use_amp = config.use_amp and config.device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Using mixed precision training (AMP)")

    # Resume from checkpoint if specified
    start_iter = 0
    best_val_loss = float('inf')
    if config.resume_from and os.path.exists(config.resume_from):
        print(f"Resuming from checkpoint: {config.resume_from}")
        checkpoint = torch.load(config.resume_from, map_location=config.device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_iter = checkpoint.get('iter', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if scaler and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        print(f"Resumed from iteration {start_iter}")

    # Training loop
    print("\nStarting training...")
    model.train()
    train_iter = iter(train_loader)

    pbar = tqdm(range(start_iter, config.max_iters), desc="Training")
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

        # Forward pass with optional AMP
        if use_amp:
            with torch.amp.autocast("cuda"):
                _, loss = model(x, y)
            # Backward pass with scaler
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        # Logging
        if it % config.log_interval == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr:.2e}'})

        # Evaluation
        if it % config.eval_interval == 0 and it > 0:
            train_loss, val_loss = estimate_loss(model, train_loader, val_loader, config, use_amp)
            print(f"\nStep {it}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(config.save_dir, config.save_best)
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter': it,
                    'best_val_loss': best_val_loss,
                    'config': asdict(config),
                }
                if scaler:
                    checkpoint['scaler'] = scaler.state_dict()
                torch.save(checkpoint, save_path)
                print(f"Saved best model with val loss {val_loss:.4f}")

            # Generate sample
            model.eval()
            prompt = train_dataset.encode("The ")[:5].unsqueeze(0).to(config.device)
            generated = model.generate(prompt, max_new_tokens=50, temperature=0.8)
            generated_text = train_dataset.decode(generated[0].tolist())
            print(f"Sample: {generated_text}")
            model.train()

    # Final evaluation
    train_loss, val_loss = estimate_loss(model, train_loader, val_loader, config, use_amp)
    print(f"\nFinal: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

    # Save final model
    save_path = os.path.join(config.save_dir, config.save_final)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter': config.max_iters,
        'best_val_loss': best_val_loss,
        'config': asdict(config),
    }, save_path)
    print(f"Saved final model to {save_path}")

    return model, train_dataset


def check_mhc_stability(model: mHCGPT):
    """Check mHC matrix stability metrics as mentioned in the paper."""
    print("\n" + "=" * 50)
    print("mHC Stability Analysis")
    print("=" * 50)

    for i, block in enumerate(model.blocks):
        for name, mhc in [('attn', block.mhc_attn), ('ffn', block.mhc_ffn)]:
            H_pre, H_res, H_post = mhc.get_matrices()

            # Move to CPU for SVD operation (not supported on MPS)
            H_res_cpu = H_res.detach().cpu()
            H_post_cpu = H_post.detach().cpu()

            # Spectral norm (should be <= 1 for doubly stochastic)
            res_spectral = torch.linalg.norm(H_res_cpu, ord=2).item()
            post_spectral = torch.linalg.norm(H_post_cpu, ord=2).item()

            # Row/column sum deviation from 1 (for doubly stochastic matrices)
            res_row_dev = (H_res.sum(dim=-1) - 1).abs().max().item()
            res_col_dev = (H_res.sum(dim=-2) - 1).abs().max().item()
            post_row_dev = (H_post.sum(dim=-1) - 1).abs().max().item()
            post_col_dev = (H_post.sum(dim=-2) - 1).abs().max().item()

            if i < 2 or i == len(model.blocks) - 1:  # Print first 2 and last layer
                print(f"Layer {i} {name}:")
                print(f"  H_pre range: [{H_pre.min().item():.4f}, {H_pre.max().item():.4f}]")
                print(f"  H_res spectral norm: {res_spectral:.6f}, row dev: {res_row_dev:.6f}, col dev: {res_col_dev:.6f}")
                print(f"  H_post spectral norm: {post_spectral:.6f}, row dev: {post_row_dev:.6f}, col dev: {post_col_dev:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train mHC-GPT model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--expansion_rate", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    config = TrainConfig(
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        max_iters=args.max_iters,
        expansion_rate=args.expansion_rate,
        seed=args.seed,
        resume_from=args.resume,
        use_amp=not args.no_amp,
        save_dir=args.save_dir
    )

    if args.device:
        config.device = args.device

    model, dataset = train(config)
    check_mhc_stability(model)
