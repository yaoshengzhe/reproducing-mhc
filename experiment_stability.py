"""
Training Stability Experiment: Baseline vs HC vs mHC

This script reproduces the training stability experiment from the mHC paper
(arXiv:2512.24880), comparing:
1. Baseline: Standard transformer with residual connections
2. HC: Hyper-Connections (unconstrained)
3. mHC: Manifold-Constrained Hyper-Connections

Metrics tracked:
- Training loss
- Gradient norm
- Amax Gain Magnitude (signal amplification across layers)
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List, Callable
from dataclasses import dataclass
from tqdm import tqdm

from mhc import HyperConnection, mHC, expand_to_mhc, contract_from_mhc


# =============================================================================
# Model Definitions
# =============================================================================

class CausalSelfAttention(nn.Module):
    """Causal self-attention module."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.causal_mask[:, :, :S, :S] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.proj_dropout(self.proj(out))


class FeedForward(nn.Module):
    """Feed-forward network."""

    def __init__(self, hidden_dim: int, ffn_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * ffn_ratio, bias=False)
        self.fc2 = nn.Linear(hidden_dim * ffn_ratio, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.dropout(F.gelu(self.fc1(x)))))


# -----------------------------------------------------------------------------
# Baseline Transformer Block (Standard Residual)
# -----------------------------------------------------------------------------

class BaselineBlock(nn.Module):
    """Standard transformer block with residual connections."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout, max_seq_len)
        self.ffn = FeedForward(hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# -----------------------------------------------------------------------------
# HC Transformer Block (Unconstrained Hyper-Connections)
# -----------------------------------------------------------------------------

class HCBlock(nn.Module):
    """Transformer block with unconstrained Hyper-Connections."""

    def __init__(self, hidden_dim: int, num_heads: int, expansion_rate: int = 4,
                 dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        self.expansion_rate = expansion_rate
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout, max_seq_len)
        self.ffn = FeedForward(hidden_dim, dropout=dropout)
        self.hc_attn = HyperConnection(expansion_rate, hidden_dim)
        self.hc_ffn = HyperConnection(expansion_rate, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hc_attn(x, lambda h: self.attn(self.norm1(h)))
        x = self.hc_ffn(x, lambda h: self.ffn(self.norm2(h)))
        return x


# -----------------------------------------------------------------------------
# mHC Transformer Block (Manifold-Constrained)
# -----------------------------------------------------------------------------

class mHCBlock(nn.Module):
    """Transformer block with manifold-constrained Hyper-Connections."""

    def __init__(self, hidden_dim: int, num_heads: int, expansion_rate: int = 4,
                 dropout: float = 0.1, max_seq_len: int = 512, sinkhorn_iters: int = 20):
        super().__init__()
        self.expansion_rate = expansion_rate
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout, max_seq_len)
        self.ffn = FeedForward(hidden_dim, dropout=dropout)
        self.mhc_attn = mHC(expansion_rate, hidden_dim, sinkhorn_iters)
        self.mhc_ffn = mHC(expansion_rate, hidden_dim, sinkhorn_iters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mhc_attn(x, lambda h: self.attn(self.norm1(h)))
        x = self.mhc_ffn(x, lambda h: self.ffn(self.norm2(h)))
        return x


# -----------------------------------------------------------------------------
# Full Models
# -----------------------------------------------------------------------------

class BaselineGPT(nn.Module):
    """Baseline GPT with standard residual connections."""

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int,
                 num_heads: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            BaselineBlock(hidden_dim, num_heads, dropout, max_seq_len)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, S = input_ids.shape
        positions = torch.arange(0, S, device=input_ids.device)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def get_activations(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """Get activations at each layer for Amax gain computation."""
        activations = []
        B, S = input_ids.shape
        positions = torch.arange(0, S, device=input_ids.device)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(positions))
        activations.append(x.detach())
        for block in self.blocks:
            x = block(x)
            activations.append(x.detach())
        return activations


class HCGPT(nn.Module):
    """GPT with unconstrained Hyper-Connections."""

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int,
                 num_heads: int, expansion_rate: int = 4, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.expansion_rate = expansion_rate
        self.tok_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            HCBlock(hidden_dim, num_heads, expansion_rate, dropout, max_seq_len)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, S = input_ids.shape
        positions = torch.arange(0, S, device=input_ids.device)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(positions))
        x = expand_to_mhc(x, self.expansion_rate)
        for block in self.blocks:
            x = block(x)
        x = contract_from_mhc(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def get_activations(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """Get activations at each layer for Amax gain computation."""
        activations = []
        B, S = input_ids.shape
        positions = torch.arange(0, S, device=input_ids.device)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(positions))
        x = expand_to_mhc(x, self.expansion_rate)
        activations.append(x.detach())
        for block in self.blocks:
            x = block(x)
            activations.append(x.detach())
        return activations


class mHCGPT(nn.Module):
    """GPT with manifold-constrained Hyper-Connections."""

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int,
                 num_heads: int, expansion_rate: int = 4, max_seq_len: int = 512,
                 dropout: float = 0.1, sinkhorn_iters: int = 10):
        super().__init__()
        self.expansion_rate = expansion_rate
        self.tok_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            mHCBlock(hidden_dim, num_heads, expansion_rate, dropout, max_seq_len, sinkhorn_iters)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, S = input_ids.shape
        positions = torch.arange(0, S, device=input_ids.device)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(positions))
        x = expand_to_mhc(x, self.expansion_rate)
        for block in self.blocks:
            x = block(x)
        x = contract_from_mhc(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def get_activations(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """Get activations at each layer for Amax gain computation."""
        activations = []
        B, S = input_ids.shape
        positions = torch.arange(0, S, device=input_ids.device)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(positions))
        x = expand_to_mhc(x, self.expansion_rate)
        activations.append(x.detach())
        for block in self.blocks:
            x = block(x)
            activations.append(x.detach())
        return activations


# =============================================================================
# Dataset
# =============================================================================

class TextDataset(torch.utils.data.Dataset):
    """Simple character-level dataset."""

    def __init__(self, text: str, block_size: int = 128):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.data = torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y


def get_wikitext2_data():
    """
    Download and return WikiText-2 dataset.

    WikiText-2 is a more challenging dataset (~2MB) extracted from
    Wikipedia Good and Featured articles. More diverse vocabulary
    and complex patterns than TinyShakespeare.

    Source: https://huggingface.co/datasets/Salesforce/wikitext
    """
    import os

    cache_file = "data/wikitext2.txt"
    os.makedirs("data", exist_ok=True)

    if os.path.exists(cache_file):
        print(f"Loading cached WikiText-2 from {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            return f.read()

    print("Downloading WikiText-2 dataset from Hugging Face...")

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "Please install the 'datasets' library: pip install datasets"
        )

    # Load wikitext-2-raw-v1 (character-level compatible)
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    # Combine all splits
    all_text = []
    for split in ["train", "validation", "test"]:
        texts = dataset[split]["text"]
        all_text.extend(texts)

    combined = "\n".join(all_text)

    # Cache for future use
    with open(cache_file, 'w', encoding='utf-8') as f:
        f.write(combined)

    print(f"Downloaded WikiText-2: {len(combined):,} characters")
    return combined


def get_tinyshakespeare_data():
    """Download TinyShakespeare as fallback."""
    import urllib.request
    import os

    cache_file = "data/tiny_shakespeare.txt"
    os.makedirs("data", exist_ok=True)

    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return f.read()

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        with urllib.request.urlopen(url) as response:
            text = response.read().decode('utf-8')
        with open(cache_file, 'w') as f:
            f.write(text)
        return text
    except:
        return "The quick brown fox jumps over the lazy dog. " * 10000


def get_wikitext103_data():
    """
    Download and return WikiText-103 dataset.

    WikiText-103 is a large-scale dataset (~500MB raw) with 100M+ tokens
    extracted from Wikipedia Good and Featured articles.

    Source: https://huggingface.co/datasets/Salesforce/wikitext
    """
    import os

    cache_file = "data/wikitext103.txt"
    os.makedirs("data", exist_ok=True)

    if os.path.exists(cache_file):
        print(f"Loading cached WikiText-103 from {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            return f.read()

    print("Downloading WikiText-103 dataset from Hugging Face...")
    print("This may take a few minutes (~500MB)...")

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "Please install the 'datasets' library: pip install datasets"
        )

    # Load wikitext-103-raw-v1 (character-level compatible)
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

    # Combine all splits
    all_text = []
    for split in ["train", "validation", "test"]:
        texts = dataset[split]["text"]
        all_text.extend(texts)

    combined = "\n".join(all_text)

    # Cache for future use
    with open(cache_file, 'w', encoding='utf-8') as f:
        f.write(combined)

    print(f"Downloaded WikiText-103: {len(combined):,} characters ({len(combined) / 1e6:.1f} MB)")
    return combined


def get_enwik8_data():
    """
    Download and return enwik8 dataset.

    enwik8 is exactly 100MB of Wikipedia XML text, commonly used for
    character-level language modeling benchmarks.

    Source: https://mattmahoney.net/dc/textdata.html
    """
    import os
    import urllib.request
    import zipfile
    import io

    cache_file = "data/enwik8.txt"
    os.makedirs("data", exist_ok=True)

    if os.path.exists(cache_file):
        print(f"Loading cached enwik8 from {cache_file}")
        with open(cache_file, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    print("Downloading enwik8 dataset (100MB)...")
    url = "http://mattmahoney.net/dc/enwik8.zip"

    try:
        with urllib.request.urlopen(url) as response:
            zip_data = response.read()

        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            with z.open('enwik8') as f:
                text = f.read().decode('utf-8', errors='ignore')

        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(text)

        print(f"Downloaded enwik8: {len(text):,} characters ({len(text) / 1e6:.1f} MB)")
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to download enwik8: {e}")


def get_enwik9_data():
    """
    Download and return enwik9 dataset.

    enwik9 is exactly 1GB of Wikipedia XML text, a larger version of enwik8.

    Source: https://mattmahoney.net/dc/textdata.html
    """
    import os
    import urllib.request
    import zipfile
    import io

    cache_file = "data/enwik9.txt"
    os.makedirs("data", exist_ok=True)

    if os.path.exists(cache_file):
        print(f"Loading cached enwik9 from {cache_file}")
        with open(cache_file, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    print("Downloading enwik9 dataset (1GB)...")
    print("This may take several minutes...")
    url = "http://mattmahoney.net/dc/enwik9.zip"

    try:
        with urllib.request.urlopen(url) as response:
            zip_data = response.read()

        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            with z.open('enwik9') as f:
                text = f.read().decode('utf-8', errors='ignore')

        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(text)

        print(f"Downloaded enwik9: {len(text):,} characters ({len(text) / 1e6:.1f} MB)")
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to download enwik9: {e}")


def get_training_data(dataset: str = "wikitext2"):
    """Get training data from specified dataset."""
    if dataset == "wikitext2":
        return get_wikitext2_data()
    elif dataset == "wikitext103":
        return get_wikitext103_data()
    elif dataset == "enwik8":
        return get_enwik8_data()
    elif dataset == "enwik9":
        return get_enwik9_data()
    else:
        return get_tinyshakespeare_data()


# =============================================================================
# Training and Metrics
# =============================================================================

def compute_gradient_norm(model: nn.Module) -> float:
    """Compute total gradient norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total_norm)


def compute_amax_gain(activations: List[torch.Tensor]) -> float:
    """
    Compute Amax Gain Magnitude.

    This measures how much the signal amplitude changes from input to output.
    High values indicate signal explosion, low values indicate vanishing.
    """
    if len(activations) < 2:
        return 1.0

    # Compute absolute max of first and last activations
    first_amax = activations[0].abs().max().item()
    last_amax = activations[-1].abs().max().item()

    if first_amax < 1e-8:
        return 0.0

    return last_amax / first_amax


@dataclass
class TrainConfig:
    """Training configuration matching paper setup where possible.

    Reference: arXiv:2512.24880 Section "Detailed Model Specifications and Hyper-parameters"

    Paper settings (for 3B model):
    - Optimizer: AdamW with betas (0.9, 0.95)
    - Weight decay: 0.1
    - Warmup steps: 2000 (scaled for smaller experiments)
    - LR schedule: Step-based decay with ratios [0.316, 0.1]
    - Expansion rate (n): 4
    - Sinkhorn iterations: 20
    """
    vocab_size: int = 50
    hidden_dim: int = 256
    num_layers: int = 8
    num_heads: int = 8
    expansion_rate: int = 4      # Paper: n=4
    sinkhorn_iters: int = 20     # Paper: t_max=20 (was 10)
    max_seq_len: int = 128
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1    # Paper: 0.1 (was 0.01)
    betas: Tuple[float, float] = (0.9, 0.95)  # Paper: (0.9, 0.95)
    warmup_ratio: float = 0.05   # ~5% warmup (paper uses ~4-7%)
    lr_decay_ratio: float = 0.1  # Final LR = initial * 0.1
    max_iters: int = 1000
    log_interval: int = 20
    dataset: str = "wikitext2"
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def get_lr_scheduler(optimizer, config: TrainConfig):
    """
    Create learning rate scheduler matching paper setup.

    Paper uses step-based decay with ratios [0.316, 0.1].
    We implement warmup + cosine decay for simplicity and comparable behavior.
    """
    warmup_steps = int(config.max_iters * config.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return (step + 1) / warmup_steps
        else:
            # Cosine decay to lr_decay_ratio of initial LR
            progress = (step - warmup_steps) / (config.max_iters - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            # Decay from 1.0 to lr_decay_ratio
            return config.lr_decay_ratio + (1 - config.lr_decay_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    config: TrainConfig,
    model_name: str,
    use_grad_clip: bool = True
) -> Dict[str, List[float]]:
    """Train a model and collect metrics."""

    model = model.to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas
    )
    scheduler = get_lr_scheduler(optimizer, config)

    metrics = {
        'loss': [],
        'grad_norm': [],
        'amax_gain': [],
        'steps': []
    }

    model.train()
    train_iter = iter(train_loader)
    pbar = tqdm(range(config.max_iters), desc=f"Training {model_name}")

    for step in pbar:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(config.device), y.to(config.device)

        # Forward pass
        _, loss = model(x, y)

        # Check for NaN/Inf
        if not torch.isfinite(loss):
            print(f"\n{model_name}: Loss became NaN/Inf at step {step}")
            # Fill remaining with NaN
            remaining_steps = list(range(step, config.max_iters, config.log_interval))
            metrics['loss'].extend([float('nan')] * len(remaining_steps))
            metrics['grad_norm'].extend([float('nan')] * len(remaining_steps))
            metrics['amax_gain'].extend([float('nan')] * len(remaining_steps))
            metrics['steps'].extend(remaining_steps)
            break

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Compute gradient norm before clipping
        grad_norm = compute_gradient_norm(model)

        # Gradient clipping (optional - disable for HC to show instability)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        # Compute Amax gain
        with torch.no_grad():
            activations = model.get_activations(x)
            amax_gain = compute_amax_gain(activations)

        # Log metrics
        if step % config.log_interval == 0:
            metrics['loss'].append(loss.item())
            metrics['grad_norm'].append(grad_norm)
            metrics['amax_gain'].append(amax_gain)
            metrics['steps'].append(step)
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'grad': f'{grad_norm:.1f}',
                'amax': f'{amax_gain:.1f}'
            })

    return metrics


# =============================================================================
# Plotting
# =============================================================================

def plot_results(results: Dict[str, Dict[str, List[float]]], save_path: str = "training_stability.png"):
    """Plot training stability comparison."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    colors = {'Baseline': '#2ecc71', 'HC': '#e74c3c', 'mHC': '#3498db'}
    linestyles = {'Baseline': '-', 'HC': '--', 'mHC': '-'}

    # Plot 1: Training Loss
    ax1 = axes[0]
    for name, metrics in results.items():
        steps = metrics['steps']
        loss = metrics['loss']
        ax1.plot(steps, loss, label=name, color=colors[name], linestyle=linestyles[name], linewidth=2)
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Plot 2: Gradient Norm
    ax2 = axes[1]
    for name, metrics in results.items():
        steps = metrics['steps']
        grad_norm = metrics['grad_norm']
        ax2.plot(steps, grad_norm, label=name, color=colors[name], linestyle=linestyles[name], linewidth=2)
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Gradient Norm', fontsize=12)
    ax2.set_title('Gradient Norm', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Plot 3: Amax Gain Magnitude
    ax3 = axes[2]
    for name, metrics in results.items():
        steps = metrics['steps']
        amax = metrics['amax_gain']
        ax3.plot(steps, amax, label=name, color=colors[name], linestyle=linestyles[name], linewidth=2)
    ax3.set_xlabel('Training Steps', fontsize=12)
    ax3.set_ylabel('Amax Gain', fontsize=12)
    ax3.set_title('Amax Gain Magnitude\n(Signal Amplification)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    plt.suptitle('Training Stability: Baseline vs HC vs mHC', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_detailed_comparison(results: Dict[str, Dict[str, List[float]]], save_path: str = "stability_detailed.png"):
    """Create detailed comparison plots."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = {'Baseline': '#2ecc71', 'HC': '#e74c3c', 'mHC': '#3498db'}

    # Plot 1: Loss comparison
    ax1 = axes[0, 0]
    for name, metrics in results.items():
        ax1.plot(metrics['steps'], metrics['loss'], label=name, color=colors[name], linewidth=2)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Gradient norm (log scale)
    ax2 = axes[0, 1]
    for name, metrics in results.items():
        ax2.plot(metrics['steps'], metrics['grad_norm'], label=name, color=colors[name], linewidth=2)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Gradient Norm (log)')
    ax2.set_title('Gradient Norm Over Training')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Amax Gain (log scale)
    ax3 = axes[1, 0]
    for name, metrics in results.items():
        ax3.plot(metrics['steps'], metrics['amax_gain'], label=name, color=colors[name], linewidth=2)
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Amax Gain (log)')
    ax3.set_title('Signal Amplification (Amax Gain)')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Final metrics bar chart
    ax4 = axes[1, 1]
    names = list(results.keys())
    x = np.arange(len(names))
    width = 0.25

    # Get final values (or last valid values)
    final_loss = []
    final_grad = []
    final_amax = []
    for name in names:
        losses = [l for l in results[name]['loss'] if not np.isnan(l)]
        grads = [g for g in results[name]['grad_norm'] if not np.isnan(g)]
        amaxs = [a for a in results[name]['amax_gain'] if not np.isnan(a)]
        final_loss.append(losses[-1] if losses else 0)
        final_grad.append(np.mean(grads[-5:]) if grads else 0)
        final_amax.append(np.mean(amaxs[-5:]) if amaxs else 0)

    # Normalize for visualization
    max_loss = max(final_loss) if max(final_loss) > 0 else 1
    max_amax = max(final_amax) if max(final_amax) > 0 else 1

    bars1 = ax4.bar(x - width, [l/max_loss for l in final_loss], width, label='Final Loss (norm)', color='#3498db')
    bars2 = ax4.bar(x, [np.log10(a+1)/np.log10(max_amax+1) for a in final_amax], width, label='Amax Gain (norm log)', color='#e74c3c')

    ax4.set_ylabel('Normalized Value')
    ax4.set_title('Final Metrics Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(names)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Detailed Training Stability Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved detailed plot to {save_path}")
    plt.close()


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(config: Optional[TrainConfig] = None):
    """Run the full training stability experiment."""

    if config is None:
        config = TrainConfig()

    print("=" * 60)
    print("Training Stability Experiment: Baseline vs HC vs mHC")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Hidden dim: {config.hidden_dim}, Num layers: {config.num_layers}")
    print(f"Expansion rate: {config.expansion_rate}, Sinkhorn iters: {config.sinkhorn_iters}")
    print(f"LR: {config.learning_rate}, Weight decay: {config.weight_decay}")
    print(f"Betas: {config.betas}, Warmup: {config.warmup_ratio*100:.0f}%")
    print(f"Max iterations: {config.max_iters}")
    print("=" * 60)

    # Create dataset
    print(f"\nLoading {config.dataset} dataset...")
    text = get_training_data(config.dataset)
    dataset = TextDataset(text, config.max_seq_len)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True
    )
    config.vocab_size = dataset.vocab_size
    print(f"Vocab size: {config.vocab_size}")

    results = {}

    # 1. Train Baseline
    print("\n" + "-" * 40)
    print("Training Baseline (Standard Residual)")
    print("-" * 40)
    baseline = BaselineGPT(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_len
    )
    print(f"Parameters: {sum(p.numel() for p in baseline.parameters()):,}")
    results['Baseline'] = train_model(baseline, train_loader, config, "Baseline")

    # 2. Train HC
    print("\n" + "-" * 40)
    print("Training HC (Unconstrained Hyper-Connections)")
    print("-" * 40)
    hc_model = HCGPT(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        expansion_rate=config.expansion_rate,
        max_seq_len=config.max_seq_len
    )
    print(f"Parameters: {sum(p.numel() for p in hc_model.parameters()):,}")
    results['HC'] = train_model(hc_model, train_loader, config, "HC")

    # 3. Train mHC
    print("\n" + "-" * 40)
    print("Training mHC (Manifold-Constrained)")
    print("-" * 40)
    mhc_model = mHCGPT(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        expansion_rate=config.expansion_rate,
        max_seq_len=config.max_seq_len,
        sinkhorn_iters=config.sinkhorn_iters
    )
    print(f"Parameters: {sum(p.numel() for p in mhc_model.parameters()):,}")
    results['mHC'] = train_model(mhc_model, train_loader, config, "mHC")

    # Generate plots - save to images/ folder
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)

    import os
    os.makedirs("images", exist_ok=True)

    plot_results(results, "images/training_stability.png")
    plot_detailed_comparison(results, "images/stability_detailed.png")

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    for name, metrics in results.items():
        valid_losses = [l for l in metrics['loss'] if not np.isnan(l)]
        valid_grads = [g for g in metrics['grad_norm'] if not np.isnan(g)]
        valid_amax = [a for a in metrics['amax_gain'] if not np.isnan(a)]

        print(f"\n{name}:")
        if valid_losses:
            print(f"  Final Loss: {valid_losses[-1]:.4f}")
            print(f"  Min Loss: {min(valid_losses):.4f}")
        if valid_grads:
            print(f"  Avg Gradient Norm: {np.mean(valid_grads):.2f}")
            print(f"  Max Gradient Norm: {max(valid_grads):.2f}")
        if valid_amax:
            print(f"  Avg Amax Gain: {np.mean(valid_amax):.2f}")
            print(f"  Max Amax Gain: {max(valid_amax):.2f}")
        print(f"  Training completed: {len(valid_losses) == len(metrics['steps'])}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training Stability Experiment")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--expansion_rate", type=int, default=4, help="mHC expansion rate (paper: n=4)")
    parser.add_argument("--sinkhorn_iters", type=int, default=20, help="Sinkhorn-Knopp iterations (paper: 20)")
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay (paper: 0.1)")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio (paper: ~5%%)")
    parser.add_argument("--dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "wikitext103", "enwik8", "enwik9", "shakespeare"],
                        help="Dataset: wikitext2 (~2MB), wikitext103 (~500MB), enwik8 (100MB), enwik9 (1GB), shakespeare (~1MB)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = TrainConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        expansion_rate=args.expansion_rate,
        sinkhorn_iters=args.sinkhorn_iters,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio
    )

    if args.device:
        config.device = args.device

    # Store dataset choice in config
    config.dataset = args.dataset

    run_experiment(config)
