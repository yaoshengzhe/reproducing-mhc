"""
GPT-style Language Model with mHC (Manifold-Constrained Hyper-Connections)

This implements a decoder-only transformer architecture enhanced with mHC
for improved training stability and scalability.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from mhc import mHC, expand_to_mhc, contract_from_mhc


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention with rotary positional embeddings option.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 1024
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout

        # QKV projection
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        # Output projection
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        # Compute QKV
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, num_heads, S, head_dim)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask
        attn = attn.masked_fill(self.causal_mask[:, :, :S, :S] == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.proj_dropout(self.proj(out))

        return out


class FeedForward(nn.Module):
    """
    Feed-forward network with GELU activation (GPT-style).
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_ratio: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        ffn_dim = hidden_dim * ffn_ratio

        self.fc1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class mHCTransformerBlock(nn.Module):
    """
    Transformer block with mHC integration.

    Each block contains:
    - mHC-wrapped self-attention
    - mHC-wrapped feed-forward network
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        expansion_rate: int = 4,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        sinkhorn_iters: int = 10
    ):
        super().__init__()
        self.expansion_rate = expansion_rate

        # Layer norms (Pre-Norm style)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Core modules
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout, max_seq_len)
        self.ffn = FeedForward(hidden_dim, ffn_ratio, dropout)

        # mHC modules
        self.mhc_attn = mHC(expansion_rate, hidden_dim, sinkhorn_iters)
        self.mhc_ffn = mHC(expansion_rate, hidden_dim, sinkhorn_iters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (B, S, n, D) where n is expansion rate

        Returns:
            Output of shape (B, S, n, D)
        """
        # mHC around attention
        def attn_fn(h):
            return self.attn(self.norm1(h))

        x = self.mhc_attn(x, attn_fn)

        # mHC around FFN
        def ffn_fn(h):
            return self.ffn(self.norm2(h))

        x = self.mhc_ffn(x, ffn_fn)

        return x


class mHCGPT(nn.Module):
    """
    GPT-style language model with mHC (Manifold-Constrained Hyper-Connections).

    Architecture:
    - Token embedding + learned positional embedding
    - Expand to mHC dimension (n copies)
    - Stack of mHC transformer blocks
    - Contract from mHC dimension
    - Final layer norm + output projection

    Args:
        vocab_size: Size of vocabulary
        hidden_dim: Model hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        expansion_rate: mHC expansion rate (n)
        ffn_ratio: FFN hidden dimension multiplier
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        sinkhorn_iters: Sinkhorn iterations for mHC
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        expansion_rate: int = 4,
        ffn_ratio: int = 4,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        sinkhorn_iters: int = 10
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.expansion_rate = expansion_rate
        self.max_seq_len = max_seq_len

        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.drop = nn.Dropout(dropout)

        # mHC transformer blocks
        self.blocks = nn.ModuleList([
            mHCTransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                expansion_rate=expansion_rate,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
                max_seq_len=max_seq_len,
                sinkhorn_iters=sinkhorn_iters
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(hidden_dim)

        # Output projection (weight tying with embedding)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight  # Weight tying

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs of shape (B, S)
            targets: Target token IDs of shape (B, S) for loss computation

        Returns:
            logits: Output logits of shape (B, S, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        B, S = input_ids.shape
        assert S <= self.max_seq_len, f"Sequence length {S} exceeds max {self.max_seq_len}"

        # Token + positional embeddings
        positions = torch.arange(0, S, dtype=torch.long, device=input_ids.device)
        tok_emb = self.tok_emb(input_ids)
        pos_emb = self.pos_emb(positions)
        x = self.drop(tok_emb + pos_emb)  # (B, S, D)

        # Expand for mHC
        x = expand_to_mhc(x, self.expansion_rate)  # (B, S, n, D)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Contract back from mHC
        x = contract_from_mhc(x)  # (B, S, D)

        # Final layer norm
        x = self.ln_f(x)

        # Compute logits
        logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100
            )

        return logits, loss

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Starting token IDs of shape (B, S)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None for greedy)

        Returns:
            Generated token IDs of shape (B, S + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Configuration presets
def mhc_gpt_small(vocab_size: int = 50257) -> mHCGPT:
    """Small mHC-GPT (~125M params equivalent)."""
    return mHCGPT(
        vocab_size=vocab_size,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        expansion_rate=4,
        ffn_ratio=4,
        max_seq_len=1024,
        dropout=0.1
    )


def mhc_gpt_medium(vocab_size: int = 50257) -> mHCGPT:
    """Medium mHC-GPT (~350M params equivalent)."""
    return mHCGPT(
        vocab_size=vocab_size,
        hidden_dim=1024,
        num_layers=24,
        num_heads=16,
        expansion_rate=4,
        ffn_ratio=4,
        max_seq_len=1024,
        dropout=0.1
    )


def mhc_gpt_tiny(vocab_size: int = 50257) -> mHCGPT:
    """Tiny mHC-GPT for testing."""
    return mHCGPT(
        vocab_size=vocab_size,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        expansion_rate=4,
        ffn_ratio=4,
        max_seq_len=256,
        dropout=0.1
    )


if __name__ == "__main__":
    print("Testing mHC-GPT model...")

    # Test tiny model
    model = mhc_gpt_tiny(vocab_size=1000)
    print(f"Model parameters: {model.count_parameters():,}")

    # Test forward pass
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    targets = torch.randint(0, 1000, (batch_size, seq_len))

    logits, loss = model(input_ids, targets)
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    # Test generation
    prompt = torch.randint(0, 1000, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0)
    print(f"Generated shape: {generated.shape}")

    # Test gradient flow
    loss.backward()
    print("Gradient flow test passed!")

    # Check mHC matrices
    block = model.blocks[0]
    H_pre, H_res, H_post = block.mhc_attn.get_matrices()
    print(f"\nmHC matrix properties (layer 0 attention):")
    print(f"  H_res row sums: {H_res.sum(dim=-1).detach()}")
    print(f"  H_res col sums: {H_res.sum(dim=-2).detach()}")
    print(f"  Spectral norm H_res: {torch.linalg.norm(H_res, ord=2).item():.4f}")

    print("\nAll tests passed!")
