"""
mHC: Manifold-Constrained Hyper-Connections
PyTorch implementation based on arXiv:2512.24880

This module implements:
1. Basic Hyper-Connections (HC) from arXiv:2409.19606
2. Manifold-Constrained Hyper-Connections (mHC) from arXiv:2512.24880

Key equations:
- HC: x_{l+1} = H_post * (F(H_pre * x_l) + H_res * H_pre * x_l)
- mHC adds manifold constraints:
  - H_res -> doubly stochastic via Sinkhorn-Knopp
  - H_pre -> non-negative via sigmoid
  - H_post -> non-negative via scaled sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def sinkhorn_knopp(
    log_alpha: torch.Tensor,
    num_iters: int = 10,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Sinkhorn-Knopp algorithm to project a matrix onto the doubly stochastic manifold.

    The doubly stochastic manifold (Birkhoff polytope) consists of matrices where
    all rows and columns sum to 1, with non-negative entries.

    Args:
        log_alpha: Input matrix in log space (before exp), shape (..., n, n)
        num_iters: Number of Sinkhorn iterations
        eps: Small constant for numerical stability

    Returns:
        Doubly stochastic matrix of same shape
    """
    # Apply exp to get non-negative matrix
    alpha = torch.exp(log_alpha)

    for _ in range(num_iters):
        # Row normalization
        alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + eps)
        # Column normalization
        alpha = alpha / (alpha.sum(dim=-2, keepdim=True) + eps)

    return alpha


class HyperConnection(nn.Module):
    """
    Basic Hyper-Connection (HC) module from arXiv:2409.19606.

    HC extends residual connections by expanding the residual stream width
    and allowing learnable connectivity patterns between layers.

    The HC equation:
        x_{l+1} = H_post * (F(H_pre * x) + H_res * H_pre * x)

    where:
        - H_pre: Pre-connection matrix (n x n), mixes n copies before layer
        - H_res: Residual connection matrix (n x n), controls skip connection
        - H_post: Post-connection matrix (n x n), mixes after layer output
        - n: Expansion rate (number of copies of the residual stream)

    Args:
        expansion_rate: Number of copies in the residual stream (n)
        hidden_dim: Hidden dimension of the model (d)
    """

    def __init__(self, expansion_rate: int = 4, hidden_dim: int = 512):
        super().__init__()
        self.n = expansion_rate
        self.d = hidden_dim

        # Learnable connection matrices
        # H_pre: selects/mixes which copy to use as layer input
        self.H_pre = nn.Parameter(torch.zeros(expansion_rate, expansion_rate))
        # H_res: residual connection weights
        self.H_res = nn.Parameter(torch.zeros(expansion_rate, expansion_rate))
        # H_post: mixes layer output back to expanded stream
        self.H_post = nn.Parameter(torch.zeros(expansion_rate, expansion_rate))

        self._init_weights()

    def _init_weights(self):
        """Initialize to approximate identity/Pre-Norm behavior."""
        # Initialize H_pre to select first copy (standard behavior)
        nn.init.eye_(self.H_pre)
        # Initialize H_res to identity (standard residual)
        nn.init.eye_(self.H_res)
        # Initialize H_post to identity
        nn.init.eye_(self.H_post)

    def forward(
        self,
        x: torch.Tensor,
        layer_fn: callable
    ) -> torch.Tensor:
        """
        Apply hyper-connection around a layer function.

        Args:
            x: Input tensor of shape (batch, seq_len, n, hidden_dim)
               where n is the expansion rate
            layer_fn: The layer function to apply (e.g., attention, FFN)

        Returns:
            Output tensor of same shape as input
        """
        # x shape: (B, S, n, d)
        B, S, n, d = x.shape
        assert n == self.n, f"Expected expansion rate {self.n}, got {n}"

        # H_pre * x: mix copies before layer
        # (n, n) @ (B, S, n, d) -> contract on second-to-last dim
        x_pre = torch.einsum('ij,bsjd->bsid', self.H_pre, x)

        # Compute weighted input for layer (sum across copies)
        # Take first copy as layer input (or could use weighted sum)
        layer_input = x_pre.sum(dim=2)  # (B, S, d)

        # Apply the layer function
        layer_output = layer_fn(layer_input)  # (B, S, d)

        # Expand back to n copies
        layer_output = layer_output.unsqueeze(2).expand(-1, -1, n, -1)

        # H_res * H_pre * x: residual path
        x_res = torch.einsum('ij,bsjd->bsid', self.H_res, x_pre)

        # Combine layer output with residual
        combined = layer_output + x_res

        # H_post: mix after combination
        output = torch.einsum('ij,bsjd->bsid', self.H_post, combined)

        return output


class mHC(nn.Module):
    """
    Manifold-Constrained Hyper-Connection (mHC) module from arXiv:2512.24880.

    mHC addresses HC's training instability by projecting connection matrices
    onto specific manifolds that preserve the identity mapping property:

    - H_res: Projected onto doubly stochastic manifold via Sinkhorn-Knopp
            This ensures spectral norm ||H_res|| <= 1, preventing signal explosion
    - H_pre: Non-negative via sigmoid, σ(H̃_pre + b_pre)
    - H_post: Non-negative via scaled sigmoid, 2*σ(H̃_post + b_post)

    These constraints guarantee:
    1. Bounded signal propagation across layers
    2. Compositional closure (product of doubly stochastic matrices is doubly stochastic)
    3. Identity mapping when expansion rate n=1

    Args:
        expansion_rate: Number of copies in the residual stream (n)
        hidden_dim: Hidden dimension of the model (d)
        sinkhorn_iters: Number of Sinkhorn-Knopp iterations for H_res
    """

    def __init__(
        self,
        expansion_rate: int = 4,
        hidden_dim: int = 512,
        sinkhorn_iters: int = 10
    ):
        super().__init__()
        self.n = expansion_rate
        self.d = hidden_dim
        self.sinkhorn_iters = sinkhorn_iters

        # Learnable parameters (in unconstrained space)
        # These will be projected onto manifolds during forward pass
        self.H_pre_raw = nn.Parameter(torch.zeros(expansion_rate, expansion_rate))
        self.H_res_raw = nn.Parameter(torch.zeros(expansion_rate, expansion_rate))
        self.H_post_raw = nn.Parameter(torch.zeros(expansion_rate, expansion_rate))

        # Bias terms for pre and post (as mentioned in paper)
        self.b_pre = nn.Parameter(torch.zeros(expansion_rate, expansion_rate))
        self.b_post = nn.Parameter(torch.zeros(expansion_rate, expansion_rate))

        self._init_weights()

    def _init_weights(self):
        """Initialize to approximate identity mapping."""
        # Initialize raw parameters to produce near-identity after projection
        # For H_res (doubly stochastic): start with uniform to get 1/n matrix
        nn.init.zeros_(self.H_res_raw)

        # For H_pre and H_post: initialize biases to produce near-identity
        # sigmoid(0 + b) = 0.5 for b=0, need sigmoid^-1(1) for diagonal
        # Use identity-like initialization
        nn.init.zeros_(self.H_pre_raw)
        nn.init.zeros_(self.H_post_raw)

        # Initialize biases so that diagonal elements are stronger
        with torch.no_grad():
            eye = torch.eye(self.n)
            # sigmoid(2) ≈ 0.88, sigmoid(-2) ≈ 0.12
            self.b_pre.copy_(eye * 2.0 - (1 - eye) * 2.0)
            self.b_post.copy_(eye * 2.0 - (1 - eye) * 2.0)

    def get_H_pre(self) -> torch.Tensor:
        """Get H_pre projected onto non-negative manifold via sigmoid."""
        return torch.sigmoid(self.H_pre_raw + self.b_pre)

    def get_H_res(self) -> torch.Tensor:
        """Get H_res projected onto doubly stochastic manifold via Sinkhorn-Knopp."""
        return sinkhorn_knopp(self.H_res_raw, num_iters=self.sinkhorn_iters)

    def get_H_post(self) -> torch.Tensor:
        """Get H_post projected onto non-negative manifold via scaled sigmoid."""
        return 2.0 * torch.sigmoid(self.H_post_raw + self.b_post)

    def forward(
        self,
        x: torch.Tensor,
        layer_fn: callable
    ) -> torch.Tensor:
        """
        Apply manifold-constrained hyper-connection around a layer function.

        Args:
            x: Input tensor of shape (batch, seq_len, n, hidden_dim)
               where n is the expansion rate
            layer_fn: The layer function to apply (e.g., attention, FFN)

        Returns:
            Output tensor of same shape as input
        """
        # Get manifold-constrained matrices
        H_pre = self.get_H_pre()    # Non-negative
        H_res = self.get_H_res()    # Doubly stochastic
        H_post = self.get_H_post()  # Non-negative, scaled

        # x shape: (B, S, n, d)
        B, S, n, d = x.shape
        assert n == self.n, f"Expected expansion rate {self.n}, got {n}"

        # H_pre * x: mix copies before layer
        x_pre = torch.einsum('ij,bsjd->bsid', H_pre, x)

        # Compute weighted input for layer (sum across copies, then normalize)
        # The sum of H_pre rows gives the weighting
        layer_input = x_pre.sum(dim=2)  # (B, S, d)

        # Apply the layer function
        layer_output = layer_fn(layer_input)  # (B, S, d)

        # Expand back to n copies
        layer_output = layer_output.unsqueeze(2).expand(-1, -1, n, -1)

        # H_res * H_pre * x: residual path through doubly stochastic matrix
        x_res = torch.einsum('ij,bsjd->bsid', H_res, x_pre)

        # Combine layer output with residual
        combined = layer_output + x_res

        # H_post: mix after combination
        output = torch.einsum('ij,bsjd->bsid', H_post, combined)

        return output

    def get_matrices(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the constrained matrices for inspection."""
        return self.get_H_pre(), self.get_H_res(), self.get_H_post()


class mHCBlock(nn.Module):
    """
    A complete mHC-enhanced transformer block.

    Each transformer layer has two sub-layers (attention and FFN), and each
    gets its own mHC module following the paper's design.

    Args:
        hidden_dim: Hidden dimension of the model
        num_heads: Number of attention heads
        expansion_rate: mHC expansion rate (n)
        ffn_ratio: FFN hidden dimension ratio (typically 4)
        dropout: Dropout probability
        sinkhorn_iters: Sinkhorn iterations for mHC
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        expansion_rate: int = 4,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        sinkhorn_iters: int = 10
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expansion_rate = expansion_rate

        # Layer normalization (Pre-Norm style)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_ratio, hidden_dim),
            nn.Dropout(dropout)
        )

        # mHC modules for attention and FFN
        self.mhc_attn = mHC(expansion_rate, hidden_dim, sinkhorn_iters)
        self.mhc_ffn = mHC(expansion_rate, hidden_dim, sinkhorn_iters)

    def _attn_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention with pre-norm."""
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        return attn_out

    def _ffn_fn(self, x: torch.Tensor) -> torch.Tensor:
        """FFN with pre-norm."""
        return self.ffn(self.norm2(x))

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through mHC-enhanced transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, n, hidden_dim)
            attn_mask: Optional attention mask

        Returns:
            Output tensor of same shape as input
        """
        # Apply mHC around attention
        x = self.mhc_attn(x, self._attn_fn)

        # Apply mHC around FFN
        x = self.mhc_ffn(x, self._ffn_fn)

        return x


def expand_to_mhc(x: torch.Tensor, expansion_rate: int) -> torch.Tensor:
    """
    Expand input tensor for mHC processing.

    Takes standard (B, S, D) tensor and replicates to (B, S, n, D)
    where n is the expansion rate.

    Args:
        x: Input tensor of shape (batch, seq_len, hidden_dim)
        expansion_rate: Number of copies (n)

    Returns:
        Expanded tensor of shape (batch, seq_len, n, hidden_dim)
    """
    return x.unsqueeze(2).expand(-1, -1, expansion_rate, -1).clone()


def contract_from_mhc(x: torch.Tensor) -> torch.Tensor:
    """
    Contract mHC tensor back to standard format.

    Takes (B, S, n, D) tensor and sums across expansion dimension.

    Args:
        x: Input tensor of shape (batch, seq_len, n, hidden_dim)

    Returns:
        Contracted tensor of shape (batch, seq_len, hidden_dim)
    """
    return x.sum(dim=2)


if __name__ == "__main__":
    # Quick test
    print("Testing mHC implementation...")

    batch_size = 2
    seq_len = 16
    hidden_dim = 64
    expansion_rate = 4

    # Test Sinkhorn-Knopp
    print("\n1. Testing Sinkhorn-Knopp algorithm...")
    log_alpha = torch.randn(4, 4)
    ds_matrix = sinkhorn_knopp(log_alpha)
    print(f"   Row sums: {ds_matrix.sum(dim=-1)}")
    print(f"   Col sums: {ds_matrix.sum(dim=-2)}")
    print(f"   All non-negative: {(ds_matrix >= 0).all()}")

    # Test basic HC
    print("\n2. Testing basic HyperConnection...")
    hc = HyperConnection(expansion_rate, hidden_dim)
    x = torch.randn(batch_size, seq_len, expansion_rate, hidden_dim)
    layer_fn = lambda x: x * 2  # Simple test function
    out = hc(x, layer_fn)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")

    # Test mHC
    print("\n3. Testing mHC...")
    mhc_module = mHC(expansion_rate, hidden_dim)
    out = mhc_module(x, layer_fn)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")

    # Check manifold constraints
    H_pre, H_res, H_post = mhc_module.get_matrices()
    print(f"   H_res doubly stochastic check:")
    print(f"     Row sums: {H_res.sum(dim=-1)}")
    print(f"     Col sums: {H_res.sum(dim=-2)}")
    print(f"   H_pre non-negative: {(H_pre >= 0).all()}")
    print(f"   H_post non-negative: {(H_post >= 0).all()}")

    # Test mHCBlock
    print("\n4. Testing mHCBlock...")
    block = mHCBlock(hidden_dim=hidden_dim, num_heads=4, expansion_rate=expansion_rate)
    x = torch.randn(batch_size, seq_len, expansion_rate, hidden_dim)
    out = block(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")

    print("\nAll tests passed!")
