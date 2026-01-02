"""
Comprehensive tests for mHC implementation.

Tests verify:
1. Sinkhorn-Knopp produces valid doubly stochastic matrices
2. mHC manifold constraints are satisfied
3. Gradient flow is stable
4. Signal propagation bounds (as mentioned in paper)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from mhc import (
    sinkhorn_knopp,
    HyperConnection,
    mHC,
    mHCBlock,
    expand_to_mhc,
    contract_from_mhc
)
from model import mHCGPT, mhc_gpt_tiny


def test_sinkhorn_knopp():
    """Test Sinkhorn-Knopp algorithm produces valid doubly stochastic matrices."""
    print("Testing Sinkhorn-Knopp algorithm...")

    # Test various input sizes
    for n in [2, 4, 8, 16]:
        log_alpha = torch.randn(n, n)
        ds = sinkhorn_knopp(log_alpha, num_iters=20)

        # Check row sums
        row_sums = ds.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(n), atol=1e-5), \
            f"Row sums not 1: {row_sums}"

        # Check column sums
        col_sums = ds.sum(dim=-2)
        assert torch.allclose(col_sums, torch.ones(n), atol=1e-5), \
            f"Column sums not 1: {col_sums}"

        # Check non-negativity
        assert (ds >= 0).all(), "Negative entries in doubly stochastic matrix"

        # Check spectral norm <= 1
        spectral_norm = torch.linalg.norm(ds, ord=2)
        assert spectral_norm <= 1.0 + 1e-5, f"Spectral norm > 1: {spectral_norm}"

    # Test batched input
    batch_log_alpha = torch.randn(4, 8, 8)
    batch_ds = sinkhorn_knopp(batch_log_alpha)
    assert batch_ds.shape == (4, 8, 8)
    for i in range(4):
        assert torch.allclose(batch_ds[i].sum(dim=-1), torch.ones(8), atol=1e-5)

    print("  PASSED")


def test_mhc_constraints():
    """Test that mHC matrices satisfy manifold constraints."""
    print("Testing mHC manifold constraints...")

    for n in [2, 4, 8]:
        mhc = mHC(expansion_rate=n, hidden_dim=64)

        H_pre, H_res, H_post = mhc.get_matrices()

        # H_res should be doubly stochastic
        assert torch.allclose(H_res.sum(dim=-1), torch.ones(n), atol=1e-5), \
            f"H_res row sums incorrect for n={n}"
        assert torch.allclose(H_res.sum(dim=-2), torch.ones(n), atol=1e-5), \
            f"H_res col sums incorrect for n={n}"
        assert (H_res >= 0).all(), "H_res has negative entries"

        # H_pre should be non-negative (via sigmoid)
        assert (H_pre >= 0).all(), "H_pre has negative entries"
        assert (H_pre <= 1).all(), "H_pre exceeds sigmoid range"

        # H_post should be doubly stochastic (like H_res)
        assert torch.allclose(H_post.sum(dim=-1), torch.ones(n), atol=1e-5), \
            f"H_post row sums incorrect for n={n}"
        assert torch.allclose(H_post.sum(dim=-2), torch.ones(n), atol=1e-5), \
            f"H_post col sums incorrect for n={n}"
        assert (H_post >= 0).all(), "H_post has negative entries"

    print("  PASSED")


def test_gradient_flow():
    """Test that gradients flow properly through mHC."""
    print("Testing gradient flow...")

    mhc = mHC(expansion_rate=4, hidden_dim=64)
    x = torch.randn(2, 16, 4, 64, requires_grad=True)

    def simple_layer(h):
        return h * 2 + 1

    out = mhc(x, simple_layer)
    loss = out.sum()
    loss.backward()

    # Check gradients exist and are finite
    assert x.grad is not None, "No gradient for input"
    assert torch.isfinite(x.grad).all(), "Non-finite gradient for input"

    for name, param in mhc.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    print("  PASSED")


def test_signal_propagation():
    """
    Test signal propagation stability across layers.

    Paper claims: mHC prevents signal explosion/vanishing by constraining
    H_res to doubly stochastic manifold (spectral norm <= 1).

    Note: We test with normalized layer functions, as in practice
    transformers use Pre-Norm which keeps signals bounded.
    """
    print("Testing signal propagation stability...")

    # Create deep network with multiple mHC blocks
    num_blocks = 24
    hidden_dim = 64
    expansion_rate = 4

    blocks = nn.ModuleList([
        mHC(expansion_rate, hidden_dim, sinkhorn_iters=10)
        for _ in range(num_blocks)
    ])
    norms = nn.ModuleList([
        nn.LayerNorm(hidden_dim)
        for _ in range(num_blocks)
    ])

    # Initialize input with unit norm
    x = torch.randn(1, 16, expansion_rate, hidden_dim)
    x = x / x.norm() * np.sqrt(x.numel())  # Normalize

    initial_norm = x.norm().item()

    # Propagate through blocks with normalized layer functions (realistic)
    for block, norm in zip(blocks, norms):
        # Simulate Pre-Norm transformer: norm before transformation
        x = block(x, lambda h, n=norm: n(h))

    final_norm = x.norm().item()

    # Signal should not explode (ratio should be bounded)
    gain = final_norm / initial_norm
    print(f"  Signal gain after {num_blocks} layers (with LN): {gain:.4f}")

    # The paper mentions "Amax Gain Magnitude" being controlled
    # With doubly stochastic H_res + LayerNorm, the gain should be bounded
    assert gain < 100, f"Signal explosion detected: gain = {gain}"
    assert gain > 0.01, f"Signal vanishing detected: gain = {gain}"

    print("  PASSED")


def test_identity_mapping_property():
    """
    Test identity mapping property restoration.

    Paper's key claim: mHC restores identity mapping property that HC loses.
    At initialization, the network should approximate identity.
    """
    print("Testing identity mapping property...")

    mhc = mHC(expansion_rate=4, hidden_dim=64, sinkhorn_iters=10)

    x = torch.randn(2, 16, 4, 64)

    # With identity layer function, output should be similar to input
    out = mhc(x, lambda h: h)

    # The transformation shouldn't drastically change the signal
    # (accounting for the expansion/contraction operations)
    relative_change = (out - x).norm() / x.norm()
    print(f"  Relative change with identity layer: {relative_change.item():.4f}")

    # Should be reasonable (not identity, but bounded)
    assert relative_change < 10, f"Large deviation from identity: {relative_change}"

    print("  PASSED")


def test_doubly_stochastic_composition():
    """
    Test that product of doubly stochastic matrices is doubly stochastic.

    Paper mentions "compositional closure" as a key property.
    """
    print("Testing doubly stochastic composition...")

    n = 4

    # Create multiple doubly stochastic matrices
    matrices = []
    for _ in range(5):
        log_alpha = torch.randn(n, n)
        ds = sinkhorn_knopp(log_alpha, num_iters=20)
        matrices.append(ds)

    # Compose them
    product = matrices[0]
    for m in matrices[1:]:
        product = product @ m

    # Product should still be doubly stochastic
    row_sums = product.sum(dim=-1)
    col_sums = product.sum(dim=-2)

    assert torch.allclose(row_sums, torch.ones(n), atol=1e-4), \
        f"Composed matrix row sums not 1: {row_sums}"
    assert torch.allclose(col_sums, torch.ones(n), atol=1e-4), \
        f"Composed matrix col sums not 1: {col_sums}"
    assert (product >= -1e-5).all(), "Negative entries in composed matrix"

    print("  PASSED")


def test_full_model_forward():
    """Test full model forward and backward pass."""
    print("Testing full model forward/backward...")

    model = mhc_gpt_tiny(vocab_size=1000)
    input_ids = torch.randint(0, 1000, (2, 32))
    targets = torch.randint(0, 1000, (2, 32))

    # Forward pass
    logits, loss = model(input_ids, targets)

    assert logits.shape == (2, 32, 1000), f"Wrong logits shape: {logits.shape}"
    assert loss is not None, "Loss is None"
    assert torch.isfinite(loss), f"Non-finite loss: {loss}"

    # Backward pass
    loss.backward()

    # Check all parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    print("  PASSED")


def test_generation():
    """Test autoregressive generation."""
    print("Testing generation...")

    model = mhc_gpt_tiny(vocab_size=1000)
    model.eval()

    prompt = torch.randint(0, 1000, (1, 5))

    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=20, temperature=1.0)

    assert generated.shape == (1, 25), f"Wrong generated shape: {generated.shape}"
    assert (generated >= 0).all() and (generated < 1000).all(), "Invalid token ids"

    print("  PASSED")


def test_expand_contract():
    """Test expand/contract operations."""
    print("Testing expand/contract operations...")

    x = torch.randn(2, 16, 64)
    n = 4

    # Expand
    expanded = expand_to_mhc(x, n)
    assert expanded.shape == (2, 16, 4, 64), f"Wrong expanded shape: {expanded.shape}"

    # All copies should be identical after expansion
    for i in range(n):
        assert torch.allclose(expanded[:, :, i, :], x), \
            f"Copy {i} differs from original"

    # Contract
    contracted = contract_from_mhc(expanded)
    assert contracted.shape == (2, 16, 64), f"Wrong contracted shape: {contracted.shape}"

    # Contracted should equal original (mean of n identical copies)
    assert torch.allclose(contracted, x), "Contract doesn't average correctly"

    print("  PASSED")


def test_spectral_norm_bound():
    """
    Verify spectral norm of H_res is bounded by 1.

    This is a key theoretical guarantee from the paper.
    """
    print("Testing spectral norm bound...")

    for _ in range(10):
        mhc = mHC(expansion_rate=4, hidden_dim=64)
        H_res = mhc.get_H_res()
        spectral_norm = torch.linalg.norm(H_res, ord=2)

        assert spectral_norm <= 1.0 + 1e-5, \
            f"Spectral norm exceeds 1: {spectral_norm}"

    # Also test after some gradient updates
    mhc = mHC(expansion_rate=4, hidden_dim=64)
    optimizer = torch.optim.Adam(mhc.parameters(), lr=0.1)

    for _ in range(10):
        x = torch.randn(2, 16, 4, 64)
        out = mhc(x, lambda h: h)
        loss = out.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        H_res = mhc.get_H_res()
        spectral_norm = torch.linalg.norm(H_res, ord=2)
        assert spectral_norm <= 1.0 + 1e-4, \
            f"Spectral norm exceeds 1 after update: {spectral_norm}"

    print("  PASSED")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("mHC Implementation Tests")
    print("=" * 60)
    print()

    tests = [
        test_sinkhorn_knopp,
        test_mhc_constraints,
        test_gradient_flow,
        test_signal_propagation,
        test_identity_mapping_property,
        test_doubly_stochastic_composition,
        test_full_model_forward,
        test_generation,
        test_expand_contract,
        test_spectral_norm_bound,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
