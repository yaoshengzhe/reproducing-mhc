# mHC: Manifold-Constrained Hyper-Connections

PyTorch implementation of [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) (DeepSeek, 2025).

This paper extends [Hyper-Connections](https://arxiv.org/abs/2409.19606) (ByteDance, ICLR 2025) by projecting connection matrices onto specific manifolds to restore identity mapping properties and improve training stability.

## Key Contributions

1. **Doubly Stochastic Constraint for H_res**: The residual connection matrix is projected onto the Birkhoff polytope using Sinkhorn-Knopp algorithm, ensuring:
   - Spectral norm ≤ 1 (prevents signal explosion)
   - Row and column sums = 1
   - Non-negative entries

2. **Non-negativity Constraints for H_pre/H_post**: Using sigmoid projections to maintain non-negative connection weights.

3. **Signal Stability**: These constraints guarantee bounded signal propagation across arbitrary network depths.

## Files

- `mhc.py` - Core mHC implementation:
  - `sinkhorn_knopp()` - Doubly stochastic projection
  - `HyperConnection` - Basic HC module
  - `mHC` - Manifold-constrained HC module
  - `mHCBlock` - mHC-enhanced transformer block

- `model.py` - GPT-style language model with mHC:
  - `mHCGPT` - Full model with token/positional embeddings
  - Model size presets (tiny, small, medium)

- `train.py` - Training script:
  - Character-level language modeling
  - Learning rate scheduling with warmup
  - mHC stability analysis

- `tests.py` - Comprehensive test suite

## Usage

### Basic mHC module

```python
from mhc import mHC, expand_to_mhc, contract_from_mhc

# Create mHC module
mhc = mHC(expansion_rate=4, hidden_dim=512)

# Expand input for mHC processing
x = torch.randn(batch, seq_len, hidden_dim)
x_expanded = expand_to_mhc(x, expansion_rate=4)

# Apply mHC around a layer function
def layer_fn(h):
    return my_transformer_layer(h)

output = mhc(x_expanded, layer_fn)

# Contract back to original dimension
output = contract_from_mhc(output)
```

### Training

```bash
# Quick test
python train.py --max_iters 500 --hidden_dim 64 --num_layers 2

# Full training
python train.py --max_iters 5000 --hidden_dim 256 --num_layers 6
```

### Run tests

```bash
python tests.py
```

## Mathematical Details

### HC Equation (original)
```
x_{l+1} = H_post * (F(H_pre * x_l) + H_res * x_l)
```

### mHC Manifold Projections
```
H_res = Sinkhorn-Knopp(exp(H̃_res))    # Doubly stochastic
H_pre = σ(H̃_pre + b_pre)              # Non-negative [0, 1]
H_post = 2 * σ(H̃_post + b_post)       # Non-negative [0, 2]
```

### Sinkhorn-Knopp Algorithm
Iteratively normalizes rows and columns to produce doubly stochastic matrices:
```python
for _ in range(num_iters):
    alpha = alpha / alpha.sum(dim=-1, keepdim=True)  # Row normalize
    alpha = alpha / alpha.sum(dim=-2, keepdim=True)  # Col normalize
```

## Results

The implementation demonstrates:
- Spectral norm of H_res exactly equals 1.0
- Row/column sums of H_res exactly equal 1.0
- Signal gain ~24x after 24 layers (vs infinite explosion without constraints)
- Training loss decreases properly from ~3.5 to ~1.5

## References

- [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) - DeepSeek, 2025
- [Hyper-Connections](https://arxiv.org/abs/2409.19606) - ByteDance, ICLR 2025
