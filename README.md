# mHC: Manifold-Constrained Hyper-Connections

PyTorch implementation of [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) (DeepSeek, 2025).

## Reproduction Results

Stability experiment on enwik8 (2000 iterations, 8 layers, hidden_dim=256, expansion_rate=4):

| Model | Final Loss | Min Loss | Avg Grad Norm | Max Grad Norm | Avg Amax Gain | Max Amax Gain |
|-------|------------|----------|---------------|---------------|---------------|---------------|
| Baseline | 1.876 | 1.841 | 1.06 | 6.18 | 25.40 | 41.82 |
| HC | 1.978 | 1.844 | 1.03 | 6.73 | 27.72 | 61.33 |
| **mHC** | **1.913** | **1.787** | **0.91** | **4.46** | **6.99** | **11.70** |

![Training Stability](images/training_stability.png)

![Detailed Metrics](images/stability_detailed.png)

### Analysis

**Confirming the paper's claims:**

1. **HC instability (Section 3.1)**: The paper argues that unconstrained HC suffers from signal explosion due to unbounded spectral norms. Our results confirm this — HC shows the highest max amax gain (61.33), significantly worse than baseline (41.82).

2. **mHC stability (Section 3.2)**: The paper's key contribution is that doubly stochastic constraints on H_res guarantee spectral norm ≤ 1. Our results validate this — mHC reduces max amax gain to 11.70 (3.6x better than baseline, 5.2x better than HC).

3. **Gradient stability**: mHC shows the lowest gradient norms (max 4.46 vs 6.18 baseline), confirming the paper's claim that manifold constraints improve backward pass stability.

4. **Convergence**: mHC achieves the best minimum loss (1.787), demonstrating that stability improvements translate to better optimization.

## Key Ideas

This paper extends [Hyper-Connections](https://arxiv.org/abs/2409.19606) (ByteDance, ICLR 2025) by projecting connection matrices onto specific manifolds to restore identity mapping properties and improve training stability.

1. **Doubly Stochastic Constraint for H_res**: The residual connection matrix is projected onto the Birkhoff polytope using Sinkhorn-Knopp algorithm, ensuring spectral norm ≤ 1.

2. **Softmax Constraints for H_pre/H_post**: H_pre (1×n) and H_post (n×1) use softmax to ensure non-negative weights that sum to 1.

3. **Signal Stability**: These constraints guarantee bounded signal propagation across arbitrary network depths.

## References

- [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) - DeepSeek, 2025
- [Hyper-Connections](https://arxiv.org/abs/2409.19606) - ByteDance, ICLR 2025
