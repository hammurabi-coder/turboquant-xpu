# Implementation Notes

## Algorithm Choice: Scalar Quantization vs Recursive Polar Transform

This implementation follows **TurboQuant (arxiv 2504.19874)**, not PolarQuant (arxiv 2502.02617). While both papers share authors, they use fundamentally different approaches:

**TurboQuant (what we implement):**
- Random rotation → scalar Lloyd-Max quantization per coordinate
- Each coordinate quantized independently — no coupling between dimensions
- Errors do NOT compound through deep models
- Provably within 2.7× of information-theoretic optimal (Theorem 1)

**PolarQuant (what we DON'T implement):**
- Random preconditioning → recursive polar transform → angle quantization
- Coordinates coupled through sin/cos reconstruction tree (7 levels for d=128)
- Errors compound multiplicatively through the reconstruction tree AND through transformer layers
- Works well for shallow models but degrades on 32+ layer models

## Random Rotation

We use the **Randomized Hadamard Transform** as the rotation matrix Π:
1. Random sign flip: multiply each coordinate by ±1 (deterministic from seed)
2. Fast Walsh-Hadamard Transform (FWHT): O(d log d) complexity
3. Scale by 1/√d

This is a practical approximation of the paper's random rotation matrix. The paper notes that any rotation satisfying P^T P = I works. The Hadamard approach is O(d log d) instead of O(d²) for a dense random matrix.

After rotation, each coordinate of Π·x follows a Beta distribution that converges to N(0, 1/d) for d ≥ 64 (Lemma 1). Critically, distinct coordinates become **near-independent** in high dimensions, which is why scalar quantization per coordinate is near-optimal.

## Lloyd-Max Codebook

The scalar codebook is computed by solving the continuous 1D k-means problem (Eq. 3) for the Beta distribution f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1-x²)^((d-3)/2).

For d ≥ 64, we use the Gaussian approximation N(0, 1/d) which is faster and equally accurate. Optimal centroids for common bit widths:
- b=1: {±√(2/(πd))}
- b=2: {±0.453/√d, ±1.51/√d}

The **same codebook** is used for all coordinates — no per-level or per-channel variation needed.

## QJL Residual Correction

The QJL (Quantized Johnson-Lindenstrauss) stage provides unbiased inner product estimation:
1. Compute residual: r = x - DeQuant_mse(Quant_mse(x))
2. Project: sign(S · r) where S is a random matrix
3. Store: 1-bit signs + FP16 residual norm
4. Reconstruct: x̂_qjl = √(π/2)/d · ‖r‖ · S^T · signs

We use Rademacher (±1) entries for S instead of Gaussian N(0,1) for efficiency. Both satisfy the JL property; Rademacher avoids storing floating-point matrix entries.

## Outlier Channel Handling

The paper's 2.5-bit and 3.5-bit modes (Table 1) allocate more bits to outlier channels:
- 2.5-bit: 32 outlier channels at 3 bits + 96 regular at 2 bits
- 3.5-bit: similar split with higher bit allocation

**Paper approach (Section 2.3):** Split channels into outlier/regular sets, apply two **independent** TurboQuant instances with separate rotations and codebooks to each subset.

**Our approximation:** We apply a single rotation over all dimensions, then detect high-variance channels post-rotation and assign different codebook bit budgets. This provides some benefit from residual variance inhomogeneity in the Hadamard approximation, but is not the theoretically optimal two-independent-instances approach described in the paper. The full paper approach requires separate Hadamard matrices for each subset, which we plan to implement in a future version.

## QJL Score Weight

The `compute_attention()` method defaults to `qjl_score_weight=1.0`, which produces the paper-correct **unbiased** inner product estimator (Theorem 2). Setting `qjl_score_weight < 1.0` trades bias for lower variance — this is a practical heuristic not present in the paper that can improve attention quality when the QJL variance is high relative to score differences.

## Backward Compatibility

The public API uses names like `PolarQuantCompressed` and `polarquant_encode/decode` for backward compatibility with existing code that imports these. Internally, these now implement scalar per-coordinate quantization (TurboQuant Algorithm 1), not recursive polar transform.
