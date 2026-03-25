# TurboQuant — Implementation Notes for Codex

## CRITICAL: Our PolarQuant is WRONG

We implemented a simplified "Hadamard + scalar Lloyd-Max quantizer" instead of the REAL PolarQuant algorithm.

### What the PolarQuant paper (arxiv 2502.02617) actually describes:

1. **Random preconditioning**: Multiply by Gaussian random matrix S (i.i.d. N(0,1) entries), NOT Hadamard
   - After preconditioning: S·x ~ N(0, ||x||² · I_m) — multivariate normal
   - The paper uses Fact 3: for S with i.i.d. N(0,1) entries, S·x is multivariate normal

2. **Recursive Polar Transformation** (the KEY innovation we missed):
   - Pair up coordinates: (x1, x2) → (r, θ) where r = sqrt(x1² + x2²), θ = atan2(x2, x1)
   - Then recursively pair up the radii from the previous level → more (r, θ) pairs
   - For d=128: 64 angles at level 1, 32 at level 2, ..., 1 at the top = 127 total angles + 1 radius
   - This is O(d) computation, very efficient

3. **Angle quantization** (NOT scalar coordinate quantization):
   - After random preconditioning, the angles follow a CONCENTRATED distribution
   - The distribution is analytically known (related to Beta distribution)
   - Build optimal Lloyd-Max codebook for this known angle distribution
   - Each angle quantized to b bits
   - The radius at the top is stored separately (just one float16 per vector = the norm)

4. **Reconstruction**:
   - Dequantize angles from codebook
   - Recursively convert back from polar to Cartesian
   - Multiply by inverse preconditioning matrix (S^T for Gaussian, or H^T for Hadamard)
   - Scale by stored norm

### What we currently do (WRONG):
1. Random sign flip + FWHT (Hadamard) — this is OK as an approximation of random preconditioning
2. Scalar Lloyd-Max 2-bit quantization of EACH COORDINATE independently — THIS IS WRONG
3. No polar transformation at all
4. Store norm + 2-bit indices per coordinate

### Why this matters:
- Polar quantization of angles is MORE EFFICIENT than scalar coordinate quantization
- The angles after preconditioning are tightly concentrated → fewer bits needed
- Scalar quantization wastes bits on the magnitude information (which is redundant since we store the norm)
- The paper achieves 4.2x compression with near-zero quality loss; we get 4.9x but with 0.92 cosine sim

### The TurboQuant algorithm (arxiv 2504.19874) combines:
- Stage 1: PolarQuant (REAL polar coordinates, 2-3 bits per angle)
- Stage 2: QJL (1-bit residual correction)
- The paper says 3.5 bits total — that's ~2.5 bits for PolarQuant + 1 bit for QJL

### What needs to change:
1. Implement recursive polar transformation: Cartesian → (angles, radius)
2. Implement inverse polar transformation: (angles, radius) → Cartesian  
3. Compute angle distribution after preconditioning (concentrated Beta-like)
4. Build Lloyd-Max codebook for the ANGLE distribution (not coordinate distribution)
5. Encode: precondition → polar transform → quantize angles → store angles + norm
6. Decode: dequantize angles → inverse polar → inverse precondition → scale by norm
7. Update attention kernel to work with polar-encoded keys

### Reference:
- PolarQuant paper: https://arxiv.org/html/2502.02617v1 (Section 3.1: Recursive Polar Transformation)
- TurboQuant paper: https://arxiv.org/abs/2504.19874
- QJL reference code: https://github.com/amirzandieh/QJL

### Practical note on preconditioning:
The paper uses Gaussian S but notes that randomized Hadamard (sign flip + FWHT) is a practical 
approximation that's O(d log d) instead of O(d²). Our Hadamard preconditioning is fine — the 
polar transformation is what we're missing.
