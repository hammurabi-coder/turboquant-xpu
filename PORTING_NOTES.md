# Porting TurboQuant Triton Kernels to Intel XPU (Arc B580)

## Status
PyTorch path: fully working, all 33 tests pass on Arc B580.
Triton path: blocked by compiler bug (see below).

## Correctness Bugs Found in Original Triton Kernels
These were found while debugging incorrect output during the XPU port. The math is wrong regardless of backend — these kernels appear to have been untested on any backend.

### 1. FWHT butterfly subtraction order
Upper element must be `partner - self`, not `self - partner`.
- Wrong:  `tl.where(is_upper, x - tl.gather(x, partner), ...)`
- Correct: `tl.where(is_upper, tl.gather(x, partner) - x, ...)`

### 2. Partner index arithmetic
FWHT partner is `offsets ^ h` (XOR), not `offsets ± h`.
The `±h` form only works for h=1 and silently corrupts h=2,4,8...

### 3. normalize= default
`fwht()` normalizes by 1/sqrt(D) by default. Self-inverse test requires `normalize=False` — H(H(x)) = D*x unnormalized.

## XPU-Specific Issue: tl.gather materialization bug (Triton 3.7.0)
Assigning `tl.gather` to a named intermediate produces zeros on XPU. Inline usage works correctly. Filed as intel/intel-xpu-backend-for-triton #6511.

```python
# broken — stored gather gives zeros on XPU
pval = tl.gather(x, partner)
x = tl.where(mask, x - pval, x + pval)

# working — inline gather
x = tl.where(mask,
             tl.gather(x, partner) - x,
             x + tl.gather(x, partner))
```

Minimal repro in the issue linked above.

## Environment
- Intel Arc B580 (Battlemage, 12GB VRAM)
- WSL2 Ubuntu, PyTorch XPU stable, Triton 3.7.0
- intel-xpu-backend-for-triton via pytorch.org/whl/xpu

## Asymmetric Attention
- `turboquant_attention()` implemented in `cache.py`
- PQ-only K scoring, full V decompression
- 0.76 avg cos_sim vs standard attention (seq_k=16, D=128)
- QJL residual correction tested — degrades ranking quality despite improving dot-product estimation
- Tradeoff reverses at longer seq_k (1024+) where softmax is softer
