# Porting TurboQuant Triton Kernels to Intel XPU (Arc B580)

## Status
PyTorch path: fully working, all 33 tests pass on Arc B580.
Triton path: blocked by compiler bug (see below).

## Bugs Found in Original Kernels (affect CUDA too)

### 1. FWHT butterfly subtraction order
Upper element must be `partner - self`, not `self - partner`.
- Wrong:  `tl.where(is_upper, x - tl.gather(x, partner), ...)`
- Correct: `tl.where(is_upper, tl.gather(x, partner) - x, ...)`

### 2. Partner index arithmetic
FWHT partner is `offsets ^ h` (XOR), not `offsets ± h`.
The ± form only works for h=1 and silently corrupts h=2,4,8...

### 3. normalize= default
`fwht()` normalizes by 1/sqrt(D) by default. Self-inverse
test requires normalize=False — H(H(x)) = D*x unnormalized.

## XPU-Specific Issues

### tl.gather materialization bug (Triton 3.7.0 XPU backend)
Assigning tl.gather to a named intermediate produces zeros:
```python
pval = tl.gather(x, partner)   # → zeros on XPU
x = tl.where(mask, x - pval, x + pval)  # wrong
```

Inline usage works:
```python
x = tl.where(mask,
             tl.gather(x, partner) - x,
             x + tl.gather(x, partner))
```

This appears to be a register allocation bug in the XPU
Triton compiler backend.

Minimal repro (D=4, inline gather works; stored gather gives zeros):
```python
import torch, triton, triton.language as tl
DEVICE = torch.device('xpu')

@triton.jit
def fwht4_broken(x_ptr, D, BS):
    row = tl.program_id(0)
    off = tl.arange(0, BS)
    x = tl.load(x_ptr + row*D + off)
    g = tl.gather(x, off ^ 1, axis=0)   # stored → zeros on XPU
    x = tl.where((off & 1) != 0, x - g, x + g)
    tl.store(x_ptr + row*D + off, x)

@triton.jit
def fwht4_working(x_ptr, D, BS):
    row = tl.program_id(0)
    off = tl.arange(0, BS)
    x = tl.load(x_ptr + row*D + off)
    x = tl.where((off & 1) != 0,
                 tl.gather(x, off ^ 1, axis=0) - x,
                 x + tl.gather(x, off ^ 1, axis=0))
    tl.store(x_ptr + row*D + off, x)
```

## Environment
- Intel Arc B580 (Battlemage, 12GB VRAM)
- WSL2 Ubuntu, PyTorch XPU stable, Triton 3.7.0
- intel-xpu-backend-for-triton via pytorch.org/whl/xpu
