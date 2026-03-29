"""
TurboQuant — Triton GPU Kernels (EXPERIMENTAL)
================================================
⚠️ WARNING: These Triton kernels use Rademacher (±1) S matrices for QJL,
while the primary implementation (cache.py) uses Gaussian N(0,1) S matrices
per the paper's Definition 1. The scaling factor √(π/2)/d is derived for
Gaussian entries. These kernels are provided as experimental GPU acceleration
and should NOT be mixed with cache.py encode/decode paths.

The primary (correct) implementation is in cache.py.

Implements the TurboQuant 3-bit KV-cache quantization scheme from:
  "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
  Zandieh, Daliri, Hadian, Mirrokni (Google Research / NYU / Google DeepMind)
  https://arxiv.org/abs/2504.19874

Kernels implemented:
  - fwht_kernel              Pure-register FWHT (d=128, BLOCK_SIZE=d)
  - polarquant_encode_kernel  float16 → 2-bit indices + float16 norm
  - polarquant_decode_kernel  2-bit indices + norm → float16

Architecture: one program instance owns the entire D=128 vector. All 128
elements are loaded into SIMD registers at once. The 7 FWHT butterfly stages
are computed purely in-register using tl.where with inline tl.gather for
partner access. No scratch buffers, no cross-program reads, no barriers.

IMPORTANT: tl.gather must be used DIRECTLY inline inside tl.where (not stored
to a variable first) due to an XPU backend issue with gather register allocation.

Constants (d=128, b_mse=2):
  CODEBOOK_CENTROIDS  = [-0.1335, -0.0400, +0.0400, +0.1335]
  CODEBOOK_BOUNDARIES = [-1.0, -0.0868, 0.0, +0.0868, +1.0]
"""

import math
import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

# Lloyd-Max 2-bit codebook for N(0, 1/128), σ ≈ 0.0884
CODEBOOK_CENTROIDS_LIST  = [-0.1335, -0.0400, +0.0400, +0.1335]
CODEBOOK_BOUNDARIES_LIST = [-1.0,   -0.0868,   0.0,   +0.0868,  1.0]

# As Triton compile-time constants
C0 = -0.1335
C1 = -0.0400
C2 = +0.0400
C3 = +0.1335
B1 = -0.0868
B2 =  0.0
B3 = +0.0868

SQRT_PI_OVER_2 = math.sqrt(math.pi / 2.0)


# ===========================================================================
# 1. Fast Walsh-Hadamard Transform — PURE REGISTER, INLINE GATHER
# ===========================================================================

@triton.jit
def fwht_kernel(
    x_ptr,
    batch: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Pure-register FWHT for D=128.
    One program owns the full vector. All 7 butterfly stages in registers.
    Upper element = partner - self (not self - partner).
    """
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + row * D + offsets)

    # h=1
    x = tl.where((offsets & 1) != 0,
        tl.gather(x, offsets ^ 1, axis=0) - x,
        x + tl.gather(x, offsets ^ 1, axis=0))

    # h=2
    x = tl.where((offsets & 2) != 0,
        tl.gather(x, offsets ^ 2, axis=0) - x,
        x + tl.gather(x, offsets ^ 2, axis=0))

    # h=4
    x = tl.where((offsets & 4) != 0,
        tl.gather(x, offsets ^ 4, axis=0) - x,
        x + tl.gather(x, offsets ^ 4, axis=0))

    # h=8
    x = tl.where((offsets & 8) != 0,
        tl.gather(x, offsets ^ 8, axis=0) - x,
        x + tl.gather(x, offsets ^ 8, axis=0))

    # h=16
    x = tl.where((offsets & 16) != 0,
        tl.gather(x, offsets ^ 16, axis=0) - x,
        x + tl.gather(x, offsets ^ 16, axis=0))

    # h=32
    x = tl.where((offsets & 32) != 0,
        tl.gather(x, offsets ^ 32, axis=0) - x,
        x + tl.gather(x, offsets ^ 32, axis=0))

    # h=64
    x = tl.where((offsets & 64) != 0,
        tl.gather(x, offsets ^ 64, axis=0) - x,
        x + tl.gather(x, offsets ^ 64, axis=0))

    tl.store(x_ptr + row * D + offsets, x)


def fwht(x: torch.Tensor, d: int = 128, normalize: bool = False) -> torch.Tensor:
    assert x.ndim == 2 and x.shape[1] == d
    assert (d & (d - 1)) == 0
    x = x.contiguous()
    batch = x.shape[0]
    grid = (batch,)
    fwht_kernel[grid](x, batch, d, d)
    if normalize:
        x = x * (1.0 / math.sqrt(d))
    return x


# ---------------------------------------------------------------------------
# PyTorch fallback for FWHT
# ---------------------------------------------------------------------------

def torch_fwht(x: torch.Tensor, d: int = 128, normalize: bool = True) -> torch.Tensor:
    assert x.shape[-1] == d
    x = x.clone().float()
    h = 1
    while h < d:
        x = x.reshape(*x.shape[:-1], d // (2 * h), 2 * h)
        a = x[..., :h].clone()
        b = x[..., h:2 * h].clone()
        x[..., :h]      = a + b
        x[..., h:2 * h] = a - b
        x = x.reshape(*x.shape[:-2], d)
        h *= 2
    if normalize:
        x = x / math.sqrt(d)
    return x


# ===========================================================================
# 2. PolarQuant Encode — PURE REGISTER, INLINE GATHER
# ===========================================================================

@triton.jit
def polarquant_encode_kernel(
    x_ptr,
    signs_ptr,
    out_idx_ptr,
    out_norm_ptr,
    batch,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + row * D + offsets).to(tl.float32)

    # L2 norm
    norm_sq = tl.sum(x * x, axis=0)
    norm    = tl.sqrt(norm_sq)
    safe_norm = tl.maximum(norm, 1e-10)
    x = x / safe_norm

    # Random sign flip
    dsigns = tl.load(signs_ptr + offsets).to(tl.float32)
    x = x * dsigns

    # Pure-register FWHT
    x = tl.where((offsets & 1) != 0,
        tl.gather(x, offsets ^ 1, axis=0) - x,
        x + tl.gather(x, offsets ^ 1, axis=0))
    x = tl.where((offsets & 2) != 0,
        tl.gather(x, offsets ^ 2, axis=0) - x,
        x + tl.gather(x, offsets ^ 2, axis=0))
    x = tl.where((offsets & 4) != 0,
        tl.gather(x, offsets ^ 4, axis=0) - x,
        x + tl.gather(x, offsets ^ 4, axis=0))
    x = tl.where((offsets & 8) != 0,
        tl.gather(x, offsets ^ 8, axis=0) - x,
        x + tl.gather(x, offsets ^ 8, axis=0))
    x = tl.where((offsets & 16) != 0,
        tl.gather(x, offsets ^ 16, axis=0) - x,
        x + tl.gather(x, offsets ^ 16, axis=0))
    x = tl.where((offsets & 32) != 0,
        tl.gather(x, offsets ^ 32, axis=0) - x,
        x + tl.gather(x, offsets ^ 32, axis=0))
    x = tl.where((offsets & 64) != 0,
        tl.gather(x, offsets ^ 64, axis=0) - x,
        x + tl.gather(x, offsets ^ 64, axis=0))

    # Normalise
    inv_sqrtD: tl.constexpr = 1.0 / tl.sqrt(float(D))
    x = x * inv_sqrtD

    # Lloyd-Max 2-bit quantisation
    idx = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    idx = tl.where(x >= B1, idx + 1, idx)
    idx = tl.where(x >= B2, idx + 1, idx)
    idx = tl.where(x >= B3, idx + 1, idx)

    # Pack 4 indices per byte
    byte_offsets = offsets // 4
    bit_shifts   = (offsets % 4) * 2
    packed_val = (idx << bit_shifts).to(tl.int32)
    out_base   = row * (D // 4)
    tl.atomic_or(out_idx_ptr + out_base + byte_offsets, packed_val)
    tl.store(out_norm_ptr + row, norm.to(tl.float16), mask=offsets == 0)


def polarquant_encode(
    x: torch.Tensor,
    d_signs: torch.Tensor,
    d: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dtype == torch.float16
    assert d_signs.dtype == torch.int8
    batch = x.shape[0]
    x_f32     = x.float().contiguous()
    d_signs_f = d_signs.float().contiguous()
    indices   = torch.zeros(batch, d // 4, dtype=torch.int32, device=x.device)
    norms     = torch.zeros(batch, dtype=torch.float16, device=x.device)
    grid = (batch,)
    polarquant_encode_kernel[grid](x_f32, d_signs_f, indices, norms, batch, d, d)
    return indices.to(torch.uint8), norms


def torch_polarquant_encode(
    x: torch.Tensor,
    d_signs: torch.Tensor,
    d: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.float()
    norms = x.norm(dim=-1, keepdim=True)
    safe_norms = norms.clamp(min=1e-10)
    x_unit = x / safe_norms
    x_flipped = x_unit * d_signs.float()
    x_rot = torch_fwht(x_flipped, d=d, normalize=True)
    b = torch.tensor(CODEBOOK_BOUNDARIES_LIST, dtype=torch.float32, device=x.device)
    idx = (x_rot >= b[1]).int() + (x_rot >= b[2]).int() + (x_rot >= b[3]).int()
    return idx, norms.squeeze(-1).to(x.dtype)


# ===========================================================================
# 3. PolarQuant Decode — PURE REGISTER, INLINE GATHER
# ===========================================================================

@triton.jit
def polarquant_decode_kernel(
    idx_ptr,
    norms_ptr,
    signs_ptr,
    x_ptr,
    batch,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    # Unpack 4×2-bit indices per byte
    byte_offsets = offsets // 4
    bit_shifts   = (offsets % 4) * 2
    idx_base     = row * (D // 4)
    packed = tl.load(idx_ptr + idx_base + byte_offsets).to(tl.int32)
    idx    = (packed >> bit_shifts) & 0x3

    # Codebook lookup
    val  = tl.where(idx == 0, C0, 0.0)
    val += tl.where(idx == 1, C1, 0.0)
    val += tl.where(idx == 2, C2, 0.0)
    val += tl.where(idx == 3, C3, 0.0)

    # Pure-register IFWHT (same butterfly as FWHT)
    val = tl.where((offsets & 1) != 0,
        tl.gather(val, offsets ^ 1, axis=0) - val,
        val + tl.gather(val, offsets ^ 1, axis=0))
    val = tl.where((offsets & 2) != 0,
        tl.gather(val, offsets ^ 2, axis=0) - val,
        val + tl.gather(val, offsets ^ 2, axis=0))
    val = tl.where((offsets & 4) != 0,
        tl.gather(val, offsets ^ 4, axis=0) - val,
        val + tl.gather(val, offsets ^ 4, axis=0))
    val = tl.where((offsets & 8) != 0,
        tl.gather(val, offsets ^ 8, axis=0) - val,
        val + tl.gather(val, offsets ^ 8, axis=0))
    val = tl.where((offsets & 16) != 0,
        tl.gather(val, offsets ^ 16, axis=0) - val,
        val + tl.gather(val, offsets ^ 16, axis=0))
    val = tl.where((offsets & 32) != 0,
        tl.gather(val, offsets ^ 32, axis=0) - val,
        val + tl.gather(val, offsets ^ 32, axis=0))
    val = tl.where((offsets & 64) != 0,
        tl.gather(val, offsets ^ 64, axis=0) - val,
        val + tl.gather(val, offsets ^ 64, axis=0))

    # Normalise
    inv_sqrtD: tl.constexpr = 1.0 / tl.sqrt(float(D))
    val = val * inv_sqrtD

    # Inverse sign flip
    dsigns = tl.load(signs_ptr + offsets).to(tl.float32)
    val = val * dsigns

    # Scale by norm
    norm = tl.load(norms_ptr + row).to(tl.float32)
    val  = val * norm

    tl.store(x_ptr + row * D + offsets, val.to(tl.float16))


def polarquant_decode(
    indices: torch.Tensor,
    norms: torch.Tensor,
    d_signs: torch.Tensor,
    d: int = 128,
) -> torch.Tensor:
    batch  = indices.shape[0]
    device = indices.device
    idx_i32  = indices.to(torch.int32).contiguous()
    norms_f16 = norms.to(torch.float16).contiguous()
    signs_i8  = d_signs.to(torch.int8).contiguous()
    x_out    = torch.zeros(batch, d, dtype=torch.float16, device=device)
    grid = (batch,)
    polarquant_decode_kernel[grid](idx_i32, norms_f16, signs_i8, x_out, batch, d, d)
    return x_out


def torch_polarquant_decode(
    indices: torch.Tensor,
    norms: torch.Tensor,
    d_signs: torch.Tensor,
    d: int = 128,
) -> torch.Tensor:
    c = torch.tensor(CODEBOOK_CENTROIDS_LIST, dtype=torch.float32, device=indices.device)
    idx_long = indices.long().clamp(0, 3)
    val = c[idx_long]
    x_rot = torch_fwht(val, d=d, normalize=True)
    x_unit = x_rot * d_signs.float()
    x_hat = x_unit * norms.float().unsqueeze(-1)
    return x_hat.to(torch.float16)
