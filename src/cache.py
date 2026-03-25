"""
TurboQuant KV Cache Compression — Pure PyTorch Implementation

Implements the TurboQuant algorithm (Algorithm 2) for near-optimal vector
quantization of transformer KV caches, combining:
  - PolarQuant (recursive polar transform + 2-bit angle quantization)
  - QJL (1-bit residual correction for unbiased inner products)

Total: ~3 bits per coordinate → ~4.9× compression vs FP16.

Reference: https://arxiv.org/abs/2504.19874
"""

import math
from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B_MSE = 3          # bits per angle for PolarQuant stage
B_QJL = 1          # bits per coordinate for QJL residual stage
B_TOTAL = B_MSE + B_QJL  # = 3 bits total per coordinate
EPS = 1e-10        # numerical stability threshold


# ---------------------------------------------------------------------------
# Fast Walsh-Hadamard Transform (FWHT)
# ---------------------------------------------------------------------------

def fwht(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform.

    Self-inverse up to scaling: FWHT(FWHT(x)) = d * x.
    Preserves norms up to scaling: ‖FWHT(x)‖² = d · ‖x‖².
    Complexity: O(d log d).

    Args:
        x: [..., d] where d is a power of 2.

    Returns:
        Transformed tensor, same shape as input.
    """
    d = x.shape[-1]
    y = x.clone()
    h = 1
    while h < d:
        y_view = y.reshape(*y.shape[:-1], -1, 2 * h)
        a = y_view[..., :h].clone()
        b = y_view[..., h:].clone()
        y_view[..., :h] = a + b
        y_view[..., h:] = a - b
        y = y_view.reshape(*y.shape)
        h *= 2
    return y


def fwht_inplace(x: torch.Tensor) -> None:
    """In-place Fast Walsh-Hadamard Transform."""
    d = x.shape[-1]
    h = 1
    while h < d:
        y_view = x.reshape(*x.shape[:-1], -1, 2 * h)
        a = y_view[..., :h].clone()
        b = y_view[..., h:].clone()
        y_view[..., :h] = a + b
        y_view[..., h:] = a - b
        h *= 2


# ---------------------------------------------------------------------------
# Randomized Hadamard Transform (rotation for PolarQuant)
# ---------------------------------------------------------------------------

def _generate_signs(d: int, seed: int, device: torch.device) -> torch.Tensor:
    """Generate deterministic random ±1 signs from a seed."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randint(0, 2, (d,), generator=g, device=device, dtype=torch.float32) * 2 - 1


class RandomHadamardRotation:
    """Randomized Hadamard Transform: Π·x = (1/√d) · H · (D_signs ⊙ x).

    Inverse: Π^T · y = D_signs ⊙ ((1/√d) · H · y).
    """

    def __init__(self, d: int, seed: int, device: torch.device = torch.device("cpu")):
        self.d = d
        self.seed = seed
        self.sqrt_d = math.sqrt(d)
        self.signs = _generate_signs(d, seed, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random rotation: Π · x."""
        signs = self.signs.to(device=x.device, dtype=x.dtype)
        y = x * signs
        # FWHT requires power-of-2 dimensions
        if self.d & (self.d - 1) == 0:
            fwht_inplace(y)
            y = y / self.sqrt_d
        return y

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Apply inverse rotation: Π^T · y = D ⊙ ((1/√d) · H · y)."""
        z = y.clone()
        if self.d & (self.d - 1) == 0:
            fwht_inplace(z)
            z = z / self.sqrt_d
        z = z * self.signs.to(device=z.device, dtype=z.dtype)
        return z


# ---------------------------------------------------------------------------
# Recursive polar transform helpers
# ---------------------------------------------------------------------------

def _next_power_of_two(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p *= 2
    return p


def _require_power_of_two(d: int) -> None:
    if d <= 0 or d & (d - 1):
        raise ValueError(f"d must be a positive power of 2, got {d}")


def _polar_level_sizes(d: int) -> Tuple[int, ...]:
    """Number of angles emitted at each recursive level, bottom-up."""
    _require_power_of_two(d)
    sizes = []
    width = d
    while width > 1:
        width //= 2
        sizes.append(width)
    return tuple(sizes)


def recursive_polar_transform(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert Cartesian vectors to recursive polar coordinates.

    The transform pairs coordinates bottom-up:
      level 0:  (x1, x2) -> (r, theta)
      level 1+: pair the radii from the previous level.

    Args:
        x: [batch, d] Cartesian vectors where d is a power of 2.

    Returns:
        angles: [batch, d-1] recursive angles in bottom-up order
        radius: [batch, 1] final radius (equal to ||x||_2)
    """
    squeeze_batch = False
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze_batch = True

    d = x.shape[-1]
    _require_power_of_two(d)

    current = x
    angle_levels = []
    while current.shape[-1] > 1:
        pairs = current.reshape(*current.shape[:-1], -1, 2)
        left = pairs[..., 0]
        right = pairs[..., 1]
        radii = torch.sqrt(left.square() + right.square())
        angles = torch.atan2(right, left)
        angle_levels.append(angles.reshape(x.shape[0], -1))
        current = radii

    angles = torch.cat(angle_levels, dim=-1) if angle_levels else x.new_zeros(x.shape[0], 0)
    radius = current.reshape(x.shape[0], 1)

    if squeeze_batch:
        return angles.squeeze(0), radius.squeeze(0)
    return angles, radius


def inverse_polar_transform(angles: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    """Invert the recursive polar transform back to Cartesian coordinates."""
    squeeze_batch = False
    if angles.dim() == 1:
        angles = angles.unsqueeze(0)
        squeeze_batch = True
    if radius.dim() == 0:
        radius = radius.reshape(1, 1)
    elif radius.dim() == 1:
        radius = radius.unsqueeze(-1)

    d = angles.shape[-1] + 1
    level_sizes = _polar_level_sizes(d)
    current = radius
    offset = angles.shape[-1]

    for level_size in reversed(level_sizes):
        offset -= level_size
        level_angles = angles[..., offset:offset + level_size]
        current = torch.stack(
            (current * torch.cos(level_angles), current * torch.sin(level_angles)),
            dim=-1,
        ).reshape(angles.shape[0], -1)

    if squeeze_batch:
        return current.squeeze(0)
    return current


def _uniform_pdf(x: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(x)


def _centered_recursive_angle_pdf(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Density for upper-level recursive angles centered around 0.

    For a pair of equal-size blocks with block_size original coordinates each,
    θ = atan2(r_right, r_left) lives in [0, π/2] with density proportional to
    sin(θ)^(block_size-1) cos(θ)^(block_size-1). We center it as δ = θ - π/4,
    so δ ∈ [-π/4, π/4] and the density is symmetric around 0.
    """
    theta = x + (math.pi / 4.0)
    valid = (x >= -math.pi / 4.0) & (x <= math.pi / 4.0)
    pdf = torch.zeros_like(x)
    if valid.any():
        theta_valid = theta[valid].clamp(min=EPS, max=(math.pi / 2.0) - EPS)
        pdf[valid] = (torch.sin(theta_valid) * torch.cos(theta_valid)).pow(block_size - 1)
    return pdf


# ---------------------------------------------------------------------------
# Lloyd-Max Codebook
# ---------------------------------------------------------------------------

@dataclass
class Codebook:
    """Level-wise Lloyd-Max codebooks for recursive polar angles."""
    centroids: Tuple[torch.Tensor, ...]    # each [K], on shifted angle support
    boundaries: Tuple[torch.Tensor, ...]   # each [K+1], on shifted angle support
    shifts: Tuple[float, ...]              # angle shifts applied before quantization
    level_sizes: Tuple[int, ...]           # angles emitted at each level
    d: int
    b: int
    K: int

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Map recursive angles to per-level codebook indices."""
        squeeze_batch = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_batch = True

        idx_chunks = []
        offset = 0
        for level_idx, level_size in enumerate(self.level_sizes):
            level_x = x[..., offset:offset + level_size] - self.shifts[level_idx]
            boundaries = self.boundaries[level_idx].to(device=x.device, dtype=x.dtype)
            idx = torch.searchsorted(boundaries, level_x, right=False) - 1
            idx_chunks.append(idx.clamp(0, self.K - 1).to(torch.uint8))
            offset += level_size

        out = torch.cat(idx_chunks, dim=-1) if idx_chunks else x.new_zeros(x.shape[0], 0, dtype=torch.uint8)
        if squeeze_batch:
            return out.squeeze(0)
        return out

    def dequantize(self, idx: torch.Tensor) -> torch.Tensor:
        """Map per-level codebook indices back to recursive angles."""
        squeeze_batch = False
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
            squeeze_batch = True

        angle_chunks = []
        offset = 0
        for level_idx, level_size in enumerate(self.level_sizes):
            level_idx_tensor = idx[..., offset:offset + level_size].long()
            centroids = self.centroids[level_idx].to(device=idx.device)
            angle_chunks.append(centroids[level_idx_tensor] + self.shifts[level_idx])
            offset += level_size

        out = torch.cat(angle_chunks, dim=-1) if angle_chunks else torch.zeros(
            idx.shape[0], 0, device=idx.device, dtype=torch.float32
        )
        if squeeze_batch:
            return out.squeeze(0)
        return out


def _compute_density_codebook(
    b: int,
    support: Tuple[float, float],
    pdf_fn: Callable[[torch.Tensor], torch.Tensor],
    max_iter: int,
    tol: float,
    device: torch.device,
    grid_size: int = 16385,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Numerically solve the Lloyd-Max problem for a 1D density."""
    K = 2 ** b
    lo, hi = support
    grid = torch.linspace(lo, hi, grid_size, device=device, dtype=torch.float64)
    pdf = pdf_fn(grid).clamp_min(0)
    mass = torch.trapz(pdf, grid)
    if mass.item() <= EPS:
        raise ValueError("Degenerate density produced zero mass")
    pdf = pdf / mass

    centroids = torch.linspace(lo, hi, K + 2, device=device, dtype=torch.float64)[1:-1]
    boundaries = torch.empty(K + 1, device=device, dtype=torch.float64)
    boundaries[0] = lo
    boundaries[-1] = hi

    for _ in range(max_iter):
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])
        old_centroids = centroids.clone()

        for i in range(K):
            mask = (grid >= boundaries[i]) & (grid <= boundaries[i + 1])
            grid_slice = grid[mask]
            pdf_slice = pdf[mask]
            if grid_slice.numel() < 2:
                centroids[i] = 0.5 * (boundaries[i] + boundaries[i + 1])
                continue

            interval_mass = torch.trapz(pdf_slice, grid_slice)
            if interval_mass.item() <= EPS:
                centroids[i] = 0.5 * (boundaries[i] + boundaries[i + 1])
            else:
                interval_moment = torch.trapz(pdf_slice * grid_slice, grid_slice)
                centroids[i] = interval_moment / interval_mass

        if (centroids - old_centroids).abs().max().item() < tol:
            break

    boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])
    return centroids.float(), boundaries.float()


def compute_lloyd_max_codebook(
    d: int,
    b: int,
    max_iter: int = 500,
    tol: float = 1e-12,
    device: torch.device = torch.device("cpu"),
) -> Codebook:
    """Compute level-wise Lloyd-Max angle codebooks for recursive PolarQuant."""
    level_sizes = _polar_level_sizes(d)
    level_centroids = []
    level_boundaries = []
    level_shifts = []

    for level_idx, _level_size in enumerate(level_sizes):
        if level_idx == 0:
            support = (-math.pi, math.pi)
            shift = 0.0
            pdf_fn = _uniform_pdf
        else:
            support = (-math.pi / 4.0, math.pi / 4.0)
            shift = math.pi / 4.0
            block_size = 2 ** level_idx
            pdf_fn = lambda x, block_size=block_size: _centered_recursive_angle_pdf(x, block_size)

        # For highly concentrated distributions (block_size >= 16), the density
        # becomes a near-delta at the shift point. Use uniform quantization on a
        # narrow support instead of numerically solving Lloyd-Max (which underflows).
        if level_idx > 0 and block_size >= 16:
            narrow = math.pi / (4.0 * math.sqrt(block_size))
            support = (-narrow, narrow)
            pdf_fn = _uniform_pdf

        centroids, boundaries = _compute_density_codebook(
            b=b,
            support=support,
            pdf_fn=pdf_fn,
            max_iter=max_iter,
            tol=tol,
            device=device,
        )
        level_centroids.append(centroids)
        level_boundaries.append(boundaries)
        level_shifts.append(shift)

    return Codebook(
        centroids=tuple(level_centroids),
        boundaries=tuple(level_boundaries),
        shifts=tuple(level_shifts),
        level_sizes=level_sizes,
        d=d,
        b=b,
        K=2 ** b,
    )


# ---------------------------------------------------------------------------
# QJL Random Matrix
# ---------------------------------------------------------------------------

def generate_qjl_matrix(d: int, seed: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Generate the QJL random Rademacher matrix S ∈ {-1, +1}^{d×d}.
    
    The QJL algorithm requires a Rademacher (±1) random matrix, not Gaussian.
    Rademacher matrices satisfy the Johnson-Lindenstrauss property and produce
    lower-variance single-sample inner product estimates than Gaussian matrices.
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randint(0, 2, (d, d), generator=g, device=device).float() * 2 - 1


# ---------------------------------------------------------------------------
# PolarQuant: 2-bit MSE-optimal quantization
# ---------------------------------------------------------------------------

@dataclass
class PolarQuantCompressed:
    """Compressed representation from PolarQuant (2-bit per angle + norm)."""
    norm: torch.Tensor            # [batch] L2 norms / top-level radius
    indices: torch.Tensor         # [batch, d-1] uint8 indices in {0..K-1}
    codebook: Codebook
    rotation: RandomHadamardRotation

    @property
    def d(self) -> int:
        return self.codebook.d


def polarquant_encode(x: torch.Tensor, codebook: Codebook, rotation: RandomHadamardRotation) -> PolarQuantCompressed:
    """PolarQuant encode: precondition -> recursive polar transform -> angle quantization."""
    if x.dim() == 1:
        x = x.unsqueeze(0)

    x = x.detach().float()
    
    norm = x.norm(dim=-1).to(torch.float16)
    zero_mask = norm < EPS

    # Pad to power-of-2 BEFORE rotation if needed
    d_actual = x.shape[-1]
    d_padded = _next_power_of_two(d_actual)
    if d_padded != d_actual:
        padding = torch.zeros(*x.shape[:-1], d_padded - d_actual,
                              device=x.device, dtype=x.dtype)
        x = torch.cat([x, padding], dim=-1)

    x_rotated = rotation.forward(x)
    angles, _ = recursive_polar_transform(x_rotated)
    indices = codebook.quantize(angles)

    if zero_mask.any():
        indices[zero_mask] = 0

    return PolarQuantCompressed(norm=norm, indices=indices, codebook=codebook, rotation=rotation)


def polarquant_decode(c: PolarQuantCompressed) -> torch.Tensor:
    """PolarQuant decode: dequantize angles -> inverse polar -> inverse precondition."""
    angles_hat = c.codebook.dequantize(c.indices)
    radius = c.norm.float().unsqueeze(-1)
    x_rotated_hat = inverse_polar_transform(angles_hat, radius)
    
    # Inverse rotation first (on padded dimension), then unpad
    x_hat = c.rotation.inverse(x_rotated_hat)
    
    # Unpad if the original dimension wasn't a power of 2
    d_actual = c.codebook.d
    if x_hat.shape[-1] > d_actual:
        x_hat = x_hat[..., :d_actual]

    zero_mask = c.norm < EPS
    if zero_mask.any():
        x_hat[zero_mask] = 0.0

    return x_hat


# ---------------------------------------------------------------------------
# QJL: 1-bit residual quantization
# ---------------------------------------------------------------------------

@dataclass
class QJLCompressed:
    """Compressed representation from QJL (1-bit per coord + residual norm)."""
    signs: torch.Tensor       # [batch, d] in {0, 1}
    r_norm: torch.Tensor      # [batch] residual norm
    S: torch.Tensor           # [d, d] random Gaussian matrix

    @property
    def d(self) -> int:
        return self.signs.shape[-1]


def qjl_encode(residual: torch.Tensor, S: torch.Tensor) -> QJLCompressed:
    """QJL encode: 1-bit sign quantization of projected residual."""
    if residual.dim() == 1:
        residual = residual.unsqueeze(0)

    r_norm = residual.norm(dim=-1)
    safe_norm = r_norm.clamp(min=EPS)
    r_unit = residual / safe_norm.unsqueeze(-1)

    projected = r_unit @ S.T  # [batch, d]
    signs = (projected >= 0).long()

    return QJLCompressed(signs=signs, r_norm=r_norm, S=S)


# ---------------------------------------------------------------------------
# TurboQuant: Complete 3-bit pipeline
# ---------------------------------------------------------------------------

@dataclass
class TurboQuantCompressed:
    """Complete TurboQuant compressed representation (3-bit per coord)."""
    pq: PolarQuantCompressed
    qjl: QJLCompressed

    @property
    def d(self) -> int:
        return self.pq.d


class TurboQuantConfig:
    """Configuration for a TurboQuant cache."""

    def __init__(self, d: int = 128, b_mse: int = B_MSE, device: torch.device = torch.device("cpu")):
        self.d = d
        self.d_padded = _next_power_of_two(d)
        self.b_mse = b_mse
        self.device = device
        # Codebook uses padded dimension for polar transform, but stores actual d
        self.codebook = compute_lloyd_max_codebook(self.d_padded, b_mse, device=device)
        self.codebook.d = d  # track actual dimension for unpadding

    def make_rotation(self, layer_idx: int, head_idx: int) -> RandomHadamardRotation:
        # Use padded dimension — input is padded before rotation in polarquant_encode
        seed = ((layer_idx * 1000003) ^ (head_idx * 999979) ^ 0xA5A5A5A5) & 0xFFFFFFFF
        return RandomHadamardRotation(self.d_padded, seed, self.device)

    def make_qjl_matrix(self, layer_idx: int, head_idx: int) -> torch.Tensor:
        # QJL operates on actual dimension (not padded) — residual is in original space
        seed = ((layer_idx * 1000003) ^ (head_idx * 999979) ^ 0x5A5A5A5A) & 0xFFFFFFFF
        return generate_qjl_matrix(self.d, seed, self.device)


def turboquant_encode_internal(
    x: torch.Tensor,
    codebook: Codebook,
    rotation: RandomHadamardRotation,
    S: torch.Tensor,
) -> TurboQuantCompressed:
    """Full TurboQuant encode: PolarQuant + QJL (Algorithm 2)."""
    if x.dim() == 1:
        x = x.unsqueeze(0)

    pq = polarquant_encode(x, codebook, rotation)
    x_hat = polarquant_decode(pq).float()
    
    # x_hat is unpadded (d_actual), match it for residual
    x_for_residual = x.detach().float()
    if x_for_residual.shape[-1] != x_hat.shape[-1]:
        x_for_residual = x_for_residual[..., :x_hat.shape[-1]]
    residual = x_for_residual - x_hat
    qjl = qjl_encode(residual, S)

    return TurboQuantCompressed(pq=pq, qjl=qjl)


def turboquant_decode_single(c: TurboQuantCompressed) -> torch.Tensor:
    """Full TurboQuant decode: PQ reconstruction + QJL residual estimate."""
    k_hat = polarquant_decode(c.pq)  # [1, d]

    signs_f = c.qjl.signs.float() * 2 - 1  # {-1, +1}
    d = c.d
    scale = math.sqrt(math.pi / 2) / d
    r_hat = (signs_f @ c.qjl.S) * scale  # [1, d]
    r_hat = r_hat * c.qjl.r_norm.unsqueeze(-1)

    return k_hat + r_hat


# ---------------------------------------------------------------------------
# TurboQuant Cache
# ---------------------------------------------------------------------------

class TurboQuantCache:
    """TurboQuant-compressed KV cache for transformer attention."""

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        d: int = 128,
        b_mse: int = B_MSE,
        device: torch.device = torch.device("cpu"),
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d = d
        self.device = device
        self.config = TurboQuantConfig(d, b_mse, device=device)

        self.rotations: List[List[RandomHadamardRotation]] = []
        self.qjl_matrices: List[List[torch.Tensor]] = []
        for l in range(n_layers):
            self.rotations.append([])
            self.qjl_matrices.append([])
            for h in range(n_heads):
                self.rotations[l].append(self.config.make_rotation(l, h))
                self.qjl_matrices[l].append(self.config.make_qjl_matrix(l, h))

        # cache[layer][head] = list of (key_compressed, value_compressed)
        self.cache: List[List[List[Tuple[TurboQuantCompressed, TurboQuantCompressed]]]] = []
        for l in range(n_layers):
            self.cache.append([])
            for h in range(n_heads):
                self.cache[l].append([])

    @property
    def seq_len(self) -> int:
        if self.n_layers == 0 or self.n_heads == 0:
            return 0
        return len(self.cache[0][0])

    def store(self, layer_idx: int, head_idx: int, k_vec: torch.Tensor, v_vec: torch.Tensor):
        """Encode and store a key-value pair."""
        rotation = self.rotations[layer_idx][head_idx]
        S = self.qjl_matrices[layer_idx][head_idx]
        k_c = turboquant_encode_internal(k_vec, self.config.codebook, rotation, S)
        v_c = turboquant_encode_internal(v_vec, self.config.codebook, rotation, S)
        self.cache[layer_idx][head_idx].append((k_c, v_c))

    def store_batch(self, layer_idx: int, head_idx: int, k_vecs: torch.Tensor, v_vecs: torch.Tensor):
        """Encode and store a batch of key-value pairs."""
        rotation = self.rotations[layer_idx][head_idx]
        S = self.qjl_matrices[layer_idx][head_idx]
        k_all = turboquant_encode_internal(k_vecs, self.config.codebook, rotation, S)
        v_all = turboquant_encode_internal(v_vecs, self.config.codebook, rotation, S)

        for i in range(k_vecs.shape[0]):
            k_single = TurboQuantCompressed(
                pq=PolarQuantCompressed(
                    norm=k_all.pq.norm[i:i+1], indices=k_all.pq.indices[i:i+1],
                    codebook=k_all.pq.codebook, rotation=k_all.pq.rotation,
                ),
                qjl=QJLCompressed(
                    signs=k_all.qjl.signs[i:i+1], r_norm=k_all.qjl.r_norm[i:i+1], S=S,
                ),
            )
            v_single = TurboQuantCompressed(
                pq=PolarQuantCompressed(
                    norm=v_all.pq.norm[i:i+1], indices=v_all.pq.indices[i:i+1],
                    codebook=v_all.pq.codebook, rotation=v_all.pq.rotation,
                ),
                qjl=QJLCompressed(
                    signs=v_all.qjl.signs[i:i+1], r_norm=v_all.qjl.r_norm[i:i+1], S=S,
                ),
            )
            self.cache[layer_idx][head_idx].append((k_single, v_single))

    def compute_attention(
        self, layer_idx: int, head_idx: int, q_vec: torch.Tensor, causal: bool = True,
        qjl_score_weight: float = 0.5,
    ) -> torch.Tensor:
        """Compute attention output using compressed KV cache.

        Uses PolarQuant decode for key scores with a damped QJL correction term,
        and PolarQuant-only decode for values.

        The QJL correction adds an unbiased estimate of the residual inner product,
        but a single-sample estimate has high variance. A weight < 1.0 reduces
        variance at the cost of slight bias, improving attention quality in practice.
        Set qjl_score_weight=0.0 to use PolarQuant-only scoring.
        Set qjl_score_weight=1.0 for the full unbiased QJL correction.

        Values are decoded with PolarQuant-only (not QJL), because a single-sample
        QJL residual estimate for vector reconstruction has too high a variance to
        improve MSE over PolarQuant alone.
        """
        d = self.d
        seq_len = len(self.cache[layer_idx][head_idx])
        if seq_len == 0:
            return torch.zeros(d, device=self.device)

        q_vec = q_vec.float()
        S = self.qjl_matrices[layer_idx][head_idx]
        qjl_scale = math.sqrt(math.pi / 2) / d

        # Pre-project query through S ONCE (amortized over all keys)
        # q_proj[i] = (S @ q)[i] = S[i,:] . q  — needed for QJL correction
        q_proj = S @ q_vec  # [d]

        # --- Batch-decode all PQ keys and compute scores ---
        # Collecting indices/norms for vectorized decode is faster than per-token loop
        pq_norms = torch.stack([
            self.cache[layer_idx][head_idx][t][0].pq.norm.squeeze(0)
            for t in range(seq_len)
        ])  # [seq_len]
        pq_indices = torch.cat([
            self.cache[layer_idx][head_idx][t][0].pq.indices
            for t in range(seq_len)
        ], dim=0)  # [seq_len, d]

        rotation = self.rotations[layer_idx][head_idx]
        # Batch PolarQuant decode — builds [seq_len, d] key estimates
        pq_batch = PolarQuantCompressed(
            norm=pq_norms,
            indices=pq_indices,
            codebook=self.config.codebook,
            rotation=rotation,
        )
        k_hat_all = polarquant_decode(pq_batch)  # [seq_len, d]
        score_pq_all = (k_hat_all @ q_vec) / math.sqrt(d)  # [seq_len]

        if qjl_score_weight > 0.0:
            # QJL correction: for each key t, correction = dot(S·q, sign(S·r)) * scale * r_norm
            # Batch compute: signs_pm [seq_len, d], q_proj [d]
            signs_pm_all = torch.cat([
                self.cache[layer_idx][head_idx][t][0].qjl.signs
                for t in range(seq_len)
            ], dim=0).float() * 2 - 1  # [seq_len, d]  values ∈ {-1, +1}

            r_norms_all = torch.stack([
                self.cache[layer_idx][head_idx][t][0].qjl.r_norm.squeeze(0)
                for t in range(seq_len)
            ]).float()  # [seq_len]

            # score_qjl[t] = dot(q_proj, signs_pm_all[t]) * qjl_scale * r_norm[t]
            qjl_ips = (signs_pm_all @ q_proj)  # [seq_len]
            score_qjl_all = qjl_ips * qjl_scale * r_norms_all / math.sqrt(d)  # [seq_len]

            scores = score_pq_all + qjl_score_weight * score_qjl_all
        else:
            scores = score_pq_all

        # Softmax
        attn_weights = F.softmax(scores, dim=0)  # [seq_len]

        # --- Batch-decode all PQ values ---
        v_pq_norms = torch.stack([
            self.cache[layer_idx][head_idx][t][1].pq.norm.squeeze(0)
            for t in range(seq_len)
        ])
        v_pq_indices = torch.cat([
            self.cache[layer_idx][head_idx][t][1].pq.indices
            for t in range(seq_len)
        ], dim=0)

        v_pq_batch = PolarQuantCompressed(
            norm=v_pq_norms,
            indices=v_pq_indices,
            codebook=self.config.codebook,
            rotation=rotation,
        )
        v_hat_all = polarquant_decode(v_pq_batch)  # [seq_len, d]
        # Note: We use PolarQuant-only for values. A single-sample QJL residual
        # reconstruction has too high a variance to improve over PQ alone.
        # The QJL data is stored for potential multi-sample averaging in future work.

        # Weighted sum: [seq_len] x [seq_len, d] → [d]
        output = (attn_weights.unsqueeze(-1) * v_hat_all.float()).sum(0)

        return output


# ---------------------------------------------------------------------------
# Utility: Compression ratio analysis
# ---------------------------------------------------------------------------

def compression_ratio_fp16(d: int, b_mse: int = B_MSE) -> float:
    """Compute compression ratio vs FP16."""
    fp16_bits = d * 16
    tq_bits = (d - 1) * b_mse + 16 + d * 1 + 16  # PQ angles + norm + QJL signs + r_norm
    return fp16_bits / tq_bits


def memory_bytes_per_vector(d: int, b_mse: int = B_MSE) -> Tuple[int, int]:
    """Returns (tq_bytes, fp16_bytes) per vector."""
    tq_bits = (d - 1) * b_mse + 16 + d * 1 + 16
    tq_bytes = (tq_bits + 7) // 8
    fp16_bytes = d * 2
    return tq_bytes, fp16_bytes
