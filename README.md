# ⚡ TurboQuant

[![arXiv](https://img.shields.io/badge/arXiv-2504.19874-b31b1b.svg)](https://arxiv.org/abs/2504.19874)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**First open-source implementation of Google's TurboQuant** — near-optimal KV cache compression for LLM inference. Compress your KV cache by **5×** with near-zero quality loss.

> 3.5 bits per channel = **identical quality** to FP16. 2.5 bits = marginal loss. Provably within 2.7× of information-theoretic optimal.

## Why TurboQuant?

| Method | KV Bits | LongBench Avg | Needle-in-Haystack |
|--------|---------|---------------|-------------------|
| Full Precision | 16 | 50.06 | 0.997 |
| **TurboQuant** | **3.5** | **50.06** | **0.997** |
| **TurboQuant** | **2.5** | **49.44** | **0.997** |
| PolarQuant | 3.9 | 49.78 | 0.995 |
| KIVI | 3 | 48.50 | 0.981 |
| SnapKV | — | 44.57 | 0.858 |

*Results from the TurboQuant paper (Llama-3.1-8B-Instruct). Our implementation targets the same algorithm.*

## How It Works

TurboQuant is a two-stage vector quantizer that achieves near-optimal compression:

```
Input KV vector (FP16, d=128)
         │
         ▼
┌─────────────────────┐
│  Random Rotation Π   │  Hadamard + random signs
│  y = Π · x          │  O(d log d), preserves norms
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Scalar Lloyd-Max    │  Each coordinate independently
│  idx = quantize(y)   │  b bits per coordinate
│                      │  Beta dist ≈ N(0, 1/d)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  QJL Residual        │  1-bit sign quantization
│  sign(S · residual)  │  Unbiased inner products
└─────────┬───────────┘
          │
          ▼
   Compressed: (b+1) bits/coord + FP16 norm
   = ~3.25 bits/value at b=2
   = ~4.9× compression vs FP16
```

**Key insight**: After random rotation, each coordinate follows a Beta distribution that's near-independent of other coordinates. This means scalar quantization per coordinate is near-optimal — no coupling, no error compounding through deep models.

## Quick Start

```bash
# Install
git clone https://github.com/OnlyTerp/turboquant.git
cd turboquant
pip install -e .

# Run demo (synthetic vectors, no GPU needed)
python src/demo.py

# Run real model validation (downloads TinyLlama or Nemotron-Nano-4B)
python src/test_real_model.py
```

## Results

### Synthetic Vector Benchmark (d=128)

| Metric | Value |
|--------|-------|
| Compression ratio | 4.92× vs FP16 |
| Bits per value | 3.25 |
| Memory saved | 79.7% |
| Avg cosine similarity | 0.90 |

### Real Model Validation

*Benchmarks on real transformer models coming soon.*

## Algorithm Details

TurboQuant implements two algorithms from the paper:

### Algorithm 1: TurboQuant_mse (MSE-optimal)

1. **Random rotation**: Multiply by randomized Hadamard matrix Π
2. **Scalar quantization**: Lloyd-Max codebook for Beta distribution, applied per coordinate
3. **Store**: b-bit index per coordinate + FP16 norm
4. **Distortion bound**: MSE ≤ √(3π/2) · 4^(-b)

### Algorithm 2: TurboQuant_prod (inner product-optimal)

1. Apply TurboQuant_mse with (b-1) bits
2. Compute residual: r = x - DeQuant(Quant(x))
3. QJL: sign(S · r) where S has i.i.d. N(0,1) entries
4. **Unbiased**: E[⟨y, x̂⟩] = ⟨y, x⟩ (no systematic bias)
5. **Total**: b bits per coordinate

### Why Not Recursive Polar Transform?

The related PolarQuant paper uses recursive polar coordinates, but TurboQuant deliberately avoids this. Recursive polar transforms couple coordinates through sin/cos operations at each level, causing errors to compound through deep models (7 levels for d=128). TurboQuant's scalar approach quantizes each coordinate independently — zero coupling, zero compounding.

## Project Structure

```
turboquant/
├── src/
│   ├── cache.py              # Core algorithm (encode/decode/cache/attention)
│   ├── demo.py               # Synthetic benchmark
│   ├── test_real_model.py    # Real transformer model validation
│   ├── test_turboquant.py    # Unit tests (33 tests)
│   ├── kernels.py            # Triton GPU kernels (experimental)
│   └── lut_attention.py      # LUT-based attention (experimental)
├── vllm_plugin/              # vLLM integration scaffold
├── deploy/                   # Docker deployment assets
├── BENCHMARKS.md             # Detailed measurements
├── IMPLEMENTATION_NOTES.md   # Implementation decisions
└── setup.py                  # Package installation
```

## Citation

```bibtex
@inproceedings{zandieh2026turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## Credits

- **Paper**: [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- **Authors**: Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni (Google Research / NYU / Google DeepMind)
- **Implementation**: [Terp AI Labs](https://github.com/OnlyTerp)

## License

MIT — see [LICENSE](LICENSE) for details.
