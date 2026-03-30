import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, 'src')

from cache import (
    TurboQuantConfig, turboquant_encode_internal,
    turboquant_attention,
)

DEVICE = torch.device('xpu')
torch.manual_seed(42)

def make_compressed(x, config, layer=0):
    """x: [seq, D] — returns list[list[TQC]] shape [1][1]"""
    rot = config.make_rotation(layer, 0)
    S   = config.make_qjl_matrix(layer, 0)
    enc = turboquant_encode_internal(
        x, config.codebook, rot, S, mixed=None
    )
    return [[enc]]   # [batch=1][head=1]

def test_output_shape():
    torch.manual_seed(42)
    B, H, seq_k, D = 1, 1, 16, 128
    cfg = TurboQuantConfig(d=D, b_mse=3, device=DEVICE,
                           mixed_precision=False)
    k = torch.randn(seq_k, D, device=DEVICE)
    v = torch.randn(seq_k, D, device=DEVICE)
    q = torch.randn(B, H, 1, D, device=DEVICE)
    ck = make_compressed(k, cfg)
    cv = make_compressed(v, cfg)
    out = turboquant_attention(q, ck, cv, cfg)
    assert out.shape == (B, H, 1, D), f"bad shape {out.shape}"
    print("test_output_shape PASS")

def test_cos_sim():
    torch.manual_seed(42)
    seq_k, D = 32, 128
    cfg = TurboQuantConfig(d=D, b_mse=3, device=DEVICE,
                           mixed_precision=False)
    k = torch.randn(seq_k, D, device=DEVICE)
    v = torch.randn(seq_k, D, device=DEVICE)
    q = torch.randn(1, 1, 1, D, device=DEVICE)
    ck = make_compressed(k, cfg)
    cv = make_compressed(v, cfg)
    tq_out = turboquant_attention(q, ck, cv, cfg)  # [1,1,1,D]

    # reference: standard attention against raw KV
    scores = (k @ q[0,0,0]) / D**0.5          # [seq_k]
    weights = F.softmax(scores, dim=0)         # [seq_k]
    ref_out = (weights.unsqueeze(-1) * v).sum(0)  # [D]

    cos = F.cosine_similarity(
        tq_out[0,0,0].unsqueeze(0),
        ref_out.unsqueeze(0)
    ).item()
    print(f"test_cos_sim: {cos:.4f}")
    assert cos > 0.65, f"cos_sim {cos:.4f} < 0.65"
    print("test_cos_sim PASS")

if __name__ == '__main__':
    test_output_shape()
    test_cos_sim()
