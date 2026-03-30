import sys
sys.path.insert(0, 'src')

import torch
import torch.nn.functional as F
from cache import (
    TurboQuantConfig,
    turboquant_encode_internal,
    polarquant_decode,
    turboquant_decode_single,
    turboquant_attention,
)


def test_turboquant_attention():
    device = torch.device('xpu')

    B, H, seq_k, D = 1, 8, 16, 128
    config = TurboQuantConfig(d=D, b_mse=3, device=device, mixed_precision=False)

    torch.manual_seed(42)
    k_full = torch.randn(B, seq_k, D, device=device)
    v_full = torch.randn(B, seq_k, D, device=device)
    q = torch.randn(B, H, 1, D, device=device)

    # Compress KV for each head per batch element
    compressed_k = []
    compressed_v = []
    for b in range(B):
        batch_k = []
        batch_v = []
        for h in range(H):
            rot = config.make_rotation(0, h)
            S = config.make_qjl_matrix(0, h)
            seed = ((b * 1000003) ^ (h * 999979) ^ 0x5A5A5A5A) & 0xFFFFFFFF
            k_enc = turboquant_encode_internal(k_full[b:b+1], config.codebook, rot, S, seed)
            v_enc = turboquant_encode_internal(v_full[b:b+1], config.codebook, rot, S, seed)
            batch_k.append(k_enc)
            batch_v.append(v_enc)
        compressed_k.append(batch_k)
        compressed_v.append(batch_v)

    # Run turboquant attention
    out_tq = turboquant_attention(q, compressed_k, compressed_v, config)

    # Reference: standard attention on FULLY decoded (PQ+QJL) K and V
    cos_sims = []
    for h in range(H):
        q_h = q[0, h, 0]  # [D]

        # Decode K and V fully per batch
        k_decoded_all = []
        v_decoded_all = []
        for b in range(B):
            rot = config.make_rotation(0, h)
            S = config.make_qjl_matrix(0, h)
            seed = ((b * 1000003) ^ (h * 999979) ^ 0x5A5A5A5A) & 0xFFFFFFFF
            k_enc = turboquant_encode_internal(k_full[b:b+1], config.codebook, rot, S, seed)
            v_enc = turboquant_encode_internal(v_full[b:b+1], config.codebook, rot, S, seed)
            k_decoded_all.append(turboquant_decode_single(k_enc))
            v_decoded_all.append(turboquant_decode_single(v_enc))

        k_decoded = torch.cat(k_decoded_all, dim=0)  # [B, seq_k, D]
        v_decoded = torch.cat(v_decoded_all, dim=0)

        # Standard attention: [B, seq_k, D] @ [B, D, 1] -> [B, seq_k, 1]
        scale = D ** -0.5
        q_exp = q_h.unsqueeze(0).unsqueeze(-1)   # [1, D, 1]
        k_exp = k_decoded                        # [1, seq_k, D]
        scores = (k_exp @ q_exp).squeeze(-1) * scale  # [1, seq_k]
        weights = F.softmax(scores, dim=1)          # [1, seq_k]
        v_out = (weights.unsqueeze(-1) * v_decoded).sum(1)  # [1, D]

        tq_out = out_tq[0, h, 0]  # [D]
        cos_sim = torch.dot(tq_out, v_out.squeeze(0)) / (tq_out.norm() * v_out.squeeze(0).norm() + 1e-8)
        cos_sim = cos_sim.item()
        cos_sims.append(cos_sim)

    avg_cos_sim = sum(cos_sims) / len(cos_sims)
    threshold = 0.85
    passed = avg_cos_sim > threshold

    print(f"Cosine similarity per head: {[f'{c:.4f}' for c in cos_sims]}")
    print(f"Average cosine similarity: {avg_cos_sim:.4f}")
    print(f"Threshold: {threshold}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")

    return passed, avg_cos_sim


if __name__ == '__main__':
    test_turboquant_attention()
