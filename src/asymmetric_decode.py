"""End-to-end asymmetric attention decode with TurboQuant.

Flow:
  1. Prefill: run model normally, get FP16 KV cache
  2. Compress: convert entire KV cache to TurboQuantCompressed (stored in custom list)
  3. Delete FP16 KV cache immediately
  4. Decode: for each new token, encode K+V with TurboQuant, add to compressed cache
     Then compute attention by scoring against compressed K (codebook lookup),
     summing with decompressed V — K stays compressed throughout decode
"""
import os, sys, time
os.environ["BNB_CUDA_TRITON"] = "0"
os.environ["PYTORCH_XPU_ALLOC_CONF"] = "expandable_segments:True"
import torch

sys.path.insert(0, "/home/hermes/turboquant-experiments/OnlyTerp-turboquant/src")
from cache import (
    TurboQuantConfig, TurboQuantCache,
    turboquant_encode_internal, polarquant_decode,
    turboquant_decode_single, turboquant_attention,
)
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("xpu")

# ── setup ─────────────────────────────────────────────────────────────────────
print("Loading tokenizer..."); sys.stdout.flush()
tok = AutoTokenizer.from_pretrained("/home/hermes/.cache/huggingface/modules",
    trust_remote_code=True, padding_side="left")
if tok.pad_token is None: tok.pad_token = tok.eos_token

MODEL = "Qwen/Qwen3-0.6B"
CTX = 512
DECODE_STEPS = 50

print(f"Loading {MODEL}..."); sys.stdout.flush()
m = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True,
)
m = m.to(device).eval()
torch.xpu.synchronize()

attn_0 = m.model.layers[0].self_attn
head_dim = attn_0.head_dim
num_q_heads = m.config.num_attention_heads        # 16
num_kv_heads = m.config.num_key_value_heads       # 8
kv_groups = num_q_heads // num_kv_heads          # 2 (GQA)
scaling = attn_0.scaling

print(f"Ready. {num_q_heads}Q/{num_kv_heads}KV, head_dim={head_dim}, groups={kv_groups}")
print(f"VRAM={torch.xpu.memory_allocated()/1e9:.2f}GB\n"); sys.stdout.flush()

# RoPE — vectorized for [heads, seq, dim]
def apply_rope(q, cos, sin):
    """q: [H, seq, D], cos/sin: [1, seq, D]. Apply rotary embedding."""
    D = q.shape[-1]
    q0, q1 = q[..., :D//2], q[..., D//2:]
    return torch.cat([-(q1 * sin) + (q0 * cos), (q0 * sin.neg()) + (q1 * cos)], dim=-1)

# ── prompt ────────────────────────────────────────────────────────────────────
text = (
    "In the northern reaches of the Ember Valley, the last remnants of the ancient "
    "Clockwork Empire lay buried beneath layers of volcanic ash. For three centuries "
    "the great brass automatons had stood frozen in their eternal gardens. Scholars "
    "from the Athenaeum of Dusk had catalogued over four thousand distinct automaton "
    "designs, from the towering War Golems to the delicate Songbirds."
)
tokens = tok(text, return_tensors="pt", truncation=True, max_length=CTX)["input_ids"]
if tokens.shape[1] > CTX:
    tokens = tokens[:, :CTX]
print(f"Prompt: {tokens.shape[1]} tokens\n"); sys.stdout.flush()

# ══════════════════════════════════════════════════════════════════════════════
# BASELINE
# ══════════════════════════════════════════════════════════════════════════════
print("=== BASELINE ==="); sys.stdout.flush()
torch.xpu.reset_peak_memory_stats()
torch.xpu.synchronize()
t0 = time.time()
with torch.no_grad():
    gen_out = m.generate(tokens.to(device), max_new_tokens=DECODE_STEPS,
        do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
torch.xpu.synchronize(); t_baseline = time.time() - t0
vram_baseline = torch.xpu.max_memory_allocated() / 1e9
baseline_text = tok.decode(gen_out[0])
print(f"  Time={t_baseline:.1f}s  Peak={vram_baseline:.3f}GB\n"); sys.stdout.flush()

# ══════════════════════════════════════════════════════════════════════════════
# TURBOQUANT PATH
# ══════════════════════════════════════════════════════════════════════════════
print("=== TURBOQUANT ==="); sys.stdout.flush()

# 1. Prefill
torch.xpu.reset_peak_memory_stats()
torch.xpu.synchronize()
t0 = time.time()
with torch.no_grad():
    prefill_out = m(input_ids=tokens.to(device), use_cache=True)
torch.xpu.synchronize(); t_prefill = time.time() - t0
vram_prefill = torch.xpu.memory_allocated() / 1e9
pkv = prefill_out.past_key_values

n_layers = len(pkv.layers)
seq_len = pkv.layers[0].keys.shape[2]
dtype = pkv.layers[0].keys.dtype

kv_bytes_total = sum(
    l.keys.element_size()*l.keys.numel() + l.values.element_size()*l.values.numel()
    for l in pkv.layers
)
print(f"  Prefill:  {t_prefill:.1f}s  VRAM={vram_prefill:.2f}GB")
print(f"  KV:       {n_layers}L x {num_kv_heads}KV x {seq_len} x {head_dim}d = {kv_bytes_total/1e6:.1f}MB\n"); sys.stdout.flush()

# 2. Build TurboQuantCache config and pre-compute per-layer/head structures
tqc_config = TurboQuantConfig(
    d=head_dim, b_mse=3, device=device,
    mixed_precision=False, use_online_codebook=False,
)
rotations = [[tqc_config.make_rotation(l, h) for h in range(num_kv_heads)] for l in range(n_layers)]
S_matrices = [[tqc_config.make_qjl_matrix(l, h) for h in range(num_kv_heads)] for l in range(n_layers)]
S_seeds = [[((l * 1000003) ^ (h * 999979) ^ 0x5A5A5A5A) & 0xFFFFFFFF for h in range(num_kv_heads)] for l in range(n_layers)]

# 3. Compress entire prefill KV into TurboQuantCompressed objects
torch.xpu.synchronize(); t0 = time.time()
# Custom cache: [layer][head] = list of TurboQuantCompressed per timestep
tqc_cache: list[list[list]] = [[list() for _ in range(num_kv_heads)] for _ in range(n_layers)]

for l in range(n_layers):
    for h in range(num_kv_heads):
        k = pkv.layers[l].keys[:, h, :, :].reshape(1, seq_len, head_dim).float()  # [1, seq, D]
        v = pkv.layers[l].values[:, h, :, :].reshape(1, seq_len, head_dim).float()
        rot = rotations[l][h]
        S = S_matrices[l][h]
        S_seed = S_seeds[l][h]

        # Encode each timestep individually (avoids batch-slicing shape issues)
        for t in range(seq_len):
            k_t = k[:, t:t+1, :]    # [1, 1, D]
            v_t = v[:, t:t+1, :]    # [1, 1, D]
            kc_t = turboquant_encode_internal(k_t.squeeze(1), tqc_config.codebook, rot, S, mixed=None)
            vc_t = turboquant_encode_internal(v_t.squeeze(1), tqc_config.codebook, rot, S, mixed=None)
            tqc_cache[l][h].append((kc_t, vc_t))

torch.xpu.synchronize(); t_compress = time.time() - t0
vram_after_compress = torch.xpu.memory_allocated() / 1e9
print(f"  Compress: {t_compress:.1f}s  VRAM={vram_after_compress:.2f}GB"); sys.stdout.flush()

# 4. Delete FP16 KV
del pkv, prefill_out
torch.xpu.synchronize()
vram_after_del = torch.xpu.memory_allocated() / 1e9
kv_saved_mb = kv_bytes_total / 1e6
print(f"  After del pkv: VRAM={vram_after_del:.2f}GB  (saved {vram_prefill - vram_after_del:.2f}GB)"); sys.stdout.flush()

# 5. Asymmetric decode loop
print(f"\n  Asymmetric decode ({DECODE_STEPS} steps)...\n"); sys.stdout.flush()
rotary = m.model.rotary_emb
seq_ids = tokens.to(device)
torch.xpu.synchronize(); t0 = time.time()

# Reusable codebook instance for decode
tqc_codebook = tqc_config.codebook

with torch.no_grad():
    for step in range(DECODE_STEPS):
        cur_len = seq_ids.shape[1]

        # Position embedding for all positions
        pos_ids = torch.arange(cur_len, device=device).unsqueeze(0)  # [1, seq]
        cos_full, sin_full = rotary(seq_ids, pos_ids)   # [1, seq, D]

        if step == 0:
            # Debug: inspect the compressed cache
            kc0, vc0 = tqc_cache[0][0][0]
            print(f"    DEBUG step0 layer0 head0: kc0.pq.norm={kc0.pq.norm.shape} kc0.pq.indices={kc0.pq.indices.shape} kc0.qjl.signs={kc0.qjl.signs.shape}")

        # Embed full sequence: [1, seq, hidden]
        hidden = m.model.embed_tokens(seq_ids)

        for layer_idx, layer in enumerate(m.model.layers):
            attn = layer.self_attn

            # ── Q: project + reshape + norm + RoPE ──────────────────────────
            # q_proj: [1, 1, hidden] -> [1, 1, num_q_heads * head_dim]
            q_flat = attn.q_proj(hidden[:, -1:, :])   # [1, 1, 2048]
            # reshape to [1, num_q_heads, 1, head_dim] = [1, 16, 1, 128]
            q_reshaped = q_flat.view(1, num_q_heads, 1, head_dim)
            # q_norm expects [1, 16, 1, 128] -> produces same
            q_normed = attn.q_norm(q_reshaped)         # [1, 16, 1, 128]
            # Apply RoPE at last position using HF's apply_rotary_pos_emb
            q_for_rope = q_normed    # [1, H, 1, D]
            cos_q = cos_full[:, -1:, :]    # [1, 1, D]
            sin_q = sin_full[:, -1:, :]    # [1, 1, D]
            q, _ = apply_rotary_pos_emb(q_for_rope, q_for_rope, cos_q, sin_q, unsqueeze_dim=1)
            # q: [1, H, 1, D]

            # ── K/V: project + reshape + norm + RoPE ──────────────────────────
            # k_proj: [1, 1, hidden] -> [1, 1, num_kv_heads * head_dim]
            k_flat = attn.k_proj(hidden[:, -1:, :])   # [1, 1, 1024]
            v_flat = attn.v_proj(hidden[:, -1:, :])   # [1, 1, 1024]
            # reshape: [1, 1, 1024] -> [1, num_kv_heads, 1, head_dim] = [1, 8, 1, 128]
            k_reshaped = k_flat.view(1, num_kv_heads, 1, head_dim)  # [1, 8, 1, 128]
            v_reshaped = v_flat.view(1, num_kv_heads, 1, head_dim)  # [1, 8, 1, 128]
            k_normed = attn.k_norm(k_reshaped)         # [1, 8, 1, 128]
            # Apply RoPE at last position using HF's apply_rotary_pos_emb
            k_for_rope = k_normed    # [1, H, 1, D]
            cos_k = cos_full[:, -1:, :]    # [1, 1, D]
            sin_k = sin_full[:, -1:, :]    # [1, 1, D]
            _, k_new = apply_rotary_pos_emb(k_for_rope, k_for_rope, cos_k, sin_k, unsqueeze_dim=1)
            # k_new: [1, H, 1, D]
            v_new = v_reshaped.squeeze(2).bfloat16()      # [1, 8, 128]

            # ── Encode new token K/V (deferred — append AFTER all layers computed) ──
            rot_l = rotations[layer_idx]
            S_l = S_matrices[layer_idx]
            encoded_kv = []  # [(kn, vn), ...] per head, added to cache after all layers
            for h in range(num_kv_heads):
                kn = turboquant_encode_internal(
                    k_new[0, h, 0, :], tqc_codebook, rot_l[h], S_l[h], mixed=None)
                vn = turboquant_encode_internal(
                    v_new[0, h, :], tqc_codebook, rot_l[h], S_l[h], mixed=None)
                encoded_kv.append((kn, vn))

            # ── Asymmetric attention per KV head ─────────────────────────────
            attn_out_h = []
            for kv_head in range(num_kv_heads):
                q_group_start = kv_head * kv_groups
                q_group_end = (kv_head + 1) * kv_groups
                q_h = q[:, q_group_start:q_group_end, :, :]  # [1, G, 1, D]

                cache_h = tqc_cache[layer_idx][kv_head]
                seq_k = len(cache_h)

                K_vals, V_vals = [], []
                for t_idx in range(seq_k):
                    kc, vc = cache_h[t_idx]

                    # Full TurboQuant decode for K, then apply RoPE at position t_idx
                    k_dec = turboquant_decode_single(kc)   # [D]
                    # k_dec is [D], make it [1, 1, D] for apply_rotary_pos_emb
                    k_for_rope = k_dec.unsqueeze(0).unsqueeze(0)    # [1, 1, D]
                    cos_t = cos_full[0, t_idx:t_idx+1, :].unsqueeze(0)  # [1, 1, D]
                    sin_t = sin_full[0, t_idx:t_idx+1, :].unsqueeze(0)  # [1, 1, D]
                    _, k_rot = apply_rotary_pos_emb(k_for_rope, k_for_rope, cos_t, sin_t, unsqueeze_dim=1)
                    k_rot = k_rot.squeeze(0).squeeze(0)  # [D]
                    K_vals.append(k_rot)

                    # Full TurboQuant decode for V
                    v_dec = turboquant_decode_single(vc)  # [D]
                    V_vals.append(v_dec)

                # K_vals/V_vals: list of [1, 1, 128] -> stack to [1, seq, 1, 128]
                K = torch.stack(K_vals, dim=1).squeeze(2).to(torch.bfloat16)   # [1, seq, D]
                V = torch.stack(V_vals, dim=1).squeeze(2).to(torch.bfloat16)   # [1, seq, D]
                seq_k = K.shape[1]

                # Expand V for GQA groups: [1, seq, D] -> [1, G, seq, D]
                V_exp = V.unsqueeze(1).expand(-1, kv_groups, -1, -1)   # [1, G, seq, D]

                # Attention scores: q_h [1, G, 1, D] @ K [1, seq, D]^T
                # K needs to be [1, 1, seq, D] for matmul
                K_exp = K.unsqueeze(1)                                  # [1, 1, seq, D]
                scores = torch.matmul(
                    q_h.to(torch.bfloat16),
                    K_exp.to(torch.bfloat16)
                ) * scaling   # [1, G, 1, seq]
                attn_w = torch.softmax(scores, dim=-1)
                out = torch.matmul(attn_w, V_exp)                        # [1, G, 1, D]
                attn_out_h.append(out)

            attn_out = torch.cat(attn_out_h, dim=1)   # [1, 16, 1, 128]

            # ── Append new token K/V to compressed cache (deferred from above) ──
            for h in range(num_kv_heads):
                tqc_cache[layer_idx][h].append(encoded_kv[h])

            # ── Output projection + residual + FFN ───────────────────────────
            # attn_out: [1, 16, 1, 128] -> [1, 2048] for o_proj
            attn_out = attn_out.reshape(1, num_q_heads * head_dim)  # [1, 2048]
            attn_out = attn.o_proj(attn_out)                        # [1, hidden]
            # Residual: hidden[:, -1:, :] + attn_out (broadcast over seq)
            residual = hidden[:, -1:, :] + attn_out.unsqueeze(1)     # [1, 1, hidden]
            hidden = layer.input_layernorm(residual)
            hidden = layer.post_attention_layernorm(hidden + layer.mlp(hidden))

        # Sample next token
        logits = m.lm_head(hidden[:, -1, :])
        next_tok = logits.argmax(dim=-1, keepdim=True)
        seq_ids = torch.cat([seq_ids, next_tok], dim=-1)

        if step % 10 == 0:
            print(f"    Step {step}/{DECODE_STEPS} ({cur_len} tokens)"); sys.stdout.flush()

torch.xpu.synchronize(); t_decode = time.time() - t0
vram_tqc = torch.xpu.max_memory_allocated() / 1e9
tqc_text = tok.decode(seq_ids[0])
print(f"  Done. Decode={t_decode:.1f}s ({t_decode/DECODE_STEPS*1000:.0f}ms/token)  Peak={vram_tqc:.3f}GB\n")

# ── results ──────────────────────────────────────────────────────────────────
print("=" * 70)
print("  RESULTS")
print("=" * 70)
print(f"  Model: {MODEL}  ctx={CTX}  decode={DECODE_STEPS}")
print(f"\n  {'Metric':<35} {'Baseline':<12} {'TurboQuant':<12}")
print(f"  {'-'*35} {'-'*12} {'-'*12}")
print(f"  {'Peak VRAM (GB)':<35} {vram_baseline:<12.3f} {vram_tqc:<12.3f}")
print(f"  {'VRAM delta':<35} {'':<12} {vram_baseline-vram_tqc:+.3f}GB")
print(f"  {'Total time (s)':<35} {t_baseline:<12.1f} {t_prefill+t_compress+t_decode:<12.1f}")
print(f"  {'Decode ms/token':<35} {t_baseline/DECODE_STEPS*1000:<12.0f} {t_decode/DECODE_STEPS*1000:<12.0f}")
print(f"\n  FP16 KV: {kv_bytes_total/1e6:.1f}MB  |  VRAM saved after del: {(vram_prefill-vram_after_del)*1024:.0f}MB")
print(f"\n  --- BASELINE OUTPUT ---")
print(f"  {baseline_text[:500]}")
print(f"\n  --- TURBOQUANT OUTPUT ---")
print(f"  {tqc_text[:500]}")
print("=" * 70)