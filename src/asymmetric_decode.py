"""End-to-end asymmetric attention decode with TurboQuant.

Design (simple working version):
  1. Prefill: run model normally, use_cache=True → get past_key_values
     Extract raw (unrotated) K/V from pkv (pre-RoPE).
     Compress these raw K/V into TurboQuantCache.
  2. Delete FP16 past_key_values immediately
  3. Decode loop (manual, per layer):
     a. Project Q from hidden (attn.q_proj + q_norm) — no RoPE needed for TQ
     b. Get cached K/V from TurboQuantCache (Hadamard-rotated via PQ encode)
     c. Apply trigonometric RoPE to Q at current position
     d. turboquant_attention: scores Q (Hadamard-rotated internally)
        against cached K (Hadamard-rotated during encode), decompresses V for
        weighted sum
     e. Encode new token's K/V and append to TurboQuantCache
     f. Run FFN
  4. Sample next token from lm_head logits
  5. Stop at EOS or DECODE_STEPS tokens

The key invariant: K/V in TurboQuantCache are stored WITHOUT trigonometric RoPE.
They are rotated by Hadamard (inside PQ encode) only. The RoPE at each
decode position is applied fresh to the query. turboquant_attention handles
Hadamard rotation for both Q and cached K via the same config.make_rotation().
"""
# STATUS: proof-of-concept only. Manual layer loop is ~100x slower than
# HF generate() on XPU. Next step: replace with HF attention_override hook.
# turboquant_attention() itself is fast (~0.07s/step).

import os, sys, time
import time as _time
os.environ["BNB_CUDA_TRITON"] = "0"
os.environ["PYTORCH_XPU_ALLOC_CONF"] = "expandable_segments:True"
import torch

sys.path.insert(0, "/home/hermes/turboquant-experiments/OnlyTerp-turboquant/src")
from cache import (
    TurboQuantConfig, TurboQuantCache,
    turboquant_encode_internal, turboquant_decode_single,
    turboquant_attention,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("xpu")

# ── config ────────────────────────────────────────────────────────────────────
MODEL = "Qwen/Qwen3-0.6B"
CTX = 512
DECODE_STEPS = 3

# ── load model + tokenizer ───────────────────────────────────────────────────
print("Loading tokenizer..."); sys.stdout.flush()
tok = AutoTokenizer.from_pretrained(
    "/home/hermes/.cache/huggingface/modules",
    trust_remote_code=True, padding_side="left"
)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print(f"Loading {MODEL}..."); sys.stdout.flush()
m = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True,
)
m = m.to(device).eval()
torch.xpu.synchronize()

attn_0 = m.model.layers[0].self_attn
config_0 = m.config

head_dim = attn_0.head_dim
num_q_heads = config_0.num_attention_heads       # 16
num_kv_heads = config_0.num_key_value_heads       # 8
kv_groups = num_q_heads // num_kv_heads          # 2 (GQA)
hidden_size = config_0.hidden_size

print(f"Ready. {num_q_heads}Q/{num_kv_heads}KV, head_dim={head_dim}, groups={kv_groups}")
print(f"VRAM={torch.xpu.memory_allocated()/1e9:.2f}GB\n"); sys.stdout.flush()

# ── prompt ───────────────────────────────────────────────────────────────────
text = (
    "In the northern reaches of the Ember Valley, the last remnants of the ancient "
    "Clockwork Empire lay buried beneath layers of volcanic ash. For three centuries "
    "the great brass automatons had stood frozen in their eternal gardens."
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
    gen_out = m.generate(
        tokens.to(device), max_new_tokens=DECODE_STEPS,
        do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id
    )
torch.xpu.synchronize()
t_baseline = time.time() - t0
vram_baseline = torch.xpu.max_memory_allocated() / 1e9
baseline_text = tok.decode(gen_out[0])
print(f"  Time={t_baseline:.1f}s  Peak={vram_baseline:.3f}GB\n"); sys.stdout.flush()

# ══════════════════════════════════════════════════════════════════════════════
# TURBOQUANT PATH
# ══════════════════════════════════════════════════════════════════════════════
print("=== TURBOQUANT ==="); sys.stdout.flush()

# ── 1. Prefill ────────────────────────────────────────────────────────────────
torch.xpu.reset_peak_memory_stats()
torch.xpu.synchronize()
t0 = time.time()
with torch.no_grad():
    prefill_out = m(input_ids=tokens.to(device), use_cache=True)
torch.xpu.synchronize()
t_prefill = time.time() - t0
vram_prefill = torch.xpu.memory_allocated() / 1e9
pkv = prefill_out.past_key_values

n_layers = len(pkv.layers)
seq_len = pkv.layers[0].keys.shape[2]   # [batch, heads, seq, head_dim]

kv_bytes_total = sum(
    l.keys.element_size() * l.keys.numel() +
    l.values.element_size() * l.values.numel()
    for l in pkv.layers
)
print(f"  Prefill:  {t_prefill:.1f}s  VRAM={vram_prefill:.2f}GB")
print(f"  KV:       {n_layers}L x {num_kv_heads}KV x {seq_len} x {head_dim}d = {kv_bytes_total/1e6:.1f}MB\n"); sys.stdout.flush()

# ── 2. Extract UNROTATED K/V and compress ───────────────────────────────────
# pkv.layers[l].keys/values are pre-RoPE (unrotated) — perfect for TQ encode.
# RoPE is applied in apply_rotary_pos_emb AFTER k_norm, but pkv stores k before RoPE.

print("  Compressing KV cache..."); sys.stdout.flush()
torch.xpu.synchronize()
t0 = time.time()

# Build TurboQuant config
tqc_config = TurboQuantConfig(
    d=head_dim, b_mse=3, device=device,
    mixed_precision=False, use_online_codebook=False,
)

# tqc_cache[layer][head] = one TurboQuantCompressed holding ALL seq_len tokens
tqc_cache: list[list[object]] = [[None] * num_kv_heads for _ in range(n_layers)]

for l in range(n_layers):
    for h in range(num_kv_heads):
        k_raw = pkv.layers[l].keys[:, h, :, :].float()    # [1, seq, D]
        v_raw = pkv.layers[l].values[:, h, :, :].float()  # [1, seq, D]

        rot = tqc_config.make_rotation(l, h)
        S = tqc_config.make_qjl_matrix(l, h)

        # Encode ALL seq_len tokens at once as one batch: [seq, D]
        k_enc = turboquant_encode_internal(
            k_raw.squeeze(0),    # [seq, D]
            tqc_config.codebook, rot, S, mixed=None,
        )
        v_enc = turboquant_encode_internal(
            v_raw.squeeze(0),
            tqc_config.codebook, rot, S, mixed=None,
        )
        tqc_cache[l][h] = (k_enc, v_enc)

torch.xpu.synchronize()
t_compress = time.time() - t0
vram_after_compress = torch.xpu.memory_allocated() / 1e9
print(f"  Compress: {t_compress:.1f}s  VRAM={vram_after_compress:.2f}GB"); sys.stdout.flush()

# ── 3. Delete FP16 past_key_values ────────────────────────────────────────────
del pkv, prefill_out
torch.xpu.synchronize()
vram_after_del = torch.xpu.memory_allocated() / 1e9
print(f"  After del pkv: VRAM={vram_after_del:.2f}GB  "
      f"(saved {vram_prefill - vram_after_del:.2f}GB)\n"); sys.stdout.flush()

# ── 4. Decode loop ────────────────────────────────────────────────────────────
print(f"  Decoding {DECODE_STEPS} steps...\n"); sys.stdout.flush()
rotary = m.model.rotary_emb
seq_ids = tokens.to(device)
torch.xpu.synchronize()
t0 = time.time()

# ── rotary helper ────────────────────────────────────────────────────────────
def apply_rotary_pos_emb_xpu(q, k, cos, sin):
    """Rotary positional embedding for [B, H, seq, D].
    Compatible with XPU tensors. Returns (q_rot, k_rot).
    """
    def _rotate_half(x):
        D = x.shape[-1]
        x0 = x[..., :D//2]
        x1 = x[..., D//2:]
        return torch.cat([-x1, x0], dim=-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


with torch.no_grad():
    for step in range(DECODE_STEPS):
        t_step0 = _time.perf_counter()

        cur_len = seq_ids.shape[1]

        # Position IDs for RoPE at current (last) position
        pos_ids = torch.arange(cur_len, device=device).unsqueeze(0)   # [1, seq]
        cos_full, sin_full = rotary(seq_ids, pos_ids)                  # [1, seq, D]

        # Embed full sequence (all tokens so far for contextual hidden states)
        t_fwd0 = _time.perf_counter()
        hidden = m.model.embed_tokens(seq_ids)    # [1, seq, hidden]

        for layer_idx, layer in enumerate(m.model.layers):
            attn = layer.self_attn

            # ── Q: project from last token only ─────────────────────────────
            q_flat = attn.q_proj(hidden[:, -1:, :])                  # [1, 1, 2048]
            q_reshaped = q_flat.view(1, num_q_heads, 1, head_dim)   # [1, 16, 1, 128]
            q_normed = attn.q_norm(q_reshaped)                        # [1, 16, 1, 128]

            # Apply trigonometric RoPE at last position to Q
            cos_q = cos_full[:, -1:, :]    # [1, 1, D]
            sin_q = sin_full[:, -1:, :]    # [1, 1, D]
            q_rot, _ = apply_rotary_pos_emb_xpu(
                q_normed, q_normed, cos_q, sin_q
            )  # q_rot: [1, 16, 1, 128]

            # ── K/V: project new token only (last position) ──────────────────
            k_flat = attn.k_proj(hidden[:, -1:, :])   # [1, 1, 1024]
            v_flat = attn.v_proj(hidden[:, -1:, :])   # [1, 1, 1024]
            k_reshaped = k_flat.view(1, num_kv_heads, 1, head_dim)   # [1, 8, 1, 128]
            v_reshaped = v_flat.view(1, num_kv_heads, 1, head_dim)   # [1, 8, 1, 128]
            k_normed = attn.k_norm(k_reshaped)                        # [1, 8, 1, 128]
            v_new = v_reshaped.squeeze(2).float()                     # [1, 8, 128]

            # Apply RoPE to new K at current position
            cos_k = cos_full[:, -1:, :]
            sin_k = sin_full[:, -1:, :]
            _, k_new_rot = apply_rotary_pos_emb_xpu(
                k_normed, k_normed, cos_k, sin_k
            )   # k_new_rot: [1, 8, 1, 128]

            # ── Asymmetric attention per KV head using turboquant_attention ─
            t_attn0 = _time.perf_counter()
            attn_out_h = []
            for kv_head in range(num_kv_heads):
                q_group_start = kv_head * kv_groups
                q_group_end = (kv_head + 1) * kv_groups
                q_h = q_rot[:, q_group_start:q_group_end, :, :]   # [1, G, 1, D]

                # Build [batch=1][head=1] structure for turboquant_attention
                k_enc, v_enc = tqc_cache[layer_idx][kv_head]
                ck = [[k_enc]]   # [batch=1][head=1]
                cv = [[v_enc]]

                # GQA: expand compressed_k/v to match Q head count
                B_q, Q_heads, _, D_q = q_h.shape
                KV_heads = len(ck[0])
                G = Q_heads // KV_heads  # group size = 2

                # expand: repeat each KV head G times
                ck_expanded = [[ck[b][h // G] for h in range(Q_heads)] for b in range(B_q)]
                cv_expanded = [[cv[b][h // G] for h in range(Q_heads)] for b in range(B_q)]

                out = turboquant_attention(
                    q_h.float(), ck_expanded, cv_expanded, tqc_config
                )   # [1, G, 1, D]
                attn_out_h.append(out)
            t_attn1 = _time.perf_counter()

            # ── Encode new token K/V and append to TurboQuantCache ──────────
            t_enc0 = _time.perf_counter()
            for h in range(num_kv_heads):
                rot_h = tqc_config.make_rotation(layer_idx, h)
                S_h = tqc_config.make_qjl_matrix(layer_idx, h)
                k_enc = turboquant_encode_internal(
                    k_new_rot[0, h, 0, :],    # [D]
                    tqc_config.codebook, rot_h, S_h, mixed=None,
                )
                v_enc = turboquant_encode_internal(
                    v_new[0, h, :],
                    tqc_config.codebook, rot_h, S_h, mixed=None,
                )
                # Append as new single-token TQC to existing per-head cache
                # tqc_cache[layer_idx][h] is (k_all, v_all) — we need to extend
                k_existing, v_existing = tqc_cache[layer_idx][h]
                # Decode both to FP16, concatenate, re-encode
                k_dec = turboquant_decode_single(k_existing)   # [seq_k, D]
                v_dec = turboquant_decode_single(v_existing)   # [seq_k, D]
                k_cat = torch.cat([k_dec, k_new_rot[0, h, 0, :].unsqueeze(0)], dim=0)  # [seq+1, D]
                v_cat = torch.cat([v_dec, v_new[0, h, :].unsqueeze(0)], dim=0)
                # Re-encode as batch
                k_new_all = turboquant_encode_internal(
                    k_cat, tqc_config.codebook, rot_h, S_h, mixed=None,
                )
                v_new_all = turboquant_encode_internal(
                    v_cat, tqc_config.codebook, rot_h, S_h, mixed=None,
                )
                tqc_cache[layer_idx][h] = (k_new_all, v_new_all)
            t_enc1 = _time.perf_counter()

            attn_out = torch.cat(attn_out_h, dim=1)   # [1, 16, 1, 128]

            # ── Output projection + residual + FFN ───────────────────────────
            attn_out = attn_out.reshape(1, num_q_heads * head_dim)   # [1, 2048]
            attn_out = attn.o_proj(attn_out.to(attn.o_proj.weight.dtype))  # [1, hidden]

            residual = hidden[:, -1:, :] + attn_out.unsqueeze(1)     # [1, 1, hidden]
            hidden = layer.input_layernorm(residual)
            hidden = layer.post_attention_layernorm(hidden + layer.mlp(hidden))
        t_fwd1 = _time.perf_counter()

        # Sample next token
        logits = m.lm_head(hidden[:, -1, :])    # [1, vocab]
        next_tok = logits.argmax(dim=-1, keepdim=True)
        seq_ids = torch.cat([seq_ids, next_tok], dim=-1)

        t_step1 = _time.perf_counter()
        total = t_step1 - t_step0
        layers_fwd = t_fwd1 - t_fwd0
        layers_attn = t_attn1 - t_attn0
        layers_enc = t_enc1 - t_enc0
        print(f"  step {step}: total={total:.2f}s  fwd={layers_fwd:.2f}s  "
              f"attn={layers_attn:.2f}s  enc={layers_enc:.2f}s")
        sys.stdout.flush()

        # Check for EOS
        if next_tok.item() == tok.eos_token_id:
            print(f"    EOS at step {step}"); sys.stdout.flush()
            break

torch.xpu.synchronize()
t_decode = time.time() - t0
vram_tqc = torch.xpu.max_memory_allocated() / 1e9
tqc_text = tok.decode(seq_ids[0])
print(f"\n  Done. Decode={t_decode:.1f}s ({t_decode/DECODE_STEPS*1000:.0f}ms/token)  "
      f"Peak={vram_tqc:.3f}GB\n")

# ── results ───────────────────────────────────────────────────────────────────
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
print(f"\n  FP16 KV: {kv_bytes_total/1e6:.1f}MB  |  VRAM saved after del: "
      f"{(vram_prefill-vram_after_del)*1024:.0f}MB")
print(f"\n  --- BASELINE OUTPUT ---")
print(f"  {baseline_text[:500]}")
print(f"\n  --- TURBOQUANT OUTPUT ---")
print(f"  {tqc_text[:500]}")
print("=" * 70)
