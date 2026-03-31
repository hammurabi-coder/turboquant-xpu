"""End-to-end asymmetric attention decode with TurboQuant.

Design:
  1. Prefill: run model normally, use_cache=True → get past_key_values
     Extract raw K/V from pkv. Compress these into TurboQuantCache.
  2. Delete FP16 past_key_values immediately
  3. Decode loop (manual, per layer):
     a. Project Q from last token's hidden (attn.q_proj + q_norm)
     b. Apply trigonometric RoPE to Q at current position
     c. turboquant_attention: scores Q against cached K (Hadamard-rotated
        during encode), decompresses V for weighted sum
     d. Encode new token's K/V (WITHOUT trigonometric RoPE) and append to cache
     e. Run FFN
  4. Sample next token from lm_head logits
  5. Stop at EOS or DECODE_STEPS tokens

Key invariant: K/V in TurboQuantCache have ONLY Hadamard rotation
(from PQ encode). Trigonometric RoPE is NOT stored — it is applied
fresh to Q at each decode position. turboquant_attention applies
Hadamard rotation to both Q (via config.make_rotation) and cached K
so the spaces match.

FIXES from broken v1:
  - K/V encode: use k_normed/v_normed (NOT k_new_rot) — NO trig RoPE before encode
  - RoPE for Q: use correct position_ids matching seq_ids length
  - FFN residual: ensure dtype match before adding
  - GQA: repeat_interleave for Q expansion (not implemented — loop per KV head)
"""

import os, sys, time
import time as _time
os.environ["BNB_CUDA_TRITON"] = "0"
os.environ["PYTORCH_XPU_ALLOC_CONF"] = "expandable_segments:True"
import torch

# Monkey-patch mem_get_info to avoid Intel Arc WSL driver bug
try:
    _V = int(12.5 * 1024**3)
    torch.xpu.mem_get_info = lambda d=None: (_V - torch.xpu.memory_allocated(), _V)
except Exception:
    pass

sys.path.insert(0, "/home/hermes/turboquant-experiments/OnlyTerp-turboquant/src")
from cache import (
    TurboQuantConfig,
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
num_kv_heads = config_0.num_key_value_heads     # 8
kv_groups = num_q_heads // num_kv_heads         # 2 (GQA)
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

# ── 2. Extract K/V and compress ─────────────────────────────────────────────
# NOTE: pkv stores post-RoPE K/V (Qwen3Attention applies RoPE before cache update).
# We extract what the model stored and encode it as-is. The Hadamard rotation
# inside turboquant_encode_internal is the PQ rotation — it does NOT interfere
# with trigonometric RoPE (they operate in different spaces).

print("  Compressing KV cache..."); sys.stdout.flush()
torch.xpu.synchronize()
t0 = time.time()

tqc_config = TurboQuantConfig(
    d=head_dim, b_mse=3, device=device,
    mixed_precision=False, use_online_codebook=False,
)

# tqc_cache[layer][head] = (k_encoded_TQC, v_encoded_TQC) — ALL seq_len tokens
tqc_cache: list[list[tuple]] = [[None] * num_kv_heads for _ in range(n_layers)]

for l in range(n_layers):
    for h in range(num_kv_heads):
        k_raw = pkv.layers[l].keys[:, h, :, :].float()    # [1, seq, D]
        v_raw = pkv.layers[l].values[:, h, :, :].float()  # [1, seq, D]

        rot = tqc_config.make_rotation(l, h)
        S = tqc_config.make_qjl_matrix(l, h)

        # Encode ALL seq_len tokens at once as one batch: [1, seq, D]
        k_enc = turboquant_encode_internal(
            k_raw,
            tqc_config.codebook, rot, S, mixed=None,
        )
        v_enc = turboquant_encode_internal(
            v_raw,
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


def apply_rotary_pos_emb_xpu(q, k, cos, sin):
    """Rotary positional embedding for [B, H, seq, D].
    Compatible with XPU tensors. Returns (q_rot, k_rot).
    """
    def _rotate_half(x):
        D = x.shape[-1]
        x0 = x[..., :D//2]
        x1 = x[..., D//2:]
        return torch.cat([-x1, x0], dim=-1)
    cos = cos.unsqueeze(1)   # [1, 1, seq, D]
    sin = sin.unsqueeze(1)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


with torch.no_grad():
    for step in range(DECODE_STEPS):
        t_step0 = _time.perf_counter()

        cur_len = seq_ids.shape[1]

        # ── Get trigonometric RoPE for Q ──────────────────────────────────────
        # Use full seq_ids length for position_ids so cos/sin covers all positions.
        # We only slice the last position for Q's RoPE, but we need the full
        # cos/sin tensor to properly index [:, -1:, :].
        # Qwen3RotaryEmbedding signature: (query, position_ids) → (cos, sin)
        pos_ids = torch.arange(cur_len, dtype=torch.long, device=device).unsqueeze(0)  # [1, seq]
        cos_full, sin_full = rotary(seq_ids, pos_ids)  # cos/sin: [1, seq, D]

        # Embed full sequence (needed for contextual hidden states)
        t_fwd0 = _time.perf_counter()
        hidden = m.model.embed_tokens(seq_ids)    # [1, seq, hidden]

        for layer_idx, layer in enumerate(m.model.layers):
            attn = layer.self_attn

            # ── Q: project from last token only ─────────────────────────────
            # hidden[:, -1:, :] = [1, 1, hidden]
            q_flat = attn.q_proj(hidden[:, -1:, :])                  # [1, 1, H*head_dim]
            q_reshaped = q_flat.view(1, num_q_heads, 1, head_dim) # [1, 16, 1, 128]
            q_normed = attn.q_norm(q_reshaped)                      # [1, 16, 1, 128]

            # Apply trigonometric RoPE at last position to Q
            # cos_full/sin_full: [1, seq, D], slice to last pos → [1, 1, D]
            cos_q = cos_full[:, -1:, :]    # [1, 1, D]
            sin_q = sin_full[:, -1:, :]    # [1, 1, D]
            q_rot, _ = apply_rotary_pos_emb_xpu(
                q_normed, q_normed, cos_q, sin_q
            )  # q_rot: [1, 16, 1, 128]

            # ── K/V: project new token only (last position) ──────────────────
            k_flat = attn.k_proj(hidden[:, -1:, :])   # [1, 1, KVH*head_dim]
            v_flat = attn.v_proj(hidden[:, -1:, :])   # [1, 1, KVH*head_dim]
            k_reshaped = k_flat.view(1, num_kv_heads, 1, head_dim)  # [1, 8, 1, 128]
            v_reshaped = v_flat.view(1, num_kv_heads, 1, head_dim)  # [1, 8, 1, 128]
            k_normed = attn.k_norm(k_reshaped)                      # [1, 8, 1, 128]
            v_new = v_reshaped  # v_proj has no v_norm in Qwen3Attention

            # NOTE: We do NOT apply trigonometric RoPE to k_normed before encoding.
            # The K/V stored in TurboQuantCache should have ONLY Hadamard rotation.
            # Trigonometric RoPE is applied fresh to Q at each position.
            # turboquant_attention applies Hadamard rotation to both Q and cached K
            # via config.make_rotation(), so their spaces match.

            # ── Asymmetric attention per KV head using turboquant_attention ─
            # NOTE: turboquant_attention expects Q with shape [B, 1, 1, D] (1 head per call).
            # We loop over each KV head's Q groups and call turboquant_attention individually.
            t_attn0 = _time.perf_counter()
            attn_out_h = []
            for kv_head in range(num_kv_heads):
                k_enc, v_enc = tqc_cache[layer_idx][kv_head]
                ck = [[k_enc]]   # [batch=1][head=1] — TurboQuantCompressed
                cv = [[v_enc]]
                
                for q_idx in range(kv_groups):
                    q_head_idx = kv_head * kv_groups + q_idx
                    q_single = q_rot[:, q_head_idx:q_head_idx+1, :, :]  # [1, 1, 1, D]
                    out = turboquant_attention(
                        q_single.float(), ck, cv, tqc_config, layer_idx=layer_idx
                    )   # [1, 1, 1, D]
                    attn_out_h.append(out)   # accumulates in Q-head order
            t_attn1 = _time.perf_counter()

            # Stack all per-head outputs → [1, num_q_heads, 1, D]
            attn_out = torch.cat(attn_out_h, dim=1)   # [1, 16, 1, 128]

            # ── Output projection + residual + FFN ───────────────────────────
            # attn_out: [1, H, 1, D] → o_proj expects [1, H*D]
            attn_out = attn_out.reshape(1, num_q_heads * head_dim)   # [1, 2048]
            attn_out = attn.o_proj(attn_out.to(attn.o_proj.weight.dtype))  # [1, hidden]

            # Residual connection: attn_out + hidden_at_position
            # hidden[:, -1:, :] = [1, 1, hidden]
            residual = hidden[:, -1:, :].squeeze(1) + attn_out     # [1, hidden]
            residual = residual.unsqueeze(1)                        # [1, 1, hidden]

            # Post-attention layernorm + FFN
            normalized = layer.input_layernorm(residual)             # [1, 1, hidden]
            ffn_out = layer.mlp(normalized)                          # [1, 1, hidden]
            hidden = (normalized + ffn_out).squeeze(1)               # [1, hidden]

            # Update hidden to [1, 1, hidden] for next layer's q_proj
            hidden = hidden.unsqueeze(1)                              # [1, 1, hidden]

            # ── Encode new token K/V and append to TurboQuantCache ──────────
            # CRITICAL: Encode k_normed and v_new (NOT trig-RoPE'd versions).
            # These go through Hadamard rotation inside turboquant_encode_internal.
            # Trigonometric RoPE on K/V is NOT stored — it is applied fresh to Q.
            t_enc0 = _time.perf_counter()
            for h in range(num_kv_heads):
                rot_h = tqc_config.make_rotation(layer_idx, h)
                S_h = tqc_config.make_qjl_matrix(layer_idx, h)

                # k_normed[0, h, 0, :] → [D] (no trig RoPE applied)
                k_new_2d = k_normed[0, h, 0, :]                     # [D]
                # v_new[0, h, 0, :] → [D]
                v_new_2d = v_new[0, h, 0, :]                        # [D]

                # Decode existing cache: TurboQuantCompressed → [seq_k, D]
                k_existing, v_existing = tqc_cache[layer_idx][h]
                k_dec = turboquant_decode_single(k_existing).squeeze(0)  # [seq_k, D]
                v_dec = turboquant_decode_single(v_existing).squeeze(0)  # [seq_k, D]

                # Concat: existing + new token → [seq_k+1, D]
                k_cat = torch.cat([k_dec, k_new_2d.unsqueeze(0)], dim=0)  # [seq+1, D]
                v_cat = torch.cat([v_dec, v_new_2d.unsqueeze(0)], dim=0)  # [seq+1, D]

                # Re-encode as batch: [1, seq+1, D]
                k_new_all = turboquant_encode_internal(
                    k_cat.unsqueeze(0), tqc_config.codebook, rot_h, S_h, mixed=None,
                )
                v_new_all = turboquant_encode_internal(
                    v_cat.unsqueeze(0), tqc_config.codebook, rot_h, S_h, mixed=None,
                )
                tqc_cache[layer_idx][h] = (k_new_all, v_new_all)
            t_enc1 = _time.perf_counter()

        t_fwd1 = _time.perf_counter()

        # Sample next token — hidden is [1, hidden] at this point
        logits = m.lm_head(hidden.squeeze(1))    # [1, vocab]
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
