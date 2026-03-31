"""
HF generate() hook for TurboQuant KV cache compression — Qwen3-4B test.
"""

# Patches must be at the very top before any model loading
_VRAM_TOTAL = 12 * 1024**3

import torch

# Set DEBUG_RAW_KV=True to bypass TQ compression and use raw K/V
DEBUG_RAW_KV = True

def _mem_get_info_patch(d=None):
    return (_VRAM_TOTAL - torch.xpu.memory_allocated(), _VRAM_TOTAL)

torch.xpu.mem_get_info = _mem_get_info_patch

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import sys
import time
from typing import Optional

sys.path.insert(0, "/home/hermes/turboquant-experiments/OnlyTerp-turboquant/src")

from cache import (
    TurboQuantConfig,
    TurboQuantCache,
    TurboQuantCompressed,
    turboquant_encode_internal,
    turboquant_decode_single,
    turboquant_attention,
    polarquant_decode,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("xpu")

# ---------------------------------------------------------------------------
# Core hook logic
# ---------------------------------------------------------------------------

def install_turboquant_hook(model, tqc_config, pkv=None):
    attn_0 = model.model.layers[0].self_attn

    num_q_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = attn_0.head_dim
    kv_groups = num_q_heads // num_kv_heads
    n_layers = len(model.model.layers)

    print(f"  Hook: {n_layers}L x {num_q_heads}Q / {num_kv_heads}KV heads, "
          f"head_dim={head_dim}, groups={kv_groups}")

    if pkv is None:
        print("  No pkv provided -- starting with empty tqc_cache (decode-only mode)")
        tqc_cache = _make_empty_tqc_cache(n_layers, num_kv_heads)
    else:
        print(f"  Compressing prefill pkv ({pkv.layers[0].keys.shape[2]} tokens)...")
        t0 = time.time()
        tqc_cache = _compress_pkv_to_tqc(pkv, tqc_config, num_kv_heads, n_layers)
        print(f"  Compress done in {time.time()-t0:.1f}s")

        del pkv
        torch.xpu.empty_cache()

    originals = []
    model._tqc_originals = originals

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        originals.append(attn.forward)

        def make_hook(layer_idx=layer_idx, _attn=attn):
            @torch.no_grad()
            def turboquant_forward(
                hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values=None,
                **kwargs,
            ):
                seq_len = hidden_states.shape[1]
                is_decode = (past_key_values is not None) and (seq_len == 1)
                if not is_decode:
                    return originals[layer_idx](
                        hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        **kwargs,
                    )

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, head_dim)

                q_proj_out = _attn.q_proj(hidden_states)
                query_states = _attn.q_norm(q_proj_out.view(hidden_shape)).transpose(1, 2)

                key_states = _attn.k_norm(_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                value_states = _attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings

                query_states, key_states = _apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )

                k_new = key_states[..., -1:, :]
                v_new = value_states[..., -1:, :]

                for head_idx in range(num_kv_heads):
                    k_vec = k_new[0, head_idx, 0, :]
                    v_vec = v_new[0, head_idx, 0, :]

                    if DEBUG_RAW_KV:
                        k_c = k_vec.float()
                        v_c = v_vec.float()
                    else:
                        rotation = tqc_config.make_rotation(layer_idx, head_idx)
                        S = tqc_config.make_qjl_matrix(layer_idx, head_idx)
                        k_c = turboquant_encode_internal(k_vec, tqc_config.codebook,
                                                         rotation, S, mixed=None)
                        v_c = turboquant_encode_internal(v_vec, tqc_config.codebook,
                                                         rotation, S, mixed=None)
                    tqc_cache[layer_idx][head_idx].append((k_c, v_c))

                compressed_k = []
                compressed_v = []

                if DEBUG_RAW_KV:
                    for head_idx in range(num_kv_heads):
                        seq_len_cached = len(tqc_cache[layer_idx][head_idx])
                        k_raw = torch.stack([
                            tqc_cache[layer_idx][head_idx][t][0].float()
                            for t in range(seq_len_cached)
                        ])
                        v_raw = torch.stack([
                            tqc_cache[layer_idx][head_idx][t][1].float()
                            for t in range(seq_len_cached)
                        ])
                        compressed_k.append(k_raw)
                        compressed_v.append(v_raw)
                else:
                    for head_idx in range(num_kv_heads):
                        seq_len_cached = len(tqc_cache[layer_idx][head_idx])
                        cached_k = tqc_cache[layer_idx][head_idx]

                        first_kpq = cached_k[0][0].pq
                        k_norm_batch = torch.stack([
                            cached_k[t][0].pq.norm.squeeze(0)
                            for t in range(seq_len_cached)
                        ])
                        k_indices_batch = torch.cat([
                            cached_k[t][0].pq.indices for t in range(seq_len_cached)
                        ], dim=0)
                        batched_pq_k = _batch_polarquant_compressed(
                            first_kpq, k_norm_batch, k_indices_batch
                        )
                        k_c_batch = TurboQuantCompressed(pq=batched_pq_k, qjl=cached_k[0][0].qjl)

                        first_vpq = cached_k[0][1].pq
                        v_norm_batch = torch.stack([
                            cached_k[t][1].pq.norm.squeeze(0)
                            for t in range(seq_len_cached)
                        ])
                        v_indices_batch = torch.cat([
                            cached_k[t][1].pq.indices
                            for t in range(seq_len_cached)
                        ], dim=0)
                        batched_pq_v = _batch_polarquant_compressed(
                            first_vpq, v_norm_batch, v_indices_batch
                        )
                        v_c_batch = TurboQuantCompressed(pq=batched_pq_v, qjl=cached_k[0][1].qjl)

                        compressed_k.append(k_c_batch)
                        compressed_v.append(v_c_batch)

                B = query_states.shape[0]
                q_expanded_list = []
                ck_expanded_list = []
                cv_expanded_list = []

                for b in range(B):
                    q_head_list = []
                    ck_head_list = []
                    cv_head_list = []
                    for h in range(num_q_heads):
                        kv_head = h // kv_groups
                        q_head_list.append(query_states[b, h, 0, :])
                        ck_head_list.append(compressed_k[kv_head])
                        cv_head_list.append(compressed_v[kv_head])
                    q_expanded_list.append(torch.stack(q_head_list))
                    ck_expanded_list.append(ck_head_list)
                    cv_expanded_list.append(cv_head_list)

                q_final = torch.stack(q_expanded_list).unsqueeze(2)

                k_decoded_list = []
                for h in range(num_q_heads):
                    kv_head = h // kv_groups
                    ck = compressed_k[kv_head]
                    if DEBUG_RAW_KV:
                        k_full = ck
                    else:
                        k_full = turboquant_decode_single(ck)
                    k_decoded_list.append(k_full)

                v_decoded_list = []
                for h in range(num_q_heads):
                    kv_head = h // kv_groups
                    cv = compressed_v[kv_head]
                    if DEBUG_RAW_KV:
                        v_full = cv
                    else:
                        v_full = turboquant_decode_single(cv)
                    v_decoded_list.append(v_full)

                all_outputs = []
                for b in range(B):
                    head_outputs = []
                    for h in range(num_q_heads):
                        k_h = k_decoded_list[h]
                        v_h = v_decoded_list[h]
                        q_h = q_final[b, h, 0, :]

                        scale = head_dim ** -0.5
                        scores = (k_h.float() @ q_h.float()) * scale
                        weights = torch.softmax(scores, dim=0)

                        out = (weights.unsqueeze(1) * v_h.float()).sum(dim=0)
                        head_outputs.append(out)
                    all_outputs.append(torch.stack(head_outputs))
                attn_out = torch.stack(all_outputs).unsqueeze(2)

                attn_out = attn_out.transpose(1, 2).contiguous()
                attn_out = attn_out.view(*input_shape, -1)
                attn_out = attn_out.to(hidden_states.dtype)
                attn_out = _attn.o_proj(attn_out)

                import sys
                print(f"[DIAG layer={layer_idx}] attn_out norm={attn_out.norm().item():.4f}", file=sys.stderr)

                return (attn_out, None)

            return turboquant_forward

        attn.forward = make_hook()

    print(f"  Hooks installed on {n_layers} layers")
    return tqc_cache


def _batch_polarquant_compressed(first_pq, norm_batch, indices_batch):
    from cache import PolarQuantCompressed
    return PolarQuantCompressed(
        norm=norm_batch,
        indices=indices_batch,
        codebook=first_pq.codebook,
        rotation=first_pq.rotation,
        original_dim=first_pq.original_dim,
        regular_norm=None,
        outlier_norm=None,
        regular_indices=first_pq.regular_indices,
        outlier_indices=first_pq.outlier_indices,
        regular_quantized_indices=None,
        outlier_quantized_indices=None,
        codebook_regular=getattr(first_pq, 'codebook_regular', None),
        codebook_outlier=getattr(first_pq, 'codebook_outlier', None),
        rotation_regular=getattr(first_pq, 'rotation_regular', None),
        rotation_outlier=getattr(first_pq, 'rotation_outlier', None),
    )


def unhook(model):
    originals = getattr(model, '_tqc_originals', None)
    if originals is None:
        print("  Warning: no originals found -- nothing to unhook")
        return

    for layer_idx, layer in enumerate(model.model.layers):
        layer.self_attn.forward = originals[layer_idx]

    print(f"  Unhooked {len(originals)} layers -- original forward restored")
    model._tqc_originals = None


def _apply_rotary_pos_emb(q, k, cos, sin):
    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)

    def _rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    q_rope = _rotate_half(q) * sin_expanded + q * cos_expanded

    if k is not None:
        k_rope = _rotate_half(k) * sin_expanded + k * cos_expanded
    else:
        k_rope = None

    return q_rope, k_rope


def _compress_pkv_to_tqc(pkv, tqc_config, num_kv_heads, n_layers):
    tqc_cache = _make_empty_tqc_cache(n_layers, num_kv_heads)
    seq_len = pkv.layers[0].keys.shape[2]

    for layer_idx in range(n_layers):
        for head_idx in range(num_kv_heads):
            for tok_idx in range(seq_len):
                k_vec = pkv.layers[layer_idx].keys[0, head_idx, tok_idx, :]
                v_vec = pkv.layers[layer_idx].values[0, head_idx, tok_idx, :]

                if DEBUG_RAW_KV:
                    k_c = k_vec.float()
                    v_c = v_vec.float()
                else:
                    rotation = tqc_config.make_rotation(layer_idx, head_idx)
                    S = tqc_config.make_qjl_matrix(layer_idx, head_idx)
                    k_c = turboquant_encode_internal(k_vec, tqc_config.codebook,
                                                      rotation, S, mixed=None)
                    v_c = turboquant_encode_internal(v_vec, tqc_config.codebook,
                                                      rotation, S, mixed=None)
                tqc_cache[layer_idx][head_idx].append((k_c, v_c))

    return tqc_cache


def _make_empty_tqc_cache(n_layers, num_kv_heads):
    cache = []
    for _ in range(n_layers):
        layer_cache = []
        for _ in range(num_kv_heads):
            layer_cache.append([])
        cache.append(layer_cache)
    return cache


# ---------------------------------------------------------------------------
# Test / __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    os.environ["BNB_CUDA_TRITON"] = "0"
    os.environ["PYTORCH_XPU_ALLOC_CONF"] = "expandable_segments:True"

    MODEL = "Qwen/Qwen3-4B"
    PREFILL_TOKENS = 128
    DECODE_TOKENS = 20

    print("=" * 60)
    print("TurboQuant HF Hook Integration Test -- Qwen3-4B")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer..."); sys.stdout.flush()
    tok = AutoTokenizer.from_pretrained(
        "/home/hermes/.cache/huggingface/modules",
        trust_remote_code=True, padding_side="left"
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load model baseline
    print(f"Loading {MODEL} (4-bit bnb)..."); sys.stdout.flush()
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="fp4",
    )

    m = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="xpu", trust_remote_code=True,
        quantization_config=bnb_config, low_cpu_mem_usage=True,
    ).eval()

    vram_loaded = torch.xpu.max_memory_allocated() / 1e9
    print(f"  Model loaded. VRAM={vram_loaded:.2f}GB\n"); sys.stdout.flush()

    # Prompt
    text = "In the northern reaches of the Ember Valley"
    tokens = tok(text, return_tensors="pt", truncation=True,
                 max_length=PREFILL_TOKENS)["input_ids"]
    print(f"  Prompt: {tokens.shape[1]} tokens\n"); sys.stdout.flush()

    # BASELINE
    print("--- BASELINE: full generate() ---"); sys.stdout.flush()
    torch.xpu.reset_peak_memory_stats()
    torch.xpu.synchronize()
    t0 = time.time()

    with torch.no_grad():
        gen_out_baseline = m.generate(
            tokens.to(device),
            max_new_tokens=DECODE_TOKENS,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

    torch.xpu.synchronize()
    t_baseline = time.time() - t0
    vram_baseline = torch.xpu.max_memory_allocated() / 1e9
    baseline_text = tok.decode(gen_out_baseline[0])
    print(f"  Baseline: {t_baseline:.1f}s, VRAM={vram_baseline:.2f}GB")
    print(f"  Output: {baseline_text[:120]}...\n"); sys.stdout.flush()

    # TURBOQUANT PATH
    del m
    torch.xpu.empty_cache()

    print("--- TURBOQUANT HOOK ---"); sys.stdout.flush()
    print("Reloading model..."); sys.stdout.flush()

    m = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="xpu", trust_remote_code=True,
        quantization_config=bnb_config, low_cpu_mem_usage=True,
    ).eval()

    torch.xpu.reset_peak_memory_stats()
    torch.xpu.synchronize()

    # Prefill: get pkv
    print("  Prefill pass (capturing pkv)..."); sys.stdout.flush()
    t0 = time.time()
    with torch.no_grad():
        prefill_out = m(input_ids=tokens.to(device), use_cache=True)
    torch.xpu.synchronize()
    t_prefill = time.time() - t0
    vram_prefill = torch.xpu.max_memory_allocated() / 1e9
    print(f"  Prefill: {t_prefill:.1f}s, VRAM={vram_prefill:.2f}GB"); sys.stdout.flush()

    pkv = prefill_out.past_key_values

    # Install hook
    print("  Installing TurboQuant hook..."); sys.stdout.flush()
    attn_0 = m.model.layers[0].self_attn

    head_dim = attn_0.head_dim
    n_layers = len(m.model.layers)
    num_kv_heads = m.config.num_key_value_heads

    tqc_config = TurboQuantConfig(
        d=head_dim, b_mse=3, device=device,
        mixed_precision=False, use_online_codebook=False,
    )

    tqc_cache = install_turboquant_hook(m, tqc_config, pkv)

    vram_hooked = torch.xpu.memory_allocated() / 1e9
    print(f"  VRAM after hook install: {vram_hooked:.2f}GB\n"); sys.stdout.flush()

    last_token = tokens[:, -1:]
    eos_token_id = tok.eos_token_id

    print(f"  Decoding {DECODE_TOKENS} tokens with hook active..."); sys.stdout.flush()
    torch.xpu.reset_peak_memory_stats()
    torch.xpu.synchronize()
    t0 = time.time()

    with torch.no_grad():
        gen_tokens = last_token.clone().to(device)
        for _ in range(DECODE_TOKENS):
            logits = m.forward(gen_tokens.to(device), use_cache=True).logits
            next_tok = logits[0, -1].argmax()
            gen_tokens = torch.cat([gen_tokens, next_tok.unsqueeze(0).unsqueeze(0)], dim=1)
            if next_tok.item() == eos_token_id:
                break
        gen_out_hooked = gen_tokens

    torch.xpu.synchronize()
    t_hooked = time.time() - t0
    vram_hooked_peak = torch.xpu.max_memory_allocated() / 1e9
    hooked_text = tok.decode(gen_out_hooked[0])

    print(f"\n  TurboQuant hooked: {t_hooked:.1f}s, VRAM={vram_hooked_peak:.2f}GB")
    print(f"  Output: {hooked_text[:120]}..."); sys.stdout.flush()

    # Unhook and verify
    print("\n--- Unhook + verify ---"); sys.stdout.flush()
    unhook(m)

    with torch.no_grad():
        verify_out = m.generate(
            tokens.to(device),
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    verify_text = tok.decode(verify_out[0])
    print(f"  After unhook: {verify_text[:80]}...")
    print("  Model works -- unhook successful.")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model:     {MODEL}")
    print(f"Baseline:  {t_baseline:.1f}s, VRAM={vram_baseline:.2f}GB")
    print(f"Hooked:    {t_hooked:.1f}s, VRAM={vram_hooked_peak:.2f}GB")
    print(f"Prefill:   {t_prefill:.1f}s (separate, not in hooked total)")
    print(f"VRAM delta vs prefill: {vram_prefill - vram_hooked_peak:.2f}GB saved")
    print(f"\nBaseline output:\n  {baseline_text[:200]}")
    print(f"\nHooked output:\n  {hooked_text[:200]}")
