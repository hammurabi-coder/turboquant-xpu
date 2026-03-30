"""Context length sweep: show where KV cache dominates and compression matters.

Tests both:
  1. Baseline: standard prefill + decode
  2. TurboQuant compress→decompress (for comparison)

The compress→decompress path is NOT the final answer — it's a step toward
asymmetric attention where K stays compressed during scoring.
"""
import os, sys, time
os.environ["BNB_CUDA_TRITON"] = "0"
os.environ["PYTORCH_XPU_ALLOC_CONF"] = "expandable_segments:True"
import torch
_V = int(12.5 * 1024**3)
torch.xpu.mem_get_info = lambda d=None: (_V - torch.xpu.memory_allocated(), _V)

sys.path.insert(0, "/home/hermes/turboquant-experiments/OnlyTerp-turboquant/src")
from cache import TurboQuantConfig, turboquant_encode_internal, polarquant_decode
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

device = torch.device("xpu")

def kv_mb(pkv):
    total = 0
    for layer in pkv.layers:
        total += layer.keys.element_size() * layer.keys.numel()
        total += layer.values.element_size() * layer.values.numel()
    return total / 1e6

# ── load once ────────────────────────────────────────────────────────────────
print("Loading tokenizer..."); sys.stdout.flush()
tok = AutoTokenizer.from_pretrained("/home/hermes/.cache/huggingface/modules",
    trust_remote_code=True, padding_side="left")
if tok.pad_token is None: tok.pad_token = tok.eos_token

MODEL = "Qwen/Qwen3-0.6B"  # smaller model — faster iteration
#MODEL = "/home/hermes/.cache/huggingface/modules/models/Qwen3-4B-4bit"

print(f"Loading {MODEL}..."); sys.stdout.flush()
m = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True,
)
m = m.to(device); m.eval(); torch.xpu.synchronize()
model_vram = torch.xpu.memory_allocated() / 1e9
print(f"Model ready. VRAM={model_vram:.2f}GB\n"); sys.stdout.flush()

# ── helpers ─────────────────────────────────────────────────────────────────
def qjl_seed(l, h):
    return ((l * 1000003) ^ (h * 999979) ^ 0x5A5A5A5A) & 0xFFFFFFFF

def compress_kv(pkv, config, n_layers, n_heads, head_dim):
    """Compress all KV cache tensors. Returns (compressed_list, rot/S precomputed)."""
    rotations = [[config.make_rotation(l, h) for h in range(n_heads)] for l in range(n_layers)]
    S_matrices = [[config.make_qjl_matrix(l, h) for h in range(n_heads)] for l in range(n_layers)]
    compressed = []
    for l in range(n_layers):
        k_parts, v_parts = [], []
        for h in range(n_heads):
            k_f = pkv.layers[l].keys[:, h:h+1, :, :].reshape(-1, head_dim).float()
            v_f = pkv.layers[l].values[:, h:h+1, :, :].reshape(-1, head_dim).float()
            kc = turboquant_encode_internal(k_f, config.codebook, rotations[l][h],
                                            S_matrices[l][h], mixed=None)
            vc = turboquant_encode_internal(v_f, config.codebook, rotations[l][h],
                                            S_matrices[l][h], mixed=None)
            k_parts.append(kc); v_parts.append(vc)
        compressed.append((k_parts, v_parts))
    return compressed

def decompress_kv(compressed, n_layers, n_heads, seq_len, head_dim, dtype):
    """Rebuild DynamicCache from compressed parts."""
    rebuilt = []
    for l in range(n_layers):
        k_parts, v_parts = [], []
        for h in range(n_heads):
            kc, vc = compressed[l][0][h], compressed[l][1][h]
            kr = polarquant_decode(kc.pq)[..., :head_dim].reshape(1, 1, seq_len, head_dim).to(dtype).contiguous()
            vr = polarquant_decode(vc.pq)[..., :head_dim].reshape(1, 1, seq_len, head_dim).to(dtype).contiguous()
            k_parts.append(kr); v_parts.append(vr)
        rebuilt.append((torch.cat(k_parts, dim=1), torch.cat(v_parts, dim=1)))
    return rebuilt

def tq_footprint(compressed, n_layers, n_heads):
    """Actual byte size of the compressed representation (PQ + QJL, no S)."""
    total = 0
    for l in range(n_layers):
        for h in range(n_heads):
            kc = compressed[l][0][h]
            vc = compressed[l][1][h]
            # PQ: indices + norm
            total += kc.pq.indices.element_size() * kc.pq.indices.numel()
            total += kc.pq.norm.element_size() * kc.pq.norm.numel()
            total += vc.pq.indices.element_size() * vc.pq.indices.numel()
            total += vc.pq.norm.element_size() * vc.pq.norm.numel()
            # QJL: signs + r_norm
            total += kc.qjl.signs.element_size() * kc.qjl.signs.numel()
            total += kc.qjl.r_norm.element_size() * kc.qjl.r_norm.numel()
            total += vc.qjl.signs.element_size() * vc.qjl.signs.numel()
            total += vc.qjl.r_norm.element_size() * vc.qjl.r_norm.numel()
    return total

# ── context sweep ───────────────────────────────────────────────────────────
# Repeatable text blocks for generating variable-length contexts
base_text = (
    "In the northern reaches of the Ember Valley, the last remnants of the ancient "
    "Clockwork Empire lay buried. Scholars had catalogued over four thousand distinct "
    "automaton designs, ranging from the towering War Golems to the delicate Songbirds."
)

context_lengths = [128, 256, 512, 1024, 2048]
b_mse = 3

print(f"{'Ctx':>6} | {'Baseline':>10} | {'TQ-kvMB':>8} | {'TQ-ActualMB':>10} | {'Ratio':>6} | {'Baseline':>8} | {'TurboQ':>8}")
print(f"{'':6} | {'VRAM GB':>10} | {'':8} | {'':>10} | {'':>6} | {'Time s':>8} | {'Time s':>8}")
print("-" * 75)

for ctx_len in context_lengths:
    # Build prompt by repeating base text
    text = (base_text + " ") * (ctx_len // 40 + 1)
    tokens = tok(text, return_tensors="pt", truncation=True, max_length=ctx_len + 20)["input_ids"]
    if tokens.shape[1] > ctx_len:
        tokens = tokens[:, :ctx_len]
    prompt = tok.decode(tokens[0])
    inp = {k: v.to(device) for k, v in tok(prompt, return_tensors="pt", truncation=True, max_length=ctx_len + 20).items()}
    if inp["input_ids"].shape[1] > ctx_len:
        inp["input_ids"] = inp["input_ids"][:, :ctx_len]

    actual_len = inp["input_ids"].shape[1]
    if actual_len < 64:  # skip tiny contexts
        continue

    # --- BASELINE ---
    torch.xpu.reset_peak_memory_stats()
    torch.xpu.synchronize(); t0 = time.time()
    with torch.no_grad():
        gen_out = m.generate(
            inp["input_ids"], max_new_tokens=10, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    torch.xpu.synchronize(); t_base = time.time() - t0
    vram_base_peak = torch.xpu.max_memory_allocated() / 1e9

    # --- TURBOQUANT path ---
    torch.xpu.reset_peak_memory_stats()
    torch.xpu.synchronize(); t0 = time.time()
    with torch.no_grad():
        prefill_out = m(input_ids=inp["input_ids"], use_cache=True)
    torch.xpu.synchronize(); t_prefill = time.time() - t0
    pkv = prefill_out.past_key_values
    n_layers = len(pkv.layers)
    n_heads = pkv.layers[0].keys.shape[1]
    head_dim = pkv.layers[0].keys.shape[3]
    seq_len = pkv.layers[0].keys.shape[2]
    dtype = pkv.layers[0].keys.dtype
    baseline_kv_mb = kv_mb(pkv)

    # compress
    t0 = time.time()
    config = TurboQuantConfig(d=head_dim, b_mse=b_mse, device=device,
        mixed_precision=False, use_online_codebook=False)
    compressed = compress_kv(pkv, config, n_layers, n_heads, head_dim)
    torch.xpu.synchronize(); t_compress = time.time() - t0
    actual_tq_mb = tq_footprint(compressed, n_layers, n_heads) / 1e6

    # rebuild and decode
    del pkv; torch.xpu.synchronize()
    vram_after_del = torch.xpu.memory_allocated() / 1e9
    rebuilt = decompress_kv(compressed, n_layers, n_heads, seq_len, head_dim, dtype)
    del compressed; torch.xpu.synchronize()
    cache = DynamicCache()
    for l in range(n_layers):
        cache.update(rebuilt[l][0], rebuilt[l][1], layer_idx=l)
    del rebuilt; torch.xpu.synchronize()
    vram_rebuilt = torch.xpu.memory_allocated() / 1e9

    # decode 10 tokens
    logits = prefill_out.logits[:, -1, :]
    next_tok = logits.argmax(dim=-1, keepdim=True)
    gen_ids = torch.cat([inp["input_ids"], next_tok], dim=-1)
    torch.xpu.synchronize(); t0 = time.time()
    for i in range(10):
        with torch.no_grad():
            o = m(input_ids=next_tok, past_key_values=cache, use_cache=True)
        next_tok = o.logits[:, -1:, :].argmax(dim=-1)
        gen_ids = torch.cat([gen_ids, next_tok], dim=-1)
    torch.xpu.synchronize(); t_decode = time.time() - t0
    vram_tq_peak = torch.xpu.max_memory_allocated() / 1e9

    ratio = baseline_kv_mb / (actual_tq_mb + 1e-9)
    tq_total = t_prefill + t_compress + t_decode
    saved = vram_base_peak - vram_tq_peak
    print(f"{actual_len:>6} | {vram_base_peak:>10.3f} | {baseline_kv_mb:>8.1f} | {actual_tq_mb:>10.1f} | {ratio:>6.2f}x | {t_base:>8.1f} | {tq_total:>8.1f}")

print()
print("Key insight: TQ-ActualMB = PQ(indices+norm) + QJL(signs+r_norm) per head/layer")
print(f"  PQ overhead (uint8): {b_mse}-bit indices, float16 norms")
print(f"  QJL overhead (uint8): 1-bit signs, float32 r_norm")
print(f"  Both stored as uint8/float32 — overhead > FP16 savings at low bits")
print(f"  Asymmetric attention needed: keep K compressed during scoring")