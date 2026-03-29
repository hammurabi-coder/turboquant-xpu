"""Quick end-to-end: prefill + TurboQuant compress + 10-token decode."""
import os, sys, time
os.environ["BNB_CUDA_TRITON"] = "0"
os.environ["PYTORCH_XPU_ALLOC_CONF"] = "expandable_segments:True"
import torch
_V = int(12.5 * 1024**3)
torch.xpu.mem_get_info = lambda d=None: (_V - torch.xpu.memory_allocated(), _V)

sys.path.insert(0, "/home/hermes/turboquant-experiments/OnlyTerp-turboquant/src")
from cache import TurboQuantConfig, N_OUTLIER_CHANNELS, polarquant_decode, turboquant_encode_internal
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache

device = torch.device("xpu")

# ── load ───────────────────────────────────────────────────────────────────────
print("Loading tokenizer..."); sys.stdout.flush()
tok = AutoTokenizer.from_pretrained("/home/hermes/.cache/huggingface/modules", trust_remote_code=True, padding_side="left")
if tok.pad_token is None: tok.pad_token = tok.eos_token

print("Loading model..."); sys.stdout.flush()
m = AutoModelForCausalLM.from_pretrained(
    "/home/hermes/.cache/huggingface/modules/models/Qwen3-4B-4bit",
    device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True,
)
m = m.to(device); m.eval(); torch.xpu.synchronize()
print(f"Model ready. VRAM={torch.xpu.memory_allocated()/1e9:.2f}GB\n"); sys.stdout.flush()

# ── prompt ────────────────────────────────────────────────────────────────────
text = ("In the northern reaches of the Ember Valley, the last remnants of the ancient "
        "Clockwork Empire lay buried. Scholars had catalogued over four thousand distinct "
        "automaton designs, ranging from towering War Golems to delicate Songbirds.")
tokens = tok.encode(text, add_special_tokens=False)
if len(tokens) > 600: tokens = tokens[:600]
prompt = tok.decode(tokens)
inp = {k: v.to(device) for k, v in tok(prompt, return_tensors="pt").items()}
print(f"Prompt: {len(tokens)} tokens\n"); sys.stdout.flush()

# ── baseline prefill + 10-token gen ───────────────────────────────────────────
print("=== BASELINE (prefill + 10-token gen) ==="); sys.stdout.flush()
torch.xpu.synchronize(); t0 = time.time()
with torch.no_grad():
    gen_out = m.generate(
        inp["input_ids"],
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
torch.xpu.synchronize(); t_base = time.time() - t0
vram_base = torch.xpu.max_memory_allocated() / 1e9
print(f"  Time:      {t_base:.1f}s")
print(f"  Peak VRAM: {vram_base:.2f}GB")
print(f"  Output:    {tok.decode(gen_out[0])[:200]}\n"); sys.stdout.flush()

# ── TurboQuant path: prefill → compress → decode ───────────────────────────────
print("=== TURBOQUANT (prefill → compress → 10-token decode) ==="); sys.stdout.flush()

# prefill
torch.xpu.synchronize(); t0 = time.time()
with torch.no_grad():
    prefill_out = m(input_ids=inp["input_ids"], use_cache=True)
torch.xpu.synchronize(); t_prefill = time.time() - t0
vram_prefill = torch.xpu.memory_allocated() / 1e9
pkv = prefill_out.past_key_values
n_layers, n_heads = len(pkv.layers), pkv.layers[0].keys.shape[1]
head_dim = pkv.layers[0].keys.shape[3]
seq_len = pkv.layers[0].keys.shape[2]
print(f"  Prefill: {t_prefill:.1f}s  VRAM={vram_prefill:.2f}GB")
print(f"  KV: {n_layers}L x {n_heads}H x seq={seq_len} x dim={head_dim}"); sys.stdout.flush()

# compress: TURBOQUANT UNIFORM (3-bit, no mixed precision)
# Rotations precomputed once, no per-head calibration
print("  Building TurboQuant uniform config..."); sys.stdout.flush()
t0 = time.time()
config = TurboQuantConfig(d=head_dim, b_mse=3, device=device,
    mixed_precision=False,  # UNIFORM — no mixed-precision overhead
    use_online_codebook=False)

# Precompute rotations and QJL matrices (called once per head)
rotations = [[config.make_rotation(l, h) for h in range(n_heads)] for l in range(n_layers)]
S_matrices = [[config.make_qjl_matrix(l, h) for h in range(n_heads)] for l in range(n_layers)]
print(f"  Precompute rotations+QJL: {time.time()-t0:.1f}s"); sys.stdout.flush()

print("  Compressing KV cache..."); sys.stdout.flush()
t0 = time.time()
rebuilt = []
for l in range(n_layers):
    new_k_parts, new_v_parts = [], []
    for h in range(n_heads):
        k_f = pkv.layers[l].keys[:, h:h+1, :, :].reshape(-1, head_dim).float()
        v_f = pkv.layers[l].values[:, h:h+1, :, :].reshape(-1, head_dim).float()
        kc = turboquant_encode_internal(k_f, config.codebook, rotations[l][h], S_matrices[l][h])
        vc = turboquant_encode_internal(v_f, config.codebook, rotations[l][h], S_matrices[l][h])
        # polarquant_decode returns float32; cast to original KV dtype (float16)
        orig_dtype = pkv.layers[l].keys.dtype
        kr = polarquant_decode(kc.pq)[..., :head_dim].reshape(1, 1, seq_len, head_dim).to(orig_dtype).contiguous()
        vr = polarquant_decode(vc.pq)[..., :head_dim].reshape(1, 1, seq_len, head_dim).to(orig_dtype).contiguous()
        new_k_parts.append(kr); new_v_parts.append(vr)
    new_k = torch.cat(new_k_parts, dim=1)
    new_v = torch.cat(new_v_parts, dim=1)
    rebuilt.append((new_k, new_v))
    if l % 6 == 0: print(f"    Layer {l}/{n_layers}: {time.time()-t0:.2f}s"); sys.stdout.flush()

torch.xpu.synchronize(); t_compress = time.time() - t0
vram_compressed = torch.xpu.memory_allocated() / 1e9
print(f"  Compress:  {t_compress:.1f}s  VRAM={vram_compressed:.2f}GB"); sys.stdout.flush()

# rebuild cache
cache = DynamicCache()
for l in range(n_layers):
    cache.update(rebuilt[l][0], rebuilt[l][1], layer_idx=l)

# first token
logits = prefill_out.logits[:, -1, :]
next_tok = logits.argmax(dim=-1, keepdim=True)
gen_ids = torch.cat([inp["input_ids"], next_tok], dim=-1)

# decode loop
print("  Decoding 10 tokens..."); sys.stdout.flush()
torch.xpu.synchronize(); t0 = time.time()
for i in range(10):
    with torch.no_grad():
        o = m(input_ids=next_tok, past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        next_tok = o.logits[:, -1:, :].argmax(dim=-1)
        gen_ids = torch.cat([gen_ids, next_tok], dim=-1)
torch.xpu.synchronize(); t_decode = time.time() - t0
vram_tq = torch.xpu.max_memory_allocated() / 1e9
print(f"  Decode:   {t_decode:.1f}s  ({(t_decode/10)*1000:.0f}ms/token)"); sys.stdout.flush()

# ── results ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RESULTS")
print("=" * 60)
print(f"\n  Model:        Qwen3-4B-4bit on Intel Arc B580")
print(f"  Context:      {len(tokens)} tokens")
print(f"  Generation:   10 new tokens")
print(f"\n  {'Pass':<20} {'Peak VRAM':<10} {'Time':<10}")
print(f"  {'-'*20} {'-'*10} {'-'*10}")
print(f"  {'Baseline':<20} {vram_base:<10.2f} {t_base:<10.1f}")
print(f"  {'TurboQuant':<20} {vram_tq:<10.2f} {t_prefill+t_compress+t_decode:<10.1f}")
print(f"\n  VRAM savings: {vram_base - vram_tq:.2f}GB  "
      f"({'BETTER' if vram_tq < vram_base else 'WORSE' if vram_tq > vram_base else 'SAME'})")
print(f"  Compress overhead: {t_compress*1000:.0f}ms")
print(f"\n  TurboQuant output:\n  {tok.decode(gen_ids[0])}")
print("\n" + "=" * 60)
